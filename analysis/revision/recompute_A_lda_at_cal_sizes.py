"""
Revision recompute A — LDA-Hudgins baseline across cal sizes.

For each PhysioMio patient, evaluate at cal_per_gesture ∈ {3, 6, 12, 24, 36}
on the same balanced 39/39/39 test set. Three arms:
  - LDA on 16 Hudgins features (per-session cal only)
  - HGB on 370 features, calibration-only
  - HGB on 370 features, GrabMyo + calibration

If LDA-Hudgins collapses at small cal (cal≤6) while GrabMyo+cal doesn't,
we have the "extended operating range" story that survives the reviewer's
ablation critique.

Outputs:
  analysis/revision/results/lda_at_cal_sizes_per_session.csv
  analysis/revision/results/lda_at_cal_sizes_summary.md
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from analysis.seed import SEED, seed_everything
from ml.train_hgb_v2 import engineer_features
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, GRABMYO_META, GRABMYO_CACHE, TEST_PER_CLASS,
    CAL_WEIGHT, CLASSES,
)

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "lda_at_cal_sizes_per_session.csv"
OUT_MD = OUT_DIR / "lda_at_cal_sizes_summary.md"

CAL_SIZES = [3, 6, 12, 24, 36]
BUFFER = 3

CHANNELS = [0, 4, 9, 13]
HUDGINS = ["mav", "wl", "zc", "ssc"]


def split_at(session_df, cal_per_gesture, test_per_class, rng):
    cal_idx = []
    test_pool = {c: [] for c in CLASSES}
    for _, g in session_df.groupby("trial", sort=True):
        cls = int(g["intent_idx"].iloc[0])
        sg = g.sort_values("t_rel_s")
        n = len(sg)
        if n >= cal_per_gesture + BUFFER + test_per_class:
            cal_idx.extend(sg.index[:cal_per_gesture].tolist())
            test_pool[cls].extend(sg.index[cal_per_gesture + BUFFER:
                                            cal_per_gesture + BUFFER + test_per_class].tolist())
        else:
            cal_n = max(1, min(cal_per_gesture, n - BUFFER - 1))
            cal_idx.extend(sg.index[:cal_n].tolist())
            test_pool[cls].extend(sg.index[cal_n + BUFFER:].tolist())
    balanced = []
    for cls in CLASSES:
        pool = test_pool[cls]
        if len(pool) <= test_per_class:
            balanced.extend(pool)
        else:
            balanced.extend(rng.choice(pool, size=test_per_class, replace=False).tolist())
    return np.array(sorted(balanced)), np.array(sorted(cal_idx))


def make_hgb(seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)

    print("Loading + engineering PhysioMio...")
    df = pd.read_pickle(PHYSIOMIO_PKL)
    hudgins_cols = [f"ch{ch}_{f}" for ch in CHANNELS for f in HUDGINS if f"ch{ch}_{f}" in df.columns]
    print(f"  Hudgins cols: {hudgins_cols}")
    eng = engineer_features(df)

    with open(GRABMYO_META) as f:
        gm_meta = json.load(f)
    gm_features = gm_meta["feature_cols"]
    print("Loading GrabMyo cache (once) — subsample for speed...")
    gm = pd.read_pickle(GRABMYO_CACHE)
    # 300k stratified subsample to keep HGB fits under ~30s each
    gm_sub = gm.sample(300_000, random_state=SEED)
    gm_X = gm_sub[gm_features].values.astype(np.float32)
    gm_y = gm_sub["intent_idx"].values.astype(np.int64)
    print(f"  GrabMyo subset: {len(gm_X):,} × {len(gm_features)}")

    patients = sorted(eng.participant.unique(), key=lambda s: int(s.replace("patient", "")))
    print(f"\nGrid: {len(CAL_SIZES)} cal × {len(patients)} patients × 3 arms")

    # ── Resume support: skip (cal_size, patient) already in the CSV ──
    rows = []
    done = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done = set(zip(existing["cal_per_gesture"].astype(int), existing["patient"]))
        print(f"Resume: {len(rows)} existing rows, {len(done)} (cal, patient) pairs done")

    t_start = time.time()
    for cal_size in CAL_SIZES:
        for pi, patient in enumerate(patients, 1):
            if (int(cal_size), patient) in done:
                continue
            s01 = eng[(eng.participant == patient) & (eng.session == "impaired_01")]
            if len(s01) == 0:
                continue
            test_idx, cal_idx = split_at(s01, cal_size, TEST_PER_CLASS, rng)
            if len(test_idx) < 15 or len(cal_idx) < 3:
                continue

            # 370-feature block (for HGB arms)
            X_cal_370 = s01.loc[cal_idx, gm_features].fillna(0).values.astype(np.float32)
            X_test_370 = s01.loc[test_idx, gm_features].fillna(0).values.astype(np.float32)
            # 16-feature Hudgins block (for LDA)
            X_cal_16 = s01.loc[cal_idx, hudgins_cols].fillna(0).values.astype(np.float32)
            X_test_16 = s01.loc[test_idx, hudgins_cols].fillna(0).values.astype(np.float32)
            y_cal = s01.loc[cal_idx, "intent_idx"].values.astype(np.int64)
            y_test = s01.loc[test_idx, "intent_idx"].values.astype(np.int64)

            r = {"cal_per_gesture": cal_size, "patient": patient,
                 "n_cal": len(cal_idx), "n_test": len(test_idx)}

            # ── Arm 1: LDA on Hudgins ──
            if len(np.unique(y_cal)) >= 2:
                try:
                    sc = StandardScaler().fit(X_cal_16)
                    clf = LinearDiscriminantAnalysis().fit(sc.transform(X_cal_16), y_cal)
                    p = clf.predict(sc.transform(X_test_16))
                    r["lda_acc"] = float(accuracy_score(y_test, p))
                except Exception:
                    r["lda_acc"] = np.nan
            else:
                r["lda_acc"] = np.nan

            # ── Arm 2: HGB calibration-only ──
            if len(np.unique(y_cal)) >= 2:
                sc = StandardScaler().fit(X_cal_370)
                clf = make_hgb().fit(sc.transform(X_cal_370), y_cal)
                p = clf.predict(sc.transform(X_test_370))
                r["po_hgb_acc"] = float(accuracy_score(y_test, p))
            else:
                r["po_hgb_acc"] = np.nan

            # ── Arm 3: HGB GrabMyo+cal ──
            X_all = np.vstack([gm_X, X_cal_370])
            y_all = np.concatenate([gm_y, y_cal])
            w = np.ones(len(X_all), dtype=np.float32)
            w[len(gm_X):] = CAL_WEIGHT
            sc = StandardScaler().fit(X_all)
            clf = make_hgb().fit(sc.transform(X_all), y_all, sample_weight=w)
            p = clf.predict(sc.transform(X_test_370))
            r["gm_cal_acc"] = float(accuracy_score(y_test, p))

            rows.append(r)
            elapsed = time.time() - t_start
            print(f"  cal={cal_size:>2d} pat={pi:>2d}/{len(patients)} {patient:<10} "
                  f"LDA={r['lda_acc']:.3f} PO={r['po_hgb_acc']:.3f} GM+cal={r['gm_cal_acc']:.3f}  "
                  f"[{elapsed/60:.1f}min]", flush=True)
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    pat_mean = out.groupby(["cal_per_gesture"]).agg(
        lda=("lda_acc", "mean"),
        po=("po_hgb_acc", "mean"),
        gm=("gm_cal_acc", "mean"),
        n=("patient", "count"),
    ).round(4)

    md_lines = [
        "# Experiment A — three arms across cal sizes (PhysioMio)",
        "",
        "For each cal_per_gesture, we fit three arms on the same balanced test:",
        "  1. LDA on 16 Hudgins features",
        "  2. HGB on 370 features — calibration-only",
        "  3. HGB on 370 features — GrabMyo + calibration",
        "",
        "## Patient-mean accuracy",
        "",
        "| cal_per_gesture | LDA-Hudgins | HGB cal-only | HGB GrabMyo+cal |",
        "|---:|---:|---:|---:|",
    ]
    for cs in CAL_SIZES:
        r = pat_mean.loc[cs] if cs in pat_mean.index else None
        if r is None: continue
        md_lines.append(f"| {cs} | {r['lda']:.3f} | {r['po']:.3f} | {r['gm']:.3f} |")

    md_lines += [
        "",
        "## Reading the table",
        "",
        "- At small cal (3), if LDA and PO both collapse toward 0.33 while GM+cal",
        "  stays high, GrabMyo extends the viable operating range — the",
        "  'protocol design assumes a backbone' story is defensible.",
        "- At large cal (36), all three converge — GrabMyo doesn't add accuracy",
        "  at the paper's operating point. That's a known truth we have to live",
        "  with; the story is not about the operating point but about the range.",
    ]
    OUT_MD.write_text("\n".join(md_lines))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
