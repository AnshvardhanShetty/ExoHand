"""
Revision — C4 leakage-free rerun: GrabMyo weight sweep with leakage-free features
and frozen splits, so cal-only baseline matches the leakage-free ladder's Row 1.

Same weights, same protocol, same statistical test as recompute_C4_grabmyo_weight_sweep.py.
Only change: features go through engineer_features_leakage_free() with per-patient
cal_mask from frozen_splits.parquet.

Outputs:
  analysis/revision/results/C4_leakage_free_per_patient.csv
  analysis/revision/results/C4_leakage_free_summary.md
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

from analysis.seed import SEED, seed_everything
from analysis.physiomio.per_session_eval import PHYSIOMIO_PKL, GRABMYO_META, GRABMYO_CACHE
from analysis.revision.recompute_leakage_free_zscore import engineer_features_leakage_free

FROZEN = PROJECT_ROOT / "analysis" / "revision" / "frozen_splits.parquet"
OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "C4_leakage_free_per_patient.csv"
OUT_MD = OUT_DIR / "C4_leakage_free_summary.md"

WEIGHTS = [0.0, 1.0, 10.0, 100.0, 1000.0]


def make_hgb(seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def fit_score(X_gm, y_gm, X_cal, y_cal, X_test, y_test, weight):
    if weight == 0.0:
        if len(np.unique(y_cal)) < 2:
            return np.nan
        sc = StandardScaler().fit(X_cal)
        clf = make_hgb().fit(sc.transform(X_cal), y_cal)
    else:
        X_all = np.vstack([X_gm, X_cal])
        y_all = np.concatenate([y_gm, y_cal])
        w = np.ones(len(X_all), dtype=np.float32)
        w[len(X_gm):] = float(weight)
        sc = StandardScaler().fit(X_all)
        clf = make_hgb().fit(sc.transform(X_all), y_all, sample_weight=w)
    return float(accuracy_score(y_test, clf.predict(sc.transform(X_test))))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    t0 = time.time()

    print("Loading PhysioMio raw + leakage-free engineering...")
    df = pd.read_pickle(PHYSIOMIO_PKL).reset_index(drop=True)

    frozen = pd.read_parquet(FROZEN)
    cal_mask = pd.Series(False, index=df.index)
    per_patient = {}
    for _, r in frozen.iterrows():
        cal_mask.loc[r["cal_idx"]] = True
        per_patient.setdefault(r["patient"], {})[r["session"]] = {
            "cal_idx": list(r["cal_idx"]),
            "test_idx": list(r["test_idx"]),
        }

    df_eng = engineer_features_leakage_free(df.copy(), cal_mask)

    with open(GRABMYO_META) as f:
        gm_features = json.load(f)["feature_cols"]

    print("Loading GrabMyo (300k subsample)...")
    gm = pd.read_pickle(GRABMYO_CACHE).sample(300_000, random_state=SEED)
    gm_X = gm[gm_features].values.astype(np.float32)
    gm_y = gm["intent_idx"].values.astype(np.int64)

    rows = []
    done = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        counts = existing.groupby("patient").size()
        done = set(counts[counts >= len(WEIGHTS)].index)
        rows = existing[existing.patient.isin(done)].to_dict("records")
        print(f"Resume: {len(rows)} rows, {len(done)} patients done")

    patients = sorted([p for p in per_patient if "impaired_01" in per_patient[p]],
                        key=lambda s: int(s.replace("patient", "")))

    for pi, patient in enumerate(patients, 1):
        if patient in done:
            continue
        meta = per_patient[patient]["impaired_01"]
        cal_idx = meta["cal_idx"]
        test_idx = meta["test_idx"]
        if len(cal_idx) < 6 or len(test_idx) < 15:
            continue

        X_cal = df_eng.loc[cal_idx, gm_features].fillna(0).values.astype(np.float32)
        y_cal = df_eng.loc[cal_idx, "intent_idx"].values.astype(np.int64)
        X_test = df_eng.loc[test_idx, gm_features].fillna(0).values.astype(np.float32)
        y_test = df_eng.loc[test_idx, "intent_idx"].values.astype(np.int64)

        for w in WEIGHTS:
            try:
                acc = fit_score(gm_X, gm_y, X_cal, y_cal, X_test, y_test, w)
            except Exception as e:
                print(f"  {patient} w={w}: {e}", flush=True)
                acc = np.nan
            rows.append({"patient": patient, "weight": w, "acc": acc,
                         "n_cal": len(X_cal), "n_test": len(X_test)})

        elapsed = time.time() - t0
        eta = elapsed / max(1, pi - len(done)) * (len(patients) - pi)
        accs_this = [r["acc"] for r in rows if r["patient"] == patient]
        print(f"[{pi}/{len(patients)}] {patient}: cal-only={accs_this[0]:.4f}  "
              f"100x={accs_this[3]:.4f}  [{elapsed/60:.1f}min, eta {eta/60:.0f}min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return

    pivot = out.pivot(index="patient", columns="weight", values="acc")
    md = [
        "# C4 leakage-free — GrabMyo weight sweep",
        "",
        f"n = {len(pivot)} patients, cal_per_gesture=36 (paper operating point).",
        "Uses leakage-free features + frozen splits (same pipeline as leakage_free_ladder).",
        "Weight = cal-weight multiplier vs. GrabMyo (1×); 0 = cal-only baseline.",
        "",
        "## Ladder",
        "",
        "| Weight (× GrabMyo) | Mean acc | Median acc | vs cal-only Δ | Wilcoxon p |",
        "|---:|---:|---:|---:|---:|",
    ]
    for w in WEIGHTS:
        col = pivot[w].dropna()
        if w == 0.0:
            md.append(f"| **0 (cal-only)** | {col.mean():.4f} | {col.median():.4f} | — | — |")
        else:
            paired = pivot[[0.0, w]].dropna()
            delta = paired[w].mean() - paired[0.0].mean()
            try:
                wtest = wilcoxon(paired[w], paired[0.0], alternative="greater")
                pval = f"{wtest.pvalue:.3e}"
            except Exception:
                pval = "—"
            md.append(f"| {w:g} | {col.mean():.4f} | {col.median():.4f} | "
                      f"{delta:+.4f} | {pval} |")

    md += [
        "",
        "## Comparison to leakage-contaminated C4 (recompute_C4_grabmyo_weight_sweep.py)",
        "",
        "The older sweep reported cal-only = 0.878; the leakage-free version should reproduce",
        "row 1 of the leakage-free ladder (~0.896). If baselines match, the paired result is",
        "confirmed leakage-invariant. If ordering flips at any weight, we investigate.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
