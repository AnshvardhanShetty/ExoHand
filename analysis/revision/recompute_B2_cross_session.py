"""
Revision B2 — cross-session generalisation.

Clinical failure mode: patient returns for session N+1 with electrodes replaced
and cal drift is real. Question: does GM+cal generalise across sessions better
than cal-only HGB?

For each patient with impaired_01 AND impaired_02:
  - within-01: fit on session_01 cal, test on session_01 test set
  - cross-01→02: fit on session_01 cal, test on session_02 test set
  - within-02: fit on session_02 cal, test on session_02 test set
  - cross-02→01: fit on session_02 cal, test on session_01 test set

Two arms: HGB cal-only vs HGB GrabMyo+cal.

The key comparison: (within − cross) drop per arm. Smaller drop for GM+cal
would mean GrabMyo cushions electrode-replacement drift.

Outputs:
  analysis/revision/results/cross_session_per_patient.csv
  analysis/revision/results/cross_session_summary.md
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

from analysis.seed import SEED, seed_everything
from ml.train_hgb_v2 import engineer_features
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, GRABMYO_META, GRABMYO_CACHE, TEST_PER_CLASS,
    CAL_WEIGHT, CLASSES, split_session,
)

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "cross_session_per_patient.csv"
OUT_MD = OUT_DIR / "cross_session_summary.md"


def make_hgb(seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def fit_arm(X_cal, y_cal, gm_X=None, gm_y=None):
    """Return (scaler, classifier) tuple for the given arm."""
    if gm_X is not None:
        X_all = np.vstack([gm_X, X_cal])
        y_all = np.concatenate([gm_y, y_cal])
        w = np.ones(len(X_all), dtype=np.float32)
        w[len(gm_X):] = CAL_WEIGHT
    else:
        X_all, y_all, w = X_cal, y_cal, None
    if len(np.unique(y_all)) < 2:
        return None
    sc = StandardScaler().fit(X_all)
    clf = make_hgb().fit(sc.transform(X_all), y_all, sample_weight=w)
    return sc, clf


def score(model, X_test, y_test):
    if model is None:
        return np.nan
    sc, clf = model
    return float(accuracy_score(y_test, clf.predict(sc.transform(X_test))))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)

    print("Loading + engineering PhysioMio...")
    df = pd.read_pickle(PHYSIOMIO_PKL)
    eng = engineer_features(df)

    with open(GRABMYO_META) as f:
        gm_meta = json.load(f)
    gm_features = gm_meta["feature_cols"]

    print("Loading GrabMyo (300k subsample)...")
    gm = pd.read_pickle(GRABMYO_CACHE).sample(300_000, random_state=SEED)
    gm_X = gm[gm_features].values.astype(np.float32)
    gm_y = gm["intent_idx"].values.astype(np.int64)
    print(f"  GrabMyo: {len(gm_X):,} × {len(gm_features)}")

    # Only patients with BOTH impaired_01 and impaired_02
    have_both = (
        eng[eng.session.isin(["impaired_01", "impaired_02"])]
        .groupby("participant").session.nunique()
        .pipe(lambda s: s[s == 2].index.tolist())
    )
    patients = sorted(have_both, key=lambda s: int(s.replace("patient", "")))
    print(f"Patients with both impaired_01 and impaired_02: {len(patients)}")

    # ── Resume support: skip patients already in the CSV ──
    rows = []
    done = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done = set(existing["patient"].tolist())
        print(f"Resume: {len(rows)} existing rows, {len(done)} patients done")

    t_start = time.time()
    for pi, patient in enumerate(patients, 1):
        if patient in done:
            continue
        session_data = {}
        skip = False
        for sess in ["impaired_01", "impaired_02"]:
            s = eng[(eng.participant == patient) & (eng.session == sess)]
            try:
                test_idx, cal_idx, _ = split_session(s, TEST_PER_CLASS, rng)
            except Exception:
                skip = True
                break
            if len(test_idx) < 15 or len(cal_idx) < 6:
                skip = True
                break
            session_data[sess] = {
                "X_cal": s.loc[cal_idx, gm_features].fillna(0).values.astype(np.float32),
                "y_cal": s.loc[cal_idx, "intent_idx"].values.astype(np.int64),
                "X_test": s.loc[test_idx, gm_features].fillna(0).values.astype(np.float32),
                "y_test": s.loc[test_idx, "intent_idx"].values.astype(np.int64),
            }
        if skip:
            continue

        # Fit 4 models (2 sessions × 2 arms)
        s01, s02 = session_data["impaired_01"], session_data["impaired_02"]
        po_01 = fit_arm(s01["X_cal"], s01["y_cal"])
        po_02 = fit_arm(s02["X_cal"], s02["y_cal"])
        gm_01 = fit_arm(s01["X_cal"], s01["y_cal"], gm_X, gm_y)
        gm_02 = fit_arm(s02["X_cal"], s02["y_cal"], gm_X, gm_y)

        row = {"patient": patient}
        # Within-session
        row["po_within_01"] = score(po_01, s01["X_test"], s01["y_test"])
        row["po_within_02"] = score(po_02, s02["X_test"], s02["y_test"])
        row["gm_within_01"] = score(gm_01, s01["X_test"], s01["y_test"])
        row["gm_within_02"] = score(gm_02, s02["X_test"], s02["y_test"])
        # Cross-session
        row["po_cross_01to02"] = score(po_01, s02["X_test"], s02["y_test"])
        row["po_cross_02to01"] = score(po_02, s01["X_test"], s01["y_test"])
        row["gm_cross_01to02"] = score(gm_01, s02["X_test"], s02["y_test"])
        row["gm_cross_02to01"] = score(gm_02, s01["X_test"], s01["y_test"])

        rows.append(row)
        elapsed = time.time() - t_start
        eta = elapsed / pi * len(patients) - elapsed
        print(f"[{pi}/{len(patients)}] {patient}  "
              f"PO within={np.mean([row['po_within_01'], row['po_within_02']]):.3f} "
              f"cross={np.mean([row['po_cross_01to02'], row['po_cross_02to01']]):.3f}  "
              f"GM within={np.mean([row['gm_within_01'], row['gm_within_02']]):.3f} "
              f"cross={np.mean([row['gm_cross_01to02'], row['gm_cross_02to01']]):.3f}  "
              f"[{elapsed/60:.1f}min eta={eta/60:.0f}min]",
              flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    po_within = out[["po_within_01", "po_within_02"]].mean(axis=1)
    po_cross = out[["po_cross_01to02", "po_cross_02to01"]].mean(axis=1)
    gm_within = out[["gm_within_01", "gm_within_02"]].mean(axis=1)
    gm_cross = out[["gm_cross_01to02", "gm_cross_02to01"]].mean(axis=1)
    po_drop = po_within - po_cross
    gm_drop = gm_within - gm_cross

    from scipy.stats import wilcoxon
    w = wilcoxon(gm_drop, po_drop, alternative="less")  # GM drop LESS than PO drop → GrabMyo helps

    md = [
        "# Experiment B2 — cross-session generalisation",
        "",
        f"Patients with both impaired_01 and impaired_02: {len(out)}",
        "",
        "| arm | within-session | cross-session | drop |",
        "|---|---:|---:|---:|",
        f"| HGB cal-only | {po_within.mean():.4f} | {po_cross.mean():.4f} | {po_drop.mean():+.4f} |",
        f"| HGB GrabMyo+cal | {gm_within.mean():.4f} | {gm_cross.mean():.4f} | {gm_drop.mean():+.4f} |",
        "",
        f"Paired Wilcoxon (GM drop < PO drop, one-sided): p = {w.pvalue:.4f}",
        "",
        "## Reading",
        "",
        "If GM's cross-session drop is meaningfully smaller than PO's (and p < 0.05),",
        "GrabMyo genuinely cushions across-session drift — a real clinical claim about",
        "electrode replacement / cal drift, not about sub-2s cal budgets. If drops are",
        "similar, GrabMyo doesn't help here either and the ablation story doesn't survive.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
