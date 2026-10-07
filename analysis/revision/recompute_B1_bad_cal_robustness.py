"""
Revision B1 — bad-cal robustness.

For each PhysioMio impaired-arm session_01, corrupt the calibration data in
two realistic ways and see whether GM+cal degrades more gracefully than
cal-only HGB.

Corruption modes:
  1. NOISE  — inject Gaussian noise scaled to per-feature std, σ ∈ {0, 0.5, 1, 2}
  2. DROP   — remove one class from cal entirely (test set still balanced across all 3)

If GM+cal keeps working when cal-only collapses, GrabMyo has a real reason to
exist beyond "extends operating range to sub-2s cal budgets" (which nobody wants).

Outputs:
  analysis/revision/results/bad_cal_robustness_per_session.csv
  analysis/revision/results/bad_cal_robustness_summary.md
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
OUT_CSV = OUT_DIR / "bad_cal_robustness_per_session.csv"
OUT_MD = OUT_DIR / "bad_cal_robustness_summary.md"

NOISE_SIGMAS = [0.0, 0.5, 1.0, 2.0]
DROP_CLASSES = [None, 0, 1, 2]   # None = full 3-class cal (baseline)


def make_hgb(seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def train_and_score(X_cal, y_cal, X_test, y_test, gm_X=None, gm_y=None):
    if len(np.unique(np.concatenate([y_cal, gm_y if gm_y is not None else y_cal]))) < 2:
        return np.nan
    if gm_X is not None:
        X_all = np.vstack([gm_X, X_cal])
        y_all = np.concatenate([gm_y, y_cal])
        w = np.ones(len(X_all), dtype=np.float32)
        w[len(gm_X):] = CAL_WEIGHT
    else:
        X_all, y_all, w = X_cal, y_cal, None
        if len(np.unique(y_all)) < 2:
            return np.nan
    sc = StandardScaler().fit(X_all)
    clf = make_hgb().fit(sc.transform(X_all), y_all, sample_weight=w)
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

    # ── Resume support: skip patients already fully processed ──
    # Each patient generates up to 4 noise + 4 drop = 8 rows. We treat a patient
    # as done if they have >= 7 rows (allowing for one skipped drop-class config).
    # Partial-patient rows are discarded so the patient re-runs cleanly.
    rows = []
    done = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        counts = existing.groupby("patient").size()
        done = set(counts[counts >= 7].index)
        keep = existing[existing.patient.isin(done)]
        rows = keep.to_dict("records")
        dropped = len(existing) - len(keep)
        print(f"Resume: {len(rows)} kept rows from {len(done)} fully-done patients "
              f"({dropped} partial rows discarded)")

    t_start = time.time()
    patients = sorted(eng.participant.unique(), key=lambda s: int(s.replace("patient", "")))
    for pi, patient in enumerate(patients, 1):
        if patient in done:
            continue
        s01 = eng[(eng.participant == patient) & (eng.session == "impaired_01")]
        if len(s01) == 0:
            continue
        try:
            test_idx, cal_idx, _ = split_session(s01, TEST_PER_CLASS, rng)
        except Exception:
            continue
        if len(test_idx) < 15 or len(cal_idx) < 6:
            continue

        X_cal_full = s01.loc[cal_idx, gm_features].fillna(0).values.astype(np.float32)
        y_cal_full = s01.loc[cal_idx, "intent_idx"].values.astype(np.int64)
        X_test = s01.loc[test_idx, gm_features].fillna(0).values.astype(np.float32)
        y_test = s01.loc[test_idx, "intent_idx"].values.astype(np.int64)

        # ── NOISE MODE ──
        for sigma in NOISE_SIGMAS:
            noise_rng = np.random.RandomState(abs(hash((patient, float(sigma)))) & 0xffffffff)
            X_cal = X_cal_full.copy()
            if sigma > 0:
                std_per_feat = X_cal.std(axis=0) + 1e-6
                X_cal = X_cal + (noise_rng.randn(*X_cal.shape).astype(np.float32) * std_per_feat * sigma)
            po = train_and_score(X_cal, y_cal_full, X_test, y_test)
            gm_arm = train_and_score(X_cal, y_cal_full, X_test, y_test, gm_X, gm_y)
            rows.append({
                "patient": patient, "mode": "noise", "level": float(sigma),
                "po_acc": po, "gm_acc": gm_arm,
                "delta": (gm_arm - po) if not (np.isnan(gm_arm) or np.isnan(po)) else np.nan,
            })

        # ── DROP-CLASS MODE ──
        for drop in DROP_CLASSES:
            if drop is None:
                X_cal, y_cal = X_cal_full, y_cal_full
                level_str = "full"
            else:
                mask = y_cal_full != drop
                if mask.sum() < 3 or len(np.unique(y_cal_full[mask])) < 2:
                    continue
                X_cal, y_cal = X_cal_full[mask], y_cal_full[mask]
                level_str = f"drop_{drop}"
            po = train_and_score(X_cal, y_cal, X_test, y_test)
            gm_arm = train_and_score(X_cal, y_cal, X_test, y_test, gm_X, gm_y)
            rows.append({
                "patient": patient, "mode": "drop", "level": level_str,
                "po_acc": po, "gm_acc": gm_arm,
                "delta": (gm_arm - po) if not (np.isnan(gm_arm) or np.isnan(po)) else np.nan,
            })

        elapsed = time.time() - t_start
        eta = elapsed / pi * len(patients) - elapsed
        print(f"[{pi}/{len(patients)}] {patient}  elapsed={elapsed/60:.1f}min  eta={eta/60:.0f}min",
              flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    lines = [
        "# Experiment B1 — bad-cal robustness",
        "",
        "For each of 48 patients, session_01 impaired. Corrupt cal, evaluate on clean balanced test.",
        "",
        "Arms:",
        "- **PO** = HGB with per-session cal only (no GrabMyo)",
        "- **GM** = HGB with GrabMyo + per-session cal (paper's method)",
        "- **Δ** = GM − PO. Positive means GrabMyo helps under this corruption.",
        "",
    ]
    for mode in ["noise", "drop"]:
        sub = out[out["mode"] == mode]
        if len(sub) == 0:
            continue
        agg = sub.groupby("level").agg(
            po_mean=("po_acc", "mean"),
            gm_mean=("gm_acc", "mean"),
            delta_mean=("delta", "mean"),
            n=("patient", "count"),
        ).round(4)
        lines += [f"## Mode: {mode}", "", agg.to_markdown(), ""]
    lines += [
        "## Reading",
        "",
        "If Δ is near zero across all corruption levels, GrabMyo doesn't buy",
        "robustness — the ablation story dies. If Δ grows with corruption level",
        "(GM degrades more gracefully than PO), we have a defensible clinical",
        "claim: 'GrabMyo is a safety net for cal-quality failures in deployment.'",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
