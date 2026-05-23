"""
PhysioMio longitudinal eval — Stream 5.

The question: does calibration done at session 1 still work at sessions 2..N?
If yes: per-session recalibration may be overkill; one-time enrollment cal
suffices for ongoing clinical use. If no: per-session recal is necessary, as
the deployed system assumes.

Protocol per patient:
  1. Build cal data from the patient's FIRST impaired-arm session (impaired_01)
     using the same per-gesture temporal split as per_session_eval.py — 432 cal
     windows pre-buffer (36 cal + 3 buffer + 39 test per gesture).
  2. Train one HGB on (GrabMyo, w=1) + (impaired_01 cal, w=100×).
  3. For EVERY session of this patient (healthy_01, healthy_02, impaired_01..N):
     extract that session's BALANCED test set (same logic as per_session_eval —
     39 per class via temporal split). Predict, record accuracy + per-class F1.
     Note: when test session == cal session, the test set is on data the model
     hasn't seen (temporal buffer guarantees no signal leakage); this row is
     the "same-session" baseline for comparison.

Per patient: 1 HGB fit (~3.5 min) + many predictions (~0.5 min total).
Full run: 48 patients ≈ ~3 hours wall time.

Output:
  analysis/physiomio/results/longitudinal_per_session.csv
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from analysis.seed import SEED, seed_everything
from ml.train_hgb_v2 import META_COLS, engineer_features


PHYSIOMIO_PKL = PROJECT_ROOT / "data" / "physiomio_features_60_per_patient.pkl"
GRABMYO_CACHE = PROJECT_ROOT / "analysis" / ".cache" / "grabmyo_features_370.pkl"
GRABMYO_META = PROJECT_ROOT / "grabmyo" / "improved_hgb_meta.json"

OUT_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "longitudinal_per_session.csv"

TEST_PER_CLASS = 39
CAL_WEIGHT = 100.0
CLASSES = [0, 1, 2]


def make_classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=300, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        class_weight="balanced",
    )


def temporal_split(session_df: pd.DataFrame, test_per_class: int, rng: np.random.RandomState) -> tuple:
    """Same per-gesture temporal split as per_session_eval.split_session."""
    cal_idx = []
    test_pool_by_class = {0: [], 1: [], 2: []}
    for _trial_id, gesture_group in session_df.groupby("trial", sort=True):
        cls = int(gesture_group["intent_idx"].iloc[0])
        sorted_group = gesture_group.sort_values("t_rel_s")
        n = len(sorted_group)
        if n >= 78:
            cal_idx.extend(sorted_group.index[:36].tolist())
            test_pool_by_class[cls].extend(sorted_group.index[39:78].tolist())
        elif n >= 8:
            cal_n = (n - 3) // 2
            cal_idx.extend(sorted_group.index[:cal_n].tolist())
            test_pool_by_class[cls].extend(sorted_group.index[cal_n + 3:].tolist())
        else:
            cal_idx.extend(sorted_group.index.tolist())
    balanced_test_idx = []
    for cls in CLASSES:
        pool = test_pool_by_class[cls]
        if len(pool) <= test_per_class:
            balanced_test_idx.extend(pool)
        else:
            balanced_test_idx.extend(rng.choice(pool, size=test_per_class, replace=False).tolist())
    return np.array(sorted(balanced_test_idx)), np.array(sorted(cal_idx))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", nargs="*", default=None)
    parser.add_argument("--patient-only", action="store_true",
                        help="Train on impaired_01 cal data ONLY (no GrabMyo). "
                             "Tests whether the GrabMyo prior regularises across sessions "
                             "or is decoration. Writes to longitudinal_patient_only.csv.")
    args = parser.parse_args()
    out_csv = (OUT_CSV.parent / "longitudinal_patient_only.csv") if args.patient_only else OUT_CSV

    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    t_start = time.time()

    print("=" * 70)
    print("PhysioMio longitudinal eval — cal on impaired_01, test all other sessions")
    print("=" * 70)

    print(f"\n[1/3] Loading + engineering PhysioMio (60 → 370)...")
    physiomio = pd.read_pickle(PHYSIOMIO_PKL)
    t = time.time()
    physiomio_eng = engineer_features(physiomio)
    print(f"      done in {time.time() - t:.1f}s  shape: {physiomio_eng.shape}")

    print(f"\n[2/3] Loading GrabMyo cache...")
    grabmyo_eng = pd.read_pickle(GRABMYO_CACHE)
    with open(GRABMYO_META) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    grabmyo_X = grabmyo_eng[feature_cols].values.astype(np.float32)
    grabmyo_y = grabmyo_eng["intent_idx"].values.astype(np.int64)
    print(f"      shape: {grabmyo_eng.shape}  features: {len(feature_cols)}")

    all_patients = sorted(
        physiomio_eng["participant"].unique(),
        key=lambda s: int(s.replace("patient", "")),
    )
    patients_to_run = args.patients if args.patients else all_patients
    print(f"\n[3/3] Longitudinal eval on {len(patients_to_run)} patients...")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for pi, patient in enumerate(patients_to_run, 1):
        p_data = physiomio_eng[physiomio_eng["participant"] == patient]
        sessions = sorted(p_data["session"].unique())

        impaired_sessions = [s for s in sessions if s.startswith("impaired_")]
        if len(impaired_sessions) < 1:
            print(f"\n  [{pi}/{len(patients_to_run)}] {patient}: no impaired sessions, skipping")
            continue
        cal_session_name = impaired_sessions[0]  # impaired_01

        cal_session_df = p_data[p_data["session"] == cal_session_name].copy()
        # Use a session-local rng for reproducibility
        local_rng = np.random.RandomState(SEED + pi)
        _, cal_idx = temporal_split(cal_session_df, TEST_PER_CLASS, local_rng)

        X_cal = cal_session_df.loc[cal_idx, feature_cols].values.astype(np.float32)
        y_cal = cal_session_df.loc[cal_idx, "intent_idx"].values.astype(np.int64)

        if args.patient_only:
            # Train on cal data ONLY — no GrabMyo
            X_all = X_cal
            y_all = y_cal
            w_all = None
        else:
            # Combine GrabMyo + cal
            X_all = np.vstack([grabmyo_X, X_cal])
            y_all = np.concatenate([grabmyo_y, y_cal])
            w_all = np.ones(len(X_all), dtype=np.float32)
            w_all[len(grabmyo_X):] = CAL_WEIGHT

        scaler = StandardScaler()
        X_all_s = scaler.fit_transform(X_all)

        clf = make_classifier(SEED)
        t0 = time.time()
        clf.fit(X_all_s, y_all, sample_weight=w_all)
        fit_time = time.time() - t0
        print(f"\n  [{pi}/{len(patients_to_run)}] {patient}: cal trained on {cal_session_name} "
              f"({len(X_cal)} windows), fit={fit_time:.0f}s")

        # Per session: extract that session's balanced test set, predict
        for ti, test_session in enumerate(sessions):
            test_session_df = p_data[p_data["session"] == test_session].copy()
            test_rng = np.random.RandomState(SEED + pi * 1000 + ti)
            test_idx, _ = temporal_split(test_session_df, TEST_PER_CLASS, test_rng)
            if len(test_idx) == 0:
                continue
            X_test = test_session_df.loc[test_idx, feature_cols].values.astype(np.float32)
            y_test = test_session_df.loc[test_idx, "intent_idx"].values.astype(np.int64)
            X_test_s = scaler.transform(X_test).astype(np.float32)
            preds = clf.predict(X_test_s)
            acc = accuracy_score(y_test, preds)
            f1m = f1_score(y_test, preds, average="macro", zero_division=0)
            cls_f1 = f1_score(y_test, preds, average=None, labels=CLASSES, zero_division=0)
            arm = test_session.split("_")[0]
            is_same_session = (test_session == cal_session_name)
            is_same_arm = (arm == "impaired")

            # Within-impaired session distance (only meaningful when arm == impaired)
            impaired_dist = None
            if is_same_arm:
                impaired_dist = int(test_session.split("_")[1]) - int(cal_session_name.split("_")[1])

            rows.append({
                "patient": patient,
                "cal_session": cal_session_name,
                "test_session": test_session,
                "arm": arm,
                "is_same_session": is_same_session,
                "impaired_session_distance": impaired_dist,
                "n_test_windows": int(len(test_idx)),
                "acc": acc,
                "f1_macro": f1m,
                "f1_rest": cls_f1[0],
                "f1_close": cls_f1[1],
                "f1_open": cls_f1[2],
            })
            tag = " (SAME)" if is_same_session else ""
            dist_str = f"  dist=+{impaired_dist}" if impaired_dist is not None else ""
            print(f"      {test_session:>15s}{tag} ({arm}){dist_str}: acc={acc:.4f}  f1={f1m:.4f}")

        # Save incrementally
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print(f"AGGREGATE ({len(df)} test-session evals from {df['patient'].nunique()} patients)")
    print("=" * 70)
    # By impaired session distance
    imp = df[df["arm"] == "impaired"]
    by_dist = imp.groupby("impaired_session_distance").agg(
        n=("acc", "count"),
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
    ).round(4)
    print(f"\nImpaired-arm accuracy vs session distance from cal (impaired_01):")
    print(by_dist.to_string())
    # Healthy-arm transfer
    h = df[df["arm"] == "healthy"]
    if len(h) > 0:
        print(f"\nHealthy-arm sessions (cross-arm from impaired_01 cal):")
        for s, g in h.groupby("test_session"):
            print(f"  {s}: n={len(g)}  acc={g['acc'].mean():.4f}  std={g['acc'].std():.4f}")
    print(f"\nTotal wall: {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
