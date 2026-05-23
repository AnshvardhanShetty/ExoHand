"""
Stream 2b with-calibration eval — variant (e) protocol on PhysioMio.

For each PhysioMio patient:
  1. Take 1200 windows (~60 s) from the patient's first healthy_arm session
     as the calibration set. Stratified by intent so all 3 classes are present.
  2. Concatenate GrabMyo training data + calibration set; weight calibration 100x.
  3. Train fresh HistGradientBoostingClassifier in --fast mode.
  4. Predict on the rest of the patient's data (all other sessions + remaining
     windows of the first healthy session).
  5. Per-patient + per-session metrics.

Pairs with zero_shot_eval.py: same per-patient row schema (acc_no_cal +
acc_with_cal + delta_acc), so aggregate_loso.py can run on the merged CSV.

Usage:
    python analysis/physiomio/with_cal_eval.py                       # all 48 patients
    python analysis/physiomio/with_cal_eval.py --patients patient1   # smoke test
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
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

OUT_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "with_cal_per_patient.csv"

CALIB_N_WINDOWS = 1200      # variant (e) — ~60 s
CALIB_WEIGHT = 100.0


def make_classifier(seed: int) -> HistGradientBoostingClassifier:
    """--fast HGB matching analysis/reproduce_headline/loso_eval.py."""
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=300, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        class_weight="balanced",
    )


def stratified_indices(y: np.ndarray, n_target: int, rng: np.random.RandomState) -> np.ndarray:
    classes = np.unique(y)
    per_class = max(1, n_target // len(classes))
    picked = []
    for c in classes:
        idx = np.where(y == c)[0]
        take = min(per_class, len(idx))
        picked.extend(rng.choice(idx, size=take, replace=False))
    picked = list(set(picked))
    if len(picked) < n_target:
        leftover = np.setdiff1d(np.arange(len(y)), picked)
        topup = min(n_target - len(picked), len(leftover))
        if topup > 0:
            picked.extend(rng.choice(leftover, size=topup, replace=False))
    return np.array(sorted(picked))


def eval_one_patient(
    patient: str,
    physiomio_eng: pd.DataFrame,
    grabmyo_eng: pd.DataFrame,
    feature_cols: list,
    seed: int,
) -> dict:
    p_data = physiomio_eng[physiomio_eng["participant"] == patient]
    if len(p_data) == 0:
        return None

    # Calibration: stratified slice from this patient's FIRST healthy session
    healthy_first = p_data[p_data["session"] == "healthy_01"]
    if len(healthy_first) == 0:
        print(f"  WARN: {patient} has no healthy_01 session; skipping")
        return None

    rng = np.random.RandomState(seed)
    y_h = healthy_first["intent_idx"].values
    calib_n = min(CALIB_N_WINDOWS, len(healthy_first) // 2)
    calib_local = stratified_indices(y_h, calib_n, rng)
    calib_global_idx = healthy_first.index[calib_local]

    calib_set = p_data.loc[calib_global_idx]
    test_set = p_data.drop(calib_global_idx)
    # Also drop windows that are FROM the same trials as calibration to avoid
    # within-trial leakage. (Each trial has ~78 windows; if any are in the
    # calibration set, the rest of that trial is contaminated.)
    calib_trials = set(zip(calib_set["session"], calib_set["trial"]))
    leak_mask = test_set.apply(lambda r: (r["session"], r["trial"]) in calib_trials, axis=1)
    if leak_mask.any():
        test_set = test_set[~leak_mask]

    # Assemble training: GrabMyo + calibration
    X_train_g = grabmyo_eng[feature_cols].values.astype(np.float32)
    y_train_g = grabmyo_eng["intent_idx"].values.astype(np.int64)
    X_train_c = calib_set[feature_cols].values.astype(np.float32)
    y_train_c = calib_set["intent_idx"].values.astype(np.int64)

    X_all = np.vstack([X_train_g, X_train_c])
    y_all = np.concatenate([y_train_g, y_train_c])
    w_all = np.ones(len(X_all), dtype=np.float32)
    w_all[len(X_train_g):] = CALIB_WEIGHT

    # Scale (refit per patient — captures the combined distribution)
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all)
    X_test = test_set[feature_cols].values.astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    y_test = test_set["intent_idx"].values.astype(np.int64)

    # Train + predict
    clf = make_classifier(seed)
    t0 = time.time()
    clf.fit(X_all_s, y_all, sample_weight=w_all)
    fit_time = time.time() - t0
    preds = clf.predict(X_test_s)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    cls_f1 = f1_score(y_test, preds, average=None, labels=[0, 1, 2], zero_division=0)

    # Per-arm
    test_set = test_set.copy()
    test_set["pred"] = preds
    healthy = test_set[test_set["session"].str.startswith("healthy_")]
    impaired = test_set[test_set["session"].str.startswith("impaired_")]
    acc_h = accuracy_score(healthy["intent_idx"], healthy["pred"]) if len(healthy) else np.nan
    acc_i = accuracy_score(impaired["intent_idx"], impaired["pred"]) if len(impaired) else np.nan

    return {
        "participant": patient,
        "n_calib_windows": int(calib_n),
        "n_test_windows": int(len(test_set)),
        "n_test_healthy": int(len(healthy)),
        "n_test_impaired": int(len(impaired)),
        "acc_with_cal": acc,
        "f1_with_cal": f1,
        "acc_healthy_arm_cal": acc_h,
        "acc_impaired_arm_cal": acc_i,
        "f1_rest_cal": cls_f1[0],
        "f1_close_cal": cls_f1[1],
        "f1_open_cal": cls_f1[2],
        "fit_time_s": fit_time,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", nargs="*", default=None)
    parser.add_argument("--out", default=str(OUT_CSV))
    args = parser.parse_args()

    seed_everything(SEED)
    t_start = time.time()

    print("=" * 70)
    print("PhysioMio with-calibration eval (variant e: 1200 windows + weight 100x)")
    print("=" * 70)

    # 1. Load PhysioMio + engineer features
    print(f"\n[1/3] Loading + engineering PhysioMio features...")
    physiomio = pd.read_pickle(PHYSIOMIO_PKL)
    t = time.time()
    physiomio_eng = engineer_features(physiomio)
    print(f"      done in {time.time() - t:.1f}s  shape: {physiomio_eng.shape}")

    # 2. Load GrabMyo engineered cache
    print(f"\n[2/3] Loading GrabMyo engineered features from cache...")
    if not GRABMYO_CACHE.exists():
        print(f"      ERROR: {GRABMYO_CACHE} not found. Run analysis/reproduce_headline/loso_eval.py first to build the cache.")
        sys.exit(1)
    grabmyo_eng = pd.read_pickle(GRABMYO_CACHE)
    print(f"      loaded  shape: {grabmyo_eng.shape}")

    # 3. Load feature column order from training metadata
    with open(GRABMYO_META) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    print(f"      feature_cols expected: {len(feature_cols)}")

    # Filter patient list
    all_patients = sorted(
        physiomio_eng["participant"].unique(),
        key=lambda s: int(s.replace("patient", "")),
    )
    patients_to_run = args.patients if args.patients else all_patients
    print(f"\nRunning {len(patients_to_run)} of {len(all_patients)} patients. Seed={SEED}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, p in enumerate(patients_to_run, 1):
        print(f"\n[{i}/{len(patients_to_run)}] {p}...")
        r = eval_one_patient(p, physiomio_eng, grabmyo_eng, feature_cols, SEED)
        if r is None:
            continue
        rows.append(r)
        # Save incrementally
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  acc_with_cal={r['acc_with_cal']:.4f}  f1={r['f1_with_cal']:.4f}  "
              f"(healthy_arm={r['acc_healthy_arm_cal']:.4f}, impaired_arm={r['acc_impaired_arm_cal']:.4f})  "
              f"fit={r['fit_time_s']:.0f}s")

    df = pd.DataFrame(rows)
    if len(df) > 0:
        print("\n" + "=" * 70)
        print(f"AGGREGATE (n={len(df)})")
        print("=" * 70)
        print(f"  acc_with_cal:        mean={df['acc_with_cal'].mean():.4f}  std={df['acc_with_cal'].std():.4f}")
        print(f"  f1_with_cal:         mean={df['f1_with_cal'].mean():.4f}  std={df['f1_with_cal'].std():.4f}")
        print(f"  acc_healthy_arm_cal: mean={df['acc_healthy_arm_cal'].mean():.4f}")
        print(f"  acc_impaired_arm_cal: mean={df['acc_impaired_arm_cal'].mean():.4f}")
    print(f"\nTotal wall: {(time.time() - t_start)/60:.1f} min")
    print(f"Results CSV: {out_path}")


if __name__ == "__main__":
    main()
