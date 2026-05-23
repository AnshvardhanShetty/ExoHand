"""
LOSO reproduction of the headline accuracy claim.

For each GrabMyo participant:
  1. Train HGB on the other 42 participants (370-feature pipeline from ml.train_hgb_v2).
  2. Evaluate on held-out participant → "no calibration" accuracy.
  3. Take a fixed-size calibration slice from the held-out participant.
  4. Refit HGB with calibration samples weighted 10x → "with calibration" accuracy.
  5. Evaluate on the remaining held-out windows.

Outputs per-subject CSV with accuracy + macro-F1 before/after. Bootstrap CIs
and aggregate analysis live in a separate script so this stays focused on the
expensive per-fold work.

Usage:
    # Smoke test (single fold)
    python analysis/reproduce_headline/loso_eval.py --participants participant1

    # Full 43-fold run
    python analysis/reproduce_headline/loso_eval.py

    # Subset for quick estimate
    python analysis/reproduce_headline/loso_eval.py --participants participant1 participant5 participant10
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so absolute imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from ml.train_hgb_v2 import META_COLS, engineer_features
from analysis.seed import SEED, seed_everything


GRABMYO_CSV = PROJECT_ROOT / "grabmyo" / "grabmyo_intent_dataset.csv"
CACHE_DIR = PROJECT_ROOT / "analysis" / ".cache"
# Pickle rather than parquet to avoid an extra dependency (pyarrow/fastparquet).
# Cache is ~1.7 GB uncompressed; that's fine for a local dev machine.
CACHE_FEATURES = CACHE_DIR / "grabmyo_features_370.pkl"

# 30 seconds at 200ms windows + 50ms stride ≈ 600 windows
CALIB_N_WINDOWS_DEFAULT = 600
CALIB_WEIGHT_DEFAULT = 10.0


def load_or_build_features(force_rebuild: bool = False) -> pd.DataFrame:
    """Load the engineered 370-feature dataset, building + caching if needed."""
    if not force_rebuild and CACHE_FEATURES.exists():
        print(f"Loading cached features from {CACHE_FEATURES}")
        return pd.read_pickle(CACHE_FEATURES)

    print(f"Loading raw dataset from {GRABMYO_CSV} (one-time cost)...")
    df = pd.read_csv(GRABMYO_CSV)
    print(f"  raw shape: {df.shape}")

    print("Engineering features (370-feature v2 pipeline)...")
    df = engineer_features(df)
    print(f"  engineered shape: {df.shape}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(CACHE_FEATURES)
    print(f"Cached features to {CACHE_FEATURES}")
    return df


def select_calibration_indices(
    X_test: np.ndarray,
    y_test: np.ndarray,
    t_rel_s: np.ndarray,
    n_target: int,
    rng: np.random.RandomState,
    protocol: str = "random_stratified",
    onset_trim_s: float = 1.0,
    outlier_z: float = 5.0,
) -> tuple:
    """Return (indices, n_dropped_onset, n_dropped_outlier) into X_test/y_test for calibration.

    protocol options:
      'random_stratified': pick n_target indices stratified across classes.
      'trimmed_stratified': mimic calibrate_patient.py — first apply onset trim
          (drop windows with t_rel_s <= onset_trim_s, matches ONSET_TRIM_S=1.0)
          and outlier rejection (drop windows where any feature has |z| > outlier_z;
          features are already per-participant z-scored upstream so this catches
          electrode pops / artifacts), then stratified sample from remaining.
    """
    n = len(X_test)
    candidate_mask = np.ones(n, dtype=bool)
    n_dropped_onset = 0
    n_dropped_outlier = 0

    if protocol == "trimmed_stratified":
        onset_keep = t_rel_s > onset_trim_s
        n_dropped_onset = int((~onset_keep).sum())
        candidate_mask &= onset_keep

        max_abs_feat = np.max(np.abs(X_test), axis=1)
        outlier_keep = max_abs_feat <= outlier_z
        n_dropped_outlier = int((~outlier_keep & candidate_mask).sum())
        candidate_mask &= outlier_keep
    elif protocol != "random_stratified":
        raise ValueError(f"Unknown calibration protocol: {protocol}")

    candidate_idx = np.where(candidate_mask)[0]
    if len(candidate_idx) < n_target:
        # Not enough clean windows — fall back to all candidates (no top-up from dirty pool)
        return np.sort(candidate_idx), n_dropped_onset, n_dropped_outlier

    candidate_y = y_test[candidate_idx]
    classes = np.unique(candidate_y)
    per_class = max(1, n_target // len(classes))
    picked_local = []
    for c in classes:
        class_idx_local = np.where(candidate_y == c)[0]
        take = min(per_class, len(class_idx_local))
        picked_local.extend(rng.choice(class_idx_local, size=take, replace=False))
    picked_local = list(set(picked_local))

    if len(picked_local) < n_target:
        leftover = np.setdiff1d(np.arange(len(candidate_idx)), picked_local)
        topup = min(n_target - len(picked_local), len(leftover))
        if topup > 0:
            picked_local.extend(rng.choice(leftover, size=topup, replace=False))

    picked_global = candidate_idx[sorted(picked_local)]
    return picked_global, n_dropped_onset, n_dropped_outlier


def make_classifier(seed: int, fast: bool = False) -> HistGradientBoostingClassifier:
    """Build the HGB classifier.

    fast=True: smaller trees + fewer iterations for development iteration speed.
    Typically 10-20x faster, ~1-3% accuracy below the full config.

    fast=False (default): matches train_hgb_v2.py hyperparameters exactly. Used
    for final paper numbers.
    """
    if fast:
        return HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_leaf_nodes=63,
            max_iter=300,
            min_samples_leaf=20,
            l2_regularization=0.01,
            max_depth=10,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            class_weight="balanced",
        )
    return HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_leaf_nodes=255,
        max_iter=2500,
        min_samples_leaf=20,
        l2_regularization=0.01,
        max_depth=18,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        class_weight="balanced",
    )


def evaluate_one_fold(
    df: pd.DataFrame,
    held_out: str,
    calib_n_windows: int = CALIB_N_WINDOWS_DEFAULT,
    calib_weight: float = CALIB_WEIGHT_DEFAULT,
    seed: int = SEED,
    fast: bool = False,
    calib_protocol: str = "random_stratified",
    onset_trim_s: float = 1.0,
    outlier_z: float = 5.0,
) -> dict:
    """LOSO fold: train on 42 others, eval on held_out, then with 30-s calibration."""
    feature_cols = [c for c in df.columns if c not in META_COLS]

    train_mask = df["participant"] != held_out
    test_mask = df["participant"] == held_out

    X_train = df.loc[train_mask, feature_cols].values.astype(np.float32)
    y_train = df.loc[train_mask, "intent_idx"].values.astype(np.int64)
    X_test = df.loc[test_mask, feature_cols].values.astype(np.float32)
    y_test = df.loc[test_mask, "intent_idx"].values.astype(np.int64)
    t_rel_s = df.loc[test_mask, "t_rel_s"].values.astype(np.float64)

    # Calibration slice from held-out participant per chosen protocol
    rng = np.random.RandomState(seed)
    calib_n = min(calib_n_windows, len(X_test) // 2)
    calib_idx, n_dropped_onset, n_dropped_outlier = select_calibration_indices(
        X_test, y_test, t_rel_s, calib_n, rng,
        protocol=calib_protocol,
        onset_trim_s=onset_trim_s,
        outlier_z=outlier_z,
    )
    eval_mask = np.ones(len(X_test), dtype=bool)
    eval_mask[calib_idx] = False

    X_calib, y_calib = X_test[calib_idx], y_test[calib_idx]
    X_eval, y_eval = X_test[eval_mask], y_test[eval_mask]

    # Global scaler fit on training participants only (LOSO-safe)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_calib_s = scaler.transform(X_calib).astype(np.float32)
    X_eval_s = scaler.transform(X_eval).astype(np.float32)

    # --- Pass 1: no calibration ---
    clf = make_classifier(seed, fast=fast)
    t0 = time.time()
    clf.fit(X_train_s, y_train)
    fit_time = time.time() - t0
    pred_no_cal = clf.predict(X_eval_s)
    acc_no_cal = accuracy_score(y_eval, pred_no_cal)
    f1_no_cal = f1_score(y_eval, pred_no_cal, average="macro")

    # --- Pass 2: with calibration (10x weight on held-out samples) ---
    X_all = np.vstack([X_train_s, X_calib_s])
    y_all = np.concatenate([y_train, y_calib])
    w_all = np.ones(len(X_all), dtype=np.float32)
    w_all[len(X_train_s):] = calib_weight

    clf_cal = make_classifier(seed, fast=fast)
    t1 = time.time()
    clf_cal.fit(X_all, y_all, sample_weight=w_all)
    fit_time_cal = time.time() - t1
    pred_cal = clf_cal.predict(X_eval_s)
    acc_cal = accuracy_score(y_eval, pred_cal)
    f1_cal = f1_score(y_eval, pred_cal, average="macro")

    return {
        "participant": held_out,
        "n_train_windows": len(X_train),
        "n_calib_windows": int(calib_n),
        "n_eval_windows": int(eval_mask.sum()),
        "acc_no_cal": acc_no_cal,
        "f1_no_cal": f1_no_cal,
        "acc_with_cal": acc_cal,
        "f1_with_cal": f1_cal,
        "delta_acc": acc_cal - acc_no_cal,
        "fit_time_s_no_cal": fit_time,
        "fit_time_s_with_cal": fit_time_cal,
        "calib_weight": calib_weight,
        "calib_n_windows_target": calib_n_windows,
        "calib_protocol": calib_protocol,
        "n_dropped_onset": n_dropped_onset,
        "n_dropped_outlier": n_dropped_outlier,
        "seed": seed,
        "fast": fast,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", nargs="*", default=None,
                        help="Subset of participants to run (default = all 43)")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "analysis" / "reproduce_headline" / "loso_results.csv"))
    parser.add_argument("--calib-n-windows", type=int, default=CALIB_N_WINDOWS_DEFAULT)
    parser.add_argument("--calib-weight", type=float, default=CALIB_WEIGHT_DEFAULT)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--fast", action="store_true",
                        help="Use reduced HGB hyperparams (max_iter=300, max_depth=10, "
                             "max_leaf_nodes=63) for ~10-20x speedup. Trades ~1-3% accuracy. "
                             "Use for development iteration; turn off for final paper numbers.")
    parser.add_argument("--calib-protocol", choices=["random_stratified", "trimmed_stratified"],
                        default="random_stratified",
                        help="random_stratified (default): stratified random sample of calibration "
                             "windows. trimmed_stratified: mirror calibrate_patient.py — drop "
                             "windows within onset_trim_s of trial start, drop outlier windows "
                             "(any feature |z| > outlier_z), then stratified sample.")
    parser.add_argument("--onset-trim-s", type=float, default=1.0,
                        help="Onset trim threshold for trimmed_stratified protocol (default 1.0s, "
                             "matches calibrate_patient.py ONSET_TRIM_S).")
    parser.add_argument("--outlier-z", type=float, default=5.0,
                        help="Outlier z-score threshold for trimmed_stratified (default 5.0). "
                             "Features are already per-participant-z-scored upstream.")
    args = parser.parse_args()

    seed_everything(SEED)
    df = load_or_build_features(force_rebuild=args.force_rebuild_cache)

    all_parts = sorted(df["participant"].unique())
    parts_to_run = args.participants if args.participants else all_parts
    mode = "FAST (reduced hyperparams)" if args.fast else "FULL (train_hgb_v2 hyperparams)"
    print(f"Running LOSO on {len(parts_to_run)} of {len(all_parts)} participants. Seed={SEED}.")
    print(f"  Mode: {mode}")
    print(f"  Calibration: protocol={args.calib_protocol}  n_windows={args.calib_n_windows}  weight={args.calib_weight}")
    if args.calib_protocol == "trimmed_stratified":
        print(f"               onset_trim_s={args.onset_trim_s}  outlier_z={args.outlier_z}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    overall_start = time.time()
    for i, p in enumerate(parts_to_run, 1):
        print(f"\n[{i}/{len(parts_to_run)}] Holding out {p}...")
        r = evaluate_one_fold(
            df, p,
            calib_n_windows=args.calib_n_windows,
            calib_weight=args.calib_weight,
            seed=SEED,
            fast=args.fast,
            calib_protocol=args.calib_protocol,
            onset_trim_s=args.onset_trim_s,
            outlier_z=args.outlier_z,
        )
        results.append(r)
        # Save incrementally; a crash mid-run shouldn't lose all prior folds
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(
            f"  no-cal: acc={r['acc_no_cal']:.4f} f1={r['f1_no_cal']:.4f}  "
            f"with-cal: acc={r['acc_with_cal']:.4f} f1={r['f1_with_cal']:.4f}  "
            f"Δacc={r['delta_acc']:+.4f}  "
            f"({r['fit_time_s_no_cal']:.0f}s + {r['fit_time_s_with_cal']:.0f}s)"
        )

    res_df = pd.DataFrame(results)
    elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"AGGREGATE over {len(res_df)} participants (seed={SEED})")
    print("=" * 60)
    print(f"  No-cal   accuracy: mean={res_df['acc_no_cal'].mean():.4f}  std={res_df['acc_no_cal'].std():.4f}")
    print(f"  With-cal accuracy: mean={res_df['acc_with_cal'].mean():.4f}  std={res_df['acc_with_cal'].std():.4f}")
    print(f"  No-cal   macro-F1: mean={res_df['f1_no_cal'].mean():.4f}  std={res_df['f1_no_cal'].std():.4f}")
    print(f"  With-cal macro-F1: mean={res_df['f1_with_cal'].mean():.4f}  std={res_df['f1_with_cal'].std():.4f}")
    print(f"  Δacc (cal − no-cal): mean={res_df['delta_acc'].mean():+.4f}")
    print(f"  Total wall time: {elapsed/60:.1f} min")
    print(f"  Results CSV: {out_path}")


if __name__ == "__main__":
    main()
