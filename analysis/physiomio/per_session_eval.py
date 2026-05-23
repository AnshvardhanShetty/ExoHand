"""
PhysioMio per-session evaluation with calibration.

Locked-in protocol (from train-ride decisions):
  - Base: GrabMyo only (no Adhi data — public-data-reproducible claim)
  - No patient initial-cal layer (single-tier; PhysioMio's 64 s/session can't
    faithfully reproduce the deployed 6-min initial calibration anyway)
  - Per session:
      * Reserve a BALANCED test set: 39 rest + 39 close + 39 open  (117 windows)
      * Use everything else as calibration:   39 rest + 741 close + 39 open  (819 windows)
      * Refit HGB on GrabMyo (weight 1) + this session's cal (weight 100)
      * Predict on test, record accuracy + macro-F1 + per-class F1

Outputs:
  - per_session_results.csv: one row per (patient, session)
  - per_patient_results.csv: aggregated per patient

Usage:
    python analysis/physiomio/per_session_eval.py --patients patient1   # smoke
    python analysis/physiomio/per_session_eval.py                       # all 48
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

OUT_DIR = PROJECT_ROOT / "analysis" / "physiomio" / "results"
OUT_SESSION_CSV = OUT_DIR / "per_session_results.csv"
OUT_PATIENT_CSV = OUT_DIR / "per_patient_results.csv"
OUT_PREDICTIONS_PARQUET = OUT_DIR / "per_window_predictions.parquet"
MODEL_CACHE_DIR = PROJECT_ROOT / "analysis" / ".cache" / "physiomio_session_models"

TEST_PER_CLASS = 39        # balanced test set
CAL_WEIGHT = 100.0
CLASSES = [0, 1, 2]        # rest, close, open
CLASS_NAMES = {0: "rest", 1: "close", 2: "open"}


def make_classifier(seed: int) -> HistGradientBoostingClassifier:
    """--fast HGB matching analysis/reproduce_headline/loso_eval.py."""
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=300, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        class_weight="balanced",
    )


def split_session(session_df: pd.DataFrame, test_per_class: int, rng: np.random.RandomState) -> tuple:
    """Per-gesture TEMPORAL split with no signal overlap.

    Each PhysioMio gesture is one continuous 4-s recording yielding 78 windows
    at 200 ms width / 50 ms stride. Adjacent windows overlap by 150 ms of raw
    signal (75 %). A random shuffle within a gesture leaks signal between cal
    and test sets. To avoid that, we split TEMPORALLY:

        per gesture (sorted by t_rel_s):
            cal     = windows 0..35       (first 36 windows, t = 0.10..1.85 s)
            buffer  = windows 36..38      (dropped — would overlap cal end + test start)
            test    = windows 39..77      (last 39 windows, t = 2.05..3.95 s)

    With 200 ms windows + 50 ms stride, a 3-window buffer guarantees the last
    cal window's signal coverage (right edge = 1.95 s) ends before the first
    test window's signal coverage (left edge = 1.95 s) begins — strictly no
    overlap at the raw-sample level.

    Per session (12 mapped gestures), this gives:
        cal pool  = 432 windows   ( 36 rest + 360 close +  36 open)
        test pool = 468 windows   ( 39 rest + 390 close +  39 open)

    We then subsample the test pool to a BALANCED `test_per_class` per class
    (close has way more than rest/open, so we randomly downsample close;
    rest and open take all 39).
    """
    cal_idx = []
    test_pool_by_class = {0: [], 1: [], 2: []}
    class_counts_raw = {0: 0, 1: 0, 2: 0}

    for _trial_id, gesture_group in session_df.groupby("trial", sort=True):
        cls = int(gesture_group["intent_idx"].iloc[0])
        sorted_group = gesture_group.sort_values("t_rel_s")
        n = len(sorted_group)
        class_counts_raw[cls] += n

        if n >= 78:
            # Standard PhysioMio gesture: 36 cal + 3 buffer + 39 test
            cal_idx.extend(sorted_group.index[:36].tolist())
            test_pool_by_class[cls].extend(sorted_group.index[39:78].tolist())
        elif n >= 8:
            # Short gesture (shouldn't happen with PhysioMio but be safe):
            # split temporally ~50/50 with at least 3-window buffer
            cal_n = (n - 3) // 2
            test_n = n - 3 - cal_n
            cal_idx.extend(sorted_group.index[:cal_n].tolist())
            test_pool_by_class[cls].extend(sorted_group.index[cal_n + 3:].tolist())
        else:
            # Too short to safely split — give to cal only, no test contribution
            cal_idx.extend(sorted_group.index.tolist())

    # Balance test set: subsample per-class pool down to test_per_class
    balanced_test_idx = []
    class_counts = {}   # actual test counts after balancing
    for cls in CLASSES:
        pool = test_pool_by_class[cls]
        if len(pool) <= test_per_class:
            balanced_test_idx.extend(pool)
            class_counts[cls] = len(pool)
        else:
            sampled = rng.choice(pool, size=test_per_class, replace=False).tolist()
            balanced_test_idx.extend(sampled)
            class_counts[cls] = test_per_class

    return np.array(sorted(balanced_test_idx)), np.array(sorted(cal_idx)), class_counts


def evaluate_session(
    patient: str,
    session: str,
    session_df: pd.DataFrame,
    grabmyo_X: np.ndarray,
    grabmyo_y: np.ndarray,
    feature_cols: list,
    rng: np.random.RandomState,
    seed: int,
    use_cached: bool = False,
) -> tuple:
    """Returns (summary_dict, predictions_dataframe).

    predictions_dataframe has per-window predictions on ALL non-cal data for
    this session, in temporal order — the input the transition-accuracy
    metric needs. Always returned, regardless of whether a cache was used.
    """
    test_idx, cal_idx, class_counts = split_session(session_df, TEST_PER_CLASS, rng)

    if len(test_idx) == 0:
        return {"participant": patient, "session": session, "status": "no_test_data"}, None

    X_cal = session_df.loc[cal_idx, feature_cols].values.astype(np.float32) if len(cal_idx) > 0 else np.zeros((0, len(feature_cols)), dtype=np.float32)
    y_cal = session_df.loc[cal_idx, "intent_idx"].values.astype(np.int64) if len(cal_idx) > 0 else np.zeros(0, dtype=np.int64)
    X_test = session_df.loc[test_idx, feature_cols].values.astype(np.float32)
    y_test = session_df.loc[test_idx, "intent_idx"].values.astype(np.int64)

    # Try cached model + scaler
    cache_path = MODEL_CACHE_DIR / f"{patient}__{session}.joblib"
    fit_time = 0.0
    if use_cached and cache_path.exists():
        bundle = joblib.load(cache_path)
        clf, scaler = bundle["clf"], bundle["scaler"]
        X_test_s = scaler.transform(X_test).astype(np.float32)
    else:
        # Assemble training: GrabMyo + cal
        X_all = np.vstack([grabmyo_X, X_cal])
        y_all = np.concatenate([grabmyo_y, y_cal])
        w_all = np.ones(len(X_all), dtype=np.float32)
        w_all[len(grabmyo_X):] = CAL_WEIGHT

        scaler = StandardScaler()
        X_all_s = scaler.fit_transform(X_all)
        X_test_s = scaler.transform(X_test).astype(np.float32)

        clf = make_classifier(seed)
        t0 = time.time()
        clf.fit(X_all_s, y_all, sample_weight=w_all)
        fit_time = time.time() - t0

        # Cache model + scaler for future re-runs (predict-only)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": clf, "scaler": scaler}, cache_path, compress=3)

    preds = clf.predict(X_test_s)

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
    cls_f1 = f1_score(y_test, preds, average=None, labels=CLASSES, zero_division=0)
    arm = session.split("_")[0]   # "healthy" or "impaired"

    summary = {
        "participant": patient,
        "session": session,
        "arm": arm,
        "n_cal_windows": int(len(cal_idx)),
        "n_test_windows": int(len(test_idx)),
        "n_cal_rest": class_counts.get(0, 0) - min(class_counts.get(0, 0), TEST_PER_CLASS),
        "n_cal_close": class_counts.get(1, 0) - min(class_counts.get(1, 0), TEST_PER_CLASS),
        "n_cal_open": class_counts.get(2, 0) - min(class_counts.get(2, 0), TEST_PER_CLASS),
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_rest": cls_f1[0],
        "f1_close": cls_f1[1],
        "f1_open": cls_f1[2],
        "fit_time_s": fit_time,
        "status": "ok",
    }

    # Predict on the FULL non-cal pool in temporal order (for transition accuracy).
    # This is a superset of the balanced 117-window test set used for `acc` above.
    non_cal_mask = ~session_df.index.isin(cal_idx)
    full_df = session_df.loc[non_cal_mask].sort_values("t_rel_s")
    X_full = full_df[feature_cols].values.astype(np.float32)
    X_full_s = scaler.transform(X_full).astype(np.float32)
    preds_full = clf.predict(X_full_s)

    predictions_df = pd.DataFrame({
        "participant": patient,
        "session": session,
        "arm": arm,
        "trial": full_df["trial"].values,
        "t_rel_s": full_df["t_rel_s"].values.astype(np.float32),
        "gt_intent": full_df["intent_idx"].values.astype(np.int8),
        "pred_intent": preds_full.astype(np.int8),
    })

    return summary, predictions_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", nargs="*", default=None,
                        help="Subset (e.g. patient1 patient2). Default = all 48.")
    parser.add_argument("--use-cached", action="store_true",
                        help="Load trained per-session models from analysis/.cache/physiomio_session_models/ "
                             "instead of refitting (skip GrabMyo + cal fit if cache exists).")
    args = parser.parse_args()

    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    t_start = time.time()

    print("=" * 70)
    print("PhysioMio per-session eval — single-tier per-session calibration")
    print("=" * 70)
    print(f"  Protocol: per session, balanced test {TEST_PER_CLASS}/class, "
          f"all-leftover cal with weight {CAL_WEIGHT}x")

    # Load engineered features
    print(f"\n[1/4] Loading + engineering PhysioMio (60 → 370)...")
    physiomio = pd.read_pickle(PHYSIOMIO_PKL)
    t = time.time()
    physiomio_eng = engineer_features(physiomio)
    print(f"      done in {time.time() - t:.1f}s  shape: {physiomio_eng.shape}")

    print(f"\n[2/4] Loading GrabMyo cache (already engineered)...")
    grabmyo_eng = pd.read_pickle(GRABMYO_CACHE)
    print(f"      shape: {grabmyo_eng.shape}")

    with open(GRABMYO_META) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    grabmyo_X = grabmyo_eng[feature_cols].values.astype(np.float32)
    grabmyo_y = grabmyo_eng["intent_idx"].values.astype(np.int64)
    print(f"      base training set: {len(grabmyo_X):,} windows × {len(feature_cols)} features")

    # Filter patient list
    all_patients = sorted(
        physiomio_eng["participant"].unique(),
        key=lambda s: int(s.replace("patient", "")),
    )
    patients_to_run = args.patients if args.patients else all_patients
    print(f"\n[3/4] Per-session eval on {len(patients_to_run)} of {len(all_patients)} patients...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session_rows = []
    prediction_dfs = []
    overall_t = time.time()

    for pi, patient in enumerate(patients_to_run, 1):
        p_data = physiomio_eng[physiomio_eng["participant"] == patient]
        sessions = sorted(p_data["session"].unique())
        print(f"\n  [{pi}/{len(patients_to_run)}] {patient}  ({len(sessions)} sessions)")
        for si, session in enumerate(sessions, 1):
            s_data = p_data[p_data["session"] == session].copy()
            result, pred_df = evaluate_session(
                patient, session, s_data, grabmyo_X, grabmyo_y,
                feature_cols, rng, SEED, use_cached=args.use_cached,
            )
            session_rows.append(result)
            if pred_df is not None:
                prediction_dfs.append(pred_df)
            pd.DataFrame(session_rows).to_csv(OUT_SESSION_CSV, index=False)
            if result.get("status") == "ok":
                cached_str = "[cached]" if args.use_cached and result["fit_time_s"] == 0.0 else f"fit={result['fit_time_s']:.0f}s"
                print(f"    [{si}/{len(sessions)}] {session}: acc={result['acc']:.4f}  f1={result['f1_macro']:.4f}  "
                      f"(rest={result['f1_rest']:.3f}, close={result['f1_close']:.3f}, open={result['f1_open']:.3f})  "
                      f"{cached_str}")
            else:
                print(f"    [{si}/{len(sessions)}] {session}: SKIP — {result.get('status')}")

        # Write predictions parquet incrementally per patient (so a crash mid-run doesn't lose everything)
        if prediction_dfs:
            pd.concat(prediction_dfs, ignore_index=True).to_parquet(OUT_PREDICTIONS_PARQUET, index=False)

    sess_df = pd.DataFrame(session_rows)
    ok_df = sess_df[sess_df["status"] == "ok"]

    # Aggregate per patient
    if len(ok_df) > 0:
        patient_rows = []
        for patient, g in ok_df.groupby("participant"):
            healthy = g[g["arm"] == "healthy"]
            impaired = g[g["arm"] == "impaired"]
            patient_rows.append({
                "participant": patient,
                "n_sessions": len(g),
                "n_healthy_sessions": len(healthy),
                "n_impaired_sessions": len(impaired),
                "acc_mean": g["acc"].mean(),
                "acc_std": g["acc"].std(),
                "f1_mean": g["f1_macro"].mean(),
                "acc_healthy_mean": healthy["acc"].mean() if len(healthy) else np.nan,
                "acc_impaired_mean": impaired["acc"].mean() if len(impaired) else np.nan,
                "f1_rest_mean": g["f1_rest"].mean(),
                "f1_close_mean": g["f1_close"].mean(),
                "f1_open_mean": g["f1_open"].mean(),
            })
        pat_df = pd.DataFrame(patient_rows).sort_values(
            "participant", key=lambda s: s.str.extract(r"(\d+)").astype(int).iloc[:, 0]
        )
        pat_df.to_csv(OUT_PATIENT_CSV, index=False)

        print("\n" + "=" * 70)
        print(f"AGGREGATE (n={len(pat_df)} patients, {len(ok_df)} sessions)")
        print("=" * 70)
        print(f"  Session-level:")
        print(f"    acc:  mean={ok_df['acc'].mean():.4f}  std={ok_df['acc'].std():.4f}  "
              f"range=[{ok_df['acc'].min():.4f}, {ok_df['acc'].max():.4f}]")
        print(f"    f1:   mean={ok_df['f1_macro'].mean():.4f}  std={ok_df['f1_macro'].std():.4f}")
        print(f"    per-class F1: rest={ok_df['f1_rest'].mean():.4f}  "
              f"close={ok_df['f1_close'].mean():.4f}  open={ok_df['f1_open'].mean():.4f}")
        print(f"  Per-arm:")
        h = ok_df[ok_df["arm"] == "healthy"]
        i = ok_df[ok_df["arm"] == "impaired"]
        print(f"    healthy arm: mean={h['acc'].mean():.4f}  std={h['acc'].std():.4f}  n={len(h)}")
        print(f"    impaired arm: mean={i['acc'].mean():.4f}  std={i['acc'].std():.4f}  n={len(i)}")
        print(f"\n  Patient-level acc mean: {pat_df['acc_mean'].mean():.4f}  std={pat_df['acc_mean'].std():.4f}")
    print(f"\n[4/4] Total wall: {(time.time() - overall_t)/60:.1f} min")
    print(f"Session CSV: {OUT_SESSION_CSV}")
    print(f"Patient CSV: {OUT_PATIENT_CSV}")


if __name__ == "__main__":
    main()
