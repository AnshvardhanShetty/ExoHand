"""
Lucchetti per-session evaluation with calibration.

Mirrors analysis/physiomio/per_session_eval.py. Each Lucchetti subject has up
to 2 sessions: 'impaired_01' (paretic arm) for stroke subjects, 'healthy_01'
(non-affected arm for stroke OR dominant arm for healthy controls).

Per session:
  - Reserve a BALANCED test set: 39 rest + 39 close + 39 open (when available)
  - Use the rest as patient calibration data (weighted 100× vs GrabMyo)
  - Refit HGB on GrabMyo + cal
  - Predict on the balanced test set (raw acc, F1) AND on the full non-cal
    stream in temporal order (for transition accuracy downstream)

Caches trained models in analysis/.cache/lucchetti_session_models/ so future
re-runs (transition acc sweeps, sensitivity analyses) take minutes.

Outputs:
  analysis/lucchetti/results/per_session_results.csv
  analysis/lucchetti/results/per_patient_results.csv
  analysis/lucchetti/results/per_window_predictions.parquet
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
from ml.train_hgb_v2 import engineer_features

LUCCHETTI_PKL = PROJECT_ROOT / "data" / "lucchetti_features_60_per_subject.pkl"
GRABMYO_CACHE = PROJECT_ROOT / "analysis" / ".cache" / "grabmyo_features_370.pkl"
GRABMYO_META = PROJECT_ROOT / "grabmyo" / "improved_hgb_meta.json"

OUT_DIR = PROJECT_ROOT / "analysis" / "lucchetti" / "results"
OUT_SESSION_CSV = OUT_DIR / "per_session_results.csv"
OUT_PATIENT_CSV = OUT_DIR / "per_patient_results.csv"
OUT_PREDICTIONS_PARQUET = OUT_DIR / "per_window_predictions.parquet"
MODEL_CACHE_DIR = PROJECT_ROOT / "analysis" / ".cache" / "lucchetti_session_models"

TEST_PER_CLASS = 39
CAL_WEIGHT = 100.0
CLASSES = [0, 1, 2]


def make_classifier(seed):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=300, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        class_weight="balanced",
    )


def split_session_lucchetti(session_df, test_per_class, rng, buffer_windows=3):
    """Per-trial temporal split with buffer between cal and test.

    Lucchetti trials are contiguous segments of same-class windows derived from
    the rep events. Within each trial: first half = cal, last half (after
    buffer) = test pool. Balanced test = TEST_PER_CLASS per class sampled from
    test pools.
    """
    cal_idx = []
    test_pool_by_class = {0: [], 1: [], 2: []}
    class_counts = {0: 0, 1: 0, 2: 0}
    for _, g in session_df.groupby("trial", sort=True):
        cls = int(g["intent_idx"].iloc[0])
        sg = g.sort_values("t_rel_s")
        n = len(sg)
        class_counts[cls] += n
        if n < 8:
            # Tiny trial — put all in cal, no test
            cal_idx.extend(sg.index.tolist())
            continue
        # Halve cal/test with a buffer between
        cal_n = max(1, (n - buffer_windows) // 2)
        test_start = cal_n + buffer_windows
        cal_idx.extend(sg.index[:cal_n].tolist())
        test_pool_by_class[cls].extend(sg.index[test_start:].tolist())

    balanced_test = []
    for cls in CLASSES:
        pool = test_pool_by_class[cls]
        if len(pool) <= test_per_class:
            balanced_test.extend(pool)
        else:
            balanced_test.extend(rng.choice(pool, size=test_per_class, replace=False).tolist())
    return np.array(sorted(balanced_test)), np.array(sorted(cal_idx)), class_counts


def evaluate_session(subject, session, session_df, grabmyo_X, grabmyo_y,
                     feature_cols, rng, seed, use_cached=False):
    test_idx, cal_idx, class_counts = split_session_lucchetti(session_df, TEST_PER_CLASS, rng)
    if len(test_idx) == 0:
        return {"participant": subject, "session": session, "status": "no_test_data"}, None
    if min(class_counts.values()) < 5:
        return {"participant": subject, "session": session, "status": "imbalanced_classes",
                **{f"n_{c}": v for c, v in class_counts.items()}}, None

    X_cal = session_df.loc[cal_idx, feature_cols].values.astype(np.float32) if len(cal_idx) > 0 else np.zeros((0, len(feature_cols)), dtype=np.float32)
    y_cal = session_df.loc[cal_idx, "intent_idx"].values.astype(np.int64) if len(cal_idx) > 0 else np.zeros(0, dtype=np.int64)
    X_test = session_df.loc[test_idx, feature_cols].values.astype(np.float32)
    y_test = session_df.loc[test_idx, "intent_idx"].values.astype(np.int64)

    cache_path = MODEL_CACHE_DIR / f"{subject}__{session}.joblib"
    fit_time = 0.0
    if use_cached and cache_path.exists():
        bundle = joblib.load(cache_path)
        clf, scaler = bundle["clf"], bundle["scaler"]
        X_test_s = scaler.transform(X_test).astype(np.float32)
    else:
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
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": clf, "scaler": scaler}, cache_path, compress=3)

    preds = clf.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    f1m = f1_score(y_test, preds, average="macro", zero_division=0)
    cls_f1 = f1_score(y_test, preds, average=None, labels=CLASSES, zero_division=0)
    arm = session.split("_")[0]

    summary = {
        "participant": subject, "session": session, "arm": arm,
        "n_cal_windows": int(len(cal_idx)), "n_test_windows": int(len(test_idx)),
        "n_rest": class_counts.get(0, 0), "n_close": class_counts.get(1, 0), "n_open": class_counts.get(2, 0),
        "acc": acc, "f1_macro": f1m,
        "f1_rest": cls_f1[0], "f1_close": cls_f1[1], "f1_open": cls_f1[2],
        "fit_time_s": fit_time, "status": "ok",
    }

    # Full-stream predictions for transition accuracy
    non_cal_mask = ~session_df.index.isin(cal_idx)
    full_df = session_df.loc[non_cal_mask].sort_values(["trial", "t_rel_s"])
    X_full = full_df[feature_cols].values.astype(np.float32)
    X_full_s = scaler.transform(X_full).astype(np.float32)
    preds_full = clf.predict(X_full_s)
    pred_df = pd.DataFrame({
        "participant": subject, "session": session, "arm": arm,
        "trial": full_df["trial"].values,
        "t_rel_s": full_df["t_rel_s"].values.astype(np.float32),
        "gt_intent": full_df["intent_idx"].values.astype(np.int8),
        "pred_intent": preds_full.astype(np.int8),
    })
    return summary, pred_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Subset (e.g. ST_01 HS_01). Default = all 20.")
    parser.add_argument("--use-cached", action="store_true",
                        help="Load cached per-session models from analysis/.cache/lucchetti_session_models/.")
    args = parser.parse_args()

    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    t_start = time.time()

    print("=" * 70)
    print("Lucchetti per-session eval — single-tier per-session calibration")
    print("=" * 70)

    print("Loading Lucchetti features + engineering 60 → 370...")
    df = pd.read_pickle(LUCCHETTI_PKL)
    t = time.time()
    eng = engineer_features(df)
    print(f"  done in {time.time()-t:.1f}s  shape: {eng.shape}")

    print("Loading GrabMyo cache...")
    g = pd.read_pickle(GRABMYO_CACHE)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]
    grabmyo_X = g[feature_cols].values.astype(np.float32)
    grabmyo_y = g["intent_idx"].values.astype(np.int64)
    print(f"  GrabMyo: {len(grabmyo_X):,} windows × {len(feature_cols)} features")

    subjects = sorted(eng["participant"].unique())
    if args.subjects:
        subjects = [s for s in subjects if s in args.subjects]
    print(f"\nPer-session eval on {len(subjects)} subjects...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session_rows = []
    prediction_dfs = []

    for pi, subject in enumerate(subjects, 1):
        sdata = eng[eng["participant"] == subject]
        sessions = sorted(sdata["session"].unique())
        print(f"\n  [{pi}/{len(subjects)}] {subject}  ({len(sessions)} arms)")
        for si, session in enumerate(sessions, 1):
            s_data = sdata[sdata["session"] == session].copy()
            result, pred_df = evaluate_session(
                subject, session, s_data, grabmyo_X, grabmyo_y,
                feature_cols, rng, SEED, use_cached=args.use_cached,
            )
            session_rows.append(result)
            if pred_df is not None:
                prediction_dfs.append(pred_df)
            pd.DataFrame(session_rows).to_csv(OUT_SESSION_CSV, index=False)
            if result.get("status") == "ok":
                cached = "[cached]" if args.use_cached and result["fit_time_s"] == 0.0 else f"fit={result['fit_time_s']:.0f}s"
                print(f"    [{si}/{len(sessions)}] {session}: acc={result['acc']:.4f}  f1={result['f1_macro']:.4f}  "
                      f"(rest={result['f1_rest']:.3f}, close={result['f1_close']:.3f}, open={result['f1_open']:.3f})  {cached}")
            else:
                print(f"    [{si}/{len(sessions)}] {session}: SKIP — {result.get('status')}")
        if prediction_dfs:
            pd.concat(prediction_dfs, ignore_index=True).to_parquet(OUT_PREDICTIONS_PARQUET, index=False)

    sess_df = pd.DataFrame(session_rows)
    ok_df = sess_df[sess_df["status"] == "ok"]
    print("\n" + "=" * 70)
    print(f"AGGREGATE (n={ok_df['participant'].nunique()} subjects, {len(ok_df)} sessions)")
    print("=" * 70)
    print(f"  session mean acc: {ok_df['acc'].mean():.4f}  std={ok_df['acc'].std():.4f}")
    print(f"  per arm: healthy={ok_df[ok_df['arm']=='healthy']['acc'].mean():.4f} (n={(ok_df['arm']=='healthy').sum()})  "
          f"impaired={ok_df[ok_df['arm']=='impaired']['acc'].mean():.4f} (n={(ok_df['arm']=='impaired').sum()})")
    print(f"  per-class F1: rest={ok_df['f1_rest'].mean():.4f}  close={ok_df['f1_close'].mean():.4f}  open={ok_df['f1_open'].mean():.4f}")

    # Per patient
    pat_rows = []
    for sid, g in ok_df.groupby("participant"):
        h = g[g["arm"] == "healthy"]
        i = g[g["arm"] == "impaired"]
        pat_rows.append({
            "participant": sid, "n_sessions": len(g),
            "acc_mean": g["acc"].mean(), "acc_std": g["acc"].std(),
            "acc_healthy": h["acc"].mean() if len(h) else np.nan,
            "acc_impaired": i["acc"].mean() if len(i) else np.nan,
            "f1_mean": g["f1_macro"].mean(),
            "f1_rest_mean": g["f1_rest"].mean(),
            "f1_close_mean": g["f1_close"].mean(),
            "f1_open_mean": g["f1_open"].mean(),
        })
    pd.DataFrame(pat_rows).to_csv(OUT_PATIENT_CSV, index=False)
    print(f"\nTotal wall: {(time.time()-t_start)/60:.1f} min")
    print(f"Session CSV: {OUT_SESSION_CSV}")
    print(f"Patient CSV: {OUT_PATIENT_CSV}")


if __name__ == "__main__":
    main()
