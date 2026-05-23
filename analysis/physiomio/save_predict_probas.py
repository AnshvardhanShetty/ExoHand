"""
Save predict_proba for the same per-window test pool that produced
per_window_predictions.parquet, using the per-session cached models.

The deployed pipeline (EMA smoothing, hysteresis, confidence floor) operates
on probability outputs, not argmax predictions. This script extracts those
probabilities once so downstream analyses (full deployed pipeline, sensitivity
sweeps) can run in seconds.

Input:
  analysis/.cache/physiomio_session_models/{patient}__{session}.joblib
  data/physiomio_features_60_per_patient.pkl  (engineered features)

Output:
  analysis/physiomio/results/per_window_probas.parquet
    columns: participant, session, arm, trial, t_rel_s, gt_intent,
             proba_rest, proba_close, proba_open

Compute: ~5-10 min wall (329 sessions, each predict_proba on ~500 windows).
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from analysis.seed import SEED, seed_everything
from ml.train_hgb_v2 import engineer_features
from analysis.physiomio.per_session_eval import (
    split_session, MODEL_CACHE_DIR, PHYSIOMIO_PKL, GRABMYO_META, TEST_PER_CLASS,
)
import json


OUT_PARQUET = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "per_window_probas.parquet"


def main():
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    t_start = time.time()

    print("Loading + engineering PhysioMio (60 → 370)...")
    physiomio = pd.read_pickle(PHYSIOMIO_PKL)
    t = time.time()
    physiomio_eng = engineer_features(physiomio)
    print(f"  done in {time.time() - t:.1f}s  shape: {physiomio_eng.shape}")

    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]

    patients = sorted(physiomio_eng["participant"].unique(),
                      key=lambda s: int(s.replace("patient", "")))

    all_rows = []
    n_done = 0
    n_total_sessions = sum(physiomio_eng[physiomio_eng["participant"] == p]["session"].nunique()
                           for p in patients)
    for pi, patient in enumerate(patients, 1):
        p_data = physiomio_eng[physiomio_eng["participant"] == patient]
        sessions = sorted(p_data["session"].unique())
        for si, session in enumerate(sessions, 1):
            s_data = p_data[p_data["session"] == session].copy()
            cache_path = MODEL_CACHE_DIR / f"{patient}__{session}.joblib"
            if not cache_path.exists():
                print(f"  [{n_done}/{n_total_sessions}] MISSING cache: {cache_path.name} — skipping")
                continue

            # Match the cal/test split that produced the cached model.
            # split_session uses the rng → must be called once per session in
            # the same order as per_session_eval.py for cal_idx to match.
            local_rng = np.random.RandomState(SEED)
            _, cal_idx, _ = split_session(s_data, TEST_PER_CLASS, local_rng)

            bundle = joblib.load(cache_path)
            clf, scaler = bundle["clf"], bundle["scaler"]

            # Full non-cal stream in temporal order
            non_cal_mask = ~s_data.index.isin(cal_idx)
            full_df = s_data.loc[non_cal_mask].sort_values(["trial", "t_rel_s"])
            X = full_df[feature_cols].values.astype(np.float32)
            X_s = scaler.transform(X).astype(np.float32)
            probas = clf.predict_proba(X_s).astype(np.float32)
            # classes_ order — for our trained HGBs this is [0, 1, 2]
            classes = list(clf.classes_)
            idx_rest = classes.index(0)
            idx_close = classes.index(1)
            idx_open = classes.index(2)

            block = pd.DataFrame({
                "participant": patient,
                "session": session,
                "arm": session.split("_")[0],
                "trial": full_df["trial"].values.astype(np.int16),
                "t_rel_s": full_df["t_rel_s"].values.astype(np.float32),
                "gt_intent": full_df["intent_idx"].values.astype(np.int8),
                "proba_rest": probas[:, idx_rest],
                "proba_close": probas[:, idx_close],
                "proba_open": probas[:, idx_open],
            })
            all_rows.append(block)
            n_done += 1
            if n_done % 30 == 0:
                pd.concat(all_rows, ignore_index=True).to_parquet(OUT_PARQUET, index=False)
                print(f"  [{n_done}/{n_total_sessions}] {patient} {session}: "
                      f"{len(block)} windows  elapsed={time.time()-t_start:.0f}s")

    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_parquet(OUT_PARQUET, index=False)
    print(f"\n{n_done} sessions × ~500 windows → {OUT_PARQUET}")
    print(f"Total wall: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
