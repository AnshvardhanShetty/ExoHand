"""
Lucchetti predict_proba extractor — mirrors analysis/physiomio/save_predict_probas.py.
Uses analysis/.cache/lucchetti_session_models/ to predict probas on the full
non-cal stream per session.

Output: analysis/lucchetti/results/per_window_probas.parquet
"""

import json
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
from analysis.lucchetti.per_session_eval import (
    LUCCHETTI_PKL, GRABMYO_META, MODEL_CACHE_DIR, TEST_PER_CLASS, split_session_lucchetti,
)

OUT_PARQUET = PROJECT_ROOT / "analysis" / "lucchetti" / "results" / "per_window_probas.parquet"


def main():
    seed_everything(SEED)
    t_start = time.time()
    print("Loading + engineering Lucchetti (60 → 370)...")
    df = pd.read_pickle(LUCCHETTI_PKL)
    eng = engineer_features(df)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]

    all_rows = []
    subjects = sorted(eng["participant"].unique())
    for pi, subject in enumerate(subjects, 1):
        sdata = eng[eng["participant"] == subject]
        for session in sorted(sdata["session"].unique()):
            s_data = sdata[sdata["session"] == session].copy()
            cache_path = MODEL_CACHE_DIR / f"{subject}__{session}.joblib"
            if not cache_path.exists():
                continue
            local_rng = np.random.RandomState(SEED)
            _, cal_idx, _ = split_session_lucchetti(s_data, TEST_PER_CLASS, local_rng)
            bundle = joblib.load(cache_path)
            clf, scaler = bundle["clf"], bundle["scaler"]
            non_cal_mask = ~s_data.index.isin(cal_idx)
            full_df = s_data.loc[non_cal_mask].sort_values(["trial", "t_rel_s"])
            X_s = scaler.transform(full_df[feature_cols].values.astype(np.float32)).astype(np.float32)
            probas = clf.predict_proba(X_s).astype(np.float32)
            classes = list(clf.classes_)
            ir, ic, io_ = classes.index(0), classes.index(1), classes.index(2)
            block = pd.DataFrame({
                "participant": subject, "session": session, "arm": session.split("_")[0],
                "trial": full_df["trial"].values.astype(np.int32),
                "t_rel_s": full_df["t_rel_s"].values.astype(np.float32),
                "gt_intent": full_df["intent_idx"].values.astype(np.int8),
                "proba_rest": probas[:, ir],
                "proba_close": probas[:, ic],
                "proba_open": probas[:, io_],
            })
            all_rows.append(block)
            print(f"  [{pi}/{len(subjects)}] {subject} {session}: {len(block)} windows")

    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_parquet(OUT_PARQUET, index=False)
    print(f"\nWrote {OUT_PARQUET}  in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
