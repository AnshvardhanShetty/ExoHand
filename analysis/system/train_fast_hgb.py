"""
One-shot: fit the lighter HGB configuration used by PhysioMio per_session_eval
on the full GrabMyo feature cache, save model + scaler, so the latency benchmark
can compare against the heavy (max_iter=2500, depth=18) shipped GrabMyo model.

Config matches make_classifier() in per_session_eval.py and longitudinal_eval.py:
  max_iter=300, max_depth=10, max_leaf_nodes=63, l2_reg=0.01, early_stopping=True.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from analysis.seed import SEED, seed_everything

CACHE = PROJECT_ROOT / "analysis" / ".cache" / "grabmyo_features_370.pkl"
META = PROJECT_ROOT / "grabmyo" / "improved_hgb_meta.json"
OUT_MODEL = PROJECT_ROOT / "analysis" / "system" / "results" / "fast_hgb_model.pkl"
OUT_SCALER = PROJECT_ROOT / "analysis" / "system" / "results" / "fast_hgb_scaler.pkl"


def main():
    seed_everything(SEED)
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    print("Loading GrabMyo 370-feature cache...")
    df = pd.read_pickle(CACHE)
    with open(META) as f:
        feat_cols = json.load(f)["feature_cols"]
    X = df[feat_cols].values.astype(np.float32)
    y = df["intent_idx"].values.astype(np.int64)
    print(f"  shape: {X.shape}")

    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    print("Fitting fast HGB (max_iter=300, max_depth=10)...")
    clf = HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=300, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=SEED,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        class_weight="balanced",
    )
    t0 = time.time()
    clf.fit(X_s, y)
    print(f"  done in {time.time() - t0:.0f}s  (n_iter_={clf.n_iter_})")
    print(f"  total trees: {sum(len(p) for p in clf._predictors)}")

    joblib.dump(clf, OUT_MODEL)
    joblib.dump(scaler, OUT_SCALER)
    print(f"  wrote {OUT_MODEL.relative_to(PROJECT_ROOT)}")
    print(f"  wrote {OUT_SCALER.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
