"""
Patient-only HGB baseline for Lucchetti — mirror of analysis/physiomio/patient_only_baseline.py.
Trains HGB on cal data only (no GrabMyo), same protocol as per_session_eval.

Output:
  analysis/lucchetti/results/patient_only_per_session.csv
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from analysis.seed import SEED, seed_everything
from ml.train_hgb_v2 import engineer_features
from analysis.lucchetti.per_session_eval import (
    LUCCHETTI_PKL, GRABMYO_META, TEST_PER_CLASS, CLASSES, split_session_lucchetti,
)

OUT_SESSION = PROJECT_ROOT / "analysis" / "lucchetti" / "results" / "patient_only_per_session.csv"


def make_classifier(seed):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=300, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        class_weight="balanced",
    )


def main():
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    t_start = time.time()

    print("Loading + engineering Lucchetti (60 → 370)...")
    df = pd.read_pickle(LUCCHETTI_PKL)
    eng = engineer_features(df)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]

    subjects = sorted(eng["participant"].unique())
    rows = []
    for pi, subject in enumerate(subjects, 1):
        for session in sorted(eng[eng["participant"] == subject]["session"].unique()):
            s_data = eng[(eng["participant"] == subject) & (eng["session"] == session)].copy()
            test_idx, cal_idx, _ = split_session_lucchetti(s_data, TEST_PER_CLASS, rng)
            if len(test_idx) == 0 or len(cal_idx) == 0:
                rows.append({"participant": subject, "session": session, "status": "skip"})
                continue
            X_cal = s_data.loc[cal_idx, feature_cols].values.astype(np.float32)
            y_cal = s_data.loc[cal_idx, "intent_idx"].values.astype(np.int64)
            X_test = s_data.loc[test_idx, feature_cols].values.astype(np.float32)
            y_test = s_data.loc[test_idx, "intent_idx"].values.astype(np.int64)
            if len(np.unique(y_cal)) < 2:
                rows.append({"participant": subject, "session": session, "status": "single_class_cal"})
                continue
            scaler = StandardScaler()
            X_cal_s = scaler.fit_transform(X_cal)
            X_test_s = scaler.transform(X_test).astype(np.float32)
            clf = make_classifier(SEED)
            t0 = time.time()
            try:
                clf.fit(X_cal_s, y_cal)
            except ValueError as e:
                rows.append({"participant": subject, "session": session,
                             "status": f"fit_error:{str(e)[:50]}"})
                continue
            preds = clf.predict(X_test_s)
            cls_f1 = f1_score(y_test, preds, average=None, labels=CLASSES, zero_division=0)
            rows.append({
                "participant": subject, "session": session, "arm": session.split("_")[0],
                "n_cal_windows": int(len(cal_idx)), "n_test_windows": int(len(test_idx)),
                "acc": accuracy_score(y_test, preds),
                "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
                "f1_rest": cls_f1[0], "f1_close": cls_f1[1], "f1_open": cls_f1[2],
                "fit_time_s": time.time() - t0, "status": "ok",
            })
            pd.DataFrame(rows).to_csv(OUT_SESSION, index=False)

    df_r = pd.DataFrame(rows)
    ok = df_r[df_r["status"] == "ok"]
    print(f"\nPATIENT-ONLY HGB on Lucchetti (n={ok['participant'].nunique()} subjects, {len(ok)} sessions)")
    print(f"  session mean acc: {ok['acc'].mean():.4f}")
    print(f"  per arm: healthy={ok[ok['arm']=='healthy']['acc'].mean():.4f}  impaired={ok[ok['arm']=='impaired']['acc'].mean():.4f}")
    print(f"  per-class F1: rest={ok['f1_rest'].mean():.4f}  close={ok['f1_close'].mean():.4f}  open={ok['f1_open'].mean():.4f}")
    print(f"\n  vs main Lucchetti eval (HGB+GrabMyo+cal): 0.828 session mean")
    print(f"  Δ from GrabMyo transfer: {0.828 - ok['acc'].mean():+.4f}")
    print(f"\nTotal wall: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
