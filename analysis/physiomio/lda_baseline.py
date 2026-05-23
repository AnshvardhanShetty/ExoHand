"""
LDA baseline: identical protocol to per_session_eval.py but uses Linear
Discriminant Analysis instead of HistGradientBoosting. Same GrabMyo + cal
training mix, same sample weighting (100× on cal), same balanced test.

Question this answers: is HGB's representational capacity the source of the
calibration recovery, or does the same recovery happen with the classical
EMG baseline (LDA)? If LDA recovers similarly → "any sufficient classifier
+ weighted refit" suffices and HGB isn't load-bearing.

Output:
  analysis/physiomio/results/lda_per_session.csv
  analysis/physiomio/results/lda_per_patient.csv
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from analysis.seed import SEED, seed_everything
from ml.train_hgb_v2 import engineer_features
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, GRABMYO_CACHE, GRABMYO_META, TEST_PER_CLASS, CAL_WEIGHT,
    CLASSES, split_session,
)

OUT_SESSION = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "lda_per_session.csv"
OUT_PATIENT = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "lda_per_patient.csv"


def main():
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    t_start = time.time()

    print("Loading + engineering PhysioMio (60 → 370)...")
    physiomio = pd.read_pickle(PHYSIOMIO_PKL)
    t = time.time()
    eng = engineer_features(physiomio)
    print(f"  done in {time.time()-t:.1f}s")

    print("Loading GrabMyo cache...")
    g = pd.read_pickle(GRABMYO_CACHE)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]
    grabmyo_X = g[feature_cols].values.astype(np.float32)
    grabmyo_y = g["intent_idx"].values.astype(np.int64)
    print(f"  GrabMyo: {len(grabmyo_X):,} windows × {len(feature_cols)} features")

    patients = sorted(eng["participant"].unique(),
                      key=lambda s: int(s.replace("patient", "")))
    rows = []
    for pi, patient in enumerate(patients, 1):
        p_data = eng[eng["participant"] == patient]
        sessions = sorted(p_data["session"].unique())
        for si, session in enumerate(sessions, 1):
            s_data = p_data[p_data["session"] == session].copy()
            test_idx, cal_idx, _ = split_session(s_data, TEST_PER_CLASS, rng)
            if len(test_idx) == 0:
                rows.append({"participant": patient, "session": session, "status": "skip"})
                continue
            X_cal = s_data.loc[cal_idx, feature_cols].values.astype(np.float32) if len(cal_idx) else np.zeros((0, len(feature_cols)), dtype=np.float32)
            y_cal = s_data.loc[cal_idx, "intent_idx"].values.astype(np.int64) if len(cal_idx) else np.zeros(0, dtype=np.int64)
            X_test = s_data.loc[test_idx, feature_cols].values.astype(np.float32)
            y_test = s_data.loc[test_idx, "intent_idx"].values.astype(np.int64)

            X_all = np.vstack([grabmyo_X, X_cal])
            y_all = np.concatenate([grabmyo_y, y_cal])
            w_all = np.ones(len(X_all), dtype=np.float32)
            w_all[len(grabmyo_X):] = CAL_WEIGHT

            scaler = StandardScaler()
            X_all_s = scaler.fit_transform(X_all)
            X_test_s = scaler.transform(X_test).astype(np.float32)

            # LDA with shrinkage (robust to high-dim small-sample)
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            t0 = time.time()
            try:
                # LDA doesn't support sample_weight in sklearn — so we use sample
                # duplication for the weighted refit. Cal weight 100× → tile cal rows 100×.
                cal_mask = np.arange(len(X_all)) >= len(grabmyo_X)
                X_grab = X_all_s[~cal_mask]
                y_grab = y_all[~cal_mask]
                X_cal_s = X_all_s[cal_mask]
                y_cal_s = y_all[cal_mask]
                # Tile cal data CAL_WEIGHT× to emulate the weighting
                X_train = np.vstack([X_grab, np.tile(X_cal_s, (int(CAL_WEIGHT), 1))])
                y_train = np.concatenate([y_grab, np.tile(y_cal_s, int(CAL_WEIGHT))])
                clf.fit(X_train, y_train)
            except (np.linalg.LinAlgError, ValueError) as e:
                rows.append({"participant": patient, "session": session,
                             "status": f"fit_error:{str(e)[:50]}"})
                continue
            fit_t = time.time() - t0
            preds = clf.predict(X_test_s)
            acc = accuracy_score(y_test, preds)
            f1m = f1_score(y_test, preds, average="macro", zero_division=0)
            cls_f1 = f1_score(y_test, preds, average=None, labels=CLASSES, zero_division=0)
            rows.append({
                "participant": patient, "session": session, "arm": session.split("_")[0],
                "n_cal_windows": int(len(cal_idx)),
                "n_test_windows": int(len(test_idx)),
                "acc": acc, "f1_macro": f1m,
                "f1_rest": cls_f1[0], "f1_close": cls_f1[1], "f1_open": cls_f1[2],
                "fit_time_s": fit_t, "status": "ok",
            })
            pd.DataFrame(rows).to_csv(OUT_SESSION, index=False)
        if pi % 5 == 0:
            print(f"  [{pi}/{len(patients)}] {patient} done, elapsed={time.time()-t_start:.0f}s")

    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"]

    pat_rows = []
    for sid, g_ in ok.groupby("participant"):
        pat_rows.append({
            "participant": sid, "n_sessions": len(g_),
            "acc_mean": g_["acc"].mean(), "acc_std": g_["acc"].std(),
            "f1_mean": g_["f1_macro"].mean(),
            "f1_rest_mean": g_["f1_rest"].mean(),
            "f1_close_mean": g_["f1_close"].mean(),
            "f1_open_mean": g_["f1_open"].mean(),
        })
    pd.DataFrame(pat_rows).to_csv(OUT_PATIENT, index=False)
    print("\n" + "=" * 70)
    print(f"LDA BASELINE (GrabMyo+cal weighted)  (n={ok['participant'].nunique()} subjects, {len(ok)} sessions)")
    print("=" * 70)
    print(f"  Session-level acc: mean={ok['acc'].mean():.4f}  std={ok['acc'].std():.4f}")
    print(f"  Per arm: healthy={ok[ok['arm']=='healthy']['acc'].mean():.4f}  impaired={ok[ok['arm']=='impaired']['acc'].mean():.4f}")
    print(f"  Per-class F1: rest={ok['f1_rest'].mean():.4f}  close={ok['f1_close'].mean():.4f}  open={ok['f1_open'].mean():.4f}")
    print(f"\n  vs HGB+GrabMyo+cal: 0.871 session mean")
    print(f"  Δ HGB-LDA: {0.871 - ok['acc'].mean():+.4f}")
    print(f"\nTotal wall: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
