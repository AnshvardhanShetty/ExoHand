"""
Patient-only HGB baseline: same cal/test split as per_session_eval.py, but
the model is trained ONLY on the patient's cal data (no GrabMyo transfer).

Question this answers: does GrabMyo pretraining matter, or is "fit anything
on the patient's 60 s of cued data" enough?

If patient-only ≈ GrabMyo+cal → transfer is decoration.
If patient-only << GrabMyo+cal → transfer is doing real work.

Same fast HGB config (max_iter=300, depth=10, class_weight='balanced') so the
only difference between this and per_session_eval.py is the training set.

Output:
  analysis/physiomio/results/patient_only_per_session.csv
  analysis/physiomio/results/patient_only_per_patient.csv
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
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, GRABMYO_META, TEST_PER_CLASS, CLASSES, split_session,
)

OUT_SESSION = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "patient_only_per_session.csv"
OUT_PATIENT = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "patient_only_per_patient.csv"


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

    print("Loading + engineering PhysioMio (60 → 370)...")
    physiomio = pd.read_pickle(PHYSIOMIO_PKL)
    t = time.time()
    eng = engineer_features(physiomio)
    print(f"  done in {time.time()-t:.1f}s  shape: {eng.shape}")

    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]

    patients = sorted(eng["participant"].unique(),
                      key=lambda s: int(s.replace("patient", "")))
    rows = []
    for pi, patient in enumerate(patients, 1):
        p_data = eng[eng["participant"] == patient]
        sessions = sorted(p_data["session"].unique())
        for si, session in enumerate(sessions, 1):
            s_data = p_data[p_data["session"] == session].copy()
            test_idx, cal_idx, class_counts = split_session(s_data, TEST_PER_CLASS, rng)
            if len(test_idx) == 0 or len(cal_idx) == 0:
                rows.append({"participant": patient, "session": session, "status": "skip"})
                continue
            X_cal = s_data.loc[cal_idx, feature_cols].values.astype(np.float32)
            y_cal = s_data.loc[cal_idx, "intent_idx"].values.astype(np.int64)
            X_test = s_data.loc[test_idx, feature_cols].values.astype(np.float32)
            y_test = s_data.loc[test_idx, "intent_idx"].values.astype(np.int64)

            # Train on cal only — no GrabMyo
            scaler = StandardScaler()
            X_cal_s = scaler.fit_transform(X_cal)
            X_test_s = scaler.transform(X_test).astype(np.float32)
            clf = make_classifier(SEED)
            t0 = time.time()
            try:
                clf.fit(X_cal_s, y_cal)
            except ValueError as e:
                # Sometimes too-few-classes / too-few-samples errors
                rows.append({"participant": patient, "session": session,
                             "status": f"fit_error:{str(e)[:50]}"})
                continue
            fit_t = time.time() - t0
            preds = clf.predict(X_test_s)
            acc = accuracy_score(y_test, preds)
            f1m = f1_score(y_test, preds, average="macro", zero_division=0)
            cls_f1 = f1_score(y_test, preds, average=None, labels=CLASSES, zero_division=0)
            rows.append({
                "participant": patient, "session": session,
                "arm": session.split("_")[0],
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

    # Per patient
    pat_rows = []
    for sid, g in ok.groupby("participant"):
        pat_rows.append({
            "participant": sid, "n_sessions": len(g),
            "acc_mean": g["acc"].mean(), "acc_std": g["acc"].std(),
            "f1_mean": g["f1_macro"].mean(),
            "f1_rest_mean": g["f1_rest"].mean(),
            "f1_close_mean": g["f1_close"].mean(),
            "f1_open_mean": g["f1_open"].mean(),
        })
    pd.DataFrame(pat_rows).to_csv(OUT_PATIENT, index=False)

    print("\n" + "=" * 70)
    print(f"PATIENT-ONLY BASELINE  (n={ok['participant'].nunique()} subjects, {len(ok)} sessions)")
    print("=" * 70)
    print(f"  Session-level acc: mean={ok['acc'].mean():.4f}  std={ok['acc'].std():.4f}")
    print(f"  Per arm: healthy={ok[ok['arm']=='healthy']['acc'].mean():.4f}  impaired={ok[ok['arm']=='impaired']['acc'].mean():.4f}")
    print(f"  Per-class F1: rest={ok['f1_rest'].mean():.4f}  close={ok['f1_close'].mean():.4f}  open={ok['f1_open'].mean():.4f}")
    print(f"\n  vs main eval (GrabMyo+cal): 0.871 session mean")
    print(f"  Δ from GrabMyo transfer: {ok['acc'].mean() - 0.871:+.4f}")
    print(f"\nTotal wall: {(time.time()-t_start)/60:.1f} min")
    print(f"Wrote {OUT_SESSION}, {OUT_PATIENT}")


if __name__ == "__main__":
    main()
