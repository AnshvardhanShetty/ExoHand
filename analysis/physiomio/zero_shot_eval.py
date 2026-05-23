"""
Stream 2b zero-shot eval — apply the GrabMyo-trained model to PhysioMio
stroke patients with NO calibration. This is the cross-population baseline:
how well does a model trained on healthy EMG generalize to stroke patients
when given no patient-specific data?

Pipeline:
  1. Load PhysioMio 60-feature dataset (produced by ml/preprocessing_physiomio.py).
  2. Apply ml.train_hgb_v2.engineer_features → 370 columns.
  3. Verify column alignment with the GrabMyo-trained model's expected schema.
  4. Apply the trained StandardScaler + HistGradientBoostingClassifier.
  5. Compute per-patient accuracy, macro-F1, per-class metrics.
  6. Per-session breakdown (healthy_arm vs impaired_arm distinction).
  7. Save per-patient CSV + aggregate summary.

The output CSV uses the same schema as loso_results_full43.csv so
aggregate_loso.py can produce bootstrap CIs and statistical tests on top.
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
from sklearn.metrics import accuracy_score, classification_report, f1_score

from analysis.seed import SEED, seed_everything
from ml.train_hgb_v2 import META_COLS, engineer_features


PHYSIOMIO_PKL = PROJECT_ROOT / "data" / "physiomio_features_60_per_patient.pkl"
GRABMYO_MODEL = PROJECT_ROOT / "grabmyo" / "improved_hgb_model.pkl"
GRABMYO_SCALER = PROJECT_ROOT / "grabmyo" / "improved_hgb_scaler.pkl"
GRABMYO_META = PROJECT_ROOT / "grabmyo" / "improved_hgb_meta.json"

OUT_PATIENT_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "zero_shot_per_patient.csv"
OUT_SESSION_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "zero_shot_per_session.csv"


def main():
    seed_everything(SEED)
    t_start = time.time()

    print("=" * 70)
    print("PhysioMio zero-shot eval (GrabMyo-trained model, no calibration)")
    print("=" * 70)

    # 1. Load PhysioMio
    print(f"\n[1/5] Loading PhysioMio features from {PHYSIOMIO_PKL.name}...")
    df = pd.read_pickle(PHYSIOMIO_PKL)
    print(f"      shape: {df.shape}  n_patients: {df['participant'].nunique()}")

    # 2. Engineer features
    print(f"\n[2/5] Engineering features (60 → 370)...")
    t = time.time()
    df = engineer_features(df)
    print(f"      done in {time.time() - t:.1f}s  new shape: {df.shape}")

    # 3. Load model + scaler + expected feature cols
    print(f"\n[3/5] Loading GrabMyo-trained model + scaler...")
    model = joblib.load(GRABMYO_MODEL)
    scaler = joblib.load(GRABMYO_SCALER)
    with open(GRABMYO_META) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    intent_to_idx = meta["intent_to_idx"]
    print(f"      model expects {len(feature_cols)} features")
    print(f"      model intent_to_idx: {intent_to_idx}")

    # Verify column alignment
    missing = [c for c in feature_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in feature_cols and c not in META_COLS]
    if missing:
        print(f"      WARN: {len(missing)} expected cols missing from PhysioMio (examples: {missing[:5]})")
        print(f"            Filling missing cols with 0 — but this means feature engineering diverged.")
        for c in missing:
            df[c] = 0.0
    if extra:
        print(f"      INFO: {len(extra)} extra cols in PhysioMio not in model (dropping them silently).")

    # 4. Predict
    print(f"\n[4/5] Predicting on {len(df):,} windows...")
    X = df[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    t = time.time()
    preds = model.predict(X_scaled)
    print(f"      predict done in {time.time() - t:.1f}s")
    y_true = df["intent_idx"].values.astype(np.int64)
    overall_acc = accuracy_score(y_true, preds)
    overall_f1 = f1_score(y_true, preds, average="macro")
    print(f"\n      OVERALL: acc={overall_acc:.4f}  macro-F1={overall_f1:.4f}")

    # 5. Per-patient + per-session metrics
    print(f"\n[5/5] Computing per-patient and per-session breakdowns...")
    df["pred"] = preds

    # Per-patient
    patient_rows = []
    for p, g in df.groupby("participant"):
        y_p = g["intent_idx"].values
        pred_p = g["pred"].values
        n = len(g)
        acc = accuracy_score(y_p, pred_p)
        f1 = f1_score(y_p, pred_p, average="macro", zero_division=0)
        # Per-class F1
        cls_f1 = f1_score(y_p, pred_p, average=None, labels=[0, 1, 2], zero_division=0)
        # Per-arm breakdown
        healthy = g[g["session"].str.startswith("healthy_")]
        impaired = g[g["session"].str.startswith("impaired_")]
        acc_healthy = accuracy_score(healthy["intent_idx"], healthy["pred"]) if len(healthy) else np.nan
        acc_impaired = accuracy_score(impaired["intent_idx"], impaired["pred"]) if len(impaired) else np.nan
        patient_rows.append({
            "participant": p,
            "n_windows": n,
            "n_healthy_windows": len(healthy),
            "n_impaired_windows": len(impaired),
            "acc_no_cal": acc,                # named to match loso_results_full43 schema
            "acc_with_cal": np.nan,           # filled in by Task #39
            "f1_no_cal": f1,
            "f1_with_cal": np.nan,
            "delta_acc": np.nan,
            "acc_healthy_arm": acc_healthy,
            "acc_impaired_arm": acc_impaired,
            "f1_rest": cls_f1[0],
            "f1_close": cls_f1[1],
            "f1_open": cls_f1[2],
        })
    patient_df = pd.DataFrame(patient_rows).sort_values(
        "participant", key=lambda s: s.str.extract(r"(\d+)").astype(int).iloc[:, 0]
    )

    # Per-session
    session_rows = []
    for (p, sess), g in df.groupby(["participant", "session"]):
        y_s = g["intent_idx"].values
        pred_s = g["pred"].values
        session_rows.append({
            "participant": p,
            "session": sess,
            "arm": "healthy" if sess.startswith("healthy_") else "impaired",
            "n_windows": len(g),
            "acc": accuracy_score(y_s, pred_s),
            "f1_macro": f1_score(y_s, pred_s, average="macro", zero_division=0),
        })
    session_df = pd.DataFrame(session_rows)

    # Save
    OUT_PATIENT_CSV.parent.mkdir(parents=True, exist_ok=True)
    patient_df.to_csv(OUT_PATIENT_CSV, index=False)
    session_df.to_csv(OUT_SESSION_CSV, index=False)
    print(f"      wrote {OUT_PATIENT_CSV}")
    print(f"      wrote {OUT_SESSION_CSV}")

    # Console summary
    print("\n" + "=" * 70)
    print(f"PER-PATIENT SUMMARY (n={len(patient_df)})")
    print("=" * 70)
    print(f"  acc_no_cal:  mean={patient_df['acc_no_cal'].mean():.4f}  std={patient_df['acc_no_cal'].std():.4f}  "
          f"range=[{patient_df['acc_no_cal'].min():.4f}, {patient_df['acc_no_cal'].max():.4f}]")
    print(f"  f1_no_cal:   mean={patient_df['f1_no_cal'].mean():.4f}  std={patient_df['f1_no_cal'].std():.4f}")
    print(f"\n  acc_healthy_arm:  mean={patient_df['acc_healthy_arm'].mean():.4f}  "
          f"std={patient_df['acc_healthy_arm'].std():.4f}  (cross-population on healthy arms)")
    print(f"  acc_impaired_arm: mean={patient_df['acc_impaired_arm'].mean():.4f}  "
          f"std={patient_df['acc_impaired_arm'].std():.4f}  (cross-population on paretic arms)")
    print(f"\n  Per-class F1:")
    print(f"    rest:  mean={patient_df['f1_rest'].mean():.4f}")
    print(f"    close: mean={patient_df['f1_close'].mean():.4f}")
    print(f"    open:  mean={patient_df['f1_open'].mean():.4f}")

    print(f"\nOVERALL (single number, all patients pooled): acc={overall_acc:.4f}  macro-F1={overall_f1:.4f}")
    print(f"\nTotal wall time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
