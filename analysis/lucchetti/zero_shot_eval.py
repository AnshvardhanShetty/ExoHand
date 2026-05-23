"""
Lucchetti zero-shot eval — apply the shipped GrabMyo HGB model with NO patient
calibration. Mirrors `analysis/physiomio/zero_shot_eval.py`.

Output:
  analysis/lucchetti/results/zero_shot_per_session.csv
  analysis/lucchetti/results/zero_shot_per_subject.csv
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
from sklearn.metrics import accuracy_score, f1_score

from analysis.seed import SEED, seed_everything
from ml.train_hgb_v2 import engineer_features

LUCCHETTI_PKL = PROJECT_ROOT / "data" / "lucchetti_features_60_per_subject.pkl"
MODEL_PKL = PROJECT_ROOT / "grabmyo" / "improved_hgb_model.pkl"
SCALER_PKL = PROJECT_ROOT / "grabmyo" / "improved_hgb_scaler.pkl"
META = PROJECT_ROOT / "grabmyo" / "improved_hgb_meta.json"

OUT_DIR = PROJECT_ROOT / "analysis" / "lucchetti" / "results"
OUT_SESSION = OUT_DIR / "zero_shot_per_session.csv"
OUT_SUBJECT = OUT_DIR / "zero_shot_per_subject.csv"
CLASSES = [0, 1, 2]


def main():
    seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Lucchetti zero-shot eval (GrabMyo HGB, no calibration)")
    print("=" * 70)
    print("Loading Lucchetti features + engineering 60 → 370...")
    df = pd.read_pickle(LUCCHETTI_PKL)
    t = time.time()
    eng = engineer_features(df)
    print(f"  done in {time.time()-t:.1f}s  shape: {eng.shape}")

    model = joblib.load(MODEL_PKL)
    scaler = joblib.load(SCALER_PKL)
    with open(META) as f:
        feature_cols = json.load(f)["feature_cols"]
    print(f"  using {len(feature_cols)} features")

    X = eng[feature_cols].values.astype(np.float32)
    y = eng["intent_idx"].values.astype(np.int64)
    X_s = scaler.transform(X).astype(np.float32)
    preds = model.predict(X_s)
    eng["pred_intent"] = preds

    rows = []
    for (subj, session), g in eng.groupby(["participant", "session"]):
        acc = accuracy_score(g["intent_idx"], g["pred_intent"])
        f1m = f1_score(g["intent_idx"], g["pred_intent"], average="macro", zero_division=0)
        cls_f1 = f1_score(g["intent_idx"], g["pred_intent"], average=None, labels=CLASSES, zero_division=0)
        rows.append({
            "participant": subj, "session": session,
            "arm": session.split("_")[0],
            "n_windows": int(len(g)),
            "acc": acc, "f1_macro": f1m,
            "f1_rest": cls_f1[0], "f1_close": cls_f1[1], "f1_open": cls_f1[2],
        })
    sdf = pd.DataFrame(rows)
    sdf.to_csv(OUT_SESSION, index=False)

    # Per subject (average across sessions/arms)
    subj_rows = []
    for subj, g in sdf.groupby("participant"):
        h = g[g["arm"] == "healthy"]
        i = g[g["arm"] == "impaired"]
        subj_rows.append({
            "participant": subj,
            "n_sessions": int(len(g)),
            "acc_mean": g["acc"].mean(),
            "acc_std": g["acc"].std(),
            "acc_healthy": h["acc"].mean() if len(h) else np.nan,
            "acc_impaired": i["acc"].mean() if len(i) else np.nan,
            "f1_mean": g["f1_macro"].mean(),
        })
    pat_df = pd.DataFrame(subj_rows)
    pat_df.to_csv(OUT_SUBJECT, index=False)

    print(f"\nSession-level: mean={sdf['acc'].mean():.4f}  std={sdf['acc'].std():.4f}  n={len(sdf)}")
    print(f"  per arm: healthy mean={sdf[sdf['arm']=='healthy']['acc'].mean():.4f} "
          f"  impaired mean={sdf[sdf['arm']=='impaired']['acc'].mean():.4f}")
    print(f"  per-class F1: rest={sdf['f1_rest'].mean():.4f}  close={sdf['f1_close'].mean():.4f}  open={sdf['f1_open'].mean():.4f}")
    print(f"  patient-level mean: {pat_df['acc_mean'].mean():.4f}")
    print(f"\nWrote {OUT_SESSION}")
    print(f"Wrote {OUT_SUBJECT}")


if __name__ == "__main__":
    main()
