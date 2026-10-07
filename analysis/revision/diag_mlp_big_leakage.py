"""
Diagnostic: is MLP-big's 0.89 cal-only test acc a leakage artefact?

Hypothesis: engineered features (per-participant z-score, per-session z-score) use
test-set statistics in their normalization, so training features carry test info.
MLP-big (1.1M params) exploits this more than HGB.

Test: fit MLP-big on the SAME 7 patients using ONLY the 60 raw base features
(no engineered features, no z-score). Compare to the leaky engineered version.

If MLP-big raw ≈ 0.89 → not a leakage artefact, MLP-big generalizes well from 432 examples.
If MLP-big raw ≪ 0.89 → engineered-feature leakage was the source.
"""

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from analysis.seed import SEED, seed_everything
from analysis.physiomio.per_session_eval import PHYSIOMIO_PKL, TEST_PER_CLASS
from analysis.revision.recompute_A_lda_at_cal_sizes import split_at

CAL_SIZE = 36
N_PATIENTS_TEST = 7  # same subset as the interim capacity sweep

# The 60 raw base features (before any engineering)
CHANNELS = [0, 4, 9, 13]
BASE_FEATS = ["rms", "mav", "var", "wl", "maxamp", "zc", "ssc", "wamp", "iemg",
              "mean_freq", "median_freq", "env_mean", "env_max", "env_std", "env_rms"]


def make_mlp_big(seed=SEED):
    return MLPClassifier(
        hidden_layer_sizes=(1024, 512, 256, 128), activation="relu", solver="adam",
        alpha=1e-6, batch_size=512, learning_rate_init=1e-3,
        max_iter=100, early_stopping=False, n_iter_no_change=100, random_state=seed,
    )


def main():
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    t0 = time.time()

    print("Loading raw PhysioMio (no engineering)...")
    df = pd.read_pickle(PHYSIOMIO_PKL)
    raw_cols = [f"ch{ch}_{f}" for ch in CHANNELS for f in BASE_FEATS
                if f"ch{ch}_{f}" in df.columns]
    print(f"  Using {len(raw_cols)} raw base features (no z-score, no temporal, no cross-channel)")

    patients = sorted(df["participant"].unique(),
                       key=lambda s: int(s.replace("patient", "")))[:N_PATIENTS_TEST]

    results = []
    for i, patient in enumerate(patients, 1):
        s01 = df[(df.participant == patient) & (df.session == "impaired_01")]
        if len(s01) == 0:
            continue
        try:
            test_idx, cal_idx = split_at(s01, CAL_SIZE, TEST_PER_CLASS, rng)
        except Exception as e:
            print(f"  {patient}: split failed ({e})")
            continue
        if len(test_idx) < 15 or len(cal_idx) < 6:
            continue

        X_cal = s01.loc[cal_idx, raw_cols].fillna(0).values.astype(np.float32)
        y_cal = s01.loc[cal_idx, "intent_idx"].values.astype(np.int64)
        X_test = s01.loc[test_idx, raw_cols].fillna(0).values.astype(np.float32)
        y_test = s01.loc[test_idx, "intent_idx"].values.astype(np.int64)

        # Fit MLP-big on raw features (cal-only)
        sc = StandardScaler().fit(X_cal)
        clf = make_mlp_big()
        clf.fit(sc.transform(X_cal), y_cal)
        train_acc = float(accuracy_score(y_cal, clf.predict(sc.transform(X_cal))))
        test_acc = float(accuracy_score(y_test, clf.predict(sc.transform(X_test))))

        elapsed = time.time() - t0
        results.append({
            "patient": patient, "n_cal": len(X_cal), "n_test": len(X_test),
            "train_acc": train_acc, "test_acc": test_acc,
        })
        print(f"[{i}/{len(patients)}] {patient}  train={train_acc:.4f}  test={test_acc:.4f}  "
              f"[{elapsed/60:.1f}min]", flush=True)

    out = pd.DataFrame(results)
    print(f"\n=== SUMMARY (n={len(out)}) ===")
    print(f"MLP-big on RAW 60 base features, cal-only, cal=36")
    print(f"Mean train_acc: {out.train_acc.mean():.4f}")
    print(f"Mean test_acc:  {out.test_acc.mean():.4f}")
    print(f"Median test:    {out.test_acc.median():.4f}")
    print()
    print(f"COMPARE: with engineered (leaky) features, MLP-big cal-only got 0.89")
    print(f"If raw ≪ 0.89 → engineered-feature leakage was propping up MLP-big")
    print(f"If raw ≈ 0.89 → clean splits, MLP-big genuinely generalizes")


if __name__ == "__main__":
    main()
