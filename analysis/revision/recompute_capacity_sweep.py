"""
Revision — capacity sweep (mentor's #1 priority).

Question: does GrabMyo's contribution (GM+cal − cal-only) grow with model capacity?

The paper's null result (GrabMyo doesn't help at 22s cal, HGB capacity) could reflect
either (a) pretraining is truly redundant regardless of capacity, or (b) HGB is
capacity-starved so it can't exploit GrabMyo. ReactEMG Stroke uses a much higher-capacity
transformer and reports pretraining benefit. If (b), their result and ours are
consistent — different points on the capacity curve. If (a), pretraining is redundant
at every capacity we can measure.

For each patient, 48 patients, cal_per_gesture ∈ {3, 6, 12, 24, 36}, two arms
(cal-only, GM+cal), three capacities:
  - LDA on 16 Hudgins features (low capacity)
  - HGB on 370 features (medium capacity)
  - MLP on 370 features, [256, 128], relu, adam (higher capacity)

Compute delta = GM+cal_acc − cal_only_acc for each (capacity, cal_size, patient).
If delta grows with capacity → capacity-starvation was the confound.
If delta stays near zero at all capacities → pretraining is redundant here.

Resumable.
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from analysis.seed import SEED, seed_everything
from ml.train_hgb_v2 import engineer_features
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, GRABMYO_META, GRABMYO_CACHE, TEST_PER_CLASS, CAL_WEIGHT,
)
from analysis.revision.recompute_A_lda_at_cal_sizes import split_at, CAL_SIZES

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "capacity_sweep_per_patient.csv"
OUT_MD = OUT_DIR / "capacity_sweep_summary.md"

CHANNELS = [0, 4, 9, 13]
HUDGINS = ["mav", "wl", "zc", "ssc"]


def make_hgb(seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def make_mlp(seed=SEED):
    # Small MLP — enough capacity to fit non-linearities of 370 features,
    # kept modest for training speed on large GrabMyo+cal sets.
    return MLPClassifier(
        hidden_layer_sizes=(256, 128), activation="relu", solver="adam",
        alpha=1e-4, batch_size=1024, learning_rate_init=1e-3,
        max_iter=40, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=5, random_state=seed,
    )


def fit_lda(X, y, sample_weight=None):
    if len(np.unique(y)) < 2:
        return None
    sc = StandardScaler().fit(X)
    clf = LinearDiscriminantAnalysis()
    try:
        # LDA in sklearn doesn't take sample_weight directly; ignore
        clf.fit(sc.transform(X), y)
    except Exception:
        return None
    return (sc, clf)


def fit_hgb(X, y, sample_weight=None):
    sc = StandardScaler().fit(X)
    clf = make_hgb()
    if sample_weight is not None:
        clf.fit(sc.transform(X), y, sample_weight=sample_weight)
    else:
        clf.fit(sc.transform(X), y)
    return (sc, clf)


def fit_mlp(X, y, sample_weight=None):
    """MLP doesn't support sample_weight; approximate via oversampling cal rows."""
    if sample_weight is not None:
        # Replicate high-weight rows to approximate 100× weighting
        w_int = np.clip(sample_weight, 1.0, None).astype(int)
        idx = np.repeat(np.arange(len(X)), w_int)
        X = X[idx]; y = y[idx]
    sc = StandardScaler().fit(X)
    clf = make_mlp()
    clf.fit(sc.transform(X), y)
    return (sc, clf)


def score(model, X_test, y_test):
    if model is None:
        return np.nan
    sc, clf = model
    return float(accuracy_score(y_test, clf.predict(sc.transform(X_test))))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    t0 = time.time()

    print("Loading + engineering PhysioMio...")
    df = pd.read_pickle(PHYSIOMIO_PKL)
    hudgins_cols = [f"ch{ch}_{f}" for ch in CHANNELS for f in HUDGINS if f"ch{ch}_{f}" in df.columns]
    print(f"  Hudgins cols: {hudgins_cols}")
    eng = engineer_features(df)
    with open(GRABMYO_META) as f:
        gm_features = json.load(f)["feature_cols"]

    print("Loading GrabMyo (300k)...")
    gm = pd.read_pickle(GRABMYO_CACHE).sample(300_000, random_state=SEED)
    gm_X_full = gm[gm_features].values.astype(np.float32)
    gm_X_hud = gm[hudgins_cols].values.astype(np.float32)
    gm_y = gm["intent_idx"].values.astype(np.int64)

    rows = []
    done_keys = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done_keys = set(zip(existing.patient, existing.cal_per_gesture,
                             existing.capacity, existing.arm))
        print(f"Resume: {len(rows)} rows")

    patients = sorted(eng["participant"].unique(), key=lambda s: int(s.replace("patient", "")))
    total_units = len(patients) * len(CAL_SIZES) * 3 * 2   # capacity × arm

    for pi, patient in enumerate(patients, 1):
        s01 = eng[(eng.participant == patient) & (eng.session == "impaired_01")]
        if len(s01) == 0:
            continue
        for cal_size in CAL_SIZES:
            try:
                test_idx, cal_idx = split_at(s01, cal_size, TEST_PER_CLASS, rng)
            except Exception:
                continue
            if len(test_idx) < 15 or len(cal_idx) < 3:
                continue
            X_cal_full = s01.loc[cal_idx, gm_features].fillna(0).values.astype(np.float32)
            X_cal_hud = s01.loc[cal_idx, hudgins_cols].fillna(0).values.astype(np.float32)
            y_cal = s01.loc[cal_idx, "intent_idx"].values.astype(np.int64)
            X_test_full = s01.loc[test_idx, gm_features].fillna(0).values.astype(np.float32)
            X_test_hud = s01.loc[test_idx, hudgins_cols].fillna(0).values.astype(np.float32)
            y_test = s01.loc[test_idx, "intent_idx"].values.astype(np.int64)

            configs = [
                ("LDA_16",  X_cal_hud,  X_test_hud,  gm_X_hud,  fit_lda),
                ("HGB_370", X_cal_full, X_test_full, gm_X_full, fit_hgb),
                ("MLP_370", X_cal_full, X_test_full, gm_X_full, fit_mlp),
            ]
            for cap_name, X_cal, X_test, gm_X_cap, fit_fn in configs:
                for arm in ["cal_only", "gm_plus_cal"]:
                    key = (patient, cal_size, cap_name, arm)
                    if key in done_keys:
                        continue
                    try:
                        if arm == "cal_only":
                            model = fit_fn(X_cal, y_cal)
                        else:
                            X_all = np.vstack([gm_X_cap, X_cal])
                            y_all = np.concatenate([gm_y, y_cal])
                            w = np.ones(len(X_all), dtype=np.float32)
                            w[len(gm_X_cap):] = CAL_WEIGHT
                            model = fit_fn(X_all, y_all, sample_weight=w)
                        acc = score(model, X_test, y_test)
                    except Exception as e:
                        print(f"  {patient} c={cal_size} cap={cap_name} arm={arm}: {e}",
                              flush=True)
                        acc = np.nan
                    rows.append({
                        "patient": patient, "cal_per_gesture": cal_size,
                        "capacity": cap_name, "arm": arm, "acc": acc,
                    })
        elapsed = time.time() - t0
        eta = elapsed / max(1, pi) * (len(patients) - pi)
        print(f"[{pi}/{len(patients)}] {patient}  [{elapsed/60:.1f}min, eta {eta/60:.0f}min]",
              flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    # Delta = (GM+cal) - (cal-only) per (patient, cal_size, capacity)
    pivot = out.pivot_table(index=["patient", "cal_per_gesture", "capacity"],
                             columns="arm", values="acc").reset_index()
    pivot["delta_gm_contribution"] = pivot.gm_plus_cal - pivot.cal_only
    delta_grid = pivot.groupby(["cal_per_gesture", "capacity"]).delta_gm_contribution.mean().unstack().round(4)
    accs_pivot = pivot.groupby(["cal_per_gesture", "capacity"]).agg(
        cal_only_mean=("cal_only", "mean"),
        gm_plus_cal_mean=("gm_plus_cal", "mean"),
    ).round(4)

    md = [
        "# Capacity sweep — does GrabMyo's contribution grow with model capacity?",
        "",
        f"n = 48 patients × {len(CAL_SIZES)} cal budgets × 3 capacities × 2 arms.",
        "",
        "## GrabMyo contribution Δ = (GM+cal acc) − (cal-only acc)",
        "",
        "Rows = cal_per_gesture (≈ cal seconds × 0.6), cols = capacity.",
        "",
        delta_grid.to_markdown(),
        "",
        "## Absolute accuracies",
        "",
        accs_pivot.to_markdown(),
        "",
        "## Reading (mentor's framing)",
        "",
        "- If Δ grows with capacity (LDA→HGB→MLP) at a fixed cal size: capacity-starvation",
        "  is the confound. Pretraining helps more with more parameters. Resolves ReactEMG",
        "  Stroke's positive result as an in-data effect.",
        "- If Δ stays near zero at every capacity: pretraining is truly redundant on this",
        "  task at these scales.",
        "- Read across cal sizes for the interaction: at small cal, pretraining should help",
        "  MORE at high capacity (more parameters starved without it).",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
