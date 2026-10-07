"""
Revision — capacity sweep v3 (mentor-directed both fixes applied).

FIXES vs v2:
  1. LDA gets oversampling (same as MLP) — treats cal weight = 100× GrabMyo fairly
  2. Features engineered leakage-free (z-score μ/σ fit on cal rows only, per patient)

Four capacities at cal_per_gesture = 36 (paper operating point):
  1. LDA-16        16 features                        ~50 params
  2. HGB-370       370 leakage-free features          ~19k trees-eq
  3. MLP-small     [256, 128] on 370 leak-free feats  ~130k params
  4. MLP-big       [1024, 512, 256, 128] alpha=1e-6, no early stop, 100 iters  ~1.1M params

For every fit, log: test_acc, train_acc, n_params, n_train_effective, params_per_train.

48 patients × 4 capacities × 2 arms = 384 fits. GM subsample = 100k.
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
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, GRABMYO_META, GRABMYO_CACHE, TEST_PER_CLASS, CAL_WEIGHT,
)
from analysis.revision.recompute_A_lda_at_cal_sizes import split_at
from analysis.revision.recompute_leakage_free_zscore import engineer_features_leakage_free

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "capacity_sweep_v3_per_patient.csv"
OUT_MD = OUT_DIR / "capacity_sweep_v3_summary.md"

CAL_SIZE = 36
GM_SUBSAMPLE = 100_000

CHANNELS = [0, 4, 9, 13]
HUDGINS = ["mav", "wl", "zc", "ssc"]


def make_hgb(seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def make_mlp_small(seed=SEED):
    return MLPClassifier(
        hidden_layer_sizes=(256, 128), activation="relu", solver="adam",
        alpha=1e-4, batch_size=1024, learning_rate_init=1e-3,
        max_iter=40, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=5, random_state=seed,
    )


def make_mlp_big(seed=SEED):
    # Underregularised — must be capable of overfitting the training set.
    return MLPClassifier(
        hidden_layer_sizes=(1024, 512, 256, 128), activation="relu", solver="adam",
        alpha=1e-6, batch_size=512, learning_rate_init=1e-3,
        max_iter=100, early_stopping=False,
        n_iter_no_change=100, random_state=seed,
    )


def n_params_lda(clf, n_features, n_classes):
    return int(n_features * n_classes + n_classes)


def n_params_hgb(clf):
    if not hasattr(clf, "_predictors"):
        return int(getattr(clf, "n_iter_", 0)) * 60 * 3
    total = 0
    for iteration_predictors in clf._predictors:
        for tree_pred in iteration_predictors:
            total += tree_pred.nodes.shape[0]
    return int(total)


def n_params_mlp(clf):
    total = 0
    for coef, intercept in zip(clf.coefs_, clf.intercepts_):
        total += coef.size + intercept.size
    return int(total)


def _oversample(X, y, sample_weight):
    if sample_weight is None:
        return X, y
    w = np.clip(sample_weight, 1.0, None).astype(int)
    idx = np.repeat(np.arange(len(X)), w)
    return X[idx], y[idx]


def fit_lda(X, y, sample_weight=None):
    """FIX: LDA now gets oversampling equivalent to the weight ratio, matching MLP."""
    if len(np.unique(y)) < 2:
        return None
    X_o, y_o = _oversample(X, y, sample_weight)
    sc = StandardScaler().fit(X_o)
    clf = LinearDiscriminantAnalysis()
    clf.fit(sc.transform(X_o), y_o)
    return (sc, clf, "lda")


def fit_hgb(X, y, sample_weight=None):
    sc = StandardScaler().fit(X)
    clf = make_hgb()
    if sample_weight is not None:
        clf.fit(sc.transform(X), y, sample_weight=sample_weight)
    else:
        clf.fit(sc.transform(X), y)
    return (sc, clf, "hgb")


def fit_mlp_small(X, y, sample_weight=None):
    X_o, y_o = _oversample(X, y, sample_weight)
    sc = StandardScaler().fit(X_o)
    clf = make_mlp_small()
    clf.fit(sc.transform(X_o), y_o)
    return (sc, clf, "mlp")


def fit_mlp_big(X, y, sample_weight=None):
    X_o, y_o = _oversample(X, y, sample_weight)
    sc = StandardScaler().fit(X_o)
    clf = make_mlp_big()
    clf.fit(sc.transform(X_o), y_o)
    return (sc, clf, "mlp")


def score_and_train(model, X_train, y_train, X_test, y_test):
    if model is None:
        return np.nan, np.nan
    sc, clf, _ = model
    train_acc = float(accuracy_score(y_train, clf.predict(sc.transform(X_train))))
    test_acc = float(accuracy_score(y_test, clf.predict(sc.transform(X_test))))
    return train_acc, test_acc


def get_n_params(model, n_features, n_classes):
    if model is None:
        return np.nan
    sc, clf, kind = model
    if kind == "lda":
        return n_params_lda(clf, n_features, n_classes)
    if kind == "hgb":
        return n_params_hgb(clf)
    if kind == "mlp":
        return n_params_mlp(clf)
    return np.nan


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    t0 = time.time()

    print("Loading raw PhysioMio...")
    df = pd.read_pickle(PHYSIOMIO_PKL).reset_index(drop=True)

    # ── Determine cal_mask across all patients' impaired_01 for leakage-free engineering ──
    print("Determining cal_mask (impaired_01 cal rows across 48 patients)...")
    cal_mask = pd.Series(False, index=df.index)
    per_patient = {}
    patients = sorted(df.participant.unique(), key=lambda s: int(s.replace("patient", "")))
    for patient in patients:
        s01 = df[(df.participant == patient) & (df.session == "impaired_01")]
        if len(s01) == 0:
            continue
        try:
            rng_p = np.random.RandomState(SEED)
            test_idx, cal_idx = split_at(s01, CAL_SIZE, TEST_PER_CLASS, rng_p)
        except Exception:
            continue
        if len(test_idx) < 15 or len(cal_idx) < 6:
            continue
        cal_mask.loc[cal_idx] = True
        per_patient[patient] = {"cal_idx": cal_idx, "test_idx": test_idx}
    print(f"  cal_mask True: {int(cal_mask.sum())} rows across {len(per_patient)} patients")

    # ── Leakage-free feature engineering ──
    print("Engineering features (leakage-free)...")
    df_eng = engineer_features_leakage_free(df.copy(), cal_mask)

    hudgins_cols = [f"ch{ch}_{f}" for ch in CHANNELS for f in HUDGINS if f"ch{ch}_{f}" in df_eng.columns]
    with open(GRABMYO_META) as f:
        gm_features = json.load(f)["feature_cols"]
    print(f"  Hudgins cols: {len(hudgins_cols)}, full features: {len(gm_features)}")

    print(f"Loading GrabMyo ({GM_SUBSAMPLE//1000}k)...")
    gm = pd.read_pickle(GRABMYO_CACHE).sample(GM_SUBSAMPLE, random_state=SEED)
    gm_X_full = gm[gm_features].values.astype(np.float32)
    gm_X_hud = gm[hudgins_cols].values.astype(np.float32)
    gm_y = gm["intent_idx"].values.astype(np.int64)

    rows = []
    done_keys = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done_keys = set(zip(existing.patient, existing.capacity, existing.arm))
        print(f"Resume: {len(rows)} rows")

    patient_list = sorted(per_patient.keys(), key=lambda s: int(s.replace("patient", "")))
    for pi, patient in enumerate(patient_list, 1):
        cal_idx = per_patient[patient]["cal_idx"]
        test_idx = per_patient[patient]["test_idx"]

        X_cal_full = df_eng.loc[cal_idx, gm_features].fillna(0).values.astype(np.float32)
        X_cal_hud = df_eng.loc[cal_idx, hudgins_cols].fillna(0).values.astype(np.float32)
        y_cal = df_eng.loc[cal_idx, "intent_idx"].values.astype(np.int64)
        X_test_full = df_eng.loc[test_idx, gm_features].fillna(0).values.astype(np.float32)
        X_test_hud = df_eng.loc[test_idx, hudgins_cols].fillna(0).values.astype(np.float32)
        y_test = df_eng.loc[test_idx, "intent_idx"].values.astype(np.int64)

        configs = [
            ("LDA_16",    X_cal_hud,  X_test_hud,  gm_X_hud,  16,             fit_lda),
            ("HGB_370",   X_cal_full, X_test_full, gm_X_full, len(gm_features), fit_hgb),
            ("MLP_small", X_cal_full, X_test_full, gm_X_full, len(gm_features), fit_mlp_small),
            ("MLP_big",   X_cal_full, X_test_full, gm_X_full, len(gm_features), fit_mlp_big),
        ]
        for cap_name, X_cal, X_test, gm_X_cap, n_feat, fit_fn in configs:
            for arm in ["cal_only", "gm_plus_cal"]:
                key = (patient, cap_name, arm)
                if key in done_keys:
                    continue
                t1 = time.time()
                try:
                    if arm == "cal_only":
                        X_tr, y_tr = X_cal, y_cal
                        model = fit_fn(X_cal, y_cal)
                    else:
                        X_tr = np.vstack([gm_X_cap, X_cal])
                        y_tr = np.concatenate([gm_y, y_cal])
                        w = np.ones(len(X_tr), dtype=np.float32)
                        w[len(gm_X_cap):] = CAL_WEIGHT
                        model = fit_fn(X_tr, y_tr, sample_weight=w)
                    # Effective training size after oversampling
                    if model is not None and model[2] in ("mlp", "lda") and arm == "gm_plus_cal":
                        n_train_effective = len(gm_X_cap) + int(CAL_WEIGHT) * len(X_cal)
                    else:
                        n_train_effective = len(X_tr)
                    train_acc, test_acc = score_and_train(model, X_tr, y_tr, X_test, y_test)
                    n_p = get_n_params(model, n_feat, 3)
                    fit_time = time.time() - t1
                except Exception as e:
                    print(f"  {patient} cap={cap_name} arm={arm}: {e}", flush=True)
                    train_acc, test_acc, n_p, n_train_effective, fit_time = (np.nan,)*5
                rows.append({
                    "patient": patient, "capacity": cap_name, "arm": arm,
                    "test_acc": test_acc, "train_acc": train_acc,
                    "n_params": n_p, "n_train_effective": n_train_effective,
                    "params_per_train": (n_p / n_train_effective) if n_train_effective and n_p else np.nan,
                    "fit_time_s": fit_time,
                })
        elapsed = time.time() - t0
        eta = elapsed / max(1, pi) * (len(patient_list) - pi)
        latest = [r for r in rows if r["patient"] == patient]
        summary = "  ".join(f"{r['capacity'][:8]}({r['arm'][:3]})={r['test_acc']:.3f}/tr{r['train_acc']:.3f}"
                             for r in latest)
        print(f"[{pi}/{len(patient_list)}] {patient}: {summary}  [{elapsed/60:.1f}min, eta {eta/60:.0f}min]",
              flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return

    pivot = out.pivot_table(index=["patient", "capacity"], columns="arm",
                             values="test_acc").reset_index()
    pivot["delta_gm_contribution"] = pivot.gm_plus_cal - pivot.cal_only
    delta_by_cap = pivot.groupby("capacity").delta_gm_contribution.mean().round(4)
    test_by_cap = out.pivot_table(index="capacity", columns="arm", values="test_acc",
                                    aggfunc="mean").round(4)
    train_by_cap = out.pivot_table(index="capacity", columns="arm", values="train_acc",
                                    aggfunc="mean").round(4)
    params_by_cap = out.groupby("capacity")["n_params"].mean().round(0)

    md = [
        "# Capacity sweep v3 — leakage-free features + LDA weighting fix",
        "",
        f"n = {out.patient.nunique()} patients, cal_per_gesture = {CAL_SIZE}, GrabMyo = {GM_SUBSAMPLE//1000}k.",
        "",
        "## Test accuracy",
        "", test_by_cap.to_markdown(), "",
        "## Train accuracy (starvation vs overfitting indicator)",
        "", train_by_cap.to_markdown(), "",
        "## GrabMyo contribution Δ = (GM+cal test) − (cal-only test), by capacity",
        "", delta_by_cap.to_markdown(), "",
        "## Approximate n_params by capacity",
        "", params_by_cap.to_markdown(), "",
        "## Interpretation",
        "",
        "- Δ growing with capacity → borrowing helps more with more parameters.",
        "- Δ flat across all four (incl. MLP_big) → pretraining truly redundant on this task",
        "  across the range we can measure.",
        "- Train_acc = 1 while test_acc < 1 → overfitting (not starving). If MLP_big shows",
        "  train ≫ test → it CAN overfit, so if it doesn't benefit from GrabMyo the null is",
        "  robust across the capacity range.",
        "- Report as the mentor advised: numbers stand, mechanism claims omitted.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
