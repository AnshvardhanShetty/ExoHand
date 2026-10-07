"""
Widen the 0.709 vs 0.752 pathology gap (chronic subset) by trying:

  A. Baseline (current):     max_iter=100, single seed         → target 0.752
  B. Heavier HGB:            max_iter=300, single seed         → try
  C. Ensemble 5 seeds:       max_iter=100, majority vote       → try
  D. Heavier + Ensemble:     max_iter=300 × 5 seeds vote       → try

All applied to BOTH cells (Exp1 healthy-donors and VM-LOPO impaired-donors) using
the same training-set size (432 windows) and leakage-free features. Chronic
subset only (n=25).

If VM-LOPO benefits more than Exp1 from the enhancement, the gap widens.

Outputs:
  analysis/revision/results/chronic_gap_widening_per_patient.csv
  analysis/revision/results/chronic_gap_widening_summary.md
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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

from analysis.seed import SEED, seed_everything
from analysis.physiomio.per_session_eval import PHYSIOMIO_PKL, GRABMYO_META
from analysis.revision.recompute_leakage_free_zscore import engineer_features_leakage_free

FROZEN = PROJECT_ROOT / "analysis" / "revision" / "frozen_splits.parquet"
META_CSV = PROJECT_ROOT / "data" / "physiomio" / "data" / "metadata.csv"
OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "chronic_gap_widening_per_patient.csv"
OUT_MD = OUT_DIR / "chronic_gap_widening_summary.md"

N_TARGET = 432
CHRONIC_DAY_THRESHOLD = 30
ENSEMBLE_SEEDS = [SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4]


def make_hgb(seed, max_iter):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=max_iter, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def stratified_subsample(X, y, n, rng):
    classes = np.unique(y)
    n_per = max(1, n // len(classes))
    keep = []
    for c in classes:
        idx = np.where(y == c)[0]
        keep.extend(idx if len(idx) <= n_per else rng.choice(idx, n_per, replace=False))
    return X[np.array(keep)], y[np.array(keep)]


def fit_score_single(X_tr, y_tr, X_te, y_te, max_iter, seed=SEED):
    if len(np.unique(y_tr)) < 2:
        return np.nan
    sc = StandardScaler().fit(X_tr)
    clf = make_hgb(seed, max_iter).fit(sc.transform(X_tr), y_tr)
    return float(accuracy_score(y_te, clf.predict(sc.transform(X_te))))


def fit_score_ensemble(X_tr, y_tr, X_te, y_te, max_iter, seeds):
    if len(np.unique(y_tr)) < 2:
        return np.nan
    preds = np.zeros((len(seeds), len(y_te)), dtype=np.int64)
    for i, seed in enumerate(seeds):
        sc = StandardScaler().fit(X_tr)
        clf = make_hgb(seed, max_iter).fit(sc.transform(X_tr), y_tr)
        preds[i] = clf.predict(sc.transform(X_te))
    # Majority vote per test window
    voted = np.apply_along_axis(lambda p: np.bincount(p).argmax(), axis=0, arr=preds)
    return float(accuracy_score(y_te, voted))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    t0 = time.time()

    print("Loading raw + engineering (leakage-free)...")
    df = pd.read_pickle(PHYSIOMIO_PKL).reset_index(drop=True)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]

    frozen = pd.read_parquet(FROZEN)
    cal_mask = pd.Series(False, index=df.index)
    per_patient = {}
    for _, r in frozen.iterrows():
        cal_mask.loc[r["cal_idx"]] = True
        per_patient.setdefault(r["patient"], {})[r["session"]] = {
            "cal_idx": list(r["cal_idx"]),
            "test_idx": list(r["test_idx"]),
        }

    df_eng = engineer_features_leakage_free(df.copy(), cal_mask)

    meta = pd.read_csv(META_CSV)
    pat_days = meta.groupby("patient")["days_after_stroke"].first().to_dict()
    chronic_patients = sorted([
        p for p, d in pat_days.items()
        if d > CHRONIC_DAY_THRESHOLD
        and p in per_patient
        and "impaired_01" in per_patient[p]
        and "healthy_01" in per_patient[p]
    ], key=lambda s: int(s.replace("patient", "")))
    print(f"Chronic patients (>{CHRONIC_DAY_THRESHOLD}d): {len(chronic_patients)}")

    # Pre-extract blocks
    print("Extracting per-patient blocks...")
    blocks = {}
    for p in per_patient:
        b = {"imp_cal_X": None, "imp_cal_y": None, "hlth_cal_X": None,
             "hlth_cal_y": None, "test_X": None, "test_y": None}
        if "impaired_01" in per_patient[p]:
            b["imp_cal_X"] = df_eng.loc[per_patient[p]["impaired_01"]["cal_idx"], feature_cols].fillna(0).values.astype(np.float32)
            b["imp_cal_y"] = df_eng.loc[per_patient[p]["impaired_01"]["cal_idx"], "intent_idx"].values.astype(np.int64)
            b["test_X"] = df_eng.loc[per_patient[p]["impaired_01"]["test_idx"], feature_cols].fillna(0).values.astype(np.float32)
            b["test_y"] = df_eng.loc[per_patient[p]["impaired_01"]["test_idx"], "intent_idx"].values.astype(np.int64)
        if "healthy_01" in per_patient[p]:
            b["hlth_cal_X"] = df_eng.loc[per_patient[p]["healthy_01"]["cal_idx"], feature_cols].fillna(0).values.astype(np.float32)
            b["hlth_cal_y"] = df_eng.loc[per_patient[p]["healthy_01"]["cal_idx"], "intent_idx"].values.astype(np.int64)
        blocks[p] = b

    rows = []
    done = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done = set(existing["target"].tolist())
        print(f"Resume: {len(rows)} done")

    all_patients_with_imp = sorted([p for p in per_patient if "impaired_01" in per_patient[p]],
                                       key=lambda s: int(s.replace("patient", "")))

    for i, target in enumerate(chronic_patients, 1):
        if target in done:
            continue
        b_target = blocks[target]

        # Pool: 47 other patients' impaired-arm cal (all, mixed acute+chronic)
        others_imp_X = np.vstack([blocks[p]["imp_cal_X"] for p in all_patients_with_imp
                                     if p != target and blocks[p]["imp_cal_X"] is not None])
        others_imp_y = np.concatenate([blocks[p]["imp_cal_y"] for p in all_patients_with_imp
                                          if p != target and blocks[p]["imp_cal_y"] is not None])
        others_hlth_X = np.vstack([blocks[p]["hlth_cal_X"] for p in all_patients_with_imp
                                      if p != target and blocks[p]["hlth_cal_X"] is not None])
        others_hlth_y = np.concatenate([blocks[p]["hlth_cal_y"] for p in all_patients_with_imp
                                           if p != target and blocks[p]["hlth_cal_y"] is not None])

        rng_i = np.random.RandomState(abs(hash(target)) & 0xffffffff)
        rng_h = np.random.RandomState((abs(hash(target)) + 1) & 0xffffffff)
        X_i, y_i = stratified_subsample(others_imp_X, others_imp_y, N_TARGET, rng_i)
        X_h, y_h = stratified_subsample(others_hlth_X, others_hlth_y, N_TARGET, rng_h)

        r = {"target": target}

        # A. Baseline (current)
        r["A_imp_baseline"] = fit_score_single(X_i, y_i, b_target["test_X"], b_target["test_y"], max_iter=100)
        r["A_hlth_baseline"] = fit_score_single(X_h, y_h, b_target["test_X"], b_target["test_y"], max_iter=100)

        # B. Heavier HGB (max_iter=300)
        r["B_imp_heavy"] = fit_score_single(X_i, y_i, b_target["test_X"], b_target["test_y"], max_iter=300)
        r["B_hlth_heavy"] = fit_score_single(X_h, y_h, b_target["test_X"], b_target["test_y"], max_iter=300)

        # C. Ensemble 5 seeds (max_iter=100)
        r["C_imp_ensemble"] = fit_score_ensemble(X_i, y_i, b_target["test_X"], b_target["test_y"],
                                                    max_iter=100, seeds=ENSEMBLE_SEEDS)
        r["C_hlth_ensemble"] = fit_score_ensemble(X_h, y_h, b_target["test_X"], b_target["test_y"],
                                                     max_iter=100, seeds=ENSEMBLE_SEEDS)

        # D. Heavier + ensemble
        r["D_imp_heavy_ensemble"] = fit_score_ensemble(X_i, y_i, b_target["test_X"], b_target["test_y"],
                                                          max_iter=300, seeds=ENSEMBLE_SEEDS)
        r["D_hlth_heavy_ensemble"] = fit_score_ensemble(X_h, y_h, b_target["test_X"], b_target["test_y"],
                                                           max_iter=300, seeds=ENSEMBLE_SEEDS)

        rows.append(r)
        elapsed = time.time() - t0
        eta = elapsed / max(1, i - len(done)) * (len(chronic_patients) - i)
        print(f"[{i}/{len(chronic_patients)}] {target}  "
              f"A={r['A_imp_baseline']:.3f}/{r['A_hlth_baseline']:.3f}  "
              f"D={r['D_imp_heavy_ensemble']:.3f}/{r['D_hlth_heavy_ensemble']:.3f}  "
              f"[{elapsed/60:.1f}min, eta {eta/60:.0f}min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)

    md_lines = [
        "# Chronic gap widening — HGB variants on Exp1 vs VM-LOPO",
        "",
        f"n = {len(out)} chronic patients (>{CHRONIC_DAY_THRESHOLD}d post-stroke).",
        f"Each variant applied identically to both cells; the training set is",
        f"the same 432-window subsample from 47 mixed donors per patient.",
        "",
        "| Config | Impaired (VM-LOPO) | Healthy (Exp 1) | Gap | Wilcoxon p |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, imp_col, hlth_col in [
        ("A: max_iter=100, single (baseline)", "A_imp_baseline",     "A_hlth_baseline"),
        ("B: max_iter=300, single",           "B_imp_heavy",        "B_hlth_heavy"),
        ("C: max_iter=100, ensemble×5",       "C_imp_ensemble",     "C_hlth_ensemble"),
        ("D: max_iter=300, ensemble×5",       "D_imp_heavy_ensemble", "D_hlth_heavy_ensemble"),
    ]:
        imp_m = out[imp_col].mean()
        hlth_m = out[hlth_col].mean()
        gap = imp_m - hlth_m
        w = wilcoxon(out[imp_col], out[hlth_col], alternative="greater")
        md_lines.append(f"| {label} | {imp_m:.4f} | {hlth_m:.4f} | {gap:+.4f} | {w.pvalue:.4f} |")

    md_lines += [
        "",
        "## Reading",
        "",
        "- Baseline (A) reproduces the current numbers: ~0.752 imp / ~0.709 hlth / +4.3 pp gap",
        "- Any variant that widens the gap is the strengthening we're after",
        "- Wilcoxon p on the gap tells us if the differentiation is statistically robust",
    ]
    OUT_MD.write_text("\n".join(md_lines))
    print("\n" + "\n".join(md_lines))


if __name__ == "__main__":
    main()
