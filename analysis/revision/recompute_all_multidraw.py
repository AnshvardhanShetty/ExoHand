"""
Multi-draw chronic 2×2 cells extended to ALL 48 patients (not just chronic).

Copy of recompute_chronic_multidraw.py with the chronic filter removed.
Writes to a separate CSV so chronic-only results stay intact.

For each of 48 targets, 5 independent 47-donor subsamples of 432 windows.

Outputs:
  analysis/revision/results/all_multidraw_per_patient.csv
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

from analysis.seed import SEED, seed_everything
from analysis.physiomio.per_session_eval import PHYSIOMIO_PKL, GRABMYO_META
from analysis.revision.recompute_leakage_free_zscore import engineer_features_leakage_free

FROZEN = PROJECT_ROOT / "analysis" / "revision" / "frozen_splits.parquet"
OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "all_multidraw_per_patient.csv"

N_TARGET = 432
N_DRAWS = 5
DRAW_SEEDS = [SEED + k for k in range(N_DRAWS)]


def make_hgb():
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=SEED,
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


def fit_score(X_tr, y_tr, X_te, y_te):
    if len(np.unique(y_tr)) < 2:
        return np.nan
    sc = StandardScaler().fit(X_tr)
    return float(accuracy_score(y_te, make_hgb().fit(sc.transform(X_tr), y_tr).predict(sc.transform(X_te))))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    t0 = time.time()

    print("Loading + leakage-free engineering...")
    df = pd.read_pickle(PHYSIOMIO_PKL).reset_index(drop=True)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]

    frozen = pd.read_parquet(FROZEN)
    cal_mask = pd.Series(False, index=df.index)
    per_patient = {}
    for _, r in frozen.iterrows():
        cal_mask.loc[r["cal_idx"]] = True
        per_patient.setdefault(r["patient"], {})[r["session"]] = {
            "cal_idx": list(r["cal_idx"]), "test_idx": list(r["test_idx"]),
        }
    df_eng = engineer_features_leakage_free(df.copy(), cal_mask)

    # ALL patients with both arms — no chronic filter
    all_patients = sorted([
        p for p in per_patient
        if "impaired_01" in per_patient[p] and "healthy_01" in per_patient[p]
    ], key=lambda s: int(s.replace("patient", "")))
    print(f"All patients with both arms: {len(all_patients)}")

    all_patients_with_imp = sorted([p for p in per_patient if "impaired_01" in per_patient[p]],
                                     key=lambda s: int(s.replace("patient", "")))

    print("Extracting blocks...")
    blocks = {}
    for p in per_patient:
        b = {}
        if "impaired_01" in per_patient[p]:
            b["imp_X"] = df_eng.loc[per_patient[p]["impaired_01"]["cal_idx"], feature_cols].fillna(0).values.astype(np.float32)
            b["imp_y"] = df_eng.loc[per_patient[p]["impaired_01"]["cal_idx"], "intent_idx"].values.astype(np.int64)
            b["test_X"] = df_eng.loc[per_patient[p]["impaired_01"]["test_idx"], feature_cols].fillna(0).values.astype(np.float32)
            b["test_y"] = df_eng.loc[per_patient[p]["impaired_01"]["test_idx"], "intent_idx"].values.astype(np.int64)
        if "healthy_01" in per_patient[p]:
            b["hlth_X"] = df_eng.loc[per_patient[p]["healthy_01"]["cal_idx"], feature_cols].fillna(0).values.astype(np.float32)
            b["hlth_y"] = df_eng.loc[per_patient[p]["healthy_01"]["cal_idx"], "intent_idx"].values.astype(np.int64)
        blocks[p] = b

    rows = []
    done = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done = set(existing["target"].tolist())
        print(f"Resume: {len(rows)} done")

    for i, target in enumerate(all_patients, 1):
        if target in done:
            continue
        b_target = blocks[target]

        others_imp_X = np.vstack([blocks[p]["imp_X"] for p in all_patients_with_imp if p != target and "imp_X" in blocks[p]])
        others_imp_y = np.concatenate([blocks[p]["imp_y"] for p in all_patients_with_imp if p != target and "imp_y" in blocks[p]])
        others_hlth_X = np.vstack([blocks[p]["hlth_X"] for p in all_patients_with_imp if p != target and "hlth_X" in blocks[p]])
        others_hlth_y = np.concatenate([blocks[p]["hlth_y"] for p in all_patients_with_imp if p != target and "hlth_y" in blocks[p]])

        imp_accs, hlth_accs = [], []
        for draw_seed in DRAW_SEEDS:
            rng_i = np.random.RandomState(draw_seed * 1000 + hash(target) % 1000)
            rng_h = np.random.RandomState(draw_seed * 1000 + hash(target) % 1000 + 1)
            X_i, y_i = stratified_subsample(others_imp_X, others_imp_y, N_TARGET, rng_i)
            X_h, y_h = stratified_subsample(others_hlth_X, others_hlth_y, N_TARGET, rng_h)
            imp_accs.append(fit_score(X_i, y_i, b_target["test_X"], b_target["test_y"]))
            hlth_accs.append(fit_score(X_h, y_h, b_target["test_X"], b_target["test_y"]))

        r = {
            "target": target,
            "imp_mean": float(np.mean(imp_accs)), "imp_std": float(np.std(imp_accs)),
            "hlth_mean": float(np.mean(hlth_accs)), "hlth_std": float(np.std(hlth_accs)),
            "gap_mean": float(np.mean(imp_accs) - np.mean(hlth_accs)),
        }
        for k, v in enumerate(imp_accs):
            r[f"imp_draw{k}"] = v
        for k, v in enumerate(hlth_accs):
            r[f"hlth_draw{k}"] = v
        rows.append(r)
        elapsed = time.time() - t0
        eta = elapsed / max(1, i - len(done)) * (len(all_patients) - i)
        print(f"[{i}/{len(all_patients)}] {target}  "
              f"imp={r['imp_mean']:.3f}  hlth={r['hlth_mean']:.3f}  gap={r['gap_mean']:+.3f}  "
              f"[{elapsed/60:.1f}min, eta {eta/60:.0f}min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    main()
