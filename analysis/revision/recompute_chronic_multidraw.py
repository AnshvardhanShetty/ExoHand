"""
Chronic 2×2 with multi-draw donor sampling (stability + honest mean).

For each of 25 chronic targets:
  For each of 5 draws:
    subsample 47 others' impaired-arm cal to 432 windows → train → score
    subsample 47 others' healthy-arm cal to 432 windows → train → score
  average the 5 scores per target per cell
across chronic targets, take the mean.

If the current 0.709 / 0.752 numbers were one favorable draw, this reveals it.

Outputs:
  analysis/revision/results/chronic_multidraw_per_patient.csv
  analysis/revision/results/chronic_multidraw_summary.md
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
OUT_CSV = OUT_DIR / "chronic_multidraw_per_patient.csv"
OUT_MD = OUT_DIR / "chronic_multidraw_summary.md"

N_TARGET = 432
CHRONIC_DAY_THRESHOLD = 30
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

    meta = pd.read_csv(META_CSV)
    pat_days = meta.groupby("patient")["days_after_stroke"].first().to_dict()
    chronic_patients = sorted([
        p for p, d in pat_days.items()
        if d > CHRONIC_DAY_THRESHOLD and p in per_patient
        and "impaired_01" in per_patient[p] and "healthy_01" in per_patient[p]
    ], key=lambda s: int(s.replace("patient", "")))
    all_patients_with_imp = sorted([p for p in per_patient if "impaired_01" in per_patient[p]],
                                     key=lambda s: int(s.replace("patient", "")))
    print(f"Chronic targets: {len(chronic_patients)}. Donor pool per target: 47.")

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

    for i, target in enumerate(chronic_patients, 1):
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
        eta = elapsed / max(1, i - len(done)) * (len(chronic_patients) - i)
        print(f"[{i}/{len(chronic_patients)}] {target}  "
              f"imp={r['imp_mean']:.3f}±{r['imp_std']:.3f}  "
              f"hlth={r['hlth_mean']:.3f}±{r['hlth_std']:.3f}  "
              f"gap={r['gap_mean']:+.3f}  "
              f"[{elapsed/60:.1f}min, eta {eta/60:.0f}min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    imp_grand = out.imp_mean.mean()
    hlth_grand = out.hlth_mean.mean()
    gap_grand = imp_grand - hlth_grand
    w = wilcoxon(out.imp_mean, out.hlth_mean, alternative="greater")

    boot = np.random.RandomState(SEED)
    gaps_boot = []
    for _ in range(2000):
        idx = boot.choice(len(out), len(out), replace=True)
        gaps_boot.append(out.imp_mean.iloc[idx].mean() - out.hlth_mean.iloc[idx].mean())
    ci_lo, ci_hi = np.percentile(gaps_boot, [2.5, 97.5])

    md = [
        "# Chronic 2×2 — multi-draw donor sampling",
        "",
        f"n = {len(out)} chronic patients (>{CHRONIC_DAY_THRESHOLD}d).",
        f"Per-target: {N_DRAWS} independent 47-donor subsamples of 432 windows, avg per target.",
        "",
        "| Cell | Chronic mean (multi-draw) | Prior single-draw |",
        "|---|---:|---:|",
        f"| 47 others' impaired → chronic imp target | **{imp_grand:.4f}** | 0.7525 |",
        f"| 47 others' healthy  → chronic imp target | **{hlth_grand:.4f}** | 0.7080 |",
        f"| Pathology gap                              | **{gap_grand:+.4f}** | +0.0445 |",
        "",
        "Statistics:",
        f"- Paired Wilcoxon (imp > hlth): p = {w.pvalue:.4f}",
        f"- Bootstrap 95% CI for gap: [{ci_lo:+.4f}, {ci_hi:+.4f}]",
        f"- Per-patient draw std, mean: imp {out.imp_std.mean():.3f}, hlth {out.hlth_std.mean():.3f}",
    ]
    OUT_MD.write_text("\n".join(md))
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
