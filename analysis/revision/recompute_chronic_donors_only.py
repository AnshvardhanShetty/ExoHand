"""
Chronic-target × chronic-donor pool refinement.

Current numbers (chronic targets × 47 mixed donors):
  47 others' healthy → 0.709
  47 others' impaired (VM-LOPO) → 0.752
  Pathology: +4.3 pp

Concern: the 47-donor pool is mixed (23 acute + 24 chronic). Acute donors'
impaired arms haven't diverged from healthy yet, so they add noise to the
"shared chronic signature" argument. Filtering donors to chronic-only should
sharpen the pathology effect.

For each chronic target patient (25 total):
  - Pool from 24 OTHER chronic patients' impaired arms (subsample 432 windows)
  - Pool from 24 OTHER chronic patients' healthy arms (subsample 432 windows)
  - Compare on the target's impaired-arm test set

Outputs:
  analysis/revision/results/chronic_donors_only_per_patient.csv
  analysis/revision/results/chronic_donors_only_summary.md
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
LADDER_CSV = PROJECT_ROOT / "analysis" / "revision" / "results" / "leakage_free_ladder_per_patient.csv"
META_CSV = PROJECT_ROOT / "data" / "physiomio" / "data" / "metadata.csv"
OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "chronic_donors_only_per_patient.csv"
OUT_MD = OUT_DIR / "chronic_donors_only_summary.md"

CHRONIC_DAY_THRESHOLD = 30
N_TARGET_WINDOWS = 432   # match cross-arm training size


def make_hgb(seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
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


def fit_score(X_tr, y_tr, X_te, y_te):
    if len(np.unique(y_tr)) < 2:
        return np.nan
    sc = StandardScaler().fit(X_tr)
    clf = make_hgb().fit(sc.transform(X_tr), y_tr)
    return float(accuracy_score(y_te, clf.predict(sc.transform(X_te))))


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

    # Determine chronic vs acute
    meta = pd.read_csv(META_CSV)
    pat_days = meta.groupby("patient")["days_after_stroke"].first().to_dict()

    chronic_patients = sorted([
        p for p, d in pat_days.items()
        if d > CHRONIC_DAY_THRESHOLD
        and p in per_patient
        and "impaired_01" in per_patient[p]
        and "healthy_01" in per_patient[p]
    ], key=lambda s: int(s.replace("patient", "")))
    print(f"Chronic patients (>{CHRONIC_DAY_THRESHOLD}d post-stroke, both arms): {len(chronic_patients)}")

    # Pre-extract per-patient blocks
    print("Extracting per-patient blocks...")
    blocks = {}
    for p in chronic_patients:
        blocks[p] = {
            "imp_cal_X": df_eng.loc[per_patient[p]["impaired_01"]["cal_idx"], feature_cols].fillna(0).values.astype(np.float32),
            "imp_cal_y": df_eng.loc[per_patient[p]["impaired_01"]["cal_idx"], "intent_idx"].values.astype(np.int64),
            "hlth_cal_X": df_eng.loc[per_patient[p]["healthy_01"]["cal_idx"], feature_cols].fillna(0).values.astype(np.float32),
            "hlth_cal_y": df_eng.loc[per_patient[p]["healthy_01"]["cal_idx"], "intent_idx"].values.astype(np.int64),
            "test_X": df_eng.loc[per_patient[p]["impaired_01"]["test_idx"], feature_cols].fillna(0).values.astype(np.float32),
            "test_y": df_eng.loc[per_patient[p]["impaired_01"]["test_idx"], "intent_idx"].values.astype(np.int64),
        }

    rows = []
    done = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done = set(existing["target"].tolist())
        print(f"Resume: {len(rows)} rows already done")

    for i, target in enumerate(chronic_patients, 1):
        if target in done:
            continue
        b = blocks[target]

        # Pool from OTHER chronic patients only
        others_imp_X = np.vstack([blocks[p]["imp_cal_X"] for p in chronic_patients if p != target])
        others_imp_y = np.concatenate([blocks[p]["imp_cal_y"] for p in chronic_patients if p != target])
        others_hlth_X = np.vstack([blocks[p]["hlth_cal_X"] for p in chronic_patients if p != target])
        others_hlth_y = np.concatenate([blocks[p]["hlth_cal_y"] for p in chronic_patients if p != target])

        rng_imp = np.random.RandomState(abs(hash(target)) & 0xffffffff)
        rng_hlth = np.random.RandomState((abs(hash(target)) + 1) & 0xffffffff)
        X_imp_sub, y_imp_sub = stratified_subsample(others_imp_X, others_imp_y, N_TARGET_WINDOWS, rng_imp)
        X_hlth_sub, y_hlth_sub = stratified_subsample(others_hlth_X, others_hlth_y, N_TARGET_WINDOWS, rng_hlth)

        acc_imp = fit_score(X_imp_sub, y_imp_sub, b["test_X"], b["test_y"])
        acc_hlth = fit_score(X_hlth_sub, y_hlth_sub, b["test_X"], b["test_y"])

        rows.append({
            "target": target,
            "n_donors": len(chronic_patients) - 1,
            "chronic_donors_impaired_acc": acc_imp,
            "chronic_donors_healthy_acc": acc_hlth,
        })
        elapsed = time.time() - t0
        eta = elapsed / max(1, i - len(done)) * (len(chronic_patients) - i)
        print(f"[{i}/{len(chronic_patients)}] {target}  imp={acc_imp:.4f}  hlth={acc_hlth:.4f}  "
              f"[{elapsed/60:.1f}min, eta {eta/60:.0f}min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)

    # Merge with existing ladder to compare against mixed-pool VM-LOPO
    ladder = pd.read_csv(LADDER_CSV)
    m = out.merge(ladder[["patient", "row3_vm_lopo", "row1_own_imp_cal",
                             "row2_cross_arm_own_hlth"]].rename(columns={"patient": "target"}),
                    on="target", how="left")

    imp_mean = out.chronic_donors_impaired_acc.mean()
    hlth_mean = out.chronic_donors_healthy_acc.mean()
    imp_median = out.chronic_donors_impaired_acc.median()
    vm_mixed_mean = m.row3_vm_lopo.mean()
    own_cal_mean = m.row1_own_imp_cal.mean()
    own_hlth_mean = m.row2_cross_arm_own_hlth.mean()

    w = wilcoxon(out.chronic_donors_impaired_acc, out.chronic_donors_healthy_acc,
                  alternative="greater")

    md = [
        "# Chronic-target × chronic-donor pool",
        "",
        f"n = {len(out)} chronic patients (>{CHRONIC_DAY_THRESHOLD}d post-stroke, both arms).",
        f"Donor pool restricted to {len(chronic_patients)-1} OTHER chronic patients (excludes acute donors).",
        f"Subsampled to {N_TARGET_WINDOWS} windows.",
        "",
        "## Chronic 2×2 with clean donor pool",
        "",
        "| Training source | 1 donor (own) | 24 chronic donors |",
        "|---|---:|---:|",
        f"| healthy arm(s) | own healthy: {own_hlth_mean:.4f} | chronic-donors healthy: **{hlth_mean:.4f}** |",
        f"| impaired arm(s) | own imp cal: {own_cal_mean:.4f} | chronic-donors impaired: **{imp_mean:.4f}** |",
        "",
        "## Improvement from filtering pool to chronic-only",
        "",
        f"- VM-LOPO (mixed 47-donor pool, chronic targets): {vm_mixed_mean:.4f}",
        f"- **Chronic-only donor pool (24 donors): {imp_mean:.4f}**",
        f"- Δ from filtering: **{imp_mean - vm_mixed_mean:+.4f}**",
        "",
        "## Pathology contribution (chronic-donors-impaired − chronic-donors-healthy)",
        "",
        f"- Chronic donors impaired: {imp_mean:.4f}",
        f"- Chronic donors healthy: {hlth_mean:.4f}",
        f"- **Pathology contribution: {imp_mean - hlth_mean:+.4f}**",
        f"- Paired Wilcoxon: p = {w.pvalue:.4f}",
        f"- Compare to mixed-pool pathology: +0.0431, p = 0.010",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print()
    print(f"=== CHRONIC-DONORS-ONLY RESULT ===")
    print(f"  Chronic donors' IMPAIRED → target impaired: {imp_mean:.4f}")
    print(f"  Chronic donors' HEALTHY  → target impaired: {hlth_mean:.4f}")
    print(f"  Pathology delta: {imp_mean - hlth_mean:+.4f} (p = {w.pvalue:.4f})")


if __name__ == "__main__":
    main()
