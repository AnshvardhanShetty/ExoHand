"""
Experiment 1 — the missing 4th cell of the pathology-vs-anatomy 2×2.

We currently have three of the four:
  own healthy arm (1 donor)  → target impaired  = 0.639  (cross-arm)
  47 strangers' impaired arms → target impaired = 0.752  (VM-LOPO)
  own impaired arm (1 donor) → target impaired  = 0.896  (own cal)

Missing: 47 strangers' HEALTHY arms → target impaired.

Without this cell, the +11 pp gap between 0.639 and 0.752 could be either:
  (a) Pathology matters more than anatomy (headline)
  (b) 47 donors > 1 donor regardless of arm (diversity)

This experiment holds donor pool fixed at 47 and flips only the arm (healthy vs impaired).

  If ~0.64 → healthy stays bad even with 47 donors → PATHOLOGY. Headline holds.
  If ~0.75 → healthy does fine with 47 donors → DIVERSITY was the story. Headline
             needs reframing.

For each target patient, subsample 47 other patients' healthy_01 cal windows to
match the anatomy-matched training size (~432 windows), stratified across classes.
Train HGB, test on target's impaired_01 test.

Outputs:
  analysis/revision/results/exp1_others_healthy_per_patient.csv
  analysis/revision/results/exp1_others_healthy_summary.md
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
OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "exp1_others_healthy_per_patient.csv"
OUT_MD = OUT_DIR / "exp1_others_healthy_summary.md"


def make_hgb(seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def stratified_subsample(X, y, n_target, rng):
    classes = np.unique(y)
    n_per_class = max(1, n_target // len(classes))
    keep = []
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) <= n_per_class:
            keep.extend(idx)
        else:
            keep.extend(rng.choice(idx, n_per_class, replace=False))
    return X[np.array(keep)], y[np.array(keep)]


def fit_and_score(X_train, y_train, X_test, y_test):
    if len(np.unique(y_train)) < 2:
        return np.nan
    sc = StandardScaler().fit(X_train)
    clf = make_hgb().fit(sc.transform(X_train), y_train)
    return float(accuracy_score(y_test, clf.predict(sc.transform(X_test))))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    t0 = time.time()

    print("Loading raw PhysioMio...")
    df = pd.read_pickle(PHYSIOMIO_PKL).reset_index(drop=True)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]

    print("Loading frozen splits...")
    frozen = pd.read_parquet(FROZEN)
    cal_mask = pd.Series(False, index=df.index)
    per_patient = {}
    for _, r in frozen.iterrows():
        cal_mask.loc[r["cal_idx"]] = True
        per_patient.setdefault(r["patient"], {})[r["session"]] = {
            "cal_idx": list(r["cal_idx"]),
            "test_idx": list(r["test_idx"]),
        }
    keep_patients = [p for p in per_patient if
                     "impaired_01" in per_patient[p] and "healthy_01" in per_patient[p]]
    print(f"  {len(keep_patients)} patients with both arms")

    print("Engineering features (leakage-free)...")
    df_eng = engineer_features_leakage_free(df.copy(), cal_mask)

    # Pre-extract per-patient blocks
    print("Pre-extracting per-patient blocks...")
    blocks = {}
    for p in keep_patients:
        blocks[p] = {
            "X_hlth_cal": df_eng.loc[per_patient[p]["healthy_01"]["cal_idx"], feature_cols].fillna(0).values.astype(np.float32),
            "y_hlth_cal": df_eng.loc[per_patient[p]["healthy_01"]["cal_idx"], "intent_idx"].values.astype(np.int64),
            "X_imp_test": df_eng.loc[per_patient[p]["impaired_01"]["test_idx"], feature_cols].fillna(0).values.astype(np.float32),
            "y_imp_test": df_eng.loc[per_patient[p]["impaired_01"]["test_idx"], "intent_idx"].values.astype(np.int64),
            "n_own_hlth_cal": len(per_patient[p]["healthy_01"]["cal_idx"]),
        }

    rows = []
    done = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done = set(existing["patient"].tolist())
        print(f"Resume: {len(rows)} rows, {len(done)} done")

    patient_list = sorted(blocks.keys(), key=lambda s: int(s.replace("patient", "")))
    for i, target in enumerate(patient_list, 1):
        if target in done:
            continue
        b = blocks[target]

        # Pool 47 OTHER patients' healthy-arm cal, subsample to match anatomy size
        others_X = np.vstack([blocks[p]["X_hlth_cal"] for p in patient_list if p != target])
        others_y = np.concatenate([blocks[p]["y_hlth_cal"] for p in patient_list if p != target])
        n_target = b["n_own_hlth_cal"]
        rng_p = np.random.RandomState(abs(hash(target)) & 0xffffffff)
        X_sub, y_sub = stratified_subsample(others_X, others_y, n_target, rng_p)

        acc = fit_and_score(X_sub, y_sub, b["X_imp_test"], b["y_imp_test"])

        rows.append({
            "target_patient": target,
            "n_donors": len(patient_list) - 1,
            "n_train": len(X_sub),
            "n_test": len(b["y_imp_test"]),
            "others_healthy_arms_acc": acc,
        })
        elapsed = time.time() - t0
        eta = elapsed / max(1, i - len(done)) * (len(patient_list) - i)
        print(f"[{i}/{len(patient_list)}] target={target}  n_train={len(X_sub)}  "
              f"acc={acc:.4f}  [{elapsed/60:.1f}min, eta {eta/60:.0f}min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)

    # Merge with cross-arm and VM-LOPO for per-patient comparisons
    ladder = pd.read_csv(LADDER_CSV)
    ca_csv = OUT_DIR / "cross_arm_same_patient_per_patient.csv"
    vm_csv = OUT_DIR / "lopo_volume_matched_per_patient.csv"

    merged = out.rename(columns={"target_patient": "patient"})
    merged = merged.merge(ladder[["patient", "row2_cross_arm_own_hlth", "row3_vm_lopo",
                                     "row1_own_imp_cal"]],
                             on="patient", how="left")

    # Comparisons
    e1_mean = merged.others_healthy_arms_acc.mean()
    ca_mean = merged.row2_cross_arm_own_hlth.mean()   # from ladder = 0.639
    vm_mean = merged.row3_vm_lopo.mean()               # from ladder = 0.752
    own_mean = merged.row1_own_imp_cal.mean()          # from ladder = 0.896

    # Key tests
    v = merged.dropna(subset=["others_healthy_arms_acc", "row3_vm_lopo"])
    w_vs_vm = wilcoxon(v.row3_vm_lopo, v.others_healthy_arms_acc, alternative="greater")   # VM > E1?

    v2 = merged.dropna(subset=["others_healthy_arms_acc", "row2_cross_arm_own_hlth"])
    w_vs_ca = wilcoxon(v2.others_healthy_arms_acc, v2.row2_cross_arm_own_hlth, alternative="greater")   # E1 > CA?

    md = [
        "# Experiment 1 — the missing 4th cell",
        "",
        "**Question:** does the +11 pp gap between own-healthy (0.639) and 47-others-impaired",
        "(0.752) come from pathology (arm changed) or from diversity (donor pool changed)?",
        "",
        f"n = {len(merged)} target patients.",
        "",
        "## The 2×2",
        "",
        "| Training source | 1 donor | 47 donors |",
        "|---|---:|---:|",
        f"| **healthy arm(s)** | own healthy: **{ca_mean:.4f}** (cross-arm) | 47 others' healthy: **{e1_mean:.4f}** (Exp 1) |",
        f"| **impaired arm(s)** | 1 stranger's impaired: — (not tested) | 47 others' impaired: **{vm_mean:.4f}** (VM-LOPO) |",
        f"| own impaired reference | **{own_mean:.4f}** (own cal) | — |",
        "",
        "## Decision (pre-registered)",
        "",
        f"- If Exp 1 ≈ {ca_mean:.2f} (own-healthy): PATHOLOGY. Healthy stays bad even with 47 donors. Headline holds.",
        f"- If Exp 1 ≈ {vm_mean:.2f} (VM-LOPO): DIVERSITY. Healthy works fine once you have 47 donors. Reframe.",
        f"- If intermediate: mixed contribution from both axes.",
        "",
        "## Statistics",
        "",
        f"- Exp 1 mean vs cross-arm (own-healthy) mean: {e1_mean:.4f} vs {ca_mean:.4f}",
        f"  Paired Wilcoxon (Exp 1 > own-healthy): p = {w_vs_ca.pvalue:.4e}",
        f"- Exp 1 mean vs VM-LOPO (own-impaired others) mean: {e1_mean:.4f} vs {vm_mean:.4f}",
        f"  Paired Wilcoxon (VM-LOPO > Exp 1): p = {w_vs_vm.pvalue:.4e}",
        "",
        "## Reading",
        "",
        f"- Exp 1 vs cross-arm: quantifies the pure diversity contribution (going 1→47 donors, all healthy)",
        f"- Exp 1 vs VM-LOPO: quantifies the pure pathology contribution (going healthy→impaired, all 47 donors)",
        f"- Sum ≈ own-healthy → 47-others-impaired total gap of {vm_mean - ca_mean:.3f}",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print()
    print(f"=== HEADLINE DECISION ===")
    print(f"47 others' healthy arms → target impaired: {e1_mean:.4f}")
    print(f"  vs own healthy (0.639):  {'HEALTHY IS STILL BAD' if abs(e1_mean - 0.639) < abs(e1_mean - 0.752) else 'HEALTHY WORKS WITH DIVERSITY'}")
    print(f"  vs VM-LOPO   (0.752):  Δ = {vm_mean - e1_mean:+.4f}")


if __name__ == "__main__":
    main()
