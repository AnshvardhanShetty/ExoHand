"""
R-STACK — pathology + anatomy stacked, matched volume, leakage-free.

For each patient with both healthy_01 and impaired_01:
  P only:   47 other patients' impaired cal, subsampled to match anatomy size
  A only:   own healthy_01 cal
  P+A:      both combined
All at cal=36, HGB classifier, leakage-free features.

Tests whether the two data sources are complementary or substitutes.

Outputs:
  analysis/revision/results/R_STACK_leakage_free_per_patient.csv
  analysis/revision/results/R_STACK_leakage_free_summary.md
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
LEGACY_CSV = PROJECT_ROOT / "analysis" / "revision" / "results" / "stacked_pathology_anatomy_per_patient.csv"
OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "R_STACK_leakage_free_per_patient.csv"
OUT_MD = OUT_DIR / "R_STACK_leakage_free_summary.md"


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
    rng = np.random.RandomState(SEED)
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

    print("Pre-extracting per-patient blocks...")
    blocks = {}
    for p in keep_patients:
        blocks[p] = {
            "X_imp_cal": df_eng.loc[per_patient[p]["impaired_01"]["cal_idx"], feature_cols].fillna(0).values.astype(np.float32),
            "y_imp_cal": df_eng.loc[per_patient[p]["impaired_01"]["cal_idx"], "intent_idx"].values.astype(np.int64),
            "X_hlth_cal": df_eng.loc[per_patient[p]["healthy_01"]["cal_idx"], feature_cols].fillna(0).values.astype(np.float32),
            "y_hlth_cal": df_eng.loc[per_patient[p]["healthy_01"]["cal_idx"], "intent_idx"].values.astype(np.int64),
            "X_test": df_eng.loc[per_patient[p]["impaired_01"]["test_idx"], feature_cols].fillna(0).values.astype(np.float32),
            "y_test": df_eng.loc[per_patient[p]["impaired_01"]["test_idx"], "intent_idx"].values.astype(np.int64),
        }

    rows = []
    done = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done = set(existing.patient.tolist())
        print(f"Resume: {len(rows)} rows, {len(done)} done")

    patient_list = sorted(blocks.keys(), key=lambda s: int(s.replace("patient", "")))
    for i, held_out in enumerate(patient_list, 1):
        if held_out in done:
            continue
        b = blocks[held_out]

        # Volume-match to healthy-arm cal size
        n_target = len(b["X_hlth_cal"])
        others_X = np.vstack([blocks[p]["X_imp_cal"] for p in patient_list if p != held_out])
        others_y = np.concatenate([blocks[p]["y_imp_cal"] for p in patient_list if p != held_out])
        rng_p = np.random.RandomState(abs(hash(held_out)) & 0xffffffff)
        X_path, y_path = stratified_subsample(others_X, others_y, n_target, rng_p)

        p_acc = fit_and_score(X_path, y_path, b["X_test"], b["y_test"])
        a_acc = fit_and_score(b["X_hlth_cal"], b["y_hlth_cal"], b["X_test"], b["y_test"])
        X_stack = np.vstack([X_path, b["X_hlth_cal"]])
        y_stack = np.concatenate([y_path, b["y_hlth_cal"]])
        pa_acc = fit_and_score(X_stack, y_stack, b["X_test"], b["y_test"])

        rows.append({
            "patient": held_out,
            "n_pathology": len(X_path),
            "n_anatomy": len(b["X_hlth_cal"]),
            "n_stacked": len(X_stack),
            "path_only_acc": p_acc,
            "anat_only_acc": a_acc,
            "stacked_acc": pa_acc,
            "leakage_free": True,
        })
        elapsed = time.time() - t0
        eta = elapsed / max(1, i - len(done)) * (len(patient_list) - i)
        print(f"[{i}/{len(patient_list)}] {held_out}: P={p_acc:.4f}  A={a_acc:.4f}  P+A={pa_acc:.4f}  "
              f"[{elapsed/60:.1f}min, eta {eta/60:.0f}min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    p_mean = out.path_only_acc.mean()
    a_mean = out.anat_only_acc.mean()
    pa_mean = out.stacked_acc.mean()
    best_single = np.maximum(out.path_only_acc, out.anat_only_acc)
    w_vs_best = wilcoxon(out.stacked_acc, best_single, alternative="greater")
    w_vs_path = wilcoxon(out.stacked_acc, out.path_only_acc, alternative="greater")
    w_vs_anat = wilcoxon(out.stacked_acc, out.anat_only_acc, alternative="greater")

    legacy_line = ""
    if LEGACY_CSV.exists():
        leg = pd.read_csv(LEGACY_CSV)
        legacy_line = (f"legacy P={leg.path_only_acc.mean():.4f}  "
                        f"A={leg.anat_only_acc.mean():.4f}  P+A={leg.stacked_acc.mean():.4f}")

    md = [
        "# R-STACK — pathology + anatomy stacked (leakage-free)",
        "",
        f"n = {len(out)} patients, matched volume ({out.n_anatomy.mean():.0f} windows per arm).",
        "",
        "## Results (leakage-free)",
        "",
        "| Arm | Mean acc |",
        "|---|---:|",
        f"| Pathology-matched only (others' impaired, subsampled) | {p_mean:.4f} |",
        f"| Anatomy-matched only (own healthy) | {a_mean:.4f} |",
        f"| Stacked P+A | {pa_mean:.4f} |",
        "",
        f"**Paired Wilcoxon:**",
        f"- Stacked > max(P, A): p = {w_vs_best.pvalue:.4e}",
        f"- Stacked > P alone: p = {w_vs_path.pvalue:.4e}",
        f"- Stacked > A alone: p = {w_vs_anat.pvalue:.4e}",
        "",
        f"Legacy (leaky): {legacy_line}",
        "",
        "## Gate (pre-registered)",
        "",
        "- If P+A > P significantly (p < 0.05) → the two are complementary. Anastasiev",
        "  contradiction is withdrawn; reframe as replication-with-caveats.",
        "- If P+A ≈ P (p ≥ 0.05) → anatomy adds nothing on top of pathology. Original",
        "  claim survives.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
