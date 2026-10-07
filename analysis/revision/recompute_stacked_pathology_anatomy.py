"""
Revision — stacked pathology+anatomy test.

Question: are pathology-matched (other patients' impaired-arm) and anatomy-matched
(same patient's healthy-arm) data COMPLEMENTARY sources of information?

If so, combining both should beat either alone. If not, they're substitutes.

For each held-out patient:
  - Arm P (pathology-matched only): 47 other patients' impaired-arm cal, volume-matched
  - Arm A (anatomy-matched only): this patient's healthy-arm cal
  - Arm P+A (stacked): both combined

All at cal-only (no GrabMyo, since we've established that adds nothing separately).
Training-size matching handled the same way as volume-matched LOPO.

Interpretation:
  - Stacked > max(P, A): complementary, combining is worth it → useful protocol finding
  - Stacked ≈ P or A alone: substitutes, no gain
  - Stacked < P or A: interference (unlikely but possible)

Resumable.

Outputs:
  analysis/revision/results/stacked_pathology_anatomy_per_patient.csv
  analysis/revision/results/stacked_pathology_anatomy_summary.md
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
from ml.train_hgb_v2 import engineer_features
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, GRABMYO_META, TEST_PER_CLASS, split_session,
)

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
CROSS_ARM_CSV = OUT_DIR / "cross_arm_same_patient_per_patient.csv"
OUT_CSV = OUT_DIR / "stacked_pathology_anatomy_per_patient.csv"
OUT_MD = OUT_DIR / "stacked_pathology_anatomy_summary.md"


def make_hgb(seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def stratified_subsample(X, y, n_target, rng):
    classes = np.unique(y)
    n_per_class = max(1, n_target // len(classes))
    idx_keep = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        if len(c_idx) <= n_per_class:
            idx_keep.extend(c_idx)
        else:
            idx_keep.extend(rng.choice(c_idx, n_per_class, replace=False))
    return X[np.array(idx_keep)], y[np.array(idx_keep)]


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

    print("Loading + engineering PhysioMio...")
    df = pd.read_pickle(PHYSIOMIO_PKL)
    eng = engineer_features(df)

    with open(GRABMYO_META) as f:
        gm_features = json.load(f)["feature_cols"]

    cross_arm = pd.read_csv(CROSS_ARM_CSV).set_index("patient")

    # Both healthy_01 and impaired_01 cal + test for each patient
    print("\nExtracting healthy_01 cal + impaired_01 cal/test blocks...")
    patients = sorted(eng["participant"].unique(), key=lambda s: int(s.replace("patient", "")))
    per_patient = {}
    for patient in patients:
        s_imp = eng[(eng.participant == patient) & (eng.session == "impaired_01")]
        s_hlth = eng[(eng.participant == patient) & (eng.session == "healthy_01")]
        if len(s_imp) == 0 or len(s_hlth) == 0:
            continue
        try:
            rng_p = np.random.RandomState(SEED)
            test_idx, imp_cal_idx, _ = split_session(s_imp, TEST_PER_CLASS, rng_p)
            rng_p2 = np.random.RandomState(SEED + 1)
            _, hlth_cal_idx, _ = split_session(s_hlth, TEST_PER_CLASS, rng_p2)
        except Exception:
            continue
        if len(test_idx) < 15 or len(imp_cal_idx) < 6 or len(hlth_cal_idx) < 6:
            continue
        per_patient[patient] = {
            "X_imp_cal": s_imp.loc[imp_cal_idx, gm_features].fillna(0).values.astype(np.float32),
            "y_imp_cal": s_imp.loc[imp_cal_idx, "intent_idx"].values.astype(np.int64),
            "X_hlth_cal": s_hlth.loc[hlth_cal_idx, gm_features].fillna(0).values.astype(np.float32),
            "y_hlth_cal": s_hlth.loc[hlth_cal_idx, "intent_idx"].values.astype(np.int64),
            "X_test": s_imp.loc[test_idx, gm_features].fillna(0).values.astype(np.float32),
            "y_test": s_imp.loc[test_idx, "intent_idx"].values.astype(np.int64),
        }
    print(f"  Kept {len(per_patient)} patients")

    rows = []
    done = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done = set(existing["held_out_patient"].tolist())
        print(f"Resume: {len(rows)} rows, {len(done)} done")

    patient_list = sorted(per_patient.keys(), key=lambda s: int(s.replace("patient", "")))
    for i, held_out in enumerate(patient_list, 1):
        if held_out in done:
            continue
        pat = per_patient[held_out]

        # Pool cal from other 47 patients
        others_X = np.vstack([per_patient[p]["X_imp_cal"] for p in patient_list if p != held_out])
        others_y = np.concatenate([per_patient[p]["y_imp_cal"] for p in patient_list if p != held_out])

        # Volume-match to cross-arm's healthy-arm training size for this patient
        if held_out not in cross_arm.index:
            continue
        n_target = int(cross_arm.loc[held_out, "n_hlth_cal"])
        rng_p = np.random.RandomState(abs(hash(held_out)) & 0xffffffff)
        X_path, y_path = stratified_subsample(others_X, others_y, n_target, rng_p)

        # Arm P: pathology-matched only (volume-matched LOPO subsample)
        p_acc = fit_and_score(X_path, y_path, pat["X_test"], pat["y_test"])
        # Arm A: anatomy-matched only (this patient's healthy-arm cal)
        a_acc = fit_and_score(pat["X_hlth_cal"], pat["y_hlth_cal"], pat["X_test"], pat["y_test"])
        # Arm P+A: stacked
        X_stack = np.vstack([X_path, pat["X_hlth_cal"]])
        y_stack = np.concatenate([y_path, pat["y_hlth_cal"]])
        pa_acc = fit_and_score(X_stack, y_stack, pat["X_test"], pat["y_test"])

        rows.append({
            "held_out_patient": held_out,
            "n_pathology": len(X_path),
            "n_anatomy": len(pat["X_hlth_cal"]),
            "n_stacked": len(X_stack),
            "path_only_acc": p_acc,
            "anat_only_acc": a_acc,
            "stacked_acc": pa_acc,
        })
        elapsed = time.time() - t0
        eta = elapsed / max(1, len(rows) - len(done)) * (len(patient_list) - len(rows))
        print(f"[{i}/{len(patient_list)}] {held_out}: P={p_acc:.4f}  A={a_acc:.4f}  P+A={pa_acc:.4f}  "
              f"[{elapsed/60:.1f}min, eta {eta/60:.0f}min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return

    p_mean = out.path_only_acc.mean()
    a_mean = out.anat_only_acc.mean()
    pa_mean = out.stacked_acc.mean()
    best_single = np.maximum(out.path_only_acc, out.anat_only_acc)

    w_vs_best = wilcoxon(out.stacked_acc, best_single, alternative="greater")
    w_vs_path = wilcoxon(out.stacked_acc, out.path_only_acc, alternative="greater")
    w_vs_anat = wilcoxon(out.stacked_acc, out.anat_only_acc, alternative="greater")

    md = [
        "# Stacked pathology + anatomy test",
        "",
        f"For {len(out)} patients: compare (a) pathology-matched only, (b) anatomy-matched only, (c) stacked.",
        f"All volume-matched to each patient's cross-arm training size (mean n≈{out.n_anatomy.mean():.0f}).",
        "",
        "| Arm | Mean acc | n windows |",
        "|---|---:|---:|",
        f"| **Pathology-matched only** (other patients' impaired arms, subsampled) | **{p_mean:.4f}** | {out.n_pathology.mean():.0f} |",
        f"| **Anatomy-matched only** (this patient's healthy arm) | **{a_mean:.4f}** | {out.n_anatomy.mean():.0f} |",
        f"| **Stacked** (both combined) | **{pa_mean:.4f}** | {out.n_stacked.mean():.0f} |",
        "",
        f"**Paired Wilcoxon:**",
        f"- Stacked > max(P, A): p = {w_vs_best.pvalue:.4e}",
        f"- Stacked > P alone: p = {w_vs_path.pvalue:.4e}",
        f"- Stacked > A alone: p = {w_vs_anat.pvalue:.4e}",
        "",
        "## Interpretation",
        "",
        "- Stacked > max(P, A) significantly → complementary. Combining is worth it.",
        "  Would be a useful protocol finding: use both when available.",
        "- Stacked ≈ P alone → anatomy adds nothing; pathology-matched is sufficient.",
        "- Stacked ≈ A alone → pathology adds nothing; anatomy-matched is sufficient.",
        "- Stacked < either alone → interference (unlikely but possible).",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(f"P only:  {p_mean:.4f}")
    print(f"A only:  {a_mean:.4f}")
    print(f"P+A:     {pa_mean:.4f}")


if __name__ == "__main__":
    main()
