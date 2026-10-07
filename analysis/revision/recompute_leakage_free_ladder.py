"""
Revision — leakage-free training-source ladder (critical pre-v3 sanity check).

Recomputes the FOUR-row ladder using engineer_features_leakage_free (z-score stats
fit on cal_mask==True rows only, per participant), to check whether the previously
reported 13.6 pp gap between cross-arm PO (0.549) and VM-LOPO (0.684) survives
clean features. If yes → downstream results (C1/C2/C3, cross-arm gap, M1) are safe.
If not → the paper's central claim needs revision before v3 runs.

Ladder rows (all with leakage-free features, cal=36):
  1. Own impaired-arm 22s cal → own impaired-arm test
  2. Own healthy-arm 22s cal → own impaired-arm test (cross-arm PO)
  3. Volume-matched LOPO: 47 other patients' impaired cal (subsampled) → held-out
  4. GrabMyo zero-shot → own impaired-arm test

Cal_mask includes each patient's healthy_01 AND impaired_01 cal windows so both
limbs contribute to per-participant z-score stats. Test rows are never in cal_mask.

Outputs:
  analysis/revision/results/leakage_free_ladder_per_patient.csv
  analysis/revision/results/leakage_free_ladder_summary.md
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
from analysis.revision.recompute_leakage_free_zscore import engineer_features_leakage_free
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, GRABMYO_META, GRABMYO_CACHE, TEST_PER_CLASS,
)
from analysis.revision.recompute_A_lda_at_cal_sizes import split_at

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "leakage_free_ladder_per_patient.csv"
OUT_MD = OUT_DIR / "leakage_free_ladder_summary.md"

CAL_SIZE = 36
GM_SUBSAMPLE = 100_000


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

    # ── Build cal_mask (True for cal rows in healthy_01 AND impaired_01 per patient) ──
    # Must be pandas Series (engineer_features_leakage_free calls .reindex on it).
    print("Determining cal_mask via split_at on both arms per patient...")
    cal_mask = pd.Series(False, index=df.index)
    per_patient_meta = {}
    for patient in sorted(df.participant.unique(), key=lambda s: int(s.replace("patient", ""))):
        # Impaired arm cal + test
        s_imp = df[(df.participant == patient) & (df.session == "impaired_01")]
        if len(s_imp) == 0:
            continue
        try:
            rng_p = np.random.RandomState(SEED)
            imp_test_idx, imp_cal_idx = split_at(s_imp, CAL_SIZE, TEST_PER_CLASS, rng_p)
        except Exception:
            continue
        if len(imp_test_idx) < 15 or len(imp_cal_idx) < 6:
            continue

        # Healthy arm cal (test not used for cross-arm — we only need cal)
        s_hlth = df[(df.participant == patient) & (df.session == "healthy_01")]
        hlth_cal_idx = None
        if len(s_hlth) > 0:
            try:
                rng_p2 = np.random.RandomState(SEED + 1)
                _, hlth_cal_idx = split_at(s_hlth, CAL_SIZE, TEST_PER_CLASS, rng_p2)
                if len(hlth_cal_idx) < 6:
                    hlth_cal_idx = None
            except Exception:
                hlth_cal_idx = None

        cal_mask.loc[imp_cal_idx] = True
        if hlth_cal_idx is not None:
            cal_mask.loc[hlth_cal_idx] = True
        per_patient_meta[patient] = {
            "imp_cal_idx": imp_cal_idx, "imp_test_idx": imp_test_idx,
            "hlth_cal_idx": hlth_cal_idx,
        }
    print(f"  cal_mask True: {int(cal_mask.sum())} / {len(df)} rows across {len(per_patient_meta)} patients")

    # ── Leakage-free engineering ──
    print("Running engineer_features_leakage_free...")
    df_eng = engineer_features_leakage_free(df.copy(), cal_mask)
    with open(GRABMYO_META) as f:
        gm_features = json.load(f)["feature_cols"]

    # ── Load GrabMyo and normalise once (its own preprocessing already applied at cache time) ──
    print(f"Loading GrabMyo ({GM_SUBSAMPLE//1000}k subsample)...")
    gm = pd.read_pickle(GRABMYO_CACHE).sample(GM_SUBSAMPLE, random_state=SEED)
    gm_X = gm[gm_features].values.astype(np.float32)
    gm_y = gm["intent_idx"].values.astype(np.int64)

    # Pre-extract per-patient cal/test blocks
    print("Pre-extracting per-patient blocks...")
    blocks = {}
    for patient, meta in per_patient_meta.items():
        blocks[patient] = {
            "X_imp_cal": df_eng.loc[meta["imp_cal_idx"], gm_features].fillna(0).values.astype(np.float32),
            "y_imp_cal": df_eng.loc[meta["imp_cal_idx"], "intent_idx"].values.astype(np.int64),
            "X_imp_test": df_eng.loc[meta["imp_test_idx"], gm_features].fillna(0).values.astype(np.float32),
            "y_imp_test": df_eng.loc[meta["imp_test_idx"], "intent_idx"].values.astype(np.int64),
        }
        if meta["hlth_cal_idx"] is not None:
            blocks[patient]["X_hlth_cal"] = df_eng.loc[meta["hlth_cal_idx"], gm_features].fillna(0).values.astype(np.float32)
            blocks[patient]["y_hlth_cal"] = df_eng.loc[meta["hlth_cal_idx"], "intent_idx"].values.astype(np.int64)

    # ── Fit GrabMyo-only classifier ONCE for zero-shot arm ──
    print("Fitting GrabMyo-only classifier once for zero-shot...")
    sc_gm = StandardScaler().fit(gm_X)
    clf_gm = make_hgb().fit(sc_gm.transform(gm_X), gm_y)

    # ── Ladder per patient ──
    print("\nRunning ladder per patient...")
    rows = []
    patient_list = sorted(blocks.keys(), key=lambda s: int(s.replace("patient", "")))
    for i, patient in enumerate(patient_list, 1):
        b = blocks[patient]

        # Row 1: own impaired cal → own impaired test
        row1 = fit_and_score(b["X_imp_cal"], b["y_imp_cal"], b["X_imp_test"], b["y_imp_test"])

        # Row 2: own healthy cal → own impaired test (cross-arm PO)
        row2 = np.nan
        if "X_hlth_cal" in b:
            row2 = fit_and_score(b["X_hlth_cal"], b["y_hlth_cal"], b["X_imp_test"], b["y_imp_test"])

        # Row 3: volume-matched LOPO (subsample others' impaired cal to match cross-arm size)
        n_target = len(b["X_hlth_cal"]) if "X_hlth_cal" in b else len(b["X_imp_cal"])
        others_X = np.vstack([blocks[p]["X_imp_cal"] for p in patient_list if p != patient])
        others_y = np.concatenate([blocks[p]["y_imp_cal"] for p in patient_list if p != patient])
        rng_p = np.random.RandomState(abs(hash(patient)) & 0xffffffff)
        X_sub, y_sub = stratified_subsample(others_X, others_y, n_target, rng_p)
        row3 = fit_and_score(X_sub, y_sub, b["X_imp_test"], b["y_imp_test"])

        # Row 4: GrabMyo zero-shot (already-fit classifier, transform test with GrabMyo scaler)
        row4 = float(accuracy_score(b["y_imp_test"], clf_gm.predict(sc_gm.transform(b["X_imp_test"]))))

        rows.append({
            "patient": patient,
            "n_imp_cal": len(b["X_imp_cal"]),
            "n_hlth_cal": len(b.get("X_hlth_cal", [])) if "X_hlth_cal" in b else 0,
            "n_test": len(b["X_imp_test"]),
            "row1_own_imp_cal": row1,
            "row2_cross_arm_own_hlth": row2,
            "row3_vm_lopo": row3,
            "row4_grabmyo_zero_shot": row4,
        })
        elapsed = time.time() - t0
        print(f"[{i}/{len(patient_list)}] {patient}  "
              f"own={row1:.3f}  x-arm={row2:.3f}  vm-lopo={row3:.3f}  zs={row4:.3f}  "
              f"[{elapsed/60:.1f}min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    # Comparisons
    xarm = out.dropna(subset=["row2_cross_arm_own_hlth"])
    vm_gap = xarm.row3_vm_lopo - xarm.row2_cross_arm_own_hlth
    w = wilcoxon(xarm.row3_vm_lopo, xarm.row2_cross_arm_own_hlth, alternative="greater")

    md = [
        "# Leakage-free training-source ladder (pre-v3 sanity check)",
        "",
        f"n = {len(out)} patients (of {len(xarm)} with both arms). cal=36. Features via ",
        "engineer_features_leakage_free (z-score μ/σ fit on cal rows only, per participant).",
        "",
        "## Ladder",
        "",
        "| Row | Training source | Mean acc | Median acc | Previous (leaky) |",
        "|---|---|---:|---:|---:|",
        f"| 1 | Own impaired 22s cal | {out.row1_own_imp_cal.mean():.4f} | {out.row1_own_imp_cal.median():.4f} | 0.877 (paper headline) |",
        f"| 2 | **Cross-arm** (own healthy cal) | {xarm.row2_cross_arm_own_hlth.mean():.4f} | {xarm.row2_cross_arm_own_hlth.median():.4f} | 0.549 |",
        f"| 3 | VM-LOPO (47 others, matched vol) | {out.row3_vm_lopo.mean():.4f} | {out.row3_vm_lopo.median():.4f} | 0.684 |",
        f"| 4 | GrabMyo zero-shot | {out.row4_grabmyo_zero_shot.mean():.4f} | {out.row4_grabmyo_zero_shot.median():.4f} | 0.346 |",
        "",
        "## Critical gap: VM-LOPO vs cross-arm",
        "",
        f"- Previous (leaky): VM-LOPO − cross-arm = 0.684 − 0.549 = +0.136 (13.6 pp)",
        f"- **Leakage-free: VM-LOPO − cross-arm = {vm_gap.mean():+.4f} ({vm_gap.mean()*100:+.1f} pp)**",
        f"- Paired Wilcoxon (VM-LOPO > cross-arm), leakage-free: p = {w.pvalue:.4e}",
        f"- Patients where VM-LOPO > cross-arm: {(vm_gap > 0).sum()}/{len(xarm)}",
        "",
        "## Decision",
        "",
        "- If ordering (row 1 >> row 3 > row 2 > row 4) survives with rough magnitudes",
        "  intact, the paper's central claim is safe. Proceed to capacity sweep v3.",
        "- If cross-arm gap collapses or ordering breaks, the paper's central claim needs",
        "  revision. Do NOT run v3 until this is understood.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(f"\nOwn imp cal:  {out.row1_own_imp_cal.mean():.4f}")
    print(f"Cross-arm:    {xarm.row2_cross_arm_own_hlth.mean():.4f}")
    print(f"VM-LOPO:      {out.row3_vm_lopo.mean():.4f}")
    print(f"Zero-shot:    {out.row4_grabmyo_zero_shot.mean():.4f}")
    print(f"VM-LOPO − cross-arm gap: {vm_gap.mean():+.4f}  (previous leaky: +0.136)")


if __name__ == "__main__":
    main()
