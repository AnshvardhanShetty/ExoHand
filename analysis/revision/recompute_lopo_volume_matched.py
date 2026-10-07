"""
Revision — volume-matched LOPO.

Addresses the confound in the cross-arm vs LOPO comparison: LOPO uses ~20K training
windows (47 patients × ~432), cross-arm uses ~432 (this patient's healthy arm).
The 47× data advantage confounds "pathology-matched > anatomy-matched" with
"more data > less data".

Fix: for each held-out patient, subsample the 47-patient pool to match THAT patient's
cross-arm training size (from cross_arm_same_patient_per_patient.csv). Then rerun LOPO
with matched volume. Also record: LOPO full pool (~20K, previous result) for reference.

Interpretations:
  - If volume-matched LOPO > cross-arm PO → pathology-matched data really is more
    valuable per-window than anatomy-matched data. Publishable claim.
  - If volume-matched LOPO ≈ cross-arm PO → the LOPO advantage was just data volume.
    Cross-arm finding shrinks to "healthy arm doesn't transfer well" without the
    "beats LOPO" claim.
  - If volume-matched LOPO < cross-arm PO → cross-arm actually IS more useful than
    matched-volume other-patient data. Would be surprising but supports per-patient
    specificity over pathology-matching.

Resumable: skips patients already in the CSV.

Outputs:
  analysis/revision/results/lopo_volume_matched_per_patient.csv
  analysis/revision/results/lopo_volume_matched_summary.md
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
OUT_CSV = OUT_DIR / "lopo_volume_matched_per_patient.csv"
OUT_MD = OUT_DIR / "lopo_volume_matched_summary.md"


def make_hgb(seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def stratified_subsample(X, y, n_target, rng):
    """Sample ~n_target windows, stratified across classes to preserve balance."""
    classes = np.unique(y)
    n_per_class = max(1, n_target // len(classes))
    idx_keep = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        if len(c_idx) <= n_per_class:
            idx_keep.extend(c_idx)
        else:
            idx_keep.extend(rng.choice(c_idx, n_per_class, replace=False))
    idx_keep = np.array(idx_keep)
    return X[idx_keep], y[idx_keep]


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

    # Get cross-arm results for per-patient training-size matching
    cross_arm = pd.read_csv(CROSS_ARM_CSV).set_index("patient")
    print(f"Cross-arm reference: mean n_hlth_cal = {cross_arm.n_hlth_cal.mean():.0f}")

    # Extract per-patient cal + test blocks (same as LOPO)
    print("\nExtracting per-patient cal + test blocks (impaired_01)...")
    patients = sorted(eng["participant"].unique(), key=lambda s: int(s.replace("patient", "")))
    per_patient = {}
    for patient in patients:
        s01 = eng[(eng.participant == patient) & (eng.session == "impaired_01")]
        if len(s01) == 0:
            continue
        rng_p = np.random.RandomState(SEED)
        try:
            test_idx, cal_idx, _ = split_session(s01, TEST_PER_CLASS, rng_p)
        except Exception:
            continue
        if len(test_idx) < 15 or len(cal_idx) < 6:
            continue
        per_patient[patient] = {
            "X_cal": s01.loc[cal_idx, gm_features].fillna(0).values.astype(np.float32),
            "y_cal": s01.loc[cal_idx, "intent_idx"].values.astype(np.int64),
            "X_test": s01.loc[test_idx, gm_features].fillna(0).values.astype(np.float32),
            "y_test": s01.loc[test_idx, "intent_idx"].values.astype(np.int64),
        }
    print(f"  Kept {len(per_patient)} patients")

    # Resume
    rows = []
    done = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done = set(existing["held_out_patient"].tolist())
        print(f"Resume: {len(rows)} rows, {len(done)} patients done")

    patient_list = sorted(per_patient.keys(), key=lambda s: int(s.replace("patient", "")))
    for i, held_out in enumerate(patient_list, 1):
        if held_out in done:
            continue

        # Get target training size from cross-arm
        if held_out not in cross_arm.index:
            print(f"  {held_out}: not in cross-arm results, skip")
            continue
        n_target = int(cross_arm.loc[held_out, "n_hlth_cal"])

        # Pool cal from other 47 patients
        others_X = np.vstack([per_patient[p]["X_cal"] for p in patient_list if p != held_out])
        others_y = np.concatenate([per_patient[p]["y_cal"] for p in patient_list if p != held_out])

        # Volume-matched subsample
        rng_p = np.random.RandomState(abs(hash(held_out)) & 0xffffffff)
        X_sub, y_sub = stratified_subsample(others_X, others_y, n_target, rng_p)

        # Test on held-out patient
        X_test = per_patient[held_out]["X_test"]
        y_test = per_patient[held_out]["y_test"]

        # Train HGB on volume-matched pool
        try:
            sc = StandardScaler().fit(X_sub)
            clf = make_hgb().fit(sc.transform(X_sub), y_sub)
            vm_acc = float(accuracy_score(y_test, clf.predict(sc.transform(X_test))))
        except Exception as e:
            print(f"  {held_out}: volume-matched LOPO failed ({e})", flush=True)
            vm_acc = np.nan

        rows.append({
            "held_out_patient": held_out,
            "n_target": n_target,
            "n_actual_sampled": len(X_sub),
            "n_test": len(X_test),
            "vm_lopo_po_acc": vm_acc,
        })
        elapsed = time.time() - t0
        eta = elapsed / max(1, len(rows) - len(done)) * (len(patient_list) - len(rows))
        print(f"[{i}/{len(patient_list)}] {held_out}  n_target={n_target}  vm_LOPO={vm_acc:.4f}  "
              f"[{elapsed/60:.1f}min, eta {eta/60:.0f}min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        print("No rows. Aborting.")
        return

    # Merge with cross-arm + LOPO for direct comparison
    lopo_full = pd.read_csv(OUT_DIR / "lopo_cross_patient_per_patient.csv").rename(
        columns={"held_out_patient": "patient", "lopo_po_acc": "lopo_full_acc"}
    )
    ca = cross_arm.reset_index()[["patient", "cross_arm_po_acc", "imp_own_cal_acc"]]
    merged = out.rename(columns={"held_out_patient": "patient"}).merge(
        lopo_full[["patient", "lopo_full_acc"]], on="patient", how="inner"
    ).merge(ca, on="patient", how="inner")

    vm_mean = merged.vm_lopo_po_acc.mean()
    ca_mean = merged.cross_arm_po_acc.mean()
    lopo_full_mean = merged.lopo_full_acc.mean()
    imp_own_mean = merged.imp_own_cal_acc.mean()

    # Key paired test
    w = wilcoxon(merged.vm_lopo_po_acc, merged.cross_arm_po_acc, alternative="greater")

    md = [
        "# Volume-matched LOPO — addressing the data-volume confound",
        "",
        f"For each of {len(merged)} PhysioMio patients: pool cal from the other 47 patients,",
        f"then subsample the pool to match THAT patient's cross-arm training-set size",
        f"(mean target n={merged.n_target.mean():.0f}). Train HGB on the volume-matched",
        f"subsample, test on held-out patient's impaired-arm test set.",
        "",
        "## Results",
        "",
        "| Regime | Mean accuracy | Training size |",
        "|---|---:|---:|",
        f"| Zero-shot (GrabMyo only) | 0.35 (reference) | 1.14M |",
        f"| **Cross-arm PO** (this patient's healthy-arm cal) | **{ca_mean:.4f}** | ~{merged.n_target.mean():.0f} |",
        f"| **Volume-matched LOPO** (47 other patients' cal, subsampled) | **{vm_mean:.4f}** | ~{merged.n_actual_sampled.mean():.0f} |",
        f"| LOPO full pool (47 patients × all cal) | {lopo_full_mean:.4f} | ~20,000 |",
        f"| Impaired-arm own cal (baseline) | {imp_own_mean:.4f} | ~{merged.n_target.mean():.0f} |",
        "",
        f"**Paired Wilcoxon (volume-matched LOPO > cross-arm): p = {w.pvalue:.4e}**",
        f"Patients where VM-LOPO > cross-arm: {(merged.vm_lopo_po_acc > merged.cross_arm_po_acc).sum()} / {len(merged)}",
        "",
        "## Interpretation",
        "",
        "- If VM-LOPO > cross-arm significantly: pathology-matched data really is more",
        "  valuable per-window than anatomy-matched. The 'stroke EMG is a distinct",
        "  distribution' claim survives, controlling for volume.",
        "- If VM-LOPO ≈ cross-arm: the previous LOPO advantage was mostly data volume.",
        "  Cross-arm finding shrinks to 'healthy arm doesn't transfer well' without the",
        "  crisp 'beats LOPO' claim.",
        "- Either way, cross-arm remains meaningfully below impaired-arm own cal (~0.87),",
        "  so the within-arm cal advantage is not in question.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(f"Volume-matched LOPO: {vm_mean:.4f}")
    print(f"Cross-arm PO:        {ca_mean:.4f}")
    print(f"LOPO full pool:      {lopo_full_mean:.4f}")


if __name__ == "__main__":
    main()
