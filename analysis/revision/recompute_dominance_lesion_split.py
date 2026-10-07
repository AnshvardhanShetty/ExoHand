"""
Revision — dominance/lesion-side post-hoc analysis.

Post-hoc analysis on existing cross-arm results, no new HGB fits.

For each patient in cross-arm results, load metadata (impaired_arm side, dominant_arm
side, age, gender, days_after_stroke). Split cross-arm accuracy by these covariates
to check for confounds:

  - Dominance confound: cross-arm compares dominant-vs-non-dominant EMG. If paretic
    arm was the dominant hand pre-stroke, we're training on non-dominant healthy →
    predicting dominant impaired (or vice versa). Dominant/non-dominant EMG differ
    even in healthy people.
  - Lesion-side confound: contralateral projections mean lesion side determines which
    arm is paretic. Splitting by lesion side checks for asymmetries.

Outputs:
  analysis/revision/results/dominance_lesion_split_summary.md
  analysis/revision/results/dominance_lesion_split_per_patient.csv
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
CROSS_ARM_CSV = OUT_DIR / "cross_arm_same_patient_per_patient.csv"
LOPO_CSV = OUT_DIR / "lopo_cross_patient_per_patient.csv"
METADATA_CSV = PROJECT_ROOT / "data" / "physiomio" / "data" / "metadata.csv"
OUT_CSV = OUT_DIR / "dominance_lesion_split_per_patient.csv"
OUT_MD = OUT_DIR / "dominance_lesion_split_summary.md"


def main():
    ca = pd.read_csv(CROSS_ARM_CSV)
    lopo = pd.read_csv(LOPO_CSV).rename(columns={"held_out_patient": "patient"})
    meta = pd.read_csv(METADATA_CSV)

    # Metadata has one row per file; collapse to one row per patient
    per_patient_meta = (
        meta.groupby("patient")
        .first()
        .reset_index()[["patient", "age_in_years", "gender", "impaired_arm", "dominant_arm",
                        "days_after_stroke"]]
    )

    # Merge everything
    merged = (ca.merge(lopo[["patient", "lopo_po_acc"]], on="patient", how="left")
                .merge(per_patient_meta, on="patient", how="left"))

    # Derive dominant_arm_affected flag
    merged["dominant_arm_affected"] = (merged.impaired_arm == merged.dominant_arm)
    merged.to_csv(OUT_CSV, index=False)

    print(f"n patients with full metadata: {merged.dropna(subset=['impaired_arm', 'dominant_arm']).shape[0]}")

    md = ["# Dominance / lesion-side split — controlling for cross-arm confounds", ""]
    md.append(f"Post-hoc analysis on cross-arm and LOPO results merged with patient metadata (n={len(merged)}).")
    md.append("")

    # ── 1. Dominant vs non-dominant arm affected ──
    md.append("## 1. Dominant-arm-affected vs non-dominant-arm-affected")
    md.append("")
    md.append("Cross-arm accuracy split by whether the paretic arm was the dominant one pre-stroke:")
    md.append("")
    md.append("| Group | n | Cross-arm PO mean | Cross-arm PO median | LOPO mean |")
    md.append("|---|---:|---:|---:|---:|")

    for label, mask in [
        ("Dominant arm affected", merged.dominant_arm_affected == True),
        ("Non-dominant arm affected", merged.dominant_arm_affected == False),
    ]:
        sub = merged[mask].dropna(subset=["cross_arm_po_acc"])
        md.append(f"| {label} | {len(sub)} | {sub.cross_arm_po_acc.mean():.4f} | "
                  f"{sub.cross_arm_po_acc.median():.4f} | {sub.lopo_po_acc.mean():.4f} |")

    dom = merged[merged.dominant_arm_affected == True].dropna(subset=["cross_arm_po_acc"])
    nondom = merged[merged.dominant_arm_affected == False].dropna(subset=["cross_arm_po_acc"])
    if len(dom) >= 3 and len(nondom) >= 3:
        u = mannwhitneyu(dom.cross_arm_po_acc, nondom.cross_arm_po_acc)
        md.append("")
        md.append(f"Mann-Whitney U (cross-arm dominant-affected vs non-dominant-affected): "
                  f"p = {u.pvalue:.4e}")
    md.append("")

    # ── 2. Split by impaired arm side (L vs R) ──
    md.append("## 2. Split by lesion side (impaired arm L vs R)")
    md.append("")
    md.append("| Impaired side | n | Cross-arm PO mean | LOPO mean |")
    md.append("|---|---:|---:|---:|")
    for side in ["l", "r"]:
        sub = merged[merged.impaired_arm == side].dropna(subset=["cross_arm_po_acc"])
        if len(sub) > 0:
            md.append(f"| {side.upper()} arm impaired | {len(sub)} | "
                      f"{sub.cross_arm_po_acc.mean():.4f} | {sub.lopo_po_acc.mean():.4f} |")
    md.append("")

    # ── 3. Correlation with continuous covariates ──
    md.append("## 3. Correlation of cross-arm accuracy with continuous covariates")
    md.append("")
    md.append("| Covariate | Spearman ρ | p-value |")
    md.append("|---|---:|---:|")
    for col in ["age_in_years", "days_after_stroke"]:
        sub = merged.dropna(subset=[col, "cross_arm_po_acc"])
        if len(sub) >= 5:
            rho, p = spearmanr(sub[col], sub.cross_arm_po_acc)
            md.append(f"| {col} | {rho:+.3f} | {p:.3e} |")
    md.append("")

    # ── 4. Split by gender ──
    md.append("## 4. Split by gender")
    md.append("")
    md.append("| Gender | n | Cross-arm PO mean | LOPO mean |")
    md.append("|---|---:|---:|---:|")
    for g in merged.gender.dropna().unique():
        sub = merged[merged.gender == g].dropna(subset=["cross_arm_po_acc"])
        md.append(f"| {g} | {len(sub)} | {sub.cross_arm_po_acc.mean():.4f} | "
                  f"{sub.lopo_po_acc.mean():.4f} |")
    md.append("")

    md += [
        "## Interpretation",
        "",
        "- If dominant-affected vs non-dominant-affected show similar cross-arm accuracy,",
        "  the dominance confound is not driving the result.",
        "- If they differ significantly, we should either report the split or restrict",
        "  analysis to one group.",
        "- Lesion-side split similarly checks for asymmetries.",
        "- Continuous covariates (age, days-post-stroke) test whether patient severity",
        "  or recovery stage drives the cross-arm gap.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
