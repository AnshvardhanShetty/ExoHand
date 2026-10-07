"""
Revision — M2: does the M1 distance predict per-patient cross-arm accuracy drop?

Post-hoc analysis (no HGB fits). Merges:
  - M1's per-patient d_within (own-hlth ↔ own-imp W₁) and d_across (own-imp ↔ others)
  - Cross-arm same-patient's per-patient cross-arm PO accuracy and impaired-arm own cal accuracy

Computes per-patient "gap" (own_cal_acc − cross_arm_po_acc), then correlates with:
  - d_within (bigger within-patient shift → larger cross-arm drop expected)
  - d_within − d_across (bigger relative within-patient shift → larger drop)

Decision (pre-registered): ρ > 0.3 with p < 0.05 converts observation to mechanism.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
M1_CSV = OUT_DIR / "M1_within_vs_across_wasserstein_per_patient.csv"
CROSS_ARM_CSV = OUT_DIR / "cross_arm_same_patient_per_patient.csv"
OUT_MD = OUT_DIR / "M2_distance_predicts_accuracy_summary.md"
OUT_CSV = OUT_DIR / "M2_distance_predicts_accuracy_per_patient.csv"


def main():
    m1 = pd.read_csv(M1_CSV)
    ca = pd.read_csv(CROSS_ARM_CSV)
    merged = m1.merge(ca, on="patient", how="inner")
    merged["gap_own_cal_minus_cross_arm"] = merged.imp_own_cal_acc - merged.cross_arm_po_acc
    merged["diff_within_minus_across"] = (
        merged.d_within_own_hlth_vs_own_imp - merged.d_across_own_imp_vs_others_imp
    )
    merged.to_csv(OUT_CSV, index=False)

    predictors = {
        "d_within (own hlth ↔ own imp)":    merged.d_within_own_hlth_vs_own_imp,
        "d_across (own imp ↔ others imp)":  merged.d_across_own_imp_vs_others_imp,
        "diff (within − across)":           merged.diff_within_minus_across,
    }
    outcomes = {
        "cross_arm_po_acc":                          merged.cross_arm_po_acc,
        "gap (own_cal_acc − cross_arm_po_acc)":      merged.gap_own_cal_minus_cross_arm,
    }

    md = [
        "# M2 — distance predicts cross-arm accuracy",
        "",
        f"n = {len(merged)} patients. Correlates M1 distances with cross-arm accuracy.",
        "",
        "## Correlations",
        "",
        "| Predictor (M1) | Outcome | Spearman ρ | p-value |",
        "|---|---|---:|---:|",
    ]
    for pname, pvals in predictors.items():
        for oname, ovals in outcomes.items():
            rho, p = spearmanr(pvals, ovals)
            md.append(f"| {pname} | {oname} | {rho:+.3f} | {p:.3e} |")

    md += [
        "",
        "## Decision (pre-registered)",
        "",
        "- If ρ > 0.3 with p < 0.05 for (d_within predicts gap), the mechanism claim is",
        "  quantitative: per-patient healthy-vs-impaired feature-space distance predicts",
        "  how much accuracy is lost when substituting healthy-arm cal for impaired-arm.",
        "- The most illuminating link is:",
        "  d_within → gap (own_cal − cross_arm)",
        "  Positive ρ = bigger within-patient shift → bigger accuracy drop.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
