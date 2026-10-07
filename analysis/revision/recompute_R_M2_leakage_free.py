"""
R-M2 — distance predicts cross-arm accuracy drop (leakage-free).

Post-hoc analysis on:
  - R-M1's leakage-free d_within, d_across
  - Leakage-free ladder's cross-arm PO acc and impaired-arm own-cal acc

For each patient:
  gap = own_cal_acc − cross_arm_po_acc

Correlate gap with d_within (should be positive: larger within-patient shift →
larger accuracy drop) and with (d_within − d_across).

Pre-registered decision: ρ > 0.3 with p < 0.05 upgrades observation to mechanism.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scipy.stats import spearmanr

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
M1 = OUT_DIR / "R_M1_leakage_free_per_patient.csv"
LADDER = OUT_DIR / "leakage_free_ladder_per_patient.csv"
OUT_MD = OUT_DIR / "R_M2_leakage_free_summary.md"


def main():
    m1 = pd.read_csv(M1)
    lad = pd.read_csv(LADDER)
    merged = m1.merge(lad[["patient", "row1_own_imp_cal", "row2_cross_arm_own_hlth"]],
                       on="patient", how="inner")
    merged = merged.dropna(subset=["row2_cross_arm_own_hlth"])
    merged["gap"] = merged.row1_own_imp_cal - merged.row2_cross_arm_own_hlth
    merged["diff_within_minus_across"] = (
        merged.d_within_own_hlth_vs_own_imp - merged.d_across_own_imp_vs_others_imp
    )

    rho_within_gap, p_within_gap = spearmanr(merged.d_within_own_hlth_vs_own_imp, merged.gap)
    rho_within_cross, p_within_cross = spearmanr(merged.d_within_own_hlth_vs_own_imp,
                                                    merged.row2_cross_arm_own_hlth)
    rho_diff_gap, p_diff_gap = spearmanr(merged.diff_within_minus_across, merged.gap)

    md = [
        "# R-M2 — distance predicts cross-arm accuracy (leakage-free)",
        "",
        f"n = {len(merged)} patients. All inputs from leakage-free re-runs.",
        "",
        "## Correlations",
        "",
        "| Predictor (leakage-free) | Outcome | Spearman ρ | p-value |",
        "|---|---|---:|---:|",
        f"| d_within (own hlth ↔ own imp) | gap (own_cal − cross_arm) | {rho_within_gap:+.3f} | {p_within_gap:.3e} |",
        f"| d_within | cross_arm acc | {rho_within_cross:+.3f} | {p_within_cross:.3e} |",
        f"| diff (within − across) | gap | {rho_diff_gap:+.3f} | {p_diff_gap:.3e} |",
        "",
        "## Legacy comparison",
        "",
        "| Predictor | Outcome | Legacy ρ | Legacy p |",
        "|---|---|---:|---:|",
        "| d_within (leaky) | gap | +0.393 | 5.769e-03 |",
        "| d_within (leaky) | cross_arm | −0.576 | 1.868e-05 |",
        "",
        "## Decision (pre-registered)",
        "",
        "- If ρ > 0.3 with p < 0.05 for d_within → gap: mechanism claim is quantitative.",
        "- Direction: positive ρ = bigger within-patient shift → bigger accuracy drop.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(open(OUT_MD).read())


if __name__ == "__main__":
    main()
