# M2 — distance predicts cross-arm accuracy

n = 48 patients. Correlates M1 distances with cross-arm accuracy.

## Correlations

| Predictor (M1) | Outcome | Spearman ρ | p-value |
|---|---|---:|---:|
| d_within (own hlth ↔ own imp) | cross_arm_po_acc | -0.576 | 1.868e-05 |
| d_within (own hlth ↔ own imp) | gap (own_cal_acc − cross_arm_po_acc) | +0.393 | 5.769e-03 |
| d_across (own imp ↔ others imp) | cross_arm_po_acc | -0.310 | 3.202e-02 |
| d_across (own imp ↔ others imp) | gap (own_cal_acc − cross_arm_po_acc) | +0.264 | 6.948e-02 |
| diff (within − across) | cross_arm_po_acc | -0.456 | 1.116e-03 |
| diff (within − across) | gap (own_cal_acc − cross_arm_po_acc) | +0.238 | 1.029e-01 |

## Decision (pre-registered)

- If ρ > 0.3 with p < 0.05 for (d_within predicts gap), the mechanism claim is
  quantitative: per-patient healthy-vs-impaired feature-space distance predicts
  how much accuracy is lost when substituting healthy-arm cal for impaired-arm.
- The most illuminating link is:
  d_within → gap (own_cal − cross_arm)
  Positive ρ = bigger within-patient shift → bigger accuracy drop.