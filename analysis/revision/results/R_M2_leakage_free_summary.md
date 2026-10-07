# R-M2 — distance predicts cross-arm accuracy (leakage-free)

n = 48 patients. All inputs from leakage-free re-runs.

## Correlations

| Predictor (leakage-free) | Outcome | Spearman ρ | p-value |
|---|---|---:|---:|
| d_within (own hlth ↔ own imp) | gap (own_cal − cross_arm) | +0.171 | 2.461e-01 |
| d_within | cross_arm acc | -0.231 | 1.137e-01 |
| diff (within − across) | gap | +0.162 | 2.723e-01 |

## Legacy comparison

| Predictor | Outcome | Legacy ρ | Legacy p |
|---|---|---:|---:|
| d_within (leaky) | gap | +0.393 | 5.769e-03 |
| d_within (leaky) | cross_arm | −0.576 | 1.868e-05 |

## Decision (pre-registered)

- If ρ > 0.3 with p < 0.05 for d_within → gap: mechanism claim is quantitative.
- Direction: positive ρ = bigger within-patient shift → bigger accuracy drop.