# Leakage-free per-participant z-score — revision recompute #2

Reviewer #2 flagged that `add_per_participant_normalisation` computes
per-patient μ/σ across ALL windows (including test). This recompute
refits those stats from cal windows only.

## Result (PhysioMio impaired-arm, n=48, balanced 39/39/39 test)

| arm | leaky (submitted) | leakage-free | Δ |
|---|---:|---:|---:|
| zero-shot | 0.346 | **0.400** | +0.054 |
| calibration-only | 0.878 | **0.922** | +0.044 |
| GrabMyo + cal | 0.860 | **0.919** | +0.059 |

## Interpretation

If |Δ| < 0.02 everywhere: leakage was not material; the submitted
numbers stand. If |Δ| > 0.02 somewhere: report leakage-free numbers.