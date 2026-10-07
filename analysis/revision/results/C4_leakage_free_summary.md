# C4 leakage-free — GrabMyo weight sweep

n = 48 patients, cal_per_gesture=36 (paper operating point).
Uses leakage-free features + frozen splits (same pipeline as leakage_free_ladder).
Weight = cal-weight multiplier vs. GrabMyo (1×); 0 = cal-only baseline.

## Ladder

| Weight (× GrabMyo) | Mean acc | Median acc | vs cal-only Δ | Wilcoxon p |
|---:|---:|---:|---:|---:|
| **0 (cal-only)** | 0.8957 | 0.9231 | — | — |
| 1 | 0.7787 | 0.7821 | -0.1170 | 1.000e+00 |
| 10 | 0.8513 | 0.8675 | -0.0443 | 9.995e-01 |
| 100 | 0.8752 | 0.9103 | -0.0205 | 9.820e-01 |
| 1000 | 0.8796 | 0.8974 | -0.0160 | 9.708e-01 |

## Comparison to leakage-contaminated C4 (recompute_C4_grabmyo_weight_sweep.py)

The older sweep reported cal-only = 0.878; the leakage-free version should reproduce
row 1 of the leakage-free ladder (~0.896). If baselines match, the paired result is
confirmed leakage-invariant. If ordering flips at any weight, we investigate.