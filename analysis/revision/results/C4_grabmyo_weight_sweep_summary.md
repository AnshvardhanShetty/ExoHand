# C4 — GrabMyo weight sweep

n = 48 patients, cal_per_gesture=36 (paper operating point).
Weight = cal-weight multiplier in the joint training (1× = GrabMyo weight; 0 = cal-only, no GrabMyo).

## Ladder

| Weight (× GrabMyo) | Mean acc | Median acc | vs cal-only Δ | Wilcoxon p |
|---:|---:|---:|---:|---:|
| **0 (cal-only)** | 0.8775 | 0.9145 | — | — |
| 1 | 0.7991 | 0.8034 | -0.0783 | 1.000e+00 |
| 10 | 0.8688 | 0.8974 | -0.0087 | 8.842e-01 |
| 100 | 0.8837 | 0.9103 | +0.0062 | 3.156e-01 |
| 1000 | 0.8805 | 0.9103 | +0.0030 | 4.839e-01 |

## Interpretation (pre-registered decision rule)

- If ANY weight ≠ 100× beats cal-only by > 1 pp with paired Wilcoxon p < 0.05,
  the null result headline dies. Paper becomes 'GrabMyo helps only at weight X'.
- If no weight beats cal-only meaningfully, the null result claim survives across
  the full ladder — pretraining doesn't help regardless of weighting scheme.