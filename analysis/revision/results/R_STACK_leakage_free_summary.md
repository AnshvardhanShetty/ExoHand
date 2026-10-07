# R-STACK — pathology + anatomy stacked (leakage-free)

n = 48 patients, matched volume (432 windows per arm).

## Results (leakage-free)

| Arm | Mean acc |
|---|---:|
| Pathology-matched only (others' impaired, subsampled) | 0.7651 |
| Anatomy-matched only (own healthy) | 0.6385 |
| Stacked P+A | 0.7781 |

**Paired Wilcoxon:**
- Stacked > max(P, A): p = 9.6954e-01
- Stacked > P alone: p = 1.2350e-01
- Stacked > A alone: p = 1.7287e-04

Legacy (leaky): legacy P=0.6820  A=0.5486  P+A=0.6825

## Gate (pre-registered)

- If P+A > P significantly (p < 0.05) → the two are complementary. Anastasiev
  contradiction is withdrawn; reframe as replication-with-caveats.
- If P+A ≈ P (p ≥ 0.05) → anatomy adds nothing on top of pathology. Original
  claim survives.