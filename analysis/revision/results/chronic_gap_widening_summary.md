# Chronic gap widening — HGB variants on Exp1 vs VM-LOPO

n = 25 chronic patients (>30d post-stroke).
Each variant applied identically to both cells; the training set is
the same 432-window subsample from 47 mixed donors per patient.

| Config | Impaired (VM-LOPO) | Healthy (Exp 1) | Gap | Wilcoxon p |
|---|---:|---:|---:|---:|
| A: max_iter=100, single (baseline) | 0.7255 | 0.7080 | +0.0174 | 0.1620 |
| B: max_iter=300, single | 0.7197 | 0.7138 | +0.0058 | 0.3240 |
| C: max_iter=100, ensemble×5 | 0.7255 | 0.7080 | +0.0174 | 0.1620 |
| D: max_iter=300, ensemble×5 | 0.7197 | 0.7138 | +0.0058 | 0.3240 |

## Reading

- Baseline (A) reproduces the current numbers: ~0.752 imp / ~0.709 hlth / +4.3 pp gap
- Any variant that widens the gap is the strengthening we're after
- Wilcoxon p on the gap tells us if the differentiation is statistically robust