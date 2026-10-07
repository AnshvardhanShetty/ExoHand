# Experiment 1 — the missing 4th cell

**Question:** does the +11 pp gap between own-healthy (0.639) and 47-others-impaired
(0.752) come from pathology (arm changed) or from diversity (donor pool changed)?

n = 48 target patients.

## The 2×2

| Training source | 1 donor | 47 donors |
|---|---:|---:|
| **healthy arm(s)** | own healthy: **0.6385** (cross-arm) | 47 others' healthy: **0.7390** (Exp 1) |
| **impaired arm(s)** | 1 stranger's impaired: — (not tested) | 47 others' impaired: **0.7520** (VM-LOPO) |
| own impaired reference | **0.8957** (own cal) | — |

## Decision (pre-registered)

- If Exp 1 ≈ 0.64 (own-healthy): PATHOLOGY. Healthy stays bad even with 47 donors. Headline holds.
- If Exp 1 ≈ 0.75 (VM-LOPO): DIVERSITY. Healthy works fine once you have 47 donors. Reframe.
- If intermediate: mixed contribution from both axes.

## Statistics

- Exp 1 mean vs cross-arm (own-healthy) mean: 0.7390 vs 0.6385
  Paired Wilcoxon (Exp 1 > own-healthy): p = 6.4612e-03
- Exp 1 mean vs VM-LOPO (own-impaired others) mean: 0.7390 vs 0.7520
  Paired Wilcoxon (VM-LOPO > Exp 1): p = 1.2226e-01

## Reading

- Exp 1 vs cross-arm: quantifies the pure diversity contribution (going 1→47 donors, all healthy)
- Exp 1 vs VM-LOPO: quantifies the pure pathology contribution (going healthy→impaired, all 47 donors)
- Sum ≈ own-healthy → 47-others-impaired total gap of 0.113