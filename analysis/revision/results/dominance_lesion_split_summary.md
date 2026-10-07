# Dominance / lesion-side split — controlling for cross-arm confounds

Post-hoc analysis on cross-arm and LOPO results merged with patient metadata (n=48).

## 1. Dominant-arm-affected vs non-dominant-arm-affected

Cross-arm accuracy split by whether the paretic arm was the dominant one pre-stroke:

| Group | n | Cross-arm PO mean | Cross-arm PO median | LOPO mean |
|---|---:|---:|---:|---:|
| Dominant arm affected | 22 | 0.5365 | 0.5684 | 0.6305 |
| Non-dominant arm affected | 26 | 0.5588 | 0.5684 | 0.6272 |

Mann-Whitney U (cross-arm dominant-affected vs non-dominant-affected): p = 8.1942e-01

## 2. Split by lesion side (impaired arm L vs R)

| Impaired side | n | Cross-arm PO mean | LOPO mean |
|---|---:|---:|---:|
| L arm impaired | 26 | 0.5516 | 0.6302 |
| R arm impaired | 22 | 0.5451 | 0.6270 |

## 3. Correlation of cross-arm accuracy with continuous covariates

| Covariate | Spearman ρ | p-value |
|---|---:|---:|
| age_in_years | +0.002 | 9.902e-01 |
| days_after_stroke | -0.210 | 1.519e-01 |

## 4. Split by gender

| Gender | n | Cross-arm PO mean | LOPO mean |
|---|---:|---:|---:|
| f | 20 | 0.5752 | 0.6017 |
| m | 28 | 0.5296 | 0.6480 |

## Interpretation

- If dominant-affected vs non-dominant-affected show similar cross-arm accuracy,
  the dominance confound is not driving the result.
- If they differ significantly, we should either report the split or restrict
  analysis to one group.
- Lesion-side split similarly checks for asymmetries.
- Continuous covariates (age, days-post-stroke) test whether patient severity
  or recovery stage drives the cross-arm gap.