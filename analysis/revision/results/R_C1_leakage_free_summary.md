# R-C1 — channel permutation (leakage-free)

n = 48 patients. All 24 permutations of the 4 canonical channels.

## Results (leakage-free)

| Config | Mean acc | Median acc |
|---|---:|---:|
| Identity (baseline) | 0.6385 | 0.6453 |
| Best permutation (per patient, oracle) | 0.7455 | 0.7265 |
| Worst permutation | 0.4866 | 0.4701 |
| Medial-lateral mirror | 0.6022 | 0.6068 |
| Impaired-arm own cal (leakage-free ref) | 0.8960 | — |

Gap (own cal − identity): +0.2575
Gap (own cal − best perm): +0.1505
Fraction of gap recovered: 41.56%

## VM-LOPO vs cross-arm-oracle (leakage-free)

VM-LOPO (leakage-free): 0.752
Cross-arm best-perm oracle (this run): 0.7455
Paired Wilcoxon (VM-LOPO > cross-arm-oracle): p = 3.8349e-01
Patients where VM-LOPO > oracle: 23/48

## Decision (pre-registered)

- If recovery > ⅓ of the gap → channel mounting is a real confound. Report
  corrected best-perm number as headline.
- If recovery < ⅓ → mounting is not the confound; original claim survives.
- Legacy (leaky) recovery: 21.11%, below threshold.