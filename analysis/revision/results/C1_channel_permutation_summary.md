# C1 — channel-permutation cross-arm test

n = 48 patients. Cross-arm PO tried across all 24 channel permutations
of the 4 canonical channels (ch0, ch4, ch9, ch13). Baseline = identity permutation.

## Results

| Config | Mean acc | Median acc |
|---|---:|---:|
| Identity (baseline) | 0.5486 | 0.5684 |
| Best permutation (per patient) | 0.6175 | 0.6667 |
| Worst permutation | 0.4158 | 0.3803 |
| Medial-lateral mirror | 0.4758 | 0.4316 |
| Impaired-arm own cal (reference) | 0.875 | — |

Cross-arm identity baseline gap from own cal: +0.3264
Cross-arm best-perm gap from own cal: +0.2575
Fraction of gap recovered by best permutation: 21.11%

## Decision (pre-registered)

- If best permutation recovers > ⅓ of the 33 pp gap (i.e., > 11 pp):
  the gap is partly a channel-mounting artefact.
  Report corrected best-perm number as the headline.
- If recovery < ⅓: mounting is not the confound; original claim survives.