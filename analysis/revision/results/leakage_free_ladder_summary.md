# Leakage-free training-source ladder (pre-v3 sanity check)

n = 48 patients (of 48 with both arms). cal=36. Features via 
engineer_features_leakage_free (z-score μ/σ fit on cal rows only, per participant).

## Ladder

| Row | Training source | Mean acc | Median acc | Previous (leaky) |
|---|---|---:|---:|---:|
| 1 | Own impaired 22s cal | 0.8957 | 0.9231 | 0.877 (paper headline) |
| 2 | **Cross-arm** (own healthy cal) | 0.6385 | 0.6453 | 0.549 |
| 3 | VM-LOPO (47 others, matched vol) | 0.7520 | 0.7692 | 0.684 |
| 4 | GrabMyo zero-shot | 0.3602 | 0.3675 | 0.346 |

## Critical gap: VM-LOPO vs cross-arm

- Previous (leaky): VM-LOPO − cross-arm = 0.684 − 0.549 = +0.136 (13.6 pp)
- **Leakage-free: VM-LOPO − cross-arm = +0.1134 (+11.3 pp)**
- Paired Wilcoxon (VM-LOPO > cross-arm), leakage-free: p = 3.7948e-03
- Patients where VM-LOPO > cross-arm: 29/48

## Decision

- If ordering (row 1 >> row 3 > row 2 > row 4) survives with rough magnitudes
  intact, the paper's central claim is safe. Proceed to capacity sweep v3.
- If cross-arm gap collapses or ordering breaks, the paper's central claim needs
  revision. Do NOT run v3 until this is understood.