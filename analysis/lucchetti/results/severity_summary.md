# Lucchetti severity stratification

n = 10 stroke subjects, scored by **L_CA = Fugl-Meyer UE level (1-5)**.
  - Level 1: FMA 0-22 (severe)  · Level 2: 23-31  · Level 3: 32-42  · Level 4: 43-52  · Level 5: 53-66 (mild)
  - L_CA distribution in cohort: {3: 4, 4: 2, 5: 4}

## Per-level accuracy (impaired arm only)

| L_CA | n | Zero-shot | + Calibration | Δ |
|---:|---:|---:|---:|---:|
| 3 | 4 | 0.1916 ± 0.0610 | **0.8542 ± 0.1551** | +0.6627 |
| 4 | 2 | 0.1812 ± 0.0269 | **0.7445 ± 0.2007** | +0.5633 |
| 5 | 4 | 0.2035 ± 0.0504 | **0.7607 ± 0.1305** | +0.5572 |

## Correlation with severity

- Spearman ρ(L_CA, calibration benefit Δacc) = **-0.350**, p = 0.321
- Spearman ρ(L_CA, calibrated accuracy)     = **-0.311**, p = 0.381

Positive ρ would mean less-impaired patients (higher L_CA) benefit more from calibration. Near-zero ρ would mean calibration helps regardless of severity (same null finding as PhysioMio).