# Lucchetti headline aggregate

**n = 20 subjects, 30 sessions** (10 stroke × 2 arms + 10 healthy × 1 arm).

## Headline

| Metric | Zero-shot | + Per-session calibration | Δ |
|---|---:|---:|---:|
| Session mean accuracy | 0.1816 [0.1681, 0.1948] | **0.8278 [0.7812, 0.8712]** | +0.6461 |
| Patient mean accuracy | 0.1791 [0.1653, 0.1936] | **0.8387 [0.7929, 0.8794]** | +0.6596 |

**Paired effect** (n = 30 matched session pairs):
- Mean per-session improvement: **+0.6461**
- Wilcoxon signed-rank: p = 1.86e-09
- Cliff's δ: +1.000

## Per-arm

| Arm | n | Zero-shot | + Calibration | Δ |
|---|---:|---:|---:|---:|
| healthy | 20 | 0.1753 [0.1622, 0.1890] | **0.8442 [0.7865, 0.8943]** | +0.6689 |
| impaired | 10 | 0.1943 [0.1675, 0.2234] | **0.7949 [0.7095, 0.8770]** | +0.6006 |

## Per-class F1 (calibrated)

| Class | Mean F1 | 95 % bootstrap CI |
|---|---:|---:|
| Rest | 0.9051 | [0.8738, 0.9329] |
| Close | 0.2543 | [0.1306, 0.3830] |
| Open | 0.3078 | [0.1801, 0.4498] |

## Variance reduction (cross-subject SD of per-patient mean acc)

- Zero-shot SD: **0.0331**
- Calibrated SD: **0.1032**
- Ratio (collapse factor): **0.32×**