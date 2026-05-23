# PhysioMio per-session calibration — aggregate summary

**n = 48 patients · 329 sessions · seed 42 · bootstrap resamples = 2000**
Protocol: GrabMyo (1.14 M windows, w=1) + 30 s per-session stratified recal (w=100×), --fast HGB, per-gesture temporal split with 3-window buffer (no signal leakage).

## Headline numbers

| Metric | Value | 95% CI |
|---|---|---|
| **Patient-level mean accuracy** (n=48) | **0.8754** | [0.8603, 0.8908] |
| **Session-level mean accuracy** (n=329) | **0.8710** | [0.8600, 0.8819] |
| Session-level macro-F1 (n=329) | 0.8621 | [0.8505, 0.8746] |
| Patient-level cross-subject std | 0.0534 | [0.0426, 0.0624] |
| Session-level cross-session std | 0.1014 | [0.0936, 0.1085] |

## Per-class F1 (session-level means, n=329)

| Class | Mean F1 | 95% CI |
|---|---|---|
| rest | 0.9818 | [0.9742, 0.9883] |
| close | 0.8423 | [0.8310, 0.8540] |
| open | 0.7622 | [0.7373, 0.7858] |

## Paired zero-shot vs calibration (per patient, n=48)

| Metric | Value |
|---|---|
| Mean acc zero-shot (no cal) | 0.2143 |
| Mean acc with cal | 0.8754 |
| **Δaccuracy (cal − no cal), bootstrap CI** | **+0.6611 [+0.6396, +0.6808]** |
| Paired Wilcoxon signed-rank, H₁: cal > no-cal | W = 1176.0000, **p = 3.553e-15** |
| Cliff's δ | **+1.0000** |
| Variance ratio (no-cal std / with-cal std) | 1.26× [0.95×, 1.57×] |

## Per-arm paired comparison (healthy contralateral vs paretic, per patient)

| Metric | Value |
|---|---|
| Healthy-arm mean acc | 0.9147  (std 0.0623) |
| Impaired-arm mean acc | 0.8603  (std 0.0696) |
| Δ (healthy − impaired) | +0.0544 [+0.0283, +0.0794] |
| Wilcoxon, H₁: healthy > impaired | W = 885.5000, **p = 8.184e-05** |

## Cross-comparison with GrabMyo headline

| Protocol | Mean acc | Cross-subject std |
|---|---|---|
| GrabMyo within-population LOSO (healthy, variant e) | 0.9732 | 0.0207 |
| **PhysioMio per-session cal (stroke, this work)** | **0.8754** | **0.0534** |
| PhysioMio zero-shot (no cal, this work) | 0.2143 | 0.0675 |

Calibration recovers **+66.11%** on stroke patients from a zero-shot baseline of 21.43%.
Residual gap to GrabMyo within-population: ~9.78%.