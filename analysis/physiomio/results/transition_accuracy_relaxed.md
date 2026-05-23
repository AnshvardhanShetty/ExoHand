# PhysioMio transition accuracy

Metric per Wang et al. (ReactEMG Stroke, arXiv 2601.22090). Reaction buffer = 10 windows (500 ms); max maintenance error = 0.10 (relaxed); min segment length = 5 windows.

## Headline numbers

| Aggregation | Mean | 95% bootstrap CI |
|---|---:|---|
| Session-level transition accuracy | **0.5699** | [0.5349, 0.6064] |
| Patient-level transition accuracy | **0.5720** | [0.5239, 0.6190] |
| Session-level raw accuracy (full-stream, for ref) | 0.9091 | [0.9027, 0.9154] |

Based on **329 sessions** (48 patients) and **658 total transitions**.

## By arm

| Arm | n sessions | Transition acc (95% CI) |
|---|---:|---:|
| healthy | 91 | 0.6923 [0.6264, 0.7582] |
| impaired | 238 | 0.5231 [0.4811, 0.5631] |

## Per transition type

| from → to | n | Buffer acc | Maint acc | Combined transition acc |
|---|---:|---:|---:|---:|
| rest → close | 329 | 1.000 | 0.766 | **0.766** |
| close → open | 329 | 0.967 | 0.374 | **0.374** |

## Comparison to ReactEMG Stroke (Wang et al. 2026)

Wang et al. report transition accuracy averaged across 3 chronic-stroke participants (FMA-UE 26-35) on 5 held-out distribution-shifted test sets:

| Method | Raw acc | Transition acc |
|---|---:|---:|
| Zero-shot (healthy-pretrained, frozen) | 0.60 | 0.13 |
| Stroke-only training from scratch | 0.69 | 0.42 |
| Head-only fine-tune | 0.75 | 0.53 |
| LoRA fine-tune | 0.78 | 0.61 |
| Full fine-tune | 0.78 | 0.61 |
| **This work (PhysioMio, per-session cal, n=48)** | **0.91** | **0.57** |

Notes for fair comparison:

- Cohort: 48 patients (PhysioMio) vs 3 (ReactEMG Stroke). Larger cohort → tighter CIs but different severity mix.
- Method: classical ML (HistGradientBoosting + weighted refit on a 43-subject GrabMyo base) vs transformer + LoRA pretrained on 650+ subjects.
- Test protocol: PhysioMio sessions average ~48 s of gesture data (16 gestures × 4 s, then 3-class-mapped to 12 gestures), with 2-3 ground-truth transitions per session. ReactEMG runs ~18-minute sessions with explicitly interleaved RCRCRCR cue sequences (5-6 transitions/set × multiple sets). The PhysioMio transition counts per session are therefore an order of magnitude smaller; aggregate transition acc estimate is more variance-stable thanks to the 48-patient cohort but per-session is noisier.
- Distribution shift: ReactEMG evaluates 5 held-out perturbation conditions (within-session drift, unseen posture, sensor placement, device-driven motion); the PhysioMio number here is on the per-session balanced split without explicit perturbations. Adding perturbation evaluation is tracked separately.
- Deployment: this metric is on the per-session calibrated model that fits inside the 50 ms Teensy cycle (~17 ms/cycle p50, CPU). ReactEMG runs a transformer that needs GPU at inference time.