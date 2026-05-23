# PhysioMio transition accuracy

Metric per Wang et al. (ReactEMG Stroke, arXiv 2601.22090). Reaction buffer = 10 windows (500 ms); max maintenance error = 0.00 (strict / zero tolerance); min segment length = 5 windows.

## Headline numbers

| Aggregation | Mean | 95% bootstrap CI |
|---|---:|---|
| Session-level transition accuracy | **0.2644** | [0.2325, 0.2979] |
| Patient-level transition accuracy | **0.2771** | [0.2230, 0.3329] |
| Session-level raw accuracy (full-stream, for ref) | 0.9091 | [0.9027, 0.9154] |

Based on **329 sessions** (48 patients) and **658 total transitions**.

## By arm

| Arm | n sessions | Transition acc (95% CI) |
|---|---:|---:|
| healthy | 91 | 0.3626 [0.2967, 0.4341] |
| impaired | 238 | 0.2269 [0.1912, 0.2647] |

## Per transition type

| from → to | n | Buffer acc | Maint acc | Combined transition acc |
|---|---:|---:|---:|---:|
| rest → close | 329 | 1.000 | 0.298 | **0.298** |
| close → open | 329 | 0.967 | 0.231 | **0.231** |

## Comparison to ReactEMG Stroke (Wang et al. 2026)

Wang et al. report transition accuracy averaged across 3 chronic-stroke participants (FMA-UE 26-35) on 5 held-out distribution-shifted test sets:

| Method | Raw acc | Transition acc |
|---|---:|---:|
| Zero-shot (healthy-pretrained, frozen) | 0.60 | 0.13 |
| Stroke-only training from scratch | 0.69 | 0.42 |
| Head-only fine-tune | 0.75 | 0.53 |
| LoRA fine-tune | 0.78 | 0.61 |
| Full fine-tune | 0.78 | 0.61 |
| **This work (PhysioMio, per-session cal, n=48)** | **0.91** | **0.26** |

Notes for fair comparison:

- Cohort: 48 patients (PhysioMio) vs 3 (ReactEMG Stroke). Larger cohort → tighter CIs but different severity mix.
- Method: classical ML (HistGradientBoosting + weighted refit on a 43-subject GrabMyo base) vs transformer + LoRA pretrained on 650+ subjects.
- Test protocol: PhysioMio sessions average ~48 s of gesture data (16 gestures × 4 s, then 3-class-mapped to 12 gestures), with 2-3 ground-truth transitions per session. ReactEMG runs ~18-minute sessions with explicitly interleaved RCRCRCR cue sequences (5-6 transitions/set × multiple sets). The PhysioMio transition counts per session are therefore an order of magnitude smaller; aggregate transition acc estimate is more variance-stable thanks to the 48-patient cohort but per-session is noisier.
- Distribution shift: ReactEMG evaluates 5 held-out perturbation conditions (within-session drift, unseen posture, sensor placement, device-driven motion); the PhysioMio number here is on the per-session balanced split without explicit perturbations. Adding perturbation evaluation is tracked separately.
- Deployment: this metric is on the per-session calibrated model that fits inside the 50 ms Teensy cycle (~17 ms/cycle p50, CPU). ReactEMG runs a transformer that needs GPU at inference time.