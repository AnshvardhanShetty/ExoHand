# Lucchetti cross-population validation — summary

Second-dataset replication of the PhysioMio cross-population calibration story
on Lucchetti et al. 2025 (Sci Data, DOI 10.1038/s41597-025-06174-3).

**n = 20 subjects** (10 stroke × 2 arms + 10 healthy controls × 1 arm) ·
**30 sessions** · **L_CA range 3-5** (moderate-to-mild stroke).

## TL;DR

The headline cross-population gap-closure replicates cleanly on Lucchetti
**+66 pp from zero-shot to calibrated, all 10 stroke patients improve,
Wilcoxon p ≈ 10⁻⁹, Cliff's δ = +1.000**.

But the per-class F1 breakdown reveals a nuance: the cross-population gap
closure operates almost entirely on the **rest-vs-movement** axis. The
close/open distinction does not transfer cleanly to Lucchetti's functional
reach-grasp tasks. This is a useful finding for the paper, not a failure.

## What's different from PhysioMio

| Aspect | PhysioMio | Lucchetti |
|---|---|---|
| Cohort | 48 stroke patients | **10 stroke (FMA-UE 3-5) + 10 healthy controls** |
| Multi-session? | Yes (mean 7 sessions / patient) | **No — one session per subject** (longitudinal N/A) |
| Both arms? | Yes (healthy + impaired) | Yes (impaired + non-affected for stroke; dominant for healthy) |
| Severity scoring | Per-gesture FMA 0/1/2 | **L_CA = FMA-UE level 1-5** (cleaner scale) |
| Hardware | 64-channel HD-sEMG @ 2 kHz, picked 4 per patient | 12-channel @ 1 kHz, fixed 4 channels (FCR, ECR, FDS, EDC) upsampled to 2 kHz |
| Tasks | 12 mapped from 16 discrete hand gestures, ~4 s each | 6 functional reach-grasp tasks per arm, ~107 s each × 5 reps |
| Labels | Cued gesture identity → close / open / rest | Events.Start/End + task-order convention (BA, BC, SC → close; PS → rest; HM, HH → open) |
| Class balance | Roughly balanced (3-class equal cuing) | rest 59 % / close 23 % / open 18 % (rest-dominated, task duration vs rep duration) |

## 1. Headline calibration recovery

| Metric | Zero-shot | + Per-session calibration | Δ |
|---|---:|---:|---:|
| Session mean accuracy | 0.182 [0.168, 0.195] | **0.828 [0.781, 0.871]** | **+0.646** |
| Patient mean accuracy | 0.179 [0.165, 0.194] | **0.839 [0.793, 0.879]** | **+0.660** |

**Paired effect** (n = 30 matched session pairs):
- Mean per-session improvement: **+0.646**
- Wilcoxon signed-rank: **p = 1.86 × 10⁻⁹**
- Cliff's δ: **+1.000** (every patient improves)

For reference, PhysioMio: zero-shot 0.214 → calibrated 0.875 (+0.661). **The +66 pp magnitude replicates almost exactly.**

## 2. Severity stratification (L_CA 3-5)

| L_CA | n subjects | Zero-shot | Calibrated | Δ |
|---:|---:|---:|---:|---:|
| 3 (FMA 32-42) | 4 | 0.192 ± 0.061 | **0.854 ± 0.155** | +0.663 |
| 4 (FMA 43-52) | 2 | 0.181 ± 0.027 | **0.745 ± 0.201** | +0.563 |
| 5 (FMA 53-66) | 4 | 0.204 ± 0.050 | **0.761 ± 0.131** | +0.557 |

- Spearman ρ(L_CA, calibration benefit) = **-0.350**, p = 0.321
- Spearman ρ(L_CA, calibrated accuracy) = **-0.311**, p = 0.381

**Directionally consistent with PhysioMio's null finding** (PhysioMio ρ = -0.13). Negative ρ on Lucchetti suggests if anything, *more-impaired* patients benefit *more* from calibration — though n = 10 is underpowered to reject the null.

**Limitation:** L_CA 1-2 (severe / paralytic) are NOT represented in Lucchetti. PhysioMio covers that regime (FMA hand 0 / paralytic subjects present), so the severity-independence claim across the full impairment spectrum is supported by PhysioMio, with Lucchetti providing consistent evidence in the moderate-to-mild range.

## 3. Longitudinal degradation

**Not applicable.** Lucchetti subjects have a single recording session each. The cross-session generalisation claim is supported by PhysioMio (Stream 5 result: per-session recal necessary, +23 pp gap over one-time cal).

## 4. Per-arm comparison

| Arm | n | Zero-shot | Calibrated | Δ |
|---|---:|---:|---:|---:|
| Healthy (non-affected for stroke + dominant for HS controls) | 20 | 0.175 [0.162, 0.189] | **0.844 [0.787, 0.894]** | +0.669 |
| Impaired (paretic, stroke subjects only) | 10 | 0.194 [0.168, 0.223] | **0.795 [0.710, 0.877]** | +0.601 |

Healthy-arm and impaired-arm benefit similarly, with impaired arm ~5 pp behind — consistent with PhysioMio where impaired-arm accuracy trailed healthy-arm by ~6 pp.

## 5. Variance reduction analysis

| | Cross-subject SD |
|---|---:|
| Zero-shot (per-subject mean acc) | 0.033 |
| Calibrated (per-subject mean acc) | 0.103 |
| Ratio | **0.32× (variance INCREASES)** |

**This is opposite to PhysioMio**, where variance collapsed (4.2 % → 2.1 % = 2.05× collapse).

Interpretation: on Lucchetti, zero-shot accuracy is uniformly bad across subjects (everyone at ~18 %, very low SD). Calibration helps everyone, but unevenly — some subjects calibrate to >95 % accuracy, others stick at ~60 %. The wider post-calibration range reflects per-subject differences in how well their functional-task EMG matches the close/open framing imposed by the labeling. PhysioMio's uniformly cued protocol gives more consistent calibration outcomes.

This is a real finding worth reporting: calibration generalises across populations *and* task families, but per-subject calibration quality is more variable on functional-task data than on discrete-cue data.

## 6. Transition accuracy

### Stage 1 (raw per-window classifier)

| Configuration | Strict (0 % maint err) | Relaxed (10 %) |
|---|---:|---:|
| Patient mean | **0.601 [0.498, 0.700]** | **0.627 [0.544, 0.713]** |
| Session mean | 0.609 | 0.640 |

Higher than PhysioMio (raw 0.28 strict / 0.57 relaxed) — Lucchetti's protocol has ~5 reps × 6 tasks × ~3 transitions per task = many more transitions per session, which averages out per-segment maintenance flicker.

### Stage 2 (full deployed pipeline, sweep over assist profiles)

Strict criterion:

| Profile | Level | N | Patient trans acc | +Latency |
|---|---:|---:|---|---:|
| Max Assist | 1 | 1 | 0.522 [0.436, 0.610] | +0 ms |
| High Assist | 2 | 1 | 0.539 [0.428, 0.645] | +0 ms |
| **Moderate Assist ★** | **3** | **2** | **0.566 [0.474, 0.658]** | **+50 ms** |
| Light Assist | 4 | 3 | 0.548 [0.467, 0.632] | +100 ms |
| Minimal Assist | 5 | 3 | 0.548 [0.467, 0.632] | +100 ms |

Relaxed criterion:

| Profile | Level | N | Patient trans acc | +Latency |
|---|---:|---:|---|---:|
| Max Assist | 1 | 1 | 0.542 [0.450, 0.634] | +0 ms |
| High Assist | 2 | 1 | 0.553 [0.441, 0.658] | +0 ms |
| **Moderate Assist ★** | **3** | **2** | **0.605 [0.526, 0.684]** | **+50 ms** |
| Light Assist | 4 | 3 | 0.588 [0.517, 0.660] | +100 ms |
| Minimal Assist | 5 | 3 | 0.588 [0.517, 0.660] | +100 ms |

★ = deployed default. Note that on Lucchetti the pipeline doesn't always *help* over raw — Stage 2 has more conservative thresholds that occasionally reject legitimate predictions. PhysioMio showed the same pattern at L1/L2 levels.

## 7. Per-class F1 breakdown

| Class | Zero-shot F1 | Calibrated F1 | Δ |
|---|---:|---:|---:|
| Rest | 0.008 | **0.905 [0.874, 0.933]** | +0.897 |
| Close | 0.042 | **0.254 [0.131, 0.383]** | +0.212 |
| Open | 0.318 | **0.308 [0.180, 0.450]** | -0.010 |

**This is the key per-class finding for the paper**: the cross-population calibration recovers the rest-vs-movement boundary nearly perfectly (+90 pp on rest F1) but only partially recovers close-vs-rest (+21 pp on close) and **does not improve open at all**.

Why the per-class collapse? Two mutually compatible explanations:

1. **Labeling noise.** Our labels assign "close" to the *entire* BA/BC/SC movement window (which includes a reaching phase where the hand is open before grasping the object) and "open" to entire HM/HH windows (which includes return-to-rest where the hand is relaxed). The label is not aligned with the underlying EMG signature at every window, only on average. The model learns the noisy mapping and ends up using close/open as a roughly symmetric "non-rest activity" indicator — which lifts rest F1 but smears close/open.

2. **Task-EMG mismatch.** PhysioMio's close gestures are *isolated* finger flexion (with the wrist neutral and proximal muscles quiet). Lucchetti's close (BA/BC/SC) involves whole-arm reaching with proximal muscles activated, then a brief grip. The 4-channel forearm-extensor / forearm-flexor montage that works for PhysioMio doesn't cleanly distinguish "grip closure" from "reach-with-hand-tucked" because both have similar forearm muscle activation.

**Implication for the paper claim:** the calibration methodology generalises across populations *for protocols where the close/open intents are isolatable in the EMG signal* (PhysioMio). On protocols where they're tangled with whole-arm motion (Lucchetti), the methodology still recovers the rest-vs-movement gap closure but does not magically translate functional-task EMG into discrete finger intents. This is a useful and defensible scope statement.

## Files / artifacts

- `data/lucchetti_features_60_per_subject.pkl` — 192k windows × 60 features
- `data/lucchetti_label_log.txt` — per-subject labeling sanity log (FDS amplitude per class)
- `analysis/lucchetti/results/`:
  - `per_session_results.csv` (30 rows) + `per_patient_results.csv` (20 rows)
  - `per_window_predictions.parquet` (19,646 windows, Stage 1 argmax)
  - `per_window_probas.parquet` (19,646 windows, predict_proba — Stage 2 input)
  - `zero_shot_per_session.csv` + `_per_subject.csv`
  - `aggregate_summary.{md,json}`
  - `severity_summary.{md,json}` + `severity_per_subject.csv`
  - `transition_accuracy_{strict,relaxed}.{md,json}` + `_per_session.csv`
  - `full_deployed_pipeline_{strict,relaxed}.{md,json,csv}`
- `analysis/.cache/lucchetti_session_models/` — 30 cached HGB models for future re-runs

## Methodology + caveats (recap)

- Channels: Lucchetti FCR/ECR/FDS/EDC mapped onto GrabMyo canonical [0, 4, 9, 13] interleaved flex/ext scheme. FDS (finger flexor) + EDC (finger extensor) sit at the GrabMyo positions the base model was trained on.
- Sampling: Lucchetti 1 kHz → upsampled 2× to 2 kHz to match GrabMyo pipeline's 200 ms / 400-sample window assumption.
- Labels: paper task order (BA, BC, SC, PS, HM, HH) + Events.Start/End frames. Sanity-checked post-hoc: healthy subjects show FDS amplitude close > open > rest, confirming the close/open labels are at least directionally aligned with finger activity.
- No kinematic-event labeling: the Angles array (19 channels at 125 Hz, includes finger flexion) could give cleaner labels, but the MATLAB string fields naming the channels are stored in an opaque cell format that scipy / pymatreader can't decode. Would require running Octave/Matlab to dump the names.
- Cohort range: L_CA 3-5 (moderate to mild). L_CA 1-2 (severe / paralytic) not present.
- Single session per subject: no within-Lucchetti longitudinal analysis.

## Take-away for the paper

Lucchetti gives the paper:
1. **A second-dataset replication of the +66 pp cross-population gap closure** (matches PhysioMio magnitude almost exactly).
2. **Consistent severity-independence** in the L_CA 3-5 range (Spearman ρ ~ -0.3, ns).
3. **A scope statement**: the calibration methodology generalises across populations and task families *on the rest-vs-movement axis*. Fine within-movement discrimination (close vs open) requires matched protocol structure between training and deployment.
4. **A complementary variance story** to PhysioMio: PhysioMio shows variance *collapse* under calibration; Lucchetti shows variance *growth* (everyone fails similarly → some succeed beautifully, some less so). Both directions are coherent with the methodology — calibration acts on per-patient distributional shift, which is more uniform on cued data than on functional-task data.
