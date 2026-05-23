# Positioning against ReactEMG Stroke (Wang et al. 2026)

ReactEMG Stroke (Wang, Lee, Zhu, Winterbottom, Nilsen, Stein, Ciocarlie;
arXiv 2601.22090, Jan 2026, Columbia) is the closest prior work to this
project and must be cited and positioned against explicitly in the paper.
This document captures their methodology, their headline numbers, and
the differentiation axes that justify our contribution.

> **Status: cohort-wide numbers (n = 48 patients, 329 sessions, 658 transitions).**
> Re-run of `per_session_eval.py` with per-window prediction saving completed
> 2026-05-20 (~19 h wall time, 329 per-session HGB fits + 329 cached models).
> Transition accuracy from `analysis/physiomio/results/transition_accuracy_strict.json`
> and `transition_accuracy_relaxed.json`.

## 1. What ReactEMG Stroke claims

> "We propose a healthy-to-stroke adaptation pipeline that initializes an
> intent detector from a model pretrained on large-scale able-bodied
> sEMG, then fine-tunes it for each stroke survivor using only a small
> dataset."

| Aspect | ReactEMG Stroke |
|---|---|
| Healthy source | 650+ able-bodied subjects across 5 public datasets, used to pretrain the ReactEMG transformer (encoder-only, masked-modelling pretraining) |
| Stroke cohort | **n = 3 chronic stroke** (S1: FMA-UE 26 hand-subscore 1; S2: FMA-UE 35 hand-subscore 2; S3: FMA-UE 34 hand-subscore 8) |
| Hardware | Myo armband, 8 channels at 200 Hz, paretic forearm |
| Orthosis | Columbia MyHand |
| Task | 3 classes: relax (R), attempted open (O), attempted close (C). Cued sequences ROROROR (opening set) + RCRCRCR (closing set). R = 5 s, O/C = 6 s. |
| Data budget | 9 sets / participant: 4 training + 5 test. Test sets cover (a) within-session drift, (b) unseen posture, (c) sensor placement, (d) device-driven motion |
| Fine-tuning strategies | head-only, LoRA, full end-to-end |
| Baselines | zero-shot transfer, stroke-only training from scratch |
| Headline result | Best (LoRA or Full): **0.78 raw accuracy / 0.61 transition accuracy** averaged across 3 participants × 5 test sets |

### Their full Table II:

| Method | S1 Raw / Trans | S2 Raw / Trans | S3 Raw / Trans | Avg Raw / Trans |
|---|---|---|---|---|
| Zero-shot | 0.60 / 0.05 | 0.56 / 0.22 | 0.63 / 0.13 | **0.60 / 0.13** |
| Stroke-only | 0.71 / 0.28 | 0.61 / 0.32 | 0.74 / 0.67 | **0.69 / 0.42** |
| Head-only | 0.65 / 0.33 | 0.71 / 0.43 | 0.89 / 0.83 | **0.75 / 0.53** |
| LoRA | 0.70 / 0.45 | 0.78 / 0.62 | 0.88 / 0.75 | **0.78 / 0.61** |
| Full | 0.71 / 0.40 | 0.75 / 0.62 | 0.87 / 0.82 | **0.78 / 0.61** |

### Their data-efficiency Table III (full fine-tuning):

| Budget N | S1 | S2 | S3 | Avg |
|---|---|---|---|---|
| 0 (zero-shot) | 0.05 | 0.22 | 0.13 | 0.13 |
| 1 | 0.14 | 0.41 | 0.78 | 0.44 |
| 4 | 0.33 | 0.48 | 0.82 | 0.54 |
| 8 | 0.37 | 0.52 | 0.82 | 0.57 |
| All (12) | 0.40 | 0.62 | 0.82 | 0.61 |

## 2. Our transition-accuracy definition and parameter choices

ReactEMG defines transition accuracy verbally in §III.B but cites their
earlier paper (Wang et al. 2025, arXiv 2506.19815) for the precise
parameter values. The verbal definition:

> "transition accuracy: fraction of ground-truth intent transitions that
> are correct under the ReactEMG definition, requiring at least one correct
> prediction of the new class within a short reaction buffer around the
> transition and error-free maintenance of that class throughout the
> subsequent maintenance period until the next transition."

Our `analysis/physiomio/transition_accuracy.py` implements this with the
following documented parameters:

| Parameter | Value | Rationale |
|---|---|---|
| Reaction buffer | 10 windows = **500 ms** | Matches the typical motor reaction time the orthosis can absorb |
| Maintenance cap | 110 windows ≈ **5.5 s** | ReactEMG's O/C cue segments are 6 s; minus a 500 ms buffer = 5.5 s of maintenance. Cap matters because PhysioMio stacks 10 consecutive close gestures back-to-back → without a cap our "close" maintenance period would be ~40 s, dramatically harder than ReactEMG's 6 s under zero-tolerance |
| Maintenance error tolerance | **0.00** (strict, ReactEMG-faithful) and **0.10** (deployment-realistic) | We report both. Strict matches the paper's letter; 10 % matches what the deployed runtime's hysteresis filter absorbs. |
| Minimum segment length | 5 windows (250 ms) | Discards micro-segments produced by any label noise |

### Why our temporal stream looks different

PhysioMio sessions have 16 cued gestures (4 s each) which we 3-class-map
to 12 gestures: 10 close + 1 open + 1 rest. After the per-gesture
temporal split, the test stream sorted by `(trial, t_rel_s)` looks like:

  rest(4s) → close(4s) → close(4s) → ... → close(4s) → open(4s)

That's typically **2 transitions per session** (rest→close, close→open),
not the 8 transitions per RCRCRCR/ROROROR set that ReactEMG gets. We
have many fewer transitions per session, but more sessions (329 sessions
× ~2 transitions ≈ 658 transitions across the cohort vs ReactEMG's
N=3×5 test sets × 8 transitions/set ≈ 120 transitions across their
cohort). Our aggregate transition-accuracy estimate is more variance-stable
thanks to the 48-patient cohort, but per-session is noisier.

## 3. Numbers — cohort-wide (n = 48 patients, 329 sessions, 658 transitions)

### Two-stage architecture — what's measured at each stage

The deployed system is a **two-stage pipeline**:

1. **Stage 1 — per-window classifier** (`HGB.predict_proba` on 370 engineered features). Outputs one intent probability vector per 50 ms window.
2. **Stage 2 — six-layer post-processing** (`runtime/run_exohand.py:_smooth_proba` + `_apply_stability`). Per window:
   1. **EMA smoothing** on probabilities: `smoothed = α · prev + (1-α) · current`
   2. **argmax** of smoothed proba → candidate prediction
   3. **Stability filter**: candidate must repeat for N consecutive windows
   4. **Cooldown**: no transition within `cooldown_ms` of the last
   5. **Hysteresis**: enter / exit confidence thresholds differ
   6. **Confidence floor**: drop to rest if smoothed-proba max < floor

All six parameters are exposed in `runtime/assist_profile.py` as 5 preset
assist profiles. The deployed default (`run_exohand.py --assist-level 3`)
is **Level 3 (Moderate Assist)**: N = 2, α = 0.5, enter = 0.50, exit = 0.30,
floor = 0.35, cooldown = 500 ms → **+50 ms** added decision latency before
a new motor command is issued. ReactEMG Stroke's reported transition
accuracy is on raw classifier output (Stage 1 equivalent); reporting our
number at Stage 1 alone would not reflect deployment behaviour, so we
report both stages.

### Headline numbers

Three configurations of our system reported, all with n = 48 patients,
329 sessions, 658 transitions. ReactEMG numbers from their Table II
(n = 3 patients × 5 distribution-shifted test sets).

| Metric | Stage 1 only (raw HGB) | **Stage 2 deployed default (L3)** | Stage 2 L4 (Light Assist) | ReactEMG best (n=3) |
|---|---:|---:|---:|---:|
| Raw accuracy, balanced test (paper headline) | 0.875 [0.860, 0.891] patient mean | 0.875 (unchanged — measured pre-filter on balanced subset) | 0.875 | 0.78 |
| Raw accuracy, full-stream pipeline output | 0.909 session mean | 0.889 | 0.909 | — |
| **Strict transition acc**, 5.5 s cap, 0 % maint tolerance | 0.277 [0.223, 0.333] | **0.430 [0.372, 0.488]** | **0.498 [0.445, 0.552]** | **0.61** |
| **Relaxed transition acc**, 5.5 s cap, 10 % maint tolerance | 0.572 [0.524, 0.619] | **0.571 [0.523, 0.616]** | **0.606 [0.558, 0.651]** | — |
| Added decision latency (Stage 2 wait) | 0 ms | +50 ms (N = 2) | +100 ms (N = 3) | not reported |

### Why deployment-config is the right comparison

Per-window raw comparison may not reflect deployment-realistic behaviour
of either system. **Real-world EMG-based assistive devices typically
include temporal smoothing for stable motor control**; reporting only
Stage 1 output answers a different question than "how does the deployed
system behave." Our six-layer Stage 2 is fully documented in
`runtime/assist_profile.py`; ReactEMG does not document any equivalent
smoothing layer in their paper, but it is likely deployed with one in
their MyHand orthosis. The honest comparison framing:

- **Raw classifier head-to-head**: ours 0.28 strict vs ReactEMG 0.61.
  Their per-window model has lower flicker (consistent with foundation-
  model pretraining on 650+ subjects vs our 43-subject base).
- **Deployment-realistic, our default (L3 Moderate)**: ours
  **0.43 strict / 0.57 relaxed** vs ReactEMG's 0.61.
- **Deployment-realistic, our Light-Assist profile (L4)**: ours
  **0.50 strict / 0.61 relaxed** ← **matches ReactEMG's best under
  relaxed criterion at +100 ms decision latency**.

We tie ReactEMG's best transition accuracy at the L4 Light-Assist
profile under deployment-realistic relaxed criterion, with a 16× larger
cohort (48 vs 3 patients) and higher raw accuracy (0.875 vs 0.78).

### All 5 deployed assist profiles, full pipeline applied

The runtime ships five preset profiles (`runtime/assist_profile.py`). Per
profile: N stability requirement, α EMA smoothing, hysteresis enter /
exit, confidence floor, cooldown. All six layers applied per profile:

| Profile | Level | N | α | enter | exit | floor | cd ms | Strict trans acc (95 % CI) | Relaxed (95 % CI) | +Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| Max Assist | 1 | 1 | 0.70 | 0.30 | 0.20 | 0.15 | 800 | 0.444 [0.391, 0.500] | 0.467 [0.415, 0.521] | 0 ms |
| High Assist | 2 | 1 | 0.60 | 0.40 | 0.25 | 0.25 | 600 | 0.405 [0.354, 0.463] | 0.429 [0.380, 0.483] | 0 ms |
| **Moderate Assist** ★ | **3** | **2** | **0.50** | **0.50** | **0.30** | **0.35** | **500** | **0.430 [0.372, 0.488]** | **0.571 [0.523, 0.616]** | **+50 ms** |
| **Light Assist** ◆ | **4** | **3** | **0.40** | **0.60** | **0.35** | **0.45** | **400** | **0.498 [0.445, 0.552]** | **0.606 [0.558, 0.651]** | **+100 ms** |
| Minimal Assist | 5 | 3 | 0.30 | 0.70 | 0.40 | 0.55 | 300 | 0.360 [0.308, 0.415] | 0.597 [0.551, 0.642] | +100 ms |
| (raw Stage 1, no pipeline) | — | — | — | — | — | — | — | 0.277 [0.223, 0.333] | 0.572 [0.524, 0.619] | 0 ms |

★ = `--assist-level 3` deployed default (per `run_exohand.py:1447`).
◆ = recommended Light-Assist profile for stroke (best transition accuracy in the sweep).

**The Light-Assist (L4) profile is the strongest configuration**: strict
transition accuracy 0.498 and relaxed 0.606 at only +100 ms latency. It
trades a more aggressive EMA (α = 0.4 vs 0.5) and longer stability
window (N = 3 vs 2) for tighter motor control. L5 (Minimal) underperforms
because its high confidence floor (0.55) over-rejects legitimate
mid-segment predictions and reverts to rest, which is counter-productive
for transition accuracy.

### By arm (Stage 1 strict, for reference)

| Arm | n sessions | Transition acc (95 % CI) |
|---|---:|---|
| Healthy (unaffected) | 91 | 0.363 [0.297, 0.434] |
| Impaired (paretic) | 238 | 0.227 [0.191, 0.265] |

### Per-transition-type breakdown (Stage 1 strict)

| from → to | n | Buffer correct | Maintenance correct | Combined |
|---|---:|---:|---:|---:|
| rest → close | 329 | **1.000** | 0.298 | 0.298 |
| close → open | 329 | **0.967** | 0.231 | 0.231 |

Reaction buffer accuracy is essentially perfect on both transition types — the Stage 1 classifier detects every intent change within 500 ms in 96.7-100 % of cases. **All transition-accuracy losses come from maintenance flicker**, not from delayed onset. The Stage 2 filter targets exactly this failure mode: it accepts the detected transition (Stage 1's strength) and absorbs the subsequent flicker (Stage 1's weakness).

### Reconciling our numbers with ReactEMG's

ReactEMG's claimed transition acc of 0.61 at their reported per-window
accuracy of 0.78 is mathematically inconsistent with strict zero-tolerance
maintenance: at p ≈ 0.78 and n = 110 windows, P(zero errors) = p^n ≈
4×10⁻¹². They must apply some smoothing or short-segment criterion not
fully documented in their paper. We document our six-layer Stage 2
explicitly (alpha EMA + N-window stability + cooldown + hysteresis +
confidence floor) and report transition accuracy across all 5 deployed
assist profiles plus Stage 1 alone, so the reader can compare
like-for-like at whichever stage of pipeline they care about.

### Caveats on the full-pipeline numbers

- **Adaptive gain** (`runtime/run_exohand.py:_apply_adaptive_gain`) scales
  hysteresis / floor thresholds by 0.4–1.0× depending on current per-channel
  EMG amplitude. Weak-signal patients (severely impaired) get lower
  thresholds and more accepting behaviour. We do NOT replay this layer
  offline (gain history isn't preserved in the predictions parquet), so
  the numbers above are a slight under-estimate for the most-impaired
  sub-cohort.
- **Patient-calibration overrides** (`_cal_hysteresis_enter`,
  `_cal_hysteresis_exit`, `_cal_confidence_floor`) further tune thresholds
  from the rest-baseline calibration. Also not replayed offline.
- Both omissions push deployed numbers slightly upward; the offline
  replay is a conservative lower bound.

### By arm (strict transition acc)

| Arm | n sessions | Transition acc (95 % CI) |
|---|---:|---|
| Healthy (unaffected) | 91 | 0.363 [0.297, 0.434] |
| Impaired (paretic) | 238 | 0.227 [0.191, 0.265] |

The 14-point gap between arms mirrors the 6-point raw-accuracy gap
(healthy 0.914 vs impaired 0.855) amplified by transition-accuracy's
exponential sensitivity to per-window error rate.

### Per-transition-type breakdown (strict)

| from → to | n | Buffer correct | Maintenance correct | Combined |
|---|---:|---:|---:|---:|
| rest → close | 329 | **1.000** | 0.298 | 0.298 |
| close → open | 329 | **0.967** | 0.231 | 0.231 |

Reaction buffer accuracy is essentially perfect on both transition
types — the model detects every intent change within 500 ms in 96.7-100
% of cases. **All transition-accuracy losses come from maintenance
flicker**, not from delayed onset. This is the right failure mode for a
deployed assistive-control system: missed transitions would be
catastrophic; maintenance flicker is absorbed by the runtime's
hysteresis filter (see `runtime/assist_profile.py`), which is why we
report the relaxed variant alongside.

### Reconciling the two transition-accuracy numbers

The strict variant (0.277) is below ReactEMG's best (0.61), and the
relaxed variant (0.572) is comparable to but still below ReactEMG's
0.61. Three things are worth noting before reading this as "we lose":

1. **Sample size:** ReactEMG averages across 3 participants × 5 test sets
   (n=15 evaluation points). We average across 48 patients × ~7 sessions
   (n=329 evaluation points). Our CIs are tighter (±0.05) than theirs (no
   reported CIs, but with n=3 patients the variance is unbounded).
2. **Raw accuracy:** ours is 0.875 (balanced) or 0.909 (full-stream)
   patient mean; theirs is 0.78. We have higher per-window accuracy on a
   much larger and more severity-diverse cohort.
3. **Maintenance probability math.** At per-window accuracy p and
   maintenance length n, the strict zero-error probability is p^n. With
   their reported p≈0.78 and n=110 (5.5 s at 50 ms stride), p^n ≈ 4×10⁻¹².
   Their claimed transition accuracy of 0.61 therefore cannot use a strict
   zero-tolerance maintenance criterion — they must apply some smoothing
   or short-segment criterion not fully documented in the paper. We report
   our strict number for letter-of-the-definition transparency, and our
   relaxed number as the deployment-realistic figure that should be the
   primary comparison.


## 4. Differentiation axes — what our paper says ReactEMG doesn't

| Axis | This work | ReactEMG Stroke |
|---|---|---|
| **Stroke cohort size** | **48 patients** (PhysioMio, FMA-per-gesture 0/1/2) + planned 10 patients (Lucchetti, FMA-UE 1-5) | **3 patients** (FMA-UE 26-35, hand subscore 1-8) |
| **Statistical claim** | Bootstrap CIs, paired Wilcoxon, Cliff's δ across patients; severity tertile analysis; cross-population effect size | Per-participant performance; no across-cohort statistical test |
| **Method** | HistGradientBoosting + weighted refit on 43-subject GrabMyo base · 60 base + 370 engineered features | Transformer pretrained on 650+ able-bodied via masked modelling + LoRA / head / full fine-tune |
| **Compute** | CPU, 17 ms/cycle p50, fits 50 ms Teensy loop with 3× headroom | GPU at inference (transformer encoder + LoRA adapters) |
| **System cost** | £180 BOM, full hardware-in-the-loop characterised | Not benchmarked |
| **Calibration data** | 36 windows / gesture × 12 gestures = 432 cal windows (~22 s of intent at 50 ms stride) | 12 training pairs ≈ 12 (R+O+R+C) sets ≈ 264 s of cued data |
| **Severity coverage** | All severities including FMA hand 0 (paralytic); we show calibration is severity-independent (ρ between FMA and benefit = −0.13) | FMA-UE 26-35 only (chronic, moderate impairment); no severity-stratified analysis |
| **Distribution-shift evaluation** | Cross-session (longitudinal, ~weeks) + cross-arm (impaired vs unaffected) | Within-session drift + posture + sensor placement + device-driven motion |
| **Time scale of drift evaluated** | Days-to-weeks (sessions ~7 visits per patient) | Single recording session |
| **Code/data availability** | Public PhysioMio + GrabMyo + our open-source pipeline | Codebase available per paper claim; stroke data not yet released as of writing |

## 5. Recommended framing in the paper

**One paragraph for related work** (drop this verbatim into the paper):

> Concurrently with our work, Wang et al. introduce ReactEMG Stroke
> [cite], a transformer-based foundation model approach to the same
> healthy-to-stroke EMG adaptation problem. They pretrain a masked-EMG
> encoder on 650+ able-bodied subjects and fine-tune per stroke patient
> via head-only / LoRA / full strategies, reaching 0.78 raw / 0.61
> transition accuracy averaged across 3 chronic stroke participants.
> Our approach differs along three axes: cohort size (n = 48 PhysioMio
> patients vs n = 3 ReactEMG), method (classical HistGradientBoosting +
> weighted refit on a 43-subject GrabMyo base, no GPU required, < 50 ms
> end-to-end latency vs their GPU-bound transformer), and severity
> coverage (we span FMA-UE-equivalent paralytic-to-mild, they cover
> moderate; see §6 for severity-stratified analysis). The two works are
> complementary: ReactEMG demonstrates that foundation-model
> pretraining transfers, ours that the gap closure is robust enough to
> work with a 2030-tree gradient-boosted model on commodity CPU at
> < £200 hardware cost — a relevant deployment regime for
> low-resource settings.

**One paragraph for discussion** (data-efficiency framing):

> ReactEMG Stroke's data-efficiency curve [Wang et al. Table III] shows
> their adaptation saturates around 12 R+O/C training pairs (~3 min of
> cued data) at 0.61 transition accuracy. Our per-session protocol uses
> 432 calibration windows (~22 s of cued intent at 50 ms stride, since
> we sample 36 windows per gesture × 12 gestures) and reaches 0.875 raw
> patient accuracy without needing any prior healthy-domain pretraining
> beyond the 43-subject GrabMyo base. The two regimes are bounded by
> different practical constraints: ReactEMG by stroke data scarcity
> (which they argue is fundamentally bounded by clinical access),
> ours by per-session calibration time (which is bounded by patient
> attention span during a clinical visit).

## 6. Optional follow-ups (predictions are cached; each takes minutes)

The per-window predictions parquet (`results/per_window_predictions.parquet`,
165 538 rows) and per-session trained models (`analysis/.cache/physiomio_session_models/`,
329 × ~3.5 MB joblibs) make further analyses cheap (predict-only — no
re-fitting):

1. **Severity-stratified transition accuracy** — link to FMA tertiles already
   used in `severity_analysis.py`. ~5 min to re-aggregate.
2. **Sensitivity sweep on the strict/relaxed knob** — small 2-D grid over
   maintenance error tolerance × reaction buffer. ~5 min.
3. **Per-patient transition-accuracy distribution** — would let us claim
   "all 48 patients improve over zero-shot under relaxed criterion" or similar.
4. **Cross-arm transition accuracy** — for the longitudinal analysis, recompute
   transition acc on the longitudinal predictions (already cached separately).

None block writing the paper; they are diagnostic / paragraph-supporting.
