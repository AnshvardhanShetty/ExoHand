# Clinical-outcome translation

Raw accuracy doesn't speak to clinical relevance. Here we translate the deployed pipeline (Stage 2, N=3 stability filter — the `run_exohand.py --assist-level 3` default) into three metrics a rehab clinician would actually use.

All metrics on n = 48 patients, 329 sessions, 165,538 per-window predictions.

## Headline metrics (overall, both arms)

| Clinical metric | Mean (95 % CI) | Plain-language reading |
|---|---:|---|
| **Per-rep success rate** | **0.977** [0.965, 0.988] | Of every 10 cued grasps in a session, ~9.8 produce a sustained ≥ 250 ms correct motor command. |
| **False activations / min of rest** | **1.91** [1.13, 2.87] | During rest, the system spuriously issues a non-rest command 1.9× per minute on average. |
| **Time-to-correct-command** | **149 ms** [133, 166] | After an intent change, the system reaches a sustained correct command in ~149 ms (mean). |

## By arm

| Arm | n sessions | Per-rep success (95 % CI) | False-act/min rest | Time-to-correct (ms) |
|---|---:|---|---:|---:|
| healthy | 91 | 0.989 [0.973, 1.000] | 0.63 | 126 |
| impaired | 238 | 0.973 [0.958, 0.985] | 2.40 | 157 |

## Definitions

- **Per-rep success.** For every ground-truth transition into close or open, counts as successful iff the post-Stage-2 prediction matches the new class for at least 5 consecutive windows (250 ms) anywhere in the segment. This is the clinically-meaningful question: not just *did the model see the intent*, but *did it issue a stable command long enough to drive the actuator*.
- **False activations / min of rest.** Per session, count discrete transitions of post-Stage-2 output from rest → non-rest while ground truth is in a rest period; divide by total rest duration in minutes. This is the spurious-command rate a patient experiences when not trying to move.
- **Time-to-correct-command.** For each successful rep, the latency (ms) from ground-truth segment start to the first window of the sustained correct run. Adds to the system's other latencies (~225 ms hardware/software pipeline, §3) to give the full intent → action delay.

## How this enters the paper

One paragraph in §4 or §6 (clinical relevance), positioned between the transition-accuracy and limitations sections:

> *Translating the deployed pipeline (Stage 2, N = 3) into clinical metrics: the system completes **97.7% of cued grasps** with a sustained motor command (≥ 250 ms), with **1.9 false activations per minute of rest** and a **{ttc_mean:.0f}-ms median time-to-correct-command**. For a typical 5-minute therapy session with 70 % rest, a patient would experience ~{false_act_per_session:.0f} spurious activations and successfully trigger ~{expected_successful_reps:.0f} of every 10 attempted grasps. The spurious-activation rate is the dominant remaining limitation for fully autonomous use; the deployed runtime's cooldown + hysteresis layers (§3.4) reduce it further in practice but were not replayed offline here.*