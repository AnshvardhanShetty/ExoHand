# External stroke EMG datasets — survey for cross-population validation

Survey conducted 2026-05-19. Purpose: identify a second stroke EMG dataset
suitable for validating the cross-population calibration story currently
resting on PhysioMio (n = 48) alone.

## Decision summary

| Dataset | Stroke N | Finger muscles? | Open access? | Severity score | Verdict |
|---|---:|---|---|---|---|
| **Lucchetti 2025** | **10** | ✓ FDS + EDC | ✓ FigShare, CC-BY | ✓ FMA-UE 1-5 levels | **Use as 2nd validation set** |
| MUSED-I (Mayo Hospital Lahore) | 2 | ✗ Myo armband (wrist) | ✓ GitHub | ✗ | Too few patients |
| Ye Sun bilateral (Jan 2026) | ? | ✗ FCU/ECU/FCR/ECR only | ✗ IEEE DataPort subscription | ✗ | Subscription wall + wrist only |
| Montecinos et al. 2025 (PMC12656458) | 2 | ✗ wrist only | Stroke data not released | ✗ | Hardware confound + closed |
| Yang et al. (PMC12823985) | 5 | ✗ flexors only, 225 Hz | Partial (supplement only) | 0-4 scale | No extensors, low sample rate |
| Anastasiev 2025 (Sensors) | unknown | unknown | unknown (MDPI blocked) | unknown | Worth a fresh look if Lucchetti pivot fails |

**Recommendation:** Lucchetti is the only viable candidate; everything else
is either too small, hardware-mismatched, or paywalled.

## Detail per candidate

### ✓ Lucchetti et al. 2025 — *the recommended second set*

- **Source:** Scientific Data, DOI 10.1038/s41597-025-06174-3 · CC-BY 4.0 · FigShare collection 7720187
- **Local status:** **Already downloaded** to `data/lucchetti/{healthy,stroke}/{HS,ST}_NN/` (one `.mat` per subject). Total ~500 MB.
- **Cohort:** 10 stroke (ST_01..ST_10), aged 62-82, FMA-UE level 1-5 + TimeFromEvent categorical (1-3 mo, 3-6 mo) + HemiSide; 10 healthy controls (HS_01..HS_10) aged 24-73.
- **EMG hardware:** 12 channels @ 1 kHz, .mat float64. Channel list per paper:
  1. Posterior Deltoid
  2. Mediolateral Deltoid
  3. Anterior Deltoid
  4. Triceps
  5. Biceps
  6. Flexor Carpi Radialis (FCR — wrist flexor)
  7. **Flexor Superficialis Digitorum (FDS — finger flexor)** ← matches GrabMyo
  8. Extensor Carpi Radialis (ECR — wrist extensor)
  9. **Extensor Digitorum Communis (EDC — finger extensor)** ← matches GrabMyo
  10. Abductor Pollicis Brevis (APB — intrinsic thumb)
  11. Abductor Digiti Minimi (ADM — intrinsic pinky)
  12. First Dorsal Interosseus (FDI — intrinsic)
- **Channel anatomy is the *right* match.** GrabMyo's 4 canonical channels (0/4/9/13) are 2× FDS + 2× EDC at different forearm positions. Lucchetti channels **7 (FDS) + 9 (EDC)** map directly to GrabMyo's targets; channels 6 + 8 are the wrist-level analogues. Pick these 4 → drop-in compatible with the current 4-channel feature pipeline.
- **Tasks (6 per arm):** BA (Grasp Ball), BC (Grasp 5cm³ Block), SC (Grasp 2.5cm³ Block), PS (Prono-supination), HM (Hand-to-Mouth), HH (Hand-to-Head). Each task has ~5 repetitions with Start/End frame markers in 125 Hz kinematics frames.
- **Both arms:** `s.DataULpleg` (impaired) and `s.DataULnonpleg` (non-impaired) for stroke patients; `s.DataULdom` (dominant) for healthy.

#### Critical limitation: task framing mismatch

Lucchetti tasks are functional reach-to-grasp movements, NOT discrete
close / open / rest gesture commands like PhysioMio. Each ~107 s task
contains alternating phases: rest → reach → grasp (close) → hold → release
(open) → return → rest. To match our 3-class problem we need to *label
sub-phases within each task*, not just classify whole tasks.

Three approaches:

1. **Kinematic-event labeling (cleanest).** Use the synchronized joint angle
   data (`Angles` field at 125 Hz, includes finger flexion angles per the
   `upperlimb_fig.m` reference code) to detect grasp onset (finger flexion
   exceeds threshold) and release onset (finger extension exceeds
   threshold). Label segments between events as close / open / rest. Cost:
   ~2 days to write a reliable event detector.

2. **Binary movement-vs-rest (simplest).** Use the Events.Start / Events.End
   frames already provided (5 events per task × 6 tasks × 2 arms ≈ 60
   movement segments per subject) as "movement," the gaps as "rest."
   Drops to 2-class problem on Lucchetti — apples-to-oranges with our
   3-class PhysioMio number, but gives a clean cross-population claim
   for the *movement detection* sub-problem. Cost: ~0.5 day.

3. **Grasp-phase-only with task identity as proxy.** Treat the central
   third of each Event window as close (grasp tasks BA/BC/SC) or open
   (assumed during release sub-phase of all tasks) or rest (inter-event
   gaps). Approximation — relies on the assumption that grasp-task event
   windows are predominantly close. Cost: ~1 day; some label noise risk.

Recommendation: **Approach 1** (kinematic-event labeling) if we want a
clean 3-class story; **Approach 2** as a safe fallback that gives us *a*
cross-population number even if the labeling is hard.

#### .mat reading caveat

The .mat files use the new MATLAB string type for `TaskCode` and `*VarName`
fields, which neither `scipy.io.loadmat` nor `pymatreader` can decode
directly — both error with `MatlabOpaque` or skip the field. Workaround
options:

- Numeric fields (EMG matrix, Marker matrix, Angles matrix, Events.Start /
  Events.End) all decode normally — these are what we actually need.
- Task and channel string identifiers are recoverable by (a) running the
  bundled `upperlimb_fig.m` once in Octave to dump them to a CSV, (b)
  using `matlab.engine` if Matlab is installed, or (c) hard-coding the
  channel order from the paper (above) and inferring task order from the
  6-task fixed protocol order.

In practice we don't need the string fields — the numeric fields are
indexed positionally and the paper documents the ordering.

#### Cost / time estimate to integrate Lucchetti

- `ml/preprocessing_lucchetti.py` — convert .mat → 60-feature parquet, with kinematic-event-driven labels: ~1.5 days
- `analysis/lucchetti/per_session_eval.py` (mirror of `analysis/physiomio/per_session_eval.py`): ~0.5 day
- `analysis/lucchetti/aggregate_results.py` — bootstrap CIs, Wilcoxon vs PhysioMio: ~0.5 day
- Wall-clock compute: ~2 hours (10 patients × few minutes/patient HGB refit)
- **Total: ~3 days of work + ~2 hours of compute.**

---

### ✗ MUSED-I (Mayo Hospital Lahore / NUST)

- **Source:** GitHub MustafaMarwat/MUSED-1, Kaggle, IEEE DataPort
- **Cohort:** **2 stroke patients** + 10 healthy
- **EMG hardware:** Myo armband (8 channels, ~200 Hz) — proximal forearm only
- **Gestures:** wrist flexion, wrist extension, hand close, wrist radial deviation, wrist ulnar deviation, rest
- **Verdict:** Only 2 stroke patients — insufficient for any statistical claim. Wrist-axis gestures don't map to our hand close/open framing. Skip.

### ✗ Ye Sun bilateral sEMG (IEEE DataPort, Jan 2026)

- **Source:** IEEE DataPort, DOI 10.21227/jn18-dv70 — *subscription required*
- **EMG:** 8 sensors bilateral (4 per arm) on ECU / FCU / ECR / FCR (wrist muscles only, no finger flexor/extensor)
- **Gestures:** 9 actions (7 fine finger + 2 wrist)
- **Severity:** none reported
- **Verdict:** Subscription wall + no finger flexor/extensor channels = poor fit. Skip.

### ✗ Montecinos et al. 2025 (Sensors, PMC12656458)

- **Cohort:** 2 chronic ischemic stroke (mild-moderate spasticity) + 40 healthy
- **Hardware:** different between groups (BIOPAC MP36 / 4 channels for healthy, Backyard Brains SpikerBox / 2 channels for stroke) — hardware confound
- **Gestures:** rest, wrist extension, wrist flexion, grip, finger abduction, supination
- **Severity:** "mild to moderate spasticity" only — no FMA
- **Data:** healthy released, **stroke data NOT publicly available** (ethical/privacy)
- **Verdict:** Closed + hardware confound + 2 patients. Skip.

### ✗ Yang et al. (PMC12823985)

- **Cohort:** 4 stroke (cerebral infarction) + 1 hemorrhage + 1 myelitis + 1 other hemiplegia + 5 healthy = 13 total
- **Hardware:** 6 channels on **flexor muscles only** (FPL, FDS, FDP) — no extensors at all
- **Sampling rate:** 225 Hz
- **Severity:** 0-4 motor function scale (coarse)
- **Data:** supplementary file only (5.5 MB)
- **Verdict:** Mismatched anatomy (no extensors), low sampling, supplementary-only. Skip.

---

## Critical prior art — ReactEMG Stroke

**Wang et al., arXiv 2601.22090 (Jan 2026, Columbia)** — *this is the closest related work and must be cited and positioned against.*

- **Method:** Transformer encoder (ReactEMG) pretrained on 5 healthy datasets totaling 650+ able-bodied subjects, then fine-tuned per stroke patient via head-only / LoRA / full strategies. Deep-learning, foundation-model-style approach.
- **Stroke cohort:** **n = 3** chronic stroke participants. S1: FMA-UE 26 (hand subscore 1). S2: FMA-UE 35 (hand subscore 2). S3: FMA-UE 34 (hand subscore 8).
- **Hardware:** Myo armband, 8 channels, 200 Hz.
- **Task:** 3-class (rest / open / close), cued sequences ROROROR + RCRCRCR with 5-6 s segments.
- **Headline result (averaged over 5 held-out test sets across 3 patients):**
  - Zero-shot 0.60 raw / 0.13 transition
  - Stroke-only 0.69 raw / 0.42 transition
  - **Best (LoRA / Full) 0.78 raw / 0.61 transition** ← their headline
- **Data-efficiency curve:** N = 0 (zero-shot) → N = 1 → 4 → 8 → All-12 sample budgets, full-fine-tune adaptation. Saturates around 0.61 transition accuracy.

### Positioning notes for our paper

- **Larger cohort.** We have **48 stroke patients (PhysioMio) + potentially 10 (Lucchetti)**; they have 3. Stronger statistical claim if we can keep our numbers.
- **Different method.** They use a 650-subject pretrained transformer + LoRA; we use HGB + weighted refit on 43-subject GrabMyo. Classical-ML vs foundation-model is a legitimate axis of contribution — *simpler, no GPU required, < 50 ms / cycle on CPU* is a deployment argument.
- **Different framing.** They emphasize **distribution-shift test sets** (within-session drift, unseen posture, sensor placement, device-driven motion) — five held-out perturbations. We don't currently have a posture/placement perturbation analysis; consider adding one as an ablation.
- **Different metric — transition accuracy.** They argue raw accuracy is misleading because the rest class dominates and "raw accuracy can obscure these issues because both delayed onsets and transient mislabels contribute only a small number of frame errors." **We should add transition accuracy to our analysis** — it's a much stronger real-time-control metric.
- **Headline numbers — caveat.** Their "0.78 raw / 0.61 transition" is averaged across 3 patients evaluated under 5 distribution-shifted held-out sets. Our "0.875 patient mean" is single-test-set per patient. Not directly comparable; we should re-evaluate ours under their distribution-shift framework before claiming a head-to-head win.

---

## Two-paragraph recommendation

**For the paper, the single most valuable next experiment is Lucchetti**
(~3 days work + 2 h compute). It is the only dataset with a) ≥ 10 stroke
patients, b) finger flexor / extensor channels that match our model anatomy,
c) per-patient FMA-UE severity scores, and d) open licensing. The task
framing is different — functional reach-to-grasp rather than discrete
hand gestures — which forces us to label sub-phases via kinematic events,
but the 12-channel + 1 kHz + both-arms structure is otherwise ideal.

**The other essential addition is direct comparison and positioning
against ReactEMG Stroke** (Wang et al., arXiv 2601.22090, Jan 2026). This
is the closest related work — same problem, deep-learning approach,
n = 3 stroke patients with FMA-UE 26-35. We should add their transition
accuracy metric to our analysis and discuss the methodological tradeoff
(classical ML + larger cohort vs foundation-model + small cohort) directly
in the paper. They went on arXiv 4 months before our target submission;
the reviewers will know it.
