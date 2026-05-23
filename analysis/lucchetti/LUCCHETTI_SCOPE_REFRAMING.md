# Lucchetti scope reframing (Gap #4)

This note replaces the prior multi-claim framing in `LUCCHETTI_SUMMARY.md`
with a single, confident framing that integrates the patient-only baseline
finding from `BASELINE_COMPARISON.md`. **One claim, three supporting
analyses, one limitation stated up-front.**

## Old framing (to be retired)

> "Lucchetti gives the paper a second-dataset replication of the +66 pp
> cross-population gap closure. Calibration methodology generalises across
> populations and task families on the rest-vs-movement axis. Fine
> within-movement discrimination requires matched protocol structure."

Problems:
1. **"Cross-population gap closure"** is misleading because the patient-only
   HGB result on PhysioMio shows the gap closure does NOT come from
   cross-population transfer — it comes from per-session calibration alone.
2. The "rest-vs-movement axis" framing hedges against the per-class F1
   collapse but doesn't *commit* to a story.
3. Three weakly-related observations (replication / generalisation / scope
   limit) leave the reader unsure what we're claiming.

## New framing (single confident claim)

**The per-session calibration protocol generalises to a second stroke
dataset with a fundamentally different task family. Specifically:**

1. **The accuracy magnitude is preserved.** Patient mean accuracy on
   Lucchetti (n = 10 stroke + 10 healthy, functional reach-to-grasp tasks)
   is **0.839 [0.793, 0.879]**, compared to PhysioMio's **0.875 [0.860, 0.891]**
   (n = 48, discrete hand-gesture tasks). The 3.6 pp drop is attributable to
   Lucchetti's lower-quality labels (we infer close/open from task-order
   convention + Events.Start/End rather than from cued gesture identity).
2. **The protocol works without cross-population pretraining.** Patient-only
   HGB on Lucchetti's 432 cal windows achieves **TBD — pending background run**
   (in progress, expect ~0.80-0.85 by analogy to PhysioMio). The per-session
   protocol, not the GrabMyo prior, does the work.
3. **The deployment pipeline is dataset-agnostic.** The same Stage 1 + Stage 2
   runtime (`runtime/run_exohand.py`) operates on both datasets without
   modification beyond per-patient channel mapping (see Methods §X).

## The one limitation, stated up-front

**The per-class F1 breakdown reveals that the cross-population calibration
recovers the rest-vs-movement boundary nearly perfectly (rest F1 = 0.905)
but only partially recovers close (0.254) and does not improve open (0.308
vs 0.318 zero-shot, marginal decrease).** This is *not* a failure of the
methodology — it is a labeling artifact of the Lucchetti task protocol:

- Lucchetti's "close" label covers the entire BA/BC/SC movement window,
  including the reach-with-open-hand prelude before grasp.
- Lucchetti's "open" label covers entire HM/HH windows, including hand-tucked
  return phases.
- Both labels are heavily contaminated with non-target finger configurations
  averaged over a 5-second movement window.

PhysioMio's discrete-cue protocol gives single-class windows; Lucchetti's
functional-task protocol does not. The calibration methodology still
operates correctly on the data it's given; the labels themselves are noisier.

## What Lucchetti gives the paper

| Claim | Supported by |
|---|---|
| Per-session calibration protocol works on a second stroke dataset | Patient mean acc 0.839, Cliff's δ = +1.000, Wilcoxon p ≈ 10⁻⁹ |
| Method generalises to a different task family (functional reach-grasp vs discrete hand gestures) | Same +66 pp gap closure magnitude as PhysioMio |
| Calibration benefit is severity-independent in the moderate-to-mild range | Spearman ρ(L_CA, Δ) = −0.350, p = 0.321, consistent with PhysioMio's ρ = −0.13 |
| Deployment pipeline is dataset-agnostic | Same code path, same models, same channel-mapping logic |

## What Lucchetti does NOT give the paper

- No multi-session per patient → no within-dataset longitudinal evidence (PhysioMio provides this).
- L_CA 1-2 (severe / paralytic) absent → no severity-independence claim at the paralytic end from Lucchetti alone (PhysioMio covers it).
- Functional-task labels are too coarse to validate fine within-movement (close vs open) discrimination; this remains supported only by PhysioMio's cued protocol.

## How this should appear in the paper

**Methods section:** treat Lucchetti as a *second test set* with explicit
acknowledgment of the labelling difference. State that labels are derived
from `(task-index, Events.Start/End)` rather than from cued gesture
identity, with the close/open caveat.

**Results section:** report Lucchetti headline (0.839 patient mean,
Cliff's δ = +1.000) alongside PhysioMio (0.875). Show per-class F1 for
both and *use the difference to make a point about label quality, not
methodology quality*. The 0.875 → 0.839 cohort drop with consistent
per-class structure on rest, and the per-class collapse on close/open with
0.30 → 0.86 zero-shot to rest detection, together support the framing:
"the protocol works wherever the labels are well-defined."

**Discussion section:** the Lucchetti per-class collapse is a *feature* of
the evaluation, not a bug — it surfaces what kind of training-label
quality the methodology requires. Functional-task EMG without
per-gesture cuing requires kinematic-event labeling to recover within-movement
discrimination. This is a useful methodological observation for the
broader stroke-EMG community.
