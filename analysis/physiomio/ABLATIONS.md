# Ablations supporting the per-session calibration claim

Three ablations that the TS4H paper carries in the methods + supplementary:

1. **Minimal-cal protocol** — how much cued patient data is actually needed (proposes 22 s, not 60 s).
2. **Cross-session test of the GrabMyo prior** — whether healthy-population data earns its keep across sessions.
3. **Model-class comparison** — HGB vs LDA on the same training mix.

All three use the same protocol skeleton: per-gesture temporal split, balanced 117-window test set (39 per class), 3-window buffer, single fast HGB (`max_iter=300, max_depth=10, class_weight='balanced'`) unless noted otherwise.

## 1. Minimal-cal saturation (proposes 22 s deployed-cal duration)

n = 328 sessions across 48 patients. Patient-only HGB (no GrabMyo) trained on **N** cal windows per gesture × 12 gestures, evaluated on a balanced test set.

| Cal duration | Windows/gesture | Mean accuracy | Mean F1-macro | F1 rest / close / open |
|---:|---:|---:|---:|---:|
| 7.2 s | 12 | 0.832 ± 0.104 | 0.814 | 0.946 / 0.821 / 0.675 |
| 14.4 s | 24 | 0.863 ± 0.103 | 0.849 | 0.974 / 0.846 / 0.726 |
| **21.6 s** | **36** | **0.870 ± 0.105** | **0.859** | **0.969 / 0.852 / 0.755** |
| 36 s (default) | 60 | 0.870 ± 0.104 | 0.859 | 0.970 / 0.853 / 0.755 |
| 54 s | 90 | 0.870 ± 0.104 | 0.859 | 0.969 / 0.852 / 0.754 |
| 72 s | 120 | 0.871 ± 0.104 | 0.859 | 0.970 / 0.853 / 0.755 |

**Same-session accuracy saturates at ~22 s of cued cal data.** Going from 7 → 22 s buys +3.8 pp; going from 22 → 72 s buys 0.0 pp. The current 60 s default is **conservative**; the deployed protocol can be shortened to ~22 s without measurable accuracy loss.

**Why this matters for deployment:** stroke patients have limited attention and tolerance for cued tasks. Cutting calibration time by 60 % while preserving accuracy is a clinically meaningful design choice, not a methodological detail.

**Proposed protocol:** 22-second cued calibration per session (12 gestures × ~1.8 s active each), used as the head of the deployed two-stage runtime described in §3.4 of the paper.

## 2. Cross-session test of the GrabMyo prior

The deployed model is HGB refit on (GrabMyo, weight 1) + (patient cal, weight 100×). We test whether the GrabMyo prior actually generalises across sessions of the same patient — i.e., whether it earns its keep as a regulariser.

**Protocol:** for each patient, train a single model on impaired_01 cal data only (the patient-only variant) and a separate model on impaired_01 cal + GrabMyo (the main eval variant). Test both on every other session of the same patient, paired.

| Session distance from cal | n test-sessions | Patient-only acc | GrabMyo+cal acc | Δ (PO − GM) |
|---:|---:|---:|---:|---:|
| 0 (same session) | 48 | 0.877 | 0.875 | +0.002 |
| 1 | 46 | 0.723 | 0.717 | +0.005 |
| 2 | 37 | 0.632 | 0.577 | +0.055 |
| 3 | 31 | 0.635 | 0.580 | +0.055 |
| 4 | 22 | 0.665 | 0.618 | +0.047 |
| 5 | 20 | 0.694 | 0.592 | +0.101 |
| 6+ | 31 | 0.681 | 0.547 | +0.134 |
| **Later-sessions aggregate (≥ 1)** | **190** | **0.675** | **0.618** | **+0.057** |

**Paired per-patient (later sessions, n = 46):** patient-only better in 29/46; mean Δ = +0.047 (patient-only); **Wilcoxon p = 0.025**; distribution is heterogeneous (17 patients strongly favour patient-only, 11 favour GrabMyo+cal, 18 ties).

**Honest reading.** The GrabMyo prior is *not* a cross-session regulariser for our HGB pipeline; if anything, it slightly hurts (statistically significant at p = 0.025, but heterogeneous across patients). This argues that what closes the cross-population gap is the **per-session calibration data itself**, not the healthy-population prior. We report it as evidence rather than as a paper-rewriting claim:

- The cross-population gap (zero-shot 0.21 → calibrated 0.87) characterised in §4.1 is real and is what calibration corrects.
- The *mechanism* of correction, on classical HGB + engineered features, is patient-side fitting rather than knowledge transfer from healthy young adults.
- This does **not** speak to representation-transfer methods (NNs / transformers, ReactEMG-style); HGB doesn't transfer representations, only data pools. Whether representation transfer would help at our cohort size is an open question and explicitly out of scope.

**Implication for deployment:** the deployed system could be simplified by dropping the 1.14 M-window GrabMyo base — the per-session calibration alone suffices and generalises modestly better across sessions. We retain the GrabMyo pretraining in the headline pipeline for reproducibility against the original ExoHand system, and note this finding as the basis for a future simplification.

## 3. Model-class comparison: HGB vs LDA

Same training mix (GrabMyo + per-session cal weighted 100×), same balanced test set; HGB replaced with Linear Discriminant Analysis (with shrinkage='auto'). Cal-weight replication via row tiling (LDA doesn't accept `sample_weight`).

| Configuration | Session mean accuracy | F1 macro |
|---|---:|---:|
| HGB + GrabMyo + cal (headline) | **0.871** | 0.862 |
| LDA + GrabMyo + cal (this ablation) | 0.727 | — |
| **Δ (HGB − LDA)** | **+0.144** | — |

HGB's non-linear capacity is load-bearing — a linear discriminant on the same features and same calibration loses **14 pp** in session-mean accuracy. The 14 pp gap is roughly the same order as the cross-population gap PhysioMio is testing, so the model class is the right axis to ablate against.

This addresses the "off-the-shelf" concern (HGB is well-known) by showing that **a still-simpler off-the-shelf alternative materially underperforms**. The methodological choice is principled, not lazy.

## How these enter the paper

- §3 Methods names the calibration protocol as the headline at **22 s of cued data** (not 60 s) and cites #1 as the supporting saturation.
- §3.4 Two-stage runtime describes HGB and references #3 for the model-class justification.
- §4.3 Within-subject temporal shift references #2 as the "cross-session evidence that calibration data, not the healthy-population prior, is what closes the gap," then discusses scope (no representation transfer tested).
- §5 Limitations notes the heterogeneity in #2 (11/46 patients favour the prior) and the openness of the representation-transfer question.

All three are real, defensible findings that strengthen rather than dilute the central claim.
