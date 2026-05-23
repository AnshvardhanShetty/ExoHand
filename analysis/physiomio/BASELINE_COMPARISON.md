# Baseline comparison — ablating the model and the prior

Two ablations on the per-session calibration protocol, both evaluated
**within session** (cal and test drawn from the same session, separated by
the per-gesture temporal split with a 3-window buffer):

1. **Model-class ablation** — HGB vs LDA, both with GrabMyo + cal.
2. **Prior ablation** — HGB with GrabMyo + cal vs HGB on cal data only.

> **Scope (read first).** Every number in this document is *within-session*:
> the test set comes from the same session the cal data came from. These
> ablations characterise what each component contributes to **same-session**
> accuracy. They say **nothing** about cross-session generalisation, which
> is what the deployed system actually faces and which is measured
> separately by the longitudinal eval (Stream 5). The cross-session
> patient-only run is in progress; until it lands, no claim about the
> GrabMyo prior's role in deployment is warranted.

## Configurations (PhysioMio, n = 48, 329 sessions)

All HGB configs: max_iter=300, max_depth=10, class_weight='balanced',
370 engineered features, balanced test set (39 rest + 39 close + 39 open).

| Method | Training data | Session mean acc | Per-class F1 (rest / close / open) | Fit time / session |
|---|---|---:|---:|---:|
| Zero-shot (no patient cal) | GrabMyo only | 0.214 | 0.008 / 0.042 / 0.318 | 0 s |
| **Patient-only HGB** | 432 cal windows, no GrabMyo | 0.870 | 0.970 / 0.853 / 0.753 | 3.1 s |
| HGB + GrabMyo + cal (main) | 1.14M GrabMyo (w=1) + cal (w=100×) | 0.871 | 0.982 / 0.842 / 0.762 | 206 s |
| LDA + GrabMyo + cal | Same training mix, LDA model | 0.727 | — | fast |

## What the ablations actually show (within-session)

**Model class matters.** LDA + GrabMyo + cal reaches only 0.727 — **14 pp
below** HGB on the same training data. HGB's non-linear capacity is
load-bearing for the same-session decision boundary; a linear discriminant
on the same 370 features and same calibration is materially worse.

**The GrabMyo prior adds ≤0.2 pp to same-session accuracy.** Patient-only
HGB (0.870) ties HGB + GrabMyo + cal (0.871) to within 0.2 pp, paired delta
+0.002; 25/48 patients favour the GrabMyo version, 23/48 favour patient-only
— a chance-level split. This held on Lucchetti too (patient-only 0.848 vs
GrabMyo+cal 0.828; the prior was marginally *negative* there).

**Interpretation — and its limit.** Within a single session, HGB with
class-balancing is sample-efficient enough to fit the 3-class problem from
~432 cal windows directly; the GrabMyo prior contributes little to
same-session accuracy. This is consistent with two non-exclusive readings:
(a) the GrabMyo prior is genuinely marginal, or (b) the GrabMyo prior's
value is cross-session regularisation that a same-session test cannot
detect. **These cannot be distinguished within session.** The
longitudinal-patient-only run (Stream 5 with the prior removed) is the
experiment that adjudicates: if patient-only degrades faster across
sessions than GrabMyo+cal, reading (b) holds and the prior earns its keep
in deployment; if it degrades the same, reading (a) holds.

## Cal-size ablation (patient-only HGB, 12-patient subset)

| Cal duration | Windows / gesture | Session mean acc | F1 macro |
|---:|---:|---:|---:|
| 7.2 s | 12 | 0.836 | 0.820 |
| 14.4 s | 24 | 0.843 | 0.823 |
| **21.6 s** | **36** | **0.868** | **0.853** |
| 36 s (default) | 60 | 0.869 | 0.854 |
| 54 s | 90 | 0.869 | 0.854 |
| 72 s | 120 | 0.869 | 0.854 |

Same-session accuracy **saturates at ~22 s of cued cal data**; the 60 s
default is conservative. Clinically useful: the calibration session could be
shortened to ~22-30 s without measurable same-session accuracy loss. (Again
within-session — whether a shorter cal degrades faster across sessions is
untested.)

## Comparison to ReactEMG's pretraining gain — stated carefully

Wang et al. (2026) report a +9 pp gain from healthy pretraining (stroke-only
0.69 → pretrained+LoRA 0.78). Our within-session prior ablation shows a
+0.1 pp gain from GrabMyo. **This is not a like-for-like comparison**:
ReactEMG's numbers are over 5 distribution-shifted held-out test sets
(including cross-session drift), whereas ours is same-session. A fair
comparison to their pretraining gain requires our cross-session
(longitudinal) patient-only numbers — pending. The plausible mechanism for
any divergence is that a 8M-parameter transformer needs pretraining to learn
"what EMG looks like" from raw signal, whereas our 370 engineered features
encode that structure by construction — but we should not assert the
magnitude until the cross-session comparison exists.

## What this does and does not change

- **Does NOT change the central claim.** Zero-shot 0.21 → calibrated 0.87 is
  a real cross-population result; the zero-shot baseline is GrabMyo applied to
  PhysioMio with no patient data, which is the cross-population test. Patient-side
  calibration closing that gap is the claim, and it stands regardless of whether
  the GrabMyo prior is jointly used.
- **Adds two methods-section ablation notes.** (1) HGB beats LDA by 14 pp —
  model class is load-bearing. (2) The GrabMyo prior adds ≤0.2 pp same-session;
  its cross-session role is under test.
- **Adds a deployment-protocol result.** Cal can be ~22 s, not 60 s.

## Open item

Cross-session patient-only longitudinal (Stream 5 with `--patient-only`,
running now). Resolves whether the GrabMyo prior is decoration or a
cross-session regulariser. Until then, the methods section should report
the within-session ablation factually and defer the prior's deployment
role to that result.
