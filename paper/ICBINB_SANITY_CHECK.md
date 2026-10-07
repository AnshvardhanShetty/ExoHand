# Sanity check — ICBINB @ NeurIPS 2026 workshop submission

I'm preparing a paper for the **ICBINB (I Can't Believe It's Not Better) workshop at NeurIPS 2026** (Sydney, deadline Aug 29). Please give me an honest, independent acceptance-probability estimate. I've deliberately not shared my own estimate — form your own first.

## Venue framing
ICBINB rewards negative/surprising results in ML that are documented rigorously and paired with a mechanism explanation, not just "it didn't work."

---

## The failure

**A classifier pretrained on 1.14M healthy EMG windows from 43 subjects achieves 0.360 mean accuracy on a 3-class stroke intent task (n=48 patients) — no detectable transfer beyond 3-class chance (0.333).**

The 22-second per-session on-body calibration baseline that this pretraining was supposed to augment achieves 0.896 on the same task. **1.14M windows of pretraining leaves 54 percentage points on the table versus the trivially available baseline.**

Every layered variant we tested — stacked healthy+cal, light-GM contribution, exhaustive GrabMyo-weight sweep — failed to improve on the plain calibration baseline. The failure is regime-invariant.

### The negative is statistical, not anecdotal

Chance for 3 classes = 0.333. Zero-shot mean = 0.360.

| Test | Result | Rejects chance? |
|---|---:|:---:|
| Bootstrap 95% CI on mean | [0.312, 0.408] | NO — contains 0.333 |
| Wilcoxon signed-rank (H1: median > 1/3) | p = 0.082 | NO |
| One-sample t-test (H1: mean > 1/3) | p = 0.134 | NO |
| Sign test above chance | 28/48 (58%), binomial p = 0.156 | NO |
| Cohen's d vs chance | 0.162 (small) | — |

No test detects transfer above chance. The 95% CI is compatible with at most modest transfer (upper bound +7.5 pp above chance) — roughly one-seventh of the +56.3 pp gap that would be needed to reach the 22-second calibration baseline (0.896).

### Why this is a surprise

GrabMyo was published explicitly as an EMG pretraining substrate. The received transfer-learning intuition — pretrain on the largest related distribution available — predicts this should work. The finding is that even at 1.14M windows and 43 subjects, healthy pretraining provides no detectable benefit for stroke intent classification.

For a stroke patient using the deployed classifier, 0.360 accuracy means intent decisions are indistinguishable from random and no assistive control loop can close. 0.896 with 22 seconds of calibration is a working device. The gap is deployment-defining.

---

## Given the failure: what does transfer?

Having ruled out healthy-population scale as a route to transfer, we asked the follow-up question — **among small-corpus alternatives to per-session calibration, does anything actually transfer to stroke intent classification?**

We evaluated four training regimes at matched training volume (432 windows per patient, one calibration session), all evaluated on the same held-out impaired-arm test set. Multi-draw averaging (5 independent 47-donor subsamples per patient) for the cross-patient regimes to eliminate single-draw sampling artifacts. Same classifier throughout (HistGradientBoostingClassifier, max_iter=100, class_weight="balanced").

**Full cohort (n=48):**

| Training source (432 windows) | Accuracy | vs. own-healthy baseline |
|---|---:|---:|
| Own healthy arm (baseline) | 0.639 | — |
| 47 others' healthy arms (diversity axis) | 0.719 | +8.1 pp, p=0.049, δ=+0.13 |
| 47 others' impaired arms (pathology axis) | 0.742 | +10.3 pp, p=0.005, δ=+0.25 |
| Own impaired arm (ceiling) | 0.896 | +25.7 pp |

**Answer: cross-patient stroke data transfers. Same-patient healthy data transfers less.**

432 windows from 47 other patients' impaired arms outperforms 432 windows from the patient's own healthy arm by **+10.3 pp** — despite the cross-patient donors being worse anatomy-matches, sharing no subject with the target, and coming from a corpus four orders of magnitude smaller than the healthy pretraining corpus that failed.

### What carries the signal — pathology, not just diversity

Holding donor pool constant at 47 people, switching from healthy arms to impaired arms adds **+2.2 pp (p=0.021, Cliff's δ=+0.375)**. Pathology-matching contributes beyond simply having more donors.

**Per-patient breakdown (n=48):**

|  | Diversity helps | Diversity does not help | Total |
|---|---:|---:|---:|
| Pathology helps | 16 | **17** | 33 (69%) |
| Pathology does not help | 11 | 4 | 15 |

**Of the 21 patients where donor diversity fails to help, pathology-matching rescues 17 of them, mean +6.1 pp.** Diversity and pathology contribute complementarily; pathology specifically catches patients that diversity leaves behind.

### The pathology contribution grows with days post-stroke

If pathology-matched transfer is what's carrying the signal, the effect should strengthen as the paretic arm has more time to diverge from healthy anatomy.

| Cutoff | n | Pathology Δ | p |
|---:|---:|---:|---:|
| ≥7 d | 48 | +2.2 pp | 0.021 |
| ≥14 d | 45 | +2.5 pp | 0.016 |
| ≥21 d | 37 | +2.8 pp | 0.012 |
| ≥30 d | 25 | +4.8 pp | 0.004 |
| ≥45 d | 16 | +4.1 pp | 0.042 |
| ≥60 d | 12 | +5.3 pp | 0.026 |
| ≥90 d | 3 | +9.5 pp | n=3, underpowered |

Monotonically increasing. In the ≥30-day subset alone (n=25): pathology contribution amplifies to **+4.8 pp, 95% bootstrap CI [+1.7, +7.9], p=0.004, Cliff's δ=+0.60**.

### Mechanism support (three orthogonal probes)

- **Wasserstein-1 distances**: impaired arms are distributionally closer to other patients' impaired arms than to their own healthy arm — full 48-patient cohort
- **Channel-permutation ablation**: shuffling channel labels erases the majority of the cross-patient pathology transfer benefit — the transferable signal lives in specific channel geometry, not global signal statistics
- **Feature importance × distribution-shift correlation**: features that shift most across the healthy/impaired boundary receive highest classifier importance in cross-patient models

---

## What the failure tells us, once we know what does work

Two findings, in sequence:

1. Massive healthy-population pretraining fails to transfer to stroke intent classification, at any weight or layering variant
2. A small corpus of cross-patient stroke data (47 donors, 432 windows each) transfers meaningfully — outperforming the patient's own anatomically-matched arm by +10.3 pp

Read together: **the transferable substrate for stroke intent classification isn't healthy anatomy at scale — it's pathology structure, small-corpus or otherwise.** The received transfer-learning intuition ("pretrain on the biggest related distribution") is optimising the wrong axis. Scale on healthy anatomy does not substitute for pathology-matched data at any volume we could test.

The implication for the field: **for pathology-specific embedded applications, curate small pathology-matched cross-patient corpora rather than chase healthy-population scale.** A dataset four orders of magnitude smaller, matched on pathology, transfers where 1.14M healthy windows do not.

---

## Deployment context

Deployed on custom embedded hardware: Teensy 4.0 + 4× MyoWare 2.0 EMG sensors + tendon-driven 3D-printed exoskeleton, £180 BOM. Firmware streams 20 Hz peak-to-peak envelope over serial; host runs HGB + 2-tier stability filter. End-to-end latency ~275 ms (component-sum). All reported numbers are reachable at deploy time via a Python mirror of the firmware pipeline (`deployed_pipeline_sim.py`).

---

## Methodological rigor

- Leakage-free features: z-score μ/σ computed from calibration rows only, per participant
- Frozen splits: (cal_idx, test_idx) fixed per patient across every analysis
- Pre-registered decision rules in `PREREGISTRATION.md`
- Multi-draw stability for all cross-patient regimes (5 subsample draws per cell)
- Independent replication on the Lucchetti stroke dataset (n=10, 12-channel @ 1 kHz, different acquisition rig)
- HistGradientBoostingClassifier, max_iter=100, class_weight="balanced" — one configuration throughout, no per-cell tuning

## Sample sizes
- PhysioMio (headline): 48 stroke patients, 64-channel HD-sEMG @ 2 kHz
- ≥30-day subset: 25 patients (days post-stroke ≥ 30)
- Lucchetti (replication): 10 stroke patients, 12-channel @ 1 kHz
- GrabMyo (pretraining source): 43 healthy subjects, 1.14M windows

## Known limitations
1. 3-class intent task (rest / hand-close / hand-open) matches the 3-DoF exoskeleton controller but is coarser than full finger-level EMG classification
2. Cross-arm same-patient comparison carries an electrode-placement confound; the cross-patient comparison across 48 patients is the primary mitigation
3. ≥30-day subset (n=25) is modest for subgroup claims; the dose-response monotonicity is the intended check. (Naming note: this subset is not "chronic" under Bernhardt et al. 2017 — under that taxonomy chronic is > 6 months post-stroke and only 1 of our 48 patients qualifies; the subset is predominantly early-subacute. We report it by day cutoff rather than by phase label to avoid mislabelling.)

## Paper structure (5 main figures/tables)
- **F1**: hardware photo + electrode placement diagram
- **F2**: 2×2 decomposition — per-patient scatter with categorical rescue breakdown
- **F3**: dose-response pathology gap vs days-post-stroke
- **T1**: full-cohort regime ladder (5 rows: zero-shot, cross-arm, Exp 1, VM-LOPO, own-cal)
- **T2**: per-patient categorical helps/hurts breakdown

Appendix: pre-registration, ablations (light-GM, stacked P+A, GrabMyo weight sweep, per-limb normalization), Lucchetti replication, hardware specifications.

---

## Please assess

1. **Acceptance probability** at ICBINB — give a specific range with reasoning
2. **The 3 biggest strengths** a reviewer would notice
3. **The 3 biggest weaknesses / likely rejection reasons**
4. **Concrete edits that would meaningfully raise acceptance chances** in <1 week of work
5. **Is the "ICBINB fit" tight or loose** — does this land as a real negative-with-mechanism, or does it feel like a positive-result paper wearing negative clothing?
6. **Any obvious methodological gap I might have missed** — specifically, is there anything a rigor-focused reviewer will jump on?
