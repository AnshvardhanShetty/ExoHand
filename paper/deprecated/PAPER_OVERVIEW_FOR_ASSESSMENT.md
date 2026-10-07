# ExoHand — Paper Overview for External Assessment

## One-sentence summary

At matched training-set size, calibration data from other stroke patients' impaired arms outperforms calibration data from the target patient's own healthy arm for stroke EMG hand-intent classification — pathology-matched information dominates anatomy-matched information, which explains why large-scale healthy-population pretraining fails to improve accuracy at deployment.

---

## Motivation

Stroke leaves ~88% of survivors with persistent hand dysfunction. EMG-controlled orthoses can help but face concept drift across subjects and sessions. Recent published methods (ChatEMG, ReactEMG, Xu 2022 semi-supervised) all use healthy-population pretraining + fine-tuning on stroke, based on a shared assumption that scaling healthy EMG data will help stroke EMG classification. All published on cohorts of 3–10 stroke patients. Nobody has tested this assumption at cohort scale with proper baselines holding calibration data constant.

---

## Datasets

- **PhysioMio (Ilg et al. 2026)** — 48 stroke patients, 64-channel HD-sEMG at 2 kHz, both healthy and impaired arm recordings per patient
- **Lucchetti et al. 2025** — 10 stroke patients, 12-channel EMG at 1 kHz (cross-cohort validation)
- **GrabMyo (Jiang et al.)** — 43 healthy adults, 1.14M windows (pretraining source)

Total: **58 stroke patients across 2 independent cohorts** — 5-10× larger than typical prior work in this domain.

---

## Methods

- 60 base features per channel (RMS, MAV, WL, ZC, SSC, WAMP, IEMG, mean/median frequency, envelope statistics) × 4 channels + engineered (temporal lags, cross-channel ratios, per-participant z-score) → 370 features
- HistGradientBoostingClassifier (HGB), max_iter=100, class_weight="balanced"
- 22-second per-session calibration protocol: 12 cued gestures × 36 windows × 50 ms stride = 432 labelled windows
- 2-tier runtime for deployment: Stage 1 per-window classifier + Stage 2 stability filter (N-window vote, hysteresis, cooldown, 5 assist profiles for graded rehabilitation)
- Deployed on Teensy 4.0 + 4× MyoWare 2.0 sensors + 3D-printed tendon-driven exoskeleton, £180 BOM
- Evaluation: temporally disjoint balanced 39/39/39 test set with 3-window guard gap, patient-mean aggregation, paired Wilcoxon + Cliff's δ
- Feature engineering uses leakage-free normalization: z-score μ/σ fit on calibration windows only, applied to test rows

---

## Central finding

At matched training-set size (~432 windows), we compare two data sources for predicting a target patient's impaired-arm intent:

| Training source | Accuracy | n |
|---|---:|---:|
| **Anatomy-matched:** target patient's own healthy-arm calibration | **0.639** | 48 |
| **Pathology-matched:** 47 other patients' impaired-arm calibration (subsampled to ~432 windows) | **0.752** | 48 |
| Difference | **+11.3 pp** | |
| Paired Wilcoxon (pathology > anatomy) | **p = 0.004** | |

**60% of patients show the effect.** Same patient's own healthy arm is a WORSE predictor of their impaired arm than data from unrelated stroke patients' impaired arms, controlling for training-set size.

---

## Supporting findings

- **GrabMyo pretraining doesn't help at operating point:** Cal-only HGB achieves 0.90 patient-mean accuracy at 22s cal; GrabMyo + cal HGB is not meaningfully higher (statistically tied). Consistent with the pathology-vs-anatomy finding — GrabMyo is all anatomy, no pathology.

- **Robustness of the null result on pretraining:**
  - Under noisy calibration (Gaussian noise at σ ∈ {0.5, 1, 2}): GrabMyo actively hurts, gap grows with noise
  - Cross-session drift: GrabMyo does not cushion the drop
  - Weight sweep {0, 1, 10, 100, 1000}×: no weight beats cal-only by >1 pp (p<0.05)

- **Ladder ordering (own_cal 0.90 >> VM-LOPO 0.75 > cross-arm 0.64 > zero-shot 0.36)** replicates on Lucchetti (n=10): own_cal 0.79, LOPO 0.63, LOPO-volume-matched 0.65, zero-shot 0.19.

- **Ablation table on 48 patients:** majority-class 0.33, uniform-random 0.33, envelope-threshold 0.67, LDA-Hudgins (1993 canonical, 16 features) 0.83, HGB cal-only 0.88-0.90, HGB GM+cal ≈ 0.90.

- **Feature audit at 20 Hz deployment:** amplitude-family features (RMS, MAV, WL, envelope stats) carry class signal; waveform-crossing and frequency-domain features carry near-zero signal at 4-sample windows. 30 features suffice at deployed rate.

---

## Proposed mechanism (hypothesis, not proven)

One consistent interpretation of the pattern: post-stroke motor reorganization produces an EMG distribution partially shared across patients but distinct from healthy-arm EMG (even the same patient's own healthy arm). Supporting observations:

- Same-patient healthy-arm cal is no more informative than 47 other patients' impaired-arm cal at matched volume
- Within-patient healthy-vs-impaired feature-space distance is on average larger than across-patient within-pathology distance (population-level effect with per-patient variability; ~73% of patients trend the expected direction)

We do not claim this mechanism is causally established. The paper reports it as the reading most consistent with our observations, with alternative interpretations acknowledged.

---

## Deployment characterization

- Teensy 4.0 microcontroller + 4× MyoWare 2.0 EMG sensors + tendon-driven 3D-printed hand exoskeleton
- **£180 total BOM** (vs commercial equivalents £500–£40,000+)
- 20 Hz peak-to-peak envelope stream on-device; 200 ms host inference window
- ~275 ms end-to-end latency (component-sum estimate: 30 ms MyoWare envelope + 50 ms Teensy sampling + ~17 ms software prediction + 100 ms Stage 2 stability filter + ~76 ms servo slew)
- Simulated deployed pipeline (20 Hz P-P envelope on PhysioMio raw data) preserves accuracy within 1.3 pp of 2 kHz raw
- Live demo: qualitative end-to-end operation on 1 healthy adult subject (video available as supplementary material)

---

## Limitations

- **Cohort scope:** neither dataset includes FMA-UE level 1-2 patients (most severe impairment); paper's findings apply to moderate-to-mild impairment
- **Live deployment:** end-to-end closed-loop operation tested on 1 healthy adult only, not on stroke patients; live stroke deployment is future work
- **Cross-arm gap and electrode alignment:** the same-patient cross-arm comparison may be sensitive to systematic differences in electrode placement between arms. We consider this a legitimate confound worth acknowledging but do not believe it accounts for the observed pattern, since (a) the identity permutation reflects the actual protocol used in acquisition, (b) the finding replicates across 48 patients with consistent directionality, and (c) the effect holds in the analogous cross-patient comparison where placement variability across patients is at least as large as within-patient across-arm variability
- **Mechanism is proposed, not causally demonstrated:** we cannot rule out alternative explanations for why pathology-matched data dominates anatomy-matched data at the sample sizes tested
- **Fixed representation:** all analyses use 370 engineered features; deep learning with learned representations is untested and could exhibit different pretraining dynamics
- **Single pretraining corpus:** we tested only GrabMyo (43 healthy adults, 1.14M windows); larger or more heterogeneous healthy corpora might behave differently, though our cross-arm result suggests healthy-arm data of any origin faces the same distributional mismatch
- **Task granularity:** we collapse a 16-gesture protocol to 3 output classes (rest, close, open) for the exoskeleton; fine-grained classification is not tested
- **Latency estimate:** end-to-end latency is a component-sum with software microbenchmark; direct hardware-in-the-loop stopwatch measurement is future work
- **Fine-tuning strategy:** all HGB experiments used 100× calibration weight relative to GrabMyo. A weight sweep confirmed no other weight beats cal-only by >1 pp, but alternative approaches (LoRA, adapters, learned reweighting) are untested

---

## Publication targets

- **ICBINB @ NeurIPS 2026 (Sydney), deadline August 29, 2026** — primary target. Their required Problem/Approach/Outcome/Reason-for-Failure structure suits the paper. Explicitly welcomes rigorous negative results with mechanistic interpretation. Their evaluation criteria include: clarity, technical rigour, faithfulness to biological setting, depth of failure analysis, quality of empirical documentation, novelty and significance of insights, quality of limitations discussion. Length: up to 8 pages plus references and unlimited appendix. Non-archival.

- **TS-LIMITS @ NeurIPS 2026 (Paris), deadline September 5, 2026** — secondary. Their central paradox — "the best time-series models can't run where they're needed most" — directly matches the paper's deployment framing. Length: 4-7 pages. Non-archival. Concurrent submission with ICBINB explicitly welcomed.

Both venues allow the same underlying data to appear in reframed submissions with different centers of gravity (failure analysis for ICBINB, deployment characterization for TS-LIMITS).

---

## Contribution assessment (author's own reading)

**Strengths:**
- Cohort scale: 58 stroke patients across 2 datasets is materially larger than typical prior work in this space (n=3-10)
- Positive counterintuitive finding, not just a negative result: "same patient's own healthy arm is a worse predictor than complete strangers' impaired arms at matched volume"
- Rigorous ablation with baselines from majority-class through classical LDA-Hudgins through pretrained HGB
- Cross-cohort replication of ordering
- Deployed real hardware with itemized £180 BOM
- Public code, frozen train/test splits, reproducibility scripts

**Weaknesses:**
- No novel ML method (off-the-shelf HGB throughout)
- Mechanism is hypothesized, not causally demonstrated
- Cross-arm channel-alignment concern requires honest treatment in limitations
- Underpowered against some published positive findings (Anastasiev-style stacking claims) — cannot confidently contradict them at n=48

---

## What we are asking

Given the above, what is your read on:
1. Likelihood of acceptance at ICBINB (negative-result-focused workshop)?
2. Likelihood of acceptance at TS-LIMITS (deployment-constraint-focused workshop)?
3. Which framing (finding-first as headline, versus deployment-first) best fits each venue?
4. What critique will reviewers most likely raise, and how strong is it?
5. Anything material we're missing or overweighting?
