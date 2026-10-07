# ExoHand — Pre-registration for ICBINB-BIO submission

**Date committed:** 2026-08-21
**Deadline:** 2026-08-29 (ICBINB-BIO @ NeurIPS 2026)
**Purpose:** Fix decision rules for each planned experiment BEFORE its results are read, so that outcomes cannot be re-interpreted post-hoc.

---

## Central claim under test

In stroke EMG, **pathology — not subject identity, anatomy, or data volume — is the axis that governs transfer**: a patient's own healthy arm is a worse training source than *other patients'* impaired arms, and 1.14M windows of healthy-population EMG add nothing to 22 s of impaired-arm calibration.

---

## Pre-registered experiments

### Kill-or-confirm controls

**C1 — Mirror / channel correspondence.** Cross-arm PO re-evaluated over all 4! = 24 channel permutations of the target patient's chosen channels + a medial–lateral reflection variant.

- **Decision rule:** if the best-performing permutation recovers >⅓ of the 33 pp gap (i.e., cross-arm accuracy ≥ 0.66 under best permutation), the gap is partly a channel-mounting artefact. **Report the corrected number as the headline** and revise the mechanism claim accordingly. If <⅓ recovered, mounting is not the confound.

**C2 — Per-limb normalisation.** Re-run cross-arm PO with z-scoring fit per limb (not pooled per participant); also amplitude-equalised variant (rescale healthy-arm windows to match impaired-arm amplitude statistics).

- **Decision rule:** if the gap closes below 20 pp under per-limb standardisation or amplitude-equalisation, the story becomes "scale/SNR mismatch" rather than "pathology". **Report both numbers regardless**; keep whichever framing survives.

**C3 — Volume-matched LOPO** (already running as of 2026-08-21).

- **Decision rule:** if volume-matched LOPO > cross-arm PO with paired Wilcoxon p < 0.05, the "pathology-matched > anatomy-matched at matched volume" claim survives. If ≈ (|Δ| < 5 pp AND p ≥ 0.05), demote to "own healthy arm doesn't transfer well" — the "beats LOPO" phrase is removed.

**C4 — GrabMyo weight sweep.** {0×, 1×, 10×, 100× (current), 1000×} cal weight in the joint HGB training, full accuracy ladder with bootstrap CIs on 48 patients.

- **Decision rule:** if any weight ≠ 100× beats cal-only by > 1 pp with paired Wilcoxon p < 0.05, the null-result headline dies. Paper becomes *"pretraining helps only under weighting X"* — still submittable, different abstract. If no weight beats cal-only, null-result claim survives across the full ladder.

**C5 — Budget × weight interaction.** Extend the 3/6/12/24/36-trial cal-size sweep across the C4 weights.

- **Decision rule:** report the full grid; the framing claim becomes *"pretraining helps only below X seconds of calibration, at weight Y"* — quantified rather than binary.

### Mechanism experiments

**M1 — Within-patient vs across-patient distributional distance.** Compute Wasserstein-1(own-healthy, own-impaired) vs Wasserstein-1(own-impaired, mean-over-others'-impaired), per patient, per feature, averaged appropriately.

- **Decision rule:** if within-patient cross-limb W₁ exceeds across-patient within-pathology W₁ with paired Wilcoxon p < 0.05 AND Cliff's δ > 0.2, the geometric explanation for the cross-arm result holds. If not, the mechanism claim collapses to observational only.

**M2 — Distance predicts accuracy drop.** Spearman correlation between per-patient M1 distance and per-patient cross-arm accuracy drop.

- **Decision rule:** if ρ > 0.3 with p < 0.05, converts observation to mechanism (accuracy loss is predicted by feature-space distance). If not, present M1 alone without the mechanism-of-accuracy claim.

**M3 — Feature-family shift ranking.** Rank the 60 base × 4 channel features by W₁ shift, group by family (amplitude, envelope, spectral, waveform-crossing).

- **Decision rule:** report as-is; feeds Section 6 and connects to the K=30 deployment finding.

### Replication

**R1 — Lucchetti replication.** Replicate rows 1, 3, 4 of the training-source ladder on Lucchetti (n=10 stroke). Cross-arm same-patient row cannot be replicated (Lucchetti's healthy cohort is separate subjects, not same-patient healthy arms).

- **Decision rule:** if all three replicated rows match the PhysioMio ranking (per-session cal >> VM-LOPO >> zero-shot) with the same qualitative pattern, replication is claimed. If the ordering differs on Lucchetti, replication is conditional and we say so.

---

## Statistics protocol (applies to every headline number)

- **Paired Wilcoxon signed-rank test + Cliff's δ** for every within-patient comparison.
- **Holm–Bonferroni correction** across the ladder's pairwise comparisons.
- **Bootstrap 95% CIs** on every headline number, patient-level resampling (n_resamples = 2000).
- **% of patients showing the effect** reported alongside every mean/median.
- **Per-patient paired lines** shown in every ladder figure.

---

## Explicit out-of-scope items (declared, not attempted)

- Deep learning with learned representations.
- Alternative pretraining corpora (Ninapro, HYSER, hypothetical larger).
- Live closed-loop testing on stroke patients (n=1 healthy adult only for demo video).
- Fine-grained 16-gesture classification (paper uses 3-class collapse).
- FMA-UE level 1-2 patients (not present in either cohort).

---

## Currently running as of commit time

- Volume-matched LOPO (C3) — pid 63260, launched 2026-08-21 ~06:57 EDT.
- Stacked pathology+anatomy — pid 63262, launched 2026-08-21 ~06:57 EDT.

---

## What this pre-registration prevents

- Interpreting a mixed C1/C2 result as "not a confound" post-hoc.
- Re-framing null-result death (C4) as "we always knew that was optimal" post-hoc.
- Cherry-picking which mechanism (M1 vs M2 vs geometry) survives after seeing results.
- Retrofitting the abstract to whichever result landed most cleanly.
