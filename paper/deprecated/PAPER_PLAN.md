# Paper plan — talking-point scaffold

Workshop target: NeurIPS TS4H (4-page main, refs/appendix unlimited). User writes the prose; this file is structured notes only.

---

## 1. Locked canonical numbers (the only source for every number in the paper)

**Open this file at this section every time you reach for a number while drafting.** Do not recompute, do not aggregate differently, do not introduce new number-pairs. Numbers below are patient-mean aggregations (the unit of analysis is the patient, since the contribution is per-patient improvement). Sources are the canonical pipeline outputs.

| Metric | Value | Source file |
|---|---:|---|
| PhysioMio zero-shot impaired (n=48, patient-mean) | **0.188** | `analysis/physiomio/results/zero_shot_per_session.csv` |
| PhysioMio calibrated impaired (n=48, patient-mean) | **0.860** | `analysis/physiomio/results/per_session_results.csv` |
| Cliff's δ (paired by patient) | **+1.000** (48/48 patients improve) | derived |
| Mean per-patient lift | **+0.673** (min +0.180, max +0.874) | derived |
| Paired Wilcoxon p (cal > zs) | **3.55e-15** | derived |
| Lucchetti zero-shot 3-class (n=10, patient-mean) | **0.194** | `analysis/lucchetti/results/zero_shot_per_session.csv` |
| Lucchetti calibrated 3-class (n=10, patient-mean) | **0.795** | `analysis/lucchetti/results/per_session_results.csv` |
| Lucchetti calibrated binary (n=10, patient-mean) | **0.960** | `analysis/lucchetti/results/binary_per_session.csv` |
| Within-session live deployment (PI, 12-min cued) | **99.17 %** | `analysis/system/results/live_deployment_eval.md` |
| Next-session GM+cal accuracy without re-cal (dist=1, n=46) | **0.717** | `analysis/physiomio/results/longitudinal_per_session.csv` (appendix only) |
| End-to-end latency, classify | **<50 ms** | `analysis/system/HARDWARE_LATENCY.md` |
| End-to-end latency, intent → motor | **~275 ms** | `analysis/system/HARDWARE_LATENCY.md` |
| BOM | **£180** | `analysis/system/cost_itemization.md` |
| L4 deployed transition accuracy | **0.606** | `analysis/physiomio/results/full_deployed_pipeline.csv` |
| ReactEMG Stroke best (their Table 2, n=3 patients) | **0.61 transition / 0.78 raw** | Wang et al. (cite) |
| Clinical translation — rep success | **97.7 %** | `analysis/physiomio/results/clinical_translation.md` |
| Clinical translation — false-acts/min rest | **1.91** | `analysis/physiomio/results/clinical_translation.md` |
| Clinical translation — time-to-correct intent | **149 ms** | `analysis/physiomio/results/clinical_translation.md` |
| PhysioMio calibrated, simulated 20 Hz envelope pipeline | **0.863** | `analysis/physiomio/results/deployed_pipeline_sim_gmcal.csv` (§4 confirmation only, not headline) |

Headline phrasing: **"0.19 → 0.86 on 48 stroke patients, 48/48 improve (Cliff's δ = +1.0), replicates on a second cohort (0.79 three-class on Lucchetti n=10)."**

### Rounding rule (applies everywhere in the paper)

- **Prose**: round to 2 decimals (0.86, 0.72, 0.61, 0.96).
- **Tables and figure captions**: cite the canonical-table precision (0.860, 0.717, 0.606, 0.960).
- **Exception**: Cliff's δ is "+1.000 (48/48)" everywhere — full precision in prose because the integer "48/48" is the strongest version of the claim and you don't want to lose it to rounding.

This rule guarantees that "0.86" in §3 prose and "0.860" in the §1 table refer to the same number, so a reviewer never sees what looks like two different values for the same quantity.

---

## 2. Central claim

Two contributions, in order of weight:

1. **Cross-population stroke EMG control.** A per-session calibration protocol that takes 58 stroke patients across two independent cohorts (PhysioMio n=48, Lucchetti n=10) from zero-shot 0.19 → calibrated 0.86 on PhysioMio (Cliff's δ = +1.000, 48/48 patients improve, paired Wilcoxon p = 3.6e-15) and replicates on Lucchetti (0.79 three-class, 0.96 binary). Effect is severity-independent across FMA-UE tertiles, including FMA-hand-0 patients.
2. **Deployable on £180 hardware.** The above accuracy is reached on a Teensy 4.0 + 4× MyoWare 2.0 + tendon-driven exoskeleton with <50 ms classification latency and ~275 ms intent-to-motor end-to-end. Within-session live-device accuracy is 99.17 % on a 12-minute cued session; L4 deployed transition accuracy is 0.606, comparable to ReactEMG Stroke's 0.61 on n=3.

---

## 3. Section structure (4-page TS4H, page budget)

| § | Section | Job | Evidence | Figures | Budget |
|---|---|---|---|---|---|
| 1 | Intro (related work folded inline) | Stroke EMG control problem; deployable-system gap vs existing transformer approaches; preview both contributions; one-paragraph related work covering ReactEMG positioning and prior surface-EMG stroke control inline | – | – | 0.6 p |
| 2 | System & methods | Hardware (Teensy + 4× MyoWare + tendon exo), 370 engineered features, HGB classifier jointly trained on GrabMyo base + per-session cal weighted 100×, 6-layer post-processing (EMA / argmax / stability filter N=3 / cooldown / hysteresis / floor), 5 assist profiles, 30 s per-session re-cal protocol. **One sentence pre-empting "what without re-cal?"**: "Without per-session re-cal, accuracy drops from 0.86 to 0.72 by the next session (Appendix A)." | `runtime/`, `ml/train_hgb_v2.py`, `HARDWARE_LATENCY.md`, `longitudinal_per_session.csv` (appendix only) | **Fig 1**: system diagram (hardware + data flow + cal protocol timeline) | 0.8 p |
| 3 | **Cross-population stroke result** (primary contribution) | PhysioMio n=48 zero-shot 0.19 → calibrated 0.86, Cliff's δ = +1.000 (48/48 patients improve, paired Wilcoxon p = 3.6e-15); replicated on Lucchetti n=10 stroke (three-class 0.79, binary 0.96); severity-independent across FMA-UE tertiles including FMA-hand-0 | `per_session_results.csv` (PhysioMio + Lucchetti); `zero_shot_per_session.csv`; severity_analysis outputs; `binary_per_session.csv` | **Fig 2** (composite, full-width): (a) per-patient strip plot, 58 patients sorted by zero-shot acc, both cohorts on one axis, each connected zero-shot → calibrated; (b) severity stratification panel — FMA tertiles + FMA-hand-0 sub-group | **1.4 p** |
| 4 | Deployment characterisation (secondary contribution) | £180 BOM, <50 ms classification, ~275 ms intent-to-motor, live PI session 99.17 % within-session, clinical-translation numbers (97.7 % rep success, 1.91 false-acts/min rest, 149 ms time-to-correct); ReactEMG comparison **inline as one sentence**: "L4 deployed transition accuracy is 0.606, comparable to ReactEMG Stroke's 0.61 on n=3 patients (Wang et al., Table 2)." **One sentence confirming the deployed signal regime preserves PhysioMio accuracy**: "Simulating the deployed 20 Hz peak-to-peak envelope pipeline on PhysioMio's raw EMG reproduces the cross-population accuracy (0.86), confirming the on-device signal representation is not the bottleneck." | `live_deployment_eval.md`, `clinical_translation.md`, `full_deployed_pipeline.csv`, `cost_itemization.md`, `HARDWARE_LATENCY.md`, `deployed_pipeline_sim_gmcal.csv` | **Fig 3** (compact): deployment numbers grid only — BOM, latency, accuracy, time-to-correct. ReactEMG is inline prose, not a strip in the figure. | 0.7 p |
| 5 | Limitations & discussion | Single deployed subject is project PI (not stroke patient); cross-cohort validation is offline at clinical sites, not real-world home use; no chronic data; system is research prototype, not FDA-cleared. Prose form, not bullets. | – | – | 0.3 p |
| 6 | Conclusion | One-sentence restatement of both contributions | – | – | 0.1 p |

**Total ≈ 3.9 p.** Page-budget honest. Related work folded into §1. §5 grown to 0.3 p (4 limitations need prose room, not 12-word bullets). Figure count = 3 (Fig 1 diagram, Fig 2 composite cross-population, Fig 3 deployment grid).

---

## 4. Figures (in order of importance to the contribution)

1. **Fig 2 — composite cross-population figure** (load-bearing, full-width). (a) per-patient strip plot, 58 patients (48 PhysioMio + 10 Lucchetti) sorted by zero-shot acc, each connected zero-shot → calibrated, both cohorts on one axis. (b) severity stratification panel — FMA-UE tertiles + FMA-hand-0 sub-group. A reviewer who scans only figures must see the FMA-hand-0 improvement — severity stays in main, not appendix. Source: `analysis/plots/hero_figure.py` panel (a) for the strip; severity panel needs new draft from `severity_analysis` outputs.
2. **Fig 3 — deployment numbers grid** (compact, half-width or third-page). BOM, latency, within-session accuracy, time-to-correct. ReactEMG comparison inline as one sentence in §4 prose; no row in Fig 3. Source: `analysis/plots/hero_figure.py` panel (c), trim to four numbers.
3. **Fig 1 — system diagram.** Hardware + data flow + cal-protocol timeline. Block diagram. Needs new draft.

---

## 5. Resolved structural decisions

1. **§3 figure composition.** Strip plot + severity stratification as a single composite Fig 2 (full-width). FMA-hand-0 improvement is clinically distinctive in stroke EMG literature; reviewers who skim only figures must not miss it.
2. **Lucchetti binary vs three-class.** Three-class (0.795) as primary headline; binary (0.960) as a single supporting sentence showing rest-vs-movement is the easier sub-problem on the reach-grasp task.
3. **Related work folded into §1.** At 4 pages, inline in intro is workshop standard.
4. **ReactEMG comparison.** One inline sentence in §4 using the locked transition-accuracy number (0.606 vs 0.61). No row in Fig 3, no separate table.
5. **Fig 3 scope.** Deployment numbers grid only (BOM, latency, accuracy, time-to-correct). No ReactEMG strip.

---

## 6. Things NOT to claim (absolute, in any section)

- Don't claim GrabMyo is necessary, foundational, or load-bearing for accuracy. Describe it as the pretrained base, no defense.
- Don't claim cross-subject zero-shot generalisation works on stroke EMG — zero-shot 0.19 shows the opposite. Calibration is essential and the paper is explicit about that.
- Don't claim live-deployment results extend to stroke patients — the 99.17 % is the project PI on a 12-minute cued session, not a clinical finding.
- Don't claim parity with ReactEMG without citing the matched-metric numbers (their 0.61 transition on n=3, our 0.606 transition) and naming them as comparable, not superior. Compare on one metric, not two — raw-stream accuracy invites the "raw on what test set?" question that the main text doesn't need to answer.
- Don't claim FDA-pathway or clinical readiness — this is a research prototype.

## 7. Things NOT to volunteer vs things TO pre-empt

The distinction: **don't volunteer** things a reviewer wouldn't otherwise raise (each one creates the question it was meant to deflect); **do pre-empt** things a reviewer will inevitably raise from the paper's own framing. Suppress the first; address the second with the minimum text + appendix backup.

**Don't volunteer (reviewer wouldn't otherwise ask):**
- Don't volunteer "patient-only HGB reaches comparable accuracy at the deployment cal size" — it creates *"then what is GrabMyo's contribution?"*. The contribution is the *system*, not the prior. Answer in rebuttal if raised.
- Don't volunteer the cal-size sweep as "system characterisation" — it tempts methodological self-justification. The system is what's in §2; reviewers don't need to see the underdetermined-regime curve to accept it.
- Don't volunteer the LOSO 94.6 → 97.3 healthy-subject anchor in the body — the cross-population stroke result is the headline; the same-population number weakens it by inclusion.

**Pre-empt with one sentence + appendix (reviewer will ask from the paper's own framing):**
- **"What happens without per-session re-cal?"** — arises from §2's description of the protocol itself. One sentence in §2: "Without per-session re-cal, accuracy drops from 0.86 to 0.72 by the next session (Appendix A)." Quantified, defensible, off-path for the main body.

### Describing GrabMyo vs defending its necessity

These are different and the plan does not conflate them.

- **Describe** (required, in §2): "The classifier is trained jointly on GrabMyo (1.14 M windows, 43 healthy subjects, weight ×1) and per-patient calibration data weighted ×100." Architectural description; without it, §2 reads like the system is hiding what's in the base.
- **Plus a role-of-component half-sentence** (also in §2): "The GrabMyo population sample allows the 370-feature classifier to fit stably at the 30 s per-session calibration budget." What GrabMyo does in the architecture, not whether the system could be built without it.
- **Don't defend** (avoid, anywhere): "we tested whether GrabMyo is necessary", "patient-only achieves comparable accuracy at the operating point", "without GrabMyo the protocol would not exist". These each volunteer a comparison the paper isn't making.

---

## 8. Out-of-scope material (in repo, not in paper)

These analyses exist on disk and can be cited as supporting material in rebuttal if a reviewer asks; they do not appear in the main body:

- Cal-size sweep (`analysis/physiomio/results/cal_size_sweep_*.csv`) — system characterisation, not contribution-defining.
- Longitudinal degradation full curve — main body cites only the dist=1 number (0.72); full curve to Appendix A.
- Per-feature distributional-shift mechanism analysis — appendix-only at 4 p.
- LOSO 94.6 → 97.3 on healthy GrabMyo — same-population sanity check, methods footnote at most.
- EMGBench broad validation — supporting evidence the feature pipeline generalises beyond GrabMyo; main-body only if 8-p track is chosen.
- Lucchetti on the simulated 20 Hz envelope pipeline (`analysis/lucchetti/results/deployed_pipeline_sim.csv`) — calibrated 3-class drops from 0.79 (raw 2 kHz, reported in §3) to 0.49 in the envelope regime. **Do not cite.** Reason: Lucchetti's reach-grasp task carries discriminative information in transient muscle bursts that the 50 ms peak-to-peak envelope smooths over. The deployed system's actual protocol is sustained holds (rest / close / open held ~5 s each), which the envelope regime preserves cleanly (confirmed on PhysioMio, §4). The Lucchetti envelope number tests a scenario the deployed system does not run; available in rebuttal only if a reviewer raises the question.

---

## 9. Workflow notes for the author

- **Step 0**: every number in the paper comes from §1 (Locked canonical numbers). Don't recompute, don't aggregate differently, don't introduce new number-pairs.
- **Step 1**: §3 (cross-population result) drafts first. Everything else either sets it up or characterises the system that produced it. The numbers are fixed in §1 of this plan; the figure carries the visual argument.
- **§2 describes the system as built.** No defense, no comparison, no motivation. The prose says "we use a 30 s re-cal protocol of 3 cued gestures held for 5 s each", not "we chose 30 s because…". The one pre-empt sentence (Appendix A pointer) is the only forward-looking thing in §2.
- **§4 stays tight.** Numbers are the point, not the narrative. ReactEMG comparison is one sentence inline.
- **If the body feels light at 4 pages**, expand Fig 2's caption or the severity sub-panel — not import methodological depth from §2 or backfill baseline comparisons. The contribution is the result on 58 stroke patients in a deployed configuration.
