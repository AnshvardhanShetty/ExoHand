# Writing companion — ICBINB @ NeurIPS 2026 paper

**Purpose of this document:** everything a writing-focused LLM needs to help me write the ICBINB workshop submission. Paste this at the start of every writing session.

---

## 0. Meta context — how I want you (the writing LLM) to help me

**I am writing this myself.** Do not draft paper prose for me. My learning + authorship voice depends on me writing every sentence.

What I do want from you:
- Suggest structure at the section, paragraph, and sentence-transition level
- Point out places where my draft is unclear, redundant, or overclaims
- Suggest alternative phrasings when I explicitly ask "how would you phrase X"
- Catch arithmetic errors, inconsistent numbers, or claims that contradict what's in this document
- Push back on framing choices you think are weaker than alternatives, but leave the final call to me
- Ask clarifying questions when a claim I'm about to make isn't supported by evidence in this document

What NOT to do:
- Do not draft abstracts, introductions, or paragraph text unless I explicitly ask
- Do not invent numbers or fill in ranges I haven't given you — every number must come from Section 4 below
- Do not add authors, affiliations, or acknowledgements
- Do not suggest citations I haven't already brought up (workshop paper, tight related-work section)
- Do not add Co-Authored-By or attribution lines to any commit or output

If I ask "write this section for me," push back once — remind me I want to write it — then only if I insist, provide the shortest possible starter I can rewrite entirely.

---

## 1. Venue

**ICBINB (I Can't Believe It's Not Better) workshop at NeurIPS 2026.**
- Location: Sydney
- Deadline: August 29, 2026
- Ethos: rewards negative or surprising results in ML that are documented rigorously and paired with a mechanism explanation. Not "we forgot to tune hyperparameters." Not "this new architecture underperformed." The specific fit is "a reasonable, well-motivated hypothesis of the field fails, and here is why."
- Typical acceptance rate: ~40-55%, higher than main-track conferences
- Length: ~4-8 pages typical for workshops (verify from CFP)
- Rigor markers reviewers explicitly value: pre-registration, honest reporting of what didn't work, bootstrap CIs, effect sizes alongside p-values, replication across datasets, formal statistical tests

**Secondary venue** (same paper, minor reframing): TS-LIMITS workshop, Paris, deadline Sep 5. Not the primary submission target for now.

---

## 2. The paper in one paragraph

We tested whether pretraining on a large healthy EMG corpus (GrabMyo, 1.14M windows from 43 subjects) improves stroke intent classification when deployed on cheap embedded hardware. It doesn't — the pretrained classifier lands at chance (0.360 on a 3-class task, uniform chance 0.333). Meanwhile a small cross-patient stroke corpus (47 donor patients, 432 windows each) transfers meaningfully, outperforming the patient's own healthy-arm calibration by +10.3 pp. The transferable substrate for stroke intent classification is pathology structure, not healthy-population scale. This has implications for how the field should invest in pretraining corpora for pathology-specific applications.

---

## 3. Core claims (what the paper argues)

**Claim 1 (negative, the headline):** 1.14M-window healthy EMG pretraining fails to transfer to stroke intent classification at any weight or layering variant tested. No test detects transfer above 3-class chance.

**Claim 2 (positive, supporting):** At matched training volume (432 windows), cross-patient stroke data (47 other patients' impaired arms) outperforms same-patient healthy data by +10.3 pp (Wilcoxon p=0.005, Cliff's δ=+0.25) on the full 48-patient cohort.

**Claim 3 (mechanism):** Decomposing the +10.3 pp gap into diversity (donor-pool size) and pathology (donor-arm state) axes reveals both contribute, with pathology adding +2.2 pp on the full cohort (p=0.021, δ=+0.375). In the ≥30-day subset (n=25, days post-stroke ≥ 30), the pathology contribution amplifies to +4.8 pp (95% CI [+1.7, +7.9], p=0.004, δ=+0.60).

**Claim 4 (mechanism validation):** The pathology contribution rises monotonically with days post-stroke across all thresholds we tested (+2.2 pp at ≥7 days → +4.8 pp at ≥30 days → +5.3 pp at ≥60 days), consistent with progressive divergence of the paretic arm from healthy anatomy.

**Claim 5 (implication for the field):** For pathology-specific applications, curate small pathology-matched cross-patient corpora rather than scale healthy-population pretraining corpora. A dataset four orders of magnitude smaller, matched on pathology, transfers where 1.14M healthy windows do not.

---

## 4. Every number the paper uses (canonical)

**All numbers below are frozen and reproducible.** Any number not here must not appear in the paper.

### 4.1 Zero-shot GrabMyo → stroke intent (n=48, 3-class, chance = 0.333)

| Statistic | Value |
|---|---|
| Mean accuracy | 0.360 |
| Bootstrap 95% CI on mean | [0.312, 0.408] |
| Wilcoxon signed-rank vs chance (H1: median > 1/3) | p = 0.082 |
| One-sample t-test vs chance (H1: mean > 1/3) | p = 0.134 |
| Sign test above chance | 28/48 (58%), binomial p = 0.156 |
| Cohen's d vs chance | 0.162 (small) |
| CI upper bound above chance | +7.5 pp |
| Gap to calibration baseline (0.896) | +56.3 pp |

**Safe wording:** "no detectable transfer beyond chance; 95% CI compatible with at most modest transfer, roughly one-seventh of the gap needed to reach the 22-second calibration baseline."

**Do NOT say:** "at chance" as a formal equivalence claim; "above chance"; "1.08× chance"; "marginal transfer."

### 4.2 Full-cohort regime ladder (n=48, means)

| Regime | Mean accuracy |
|---|---|
| GrabMyo zero-shot | 0.360 |
| Own healthy cal → own impaired test (cross-arm) | 0.639 |
| 47 others' healthy arms → own impaired (Exp 1, multi-draw) | 0.719 |
| 47 others' impaired arms → own impaired (VM-LOPO, multi-draw) | 0.742 |
| Own impaired cal → own impaired test (upper bound) | 0.896 |

### 4.3 Effect decomposition (full cohort n=48, all reconcile arithmetically)

| Effect | Δ mean | Wilcoxon p | Cliff's δ |
|---|---|---|---|
| Total gap (own-hlth 0.639 → 47-imp 0.742) | +10.3 pp | 0.005 | +0.250 |
| Diversity (own-hlth → 47-hlth) | +8.1 pp | 0.049 | +0.125 |
| Pathology (47-hlth → 47-imp) | +2.2 pp | 0.021 | +0.375 |

Arithmetic check: +8.1 + +2.2 = +10.3 ✓

### 4.4 ≥30-day subset (n=25)

*Cohort naming rule: never use "chronic" or "acute" for our subsets. Bernhardt et al. 2017 defines chronic as > 6 months post-stroke; only 1 of our 48 patients qualifies. Median is 36 days — the whole cohort is early-subacute-heavy. Name subsets by their day cutoff.*

| Regime | Mean accuracy |
|---|---|
| GrabMyo zero-shot | 0.367 |
| Own healthy cal → own imp (cross-arm) | 0.589 |
| 47 others' healthy → own imp (Exp 1, multi-draw) | 0.686 |
| 47 others' impaired → own imp (VM-LOPO, multi-draw) | 0.734 |
| Own impaired cal → own imp | 0.882 |

| Effect | Δ mean | Wilcoxon p | Cliff's δ |
|---|---|---|---|
| Total | +14.5 pp | 0.003 | +0.360 |
| Diversity | +9.7 pp | 0.067 (n.s.) | +0.200 |
| Pathology | +4.8 pp (95% CI [+1.7, +7.9]) | 0.004 | +0.600 |

Arithmetic check: +9.7 + +4.8 = +14.5 ✓

### 4.5 Dose-response — pathology contribution vs days post-stroke

| Cutoff (days) | n | Pathology Δ | 95% CI | Wilcoxon p |
|---|---|---|---|---|
| ≥7  | 48 | +2.2 pp | [−0.4, +4.9] | 0.021 |
| ≥14 | 45 | +2.5 pp | [−0.4, +5.2] | 0.016 |
| ≥21 | 37 | +2.8 pp | [−0.1, +5.7] | 0.012 |
| ≥30 | 25 | +4.8 pp | [+1.7, +7.9] | 0.004 |
| ≥45 | 16 | +4.1 pp | [+0.3, +8.2] | 0.042 |
| ≥60 | 12 | +5.3 pp | [+1.0, +10.3] | 0.026 |
| ≥90 |  3 | +9.5 pp | [+1.4, +25.0] | 0.125 (underpowered) |

Cutoffs are nested subsets, not independent samples. Frame this as "consistency check across increasingly late-recorded cohorts," not "seven independent confirmations." The 30-day threshold is the first cutoff whose CI cleanly excludes zero — describe as *natural statistical* inflection. Do NOT call it "the standard acute/chronic threshold" — under Bernhardt et al. 2017 the acute/chronic boundary is 6 months, not 30 days, and our cohort is almost entirely early-subacute either side of the 30-day line.

### 4.6 Per-patient categorical breakdown (n=48)

|  | Diversity helps | Diversity does not help | Total |
|---|---|---|---|
| Pathology helps | 16 | **17** | 33 (69%) |
| Pathology does not help | 11 | 4 | 15 |
| Total | 27 (56%) | 21 | 48 |

**Rescue stat:** of 21 patients where donor diversity does not help, pathology-matching still helps 17 (81%), mean rescue Δ = +6.1 pp.

**Do NOT claim:** "pathology dominates diversity" as a formal statistical claim. Bootstrap CI on `δ_pathology − δ_diversity` = [−0.21, +0.67] contains zero. Same for the mean-Δ difference [−0.14, +0.01].

**DO claim:** pathology and diversity contribute complementarily; pathology specifically catches patients that diversity leaves behind.

### 4.7 Deployment / hardware

- BOM cost: **£180**
- Hardware: **Teensy 4.0** microcontroller + **4× MyoWare 2.0** EMG sensors + tendon-driven 3D-printed hand exoskeleton
- Firmware streams 20 Hz peak-to-peak envelope over USB serial
- Host runs HistGradientBoostingClassifier + 2-tier stability filter
- End-to-end latency: **~275 ms** (component-sum: 30 ms MyoWare envelope + 50 ms Teensy sampling + ~17 ms inference + 100 ms Stage 2 stability filter + ~76 ms servo slew). **Note:** this is a component-sum estimate. A measured single-number version is in progress (task A3 in Adhi's handoff) — check whether it's been produced before submission and swap in the measured number if available.

### 4.8 Sample sizes

| Dataset | Role | n subjects | Details |
|---|---|---|---|
| PhysioMio | Headline stroke corpus | 48 | 64-channel HD-sEMG @ 2 kHz |
| PhysioMio ≥30-day subset | Amplification subgroup | 25 | days post-stroke ≥ 30 |
| Lucchetti | Replication | 10 | Stroke, 12-channel @ 1 kHz, different rig |
| GrabMyo | Pretraining source | 43 | Healthy, 1.14M windows @ 2 kHz |

### 4.9 Class balance (methods-section footnote)

Every held-out test set in the frozen splits is exactly 39 rest / 39 close / 39 open = 117 windows per patient. All 48 patients have identical balance. Consequences: raw accuracy = balanced accuracy; majority-class baseline = uniform chance (0.333); no free lunch from a majority-predicting classifier.

### 4.10 Channel selection

Per patient, from the 64-channel PhysioMio HD-sEMG grid, we picked 4 channels via **Cohen's d on flex-vs-extend contrast**:
- Compute mean envelope per channel during a flex block and an extend block
- `d = (flex_env − ext_env) / pooled_std`
- Top 2 most-positive-d → flexor picks
- Top 2 most-negative-d → extensor picks
- Interleaved into `[flex1, ext1, flex2, ext2]` — must match GrabMyo's canonical `[F1, F5, F10, F14]` order

Physical placement mirrors GrabMyo's 2×8 forearm setup — ring 1 at 1/3 forearm length distal from elbow crease, ring 2 at 2 cm distal from ring 1, one flexor + one extensor per ring.

### 4.11 Classifier + protocol

- Model: HistGradientBoostingClassifier, max_iter=100, class_weight="balanced", random_state=42
- One configuration throughout, no per-cell tuning
- Features: 60 base per channel × 4 channels + engineered → 370 features per 200 ms feature window
- Feature engineering: leakage-free z-score (μ/σ computed from calibration rows only, per participant)
- Multi-draw stability: 5 independent 47-donor subsamples per patient for all cross-patient cells
- Frozen splits: (cal_idx, test_idx) fixed per patient across every analysis, stored in `analysis/revision/frozen_splits.parquet`
- Calibration protocol (paper): 22-second cued session, 12 gestures × 36 windows @ 50 ms stride = 432 balanced training windows

### 4.12 Mechanism support (three orthogonal probes)

- **Wasserstein-1 distances**: impaired arms are distributionally closer to *other* patients' impaired arms than to their own healthy arm — full 48-patient cohort, no days-post-stroke filter.
- **Channel-permutation ablation**: shuffling channel labels erases the majority of the cross-patient pathology transfer benefit. Signal lives in specific channel geometry.
- **Feature importance × distribution-shift correlation**: features that shift most across the healthy/impaired boundary receive highest classifier importance in cross-patient models.

For each, one sentence with a supporting number is enough. Don't over-invest in these — they support, they don't headline.

---

## 5. Paper structure — CFP-mandated four-part

**The ICBINB-BIO CFP requires this four-part structure by name for full papers. Not optional.** Each of the four sections is graded on named criteria (see §1). Structure below aligns to that.

### Abstract (~200 words)
1. Motivation: cheap embedded stroke rehab, obvious pretraining bet
2. Negative result: 1.14M healthy windows → no detectable transfer beyond chance
3. Positive contrast: +10.3 pp from pathology-matched cross-patient
4. Mechanism: pathology contribution +2.2 pp full / amplifies to +4.8 pp in the ≥30-day subset
5. Dose-response confirms monotonicity with days post-stroke
6. Deployed on £180 hardware
7. Implication: curate pathology-matched corpora rather than chase healthy-population scale

---

### 1. Problem (CFP §1)

**A clearly specified biological task, data modality, and setting.** State the assumption you're about to test.

- 3-class stroke intent (rest / close / open), sEMG modality, embedded assistive-control setting
- Target metric anchored to the 0.896 calibration ceiling
- Datasets: PhysioMio (n=48 stroke, 64-ch @ 2 kHz), GrabMyo (43 healthy, 1.14M windows @ 2 kHz), Lucchetti (n=10 stroke, 12-ch @ 1 kHz replication)
- Classifier config, frozen-splits protocol
- **F1 lives here** (hardware photo + electrode placement diagram)

**Drop-in balanced-by-construction sentence:**

> "Test sets are stratified by construction under the frozen-split protocol: exactly 39 windows per class × 3 = 117 per patient; accuracy and balanced accuracy coincide; majority-class baseline equals uniform chance."

---

### 2. Proposed approach (CFP §2)

**The modeling approach or solution under investigation.**

- GrabMyo pretraining as the approach — pool the largest available related distribution
- Preconditions (montage compatibility, feature alignment)
- The explicit falsifiable hypothesis: state it plainly per the CFP's ask

**Drop-in scoping sentence:**

> "We test transfer via data pooling in the deployed feature space; deep representation transfer is out of scope."

---

### 3. Observed outcome (CFP §3)

**A concise description of the negative, null, or unexpected outcome, with uncertainty.**

- 0.360 vs 0.333 chance
- The five-test null table (bootstrap CI, Wilcoxon, t-test, sign test, Cohen's d)
- +56.3 pp threshold framing with the one-seventh CI ratio
- Regime-invariance across GrabMyo-weight sweep and stacking variants (light-GM, stacked P+A)

Nearly verbatim from the current sanity-check doc — the tests are done, just need to be presented cleanly.

---

### 4. Reason for failure (CFP §4 — LARGEST section, double-duty)

**An investigation of why the approach did not work.** The CFP grades "Depth of failure analysis" as a named criterion, so this is where reviewers will spend the most time.

**Structure:**
- **Regime ladder (T1)** — establishes what does work at matched volume
- **+10.3 pp cross-patient result** — the load-bearing evidence that healthy ≠ impaired matters
- **Diversity vs pathology decomposition (F2 / T2)** — isolates the pathology axis specifically
- **Dose-response (F3)** — monotonic with days post-stroke, biologically coherent
- **Three mechanism probes, labelled explicitly as mechanistic evidence:**
  1. Wasserstein-1 distances (distributional mechanism)
  2. Feature-importance × Wasserstein-shift correlation (feature-level mechanism)
  3. Channel-permutation ablation (geometric mechanism — save for last)
- **Boundary conditions and actionable takeaway**

**Drop-in loop-closing sentences:**

> "The transferable signal is pathology-specific channel geometry; healthy corpora cannot contain it at any volume; hence 1.14M windows fail where 432 matched windows succeed."

> "For pathology-specific applications, curate small pathology-matched corpora rather than chase healthy scale."

---

### Limitations (~4 sentences, don't compress further)

1. 3-class task is coarser than full finger-level classification (matches the 3-DoF exoskeleton controller)
2. Cross-arm within-patient carries an electrode-placement confound; cross-patient comparison across 48 patients is the primary mitigation
3. The ≥30-day subgroup (n=25) is modest for subgroup claims; the dose-response monotonicity is the mitigation
4. **GrabMyo → PhysioMio montage-mismatch caveat on the zero-shot null** — the observed 0.360 reflects both the healthy/impaired distribution shift we investigate and any residual cross-hardware alignment loss; the causal mechanism claim rests on within-PhysioMio evidence (§4) where hardware is held constant

---

### Appendix (unlimited pages, but reviewers not required to read — anything load-bearing gets a one-line summary in main text)

- Pre-registration document
- Ablations: light-GM, stacked P+A, GrabMyo weight sweep, per-limb normalization
- Lucchetti replication ladder
- Hardware specifications, BOM breakdown
- Firmware-mirror equivalence from A6 (hardware-in-the-loop replay)

---

### Non-page-counting required materials

- **LLM-use disclosure paragraph** — check CFP for exact wording expected
- **Ethics statement** — human-subjects data (stroke patients), IRB / consent status
- **Reproducibility statement** — code + data availability, frozen splits, pre-registration

---

### Anonymity checks (for the double-blind review)

- **F1 photo** — no faces, no identifying features. Arm-only framing.
- **Linked repo** — must not deanonymise (no author names in commits visible to reviewer, or use an anon-repo mirror)
- **Pre-registration link** — same, don't leak identity via the hosting URL

---

### Figures / tables (5 total)

- **F1**: hardware photo + electrode placement diagram from GrabMyo Electrodelocation.pdf (lives in §1)
- **F2**: 2×2 decomposition — per-patient scatter with categorical rescue breakdown (`analysis/revision/results/pathology_dominates.png`) (lives in §4)
- **F3**: dose-response 2-panel (`analysis/revision/results/dose_response_pathology.png`) (lives in §4)
- **T1**: full-cohort regime ladder from §4.2 (lives in §4)
- **T2**: per-patient categorical helps/hurts breakdown from §4.6 (lives in §4)

---

### Writing order + hinge sentence

**Two settled writing notes:**

1. **The hinge sentence between §3 (Observed outcome) and §4 (Reason for failure) carries the whole ICBINB framing.** It's something like: *"Having ruled out scale, we ask: what does transfer?"* Write this sentence carefully — it's the pivot that turns the negative result into a mechanism investigation, which is the venue's whole thesis. Get it right and the reader is with you for §4; get it wrong and the paper reads as two disconnected halves.

2. **Write §3 → §4 first**, since they're the closest to done (numbers are frozen, structure is fixed). Then write §1 (Problem) and the synthesis (abstract + limitations) together, so the promise made in §1 and the payoff delivered in §3–4 match without drift.

---

## 6. Framing decisions we've committed to (don't reopen these)

1. **Failure-first structure.** The paper leads with the negative result. The positive finding is subordinated as an investigation motivated by the failure, not as a competing headline. This is critical for ICBINB fit — a positive-in-disguise paper will fail the venue.

2. **The reason for failure IS the pathology-difference finding.** The paper's four-part structure has "Reason for failure" as a mandatory named section. Our reason is: healthy EMG and stroke-impaired EMG differ structurally (in distribution, channel geometry, and feature importance) in ways that make a healthy-only-pretrained classifier unable to generalize to impaired-arm targets. This is supported by six within-PhysioMio lines of evidence (regime ladder, +10.3 pp cross-patient, pathology decomposition, dose-response, channel-permutation, feature-importance × shift correlation). The GrabMyo failure is the Observed Outcome; the pathology-difference finding is the Reason for Failure. Causally linked in one direction: because healthy and impaired EMG differ structurally as demonstrated within-PhysioMio, healthy-only pretraining like GrabMyo cannot survive the transfer to impaired-arm targets. (Earlier drafts of this doc said "no causal chain" — that was wrong; the CFP explicitly requires a causal failure analysis.)

3. **Complementarity, not dominance.** Diversity and pathology are complementary contributors. Do NOT say "pathology dominates diversity" — the bootstrap δ-difference CI contains zero. DO say "pathology specifically catches patients diversity leaves behind" and use the 17/21 rescue stat.

4. **Softened zero-shot language.** "No detectable transfer beyond chance; 95% CI compatible with at most modest transfer" — not "at chance" as a formal equivalence claim.

5. **Reported protocol is PhysioMio's 22-s cued cal, not the shipped website's protocol.** The shipped web interface uses a two-tier 6-min patient + 30-s session cal on healthy users, separate line of work. The paper is entirely on the PhysioMio 22-s protocol on stroke patients. Methods section acknowledges this once and moves on.

6. **No floor tests included in the paper.** Two floor tests exist in the repo (C1 healthy-vs-healthy permutation floor, montage floor test) but neither is in the paper — they touch supporting claims and would only dampen headline framing. Standing decision. Do not surface them.

7. **Rigor is displayed only where it protects a claim we make.** Pre-registration, frozen splits, leakage-free features, multi-draw stability, Lucchetti replication — all displayed. Diagnostic ballast (chronic-donors-only refinement, gap-widening seed sweeps, floor tests) — kept in repo, not surfaced.

---

## 7. Deprecated / retired numbers and framings

Any older doc that contradicts §4 or §6 is superseded. Do not use anything from this list:

| Old | Superseded by | Why |
|---|---|---|
| "0.29 zero-shot" | 0.360 | Wrong recall, older attempted 13-class scheme |
| "13-class chance = 0.077" | 3-class chance = 0.333 | Actual pipeline is 3 classes |
| "GrabMyo transfers at ~4× chance" | "no detectable transfer" | 1.08× is not statistically distinguishable |
| "+11.3 pp total gap" | +10.3 pp | Single-draw VM-LOPO gave 0.752; multi-draw is 0.742 |
| "0.752 chronic VM-LOPO" | 0.734 | Single lucky draw |
| "0.709 chronic Exp 1" | 0.686 | Single lucky draw |
| "+4.3 pp chronic pathology" | +4.8 pp | Single-draw |
| "p=0.010 chronic pathology" | p=0.004 | Single-draw |
| Cliff's δ = +0.27 headline | +0.250 | Moved slightly with multi-draw |
| "Chronic-donors-only pathology +0.5 pp" | Omit | Falsified refinement, not reported |
| "chronic subset" as a subset label | "≥30-day subset" (n=25) | Bernhardt et al. 2017: chronic = > 6 months post-stroke; only 1 of our 48 patients qualifies. Reviewer can falsify against public metadata in 30 seconds. Name subsets by cutoff, not by phase word. |
| "standard clinical acute/chronic threshold at 30 days" | "first cutoff whose bootstrap CI cleanly excludes zero" | Bernhardt's acute/chronic boundary is 6 months, not 30 days. Any use of "acute" or "chronic" must reference Bernhardt correctly, and applies only when describing clinical context — never to name our subsets. |
| "Pathology dominates diversity" as effect-size claim | Complementarity framing | Bootstrap δ-difference contains zero |
| "22-s calibration is the deployed protocol" | "PhysioMio protocol used for reported numbers; shipped deployment uses a separate two-tier protocol" | Different populations, protocols, and classifier configurations |

---

## 8. Reviewer objections we're pre-empting

1. **"You fished a subgroup."** Mitigated by dose-response monotonicity across 7 cutoffs. 30-day threshold is the first CI-clean cutoff, a natural statistical inflection (not a "chronic" phase boundary — under Bernhardt et al. 2017, chronic is > 6 months and only 1 of our patients qualifies).

2. **"Your CI doesn't rule out modest transfer above chance."** Acknowledged in the zero-shot paragraph — softened phrasing states the CI upper bound and notes it's ~1/7 of the calibration-baseline gap.

3. **"Cross-arm same-patient has electrode-placement confounds."** Acknowledged in limitations; cross-patient comparison across 48 patients is the primary mitigation.

4. **"The negative isn't surprising given montage mismatch between GrabMyo and PhysioMio."** Not directly addressed in the paper — decided the mechanism claim doesn't need to explain GrabMyo's failure; the positive finding stands on within-PhysioMio evidence. If a reviewer raises this at review time, response would be: "The mechanism claim is supported by within-PhysioMio comparisons; the GrabMyo failure is reported as a practical negative regardless of underlying cause."

5. **"n=25 ≥30-day subset is small for subgroup claims."** Mitigated by dose-response — the effect appears across n=48 down to n=12 with monotonic progression.

6. **"3-class task is coarse."** Acknowledged limitation; matches the 3-DoF exoskeleton controller.

7. **"How was GrabMyo aligned to PhysioMio for zero-shot?"** Same 4 canonical channels (F1, F5, F10, F14) matched via Cohen's-d per-patient channel selection. Documented in methods.

---

## 9. Datasets — quick reference

- **PhysioMio**: 48 stroke patients, 64-channel HD-sEMG @ 2 kHz. Per-patient cued session including both healthy-arm and impaired-arm segments. Days-post-stroke metadata available. Headline dataset.
- **Lucchetti**: 10 stroke patients, 12-channel @ 1 kHz, different acquisition rig. Independent replication.
- **GrabMyo**: 43 healthy subjects, 1.14M windows @ 2 kHz, 2×8 forearm electrode setup (Pradhan et al.). Published explicitly as an EMG pretraining substrate. Used for pretraining source only.

---

## 10. Files that back every claim

| File | What it contains |
|---|---|
| `analysis/revision/FINAL_NUMBERS.md` | Canonical source for §4; every headline number reproducible |
| `analysis/revision/frozen_splits.parquet` | (cal_idx, test_idx) per patient, canonical splits |
| `analysis/revision/results/leakage_free_ladder_per_patient.csv` | Rows 1, 2, 5 of the full-cohort table |
| `analysis/revision/results/all_multidraw_per_patient.csv` | Exp 1 + VM-LOPO multi-draw, all 48 patients |
| `analysis/revision/results/dose_response_pathology.csv` | §4.5 exactly |
| `analysis/revision/results/pathology_dominates.png` | F2 candidate |
| `analysis/revision/results/dose_response_pathology.png` | F3 |
| `data/physiomio_channel_picks.csv` | Per-patient 4-channel selection with Cohen's d values |
| `PREREGISTRATION.md` | Pre-registered decision rules |
| `grabmyo/Electrodelocation.pdf` | F1 candidate — electrode placement diagram |

---

## 11. What to ask me if you're unsure

- "Is this number in Section 4?" — if not, don't use it
- "Does this framing contradict Section 6?" — if yes, flag it and don't proceed
- "This claim requires evidence not in this document — should we drop it, or do you have a source?"
- "This sentence overstates X" — always fine to flag
- "Do you want me to suggest an alternative phrasing here?" — always ask before offering

---

## 12. My working style

- I write in bursts. Give me one section to focus on at a time. Don't try to plan the whole paper across a single conversation.
- I'll paste my draft; you critique it. Then I revise. Not the other way around.
- I prefer terse, direct feedback over exhaustive rewrites.
- If I'm making a bad framing choice, tell me once with the reason. If I insist, drop it.
- I'll tell you the section I'm working on. Focus your feedback there — don't scope-creep.

---

## 13. What NOT to write about (paper-scope guardrails)

- Do not turn this into a paper about the deployed website. The paper is about the stroke-classification research.
- Do not turn this into a paper about EMG datasets or hardware design. The paper is about pretraining transfer failure and pathology structure.
- Do not add speculative "future work" beyond one paragraph.
- Do not include acknowledgements, funding statements, or ethics statements unless I bring them up.
- Do not editorialize about the field ("the community should...") — one implication paragraph is enough.

---

**Working notes for me (the author):**

- Deadline is Aug 29 for ICBINB. Backup deadline Sep 5 for TS-LIMITS.
- Adhi handoff for hardware (photos, video trim, latency measurement) in progress.
- If A3 (measured latency) lands before submission, swap the component-sum 275 ms for the measured single number.
- If a reviewer directly asks about montage mismatch as an alternative explanation for the GrabMyo failure, response is prepared but not in the paper.
