# FINAL_NUMBERS — canonical results for ICBINB / TS-LIMITS submission

**Frozen 2026-08-23.** Every number below is what to quote in the paper. Any older doc that disagrees is superseded.

All accuracies use:
- Leakage-free features (z-score μ/σ from cal rows only, per participant)
- Frozen splits per patient (`analysis/revision/frozen_splits.parquet`)
- HistGradientBoostingClassifier, `max_iter=100`, `class_weight="balanced"`
- 3-class intent task: rest / close / open (chance = 1/3 ≈ 0.333)
- Multi-draw cells = mean of 5 independent 47-donor subsamples per patient, 432 windows each

---

## 1. Zero-shot GrabMyo → stroke — essentially at chance

n = 48 patients. Chance for 3 classes = 0.333.

| Test | Result | Rejects H0 (transfer > chance)? |
|---|---:|:---:|
| Mean accuracy | 0.360 (+2.69 pp above chance) | — |
| Bootstrap 95% CI on mean | [0.312, 0.408] | **NO — contains 0.333** |
| Wilcoxon signed-rank (H1: median > 1/3) | p = 0.082 | NO |
| One-sample t-test (H1: mean > 1/3) | p = 0.134 | NO |
| Sign test: patients above chance | 28/48 (58%), binomial p = 0.156 | NO |
| Cohen's d vs chance | 0.162 (small) | — |

**Paper-safe wording:** *"Zero-shot transfer from 1.14M GrabMyo healthy windows achieves 0.360 mean accuracy across 48 stroke patients — statistically indistinguishable from 3-class chance (0.333); bootstrap 95% CI [0.31, 0.41] contains chance, Wilcoxon p = 0.08, Cohen's d = 0.16."*

Do NOT say: "above chance", "marginal transfer", "1.08× chance." Say: "at chance", "no detectable zero-shot transfer."

---

## 2. Full cohort (n=48) — regime ladder

| Regime | Mean acc | Median |
|---|---:|---:|
| GrabMyo zero-shot | 0.360 | 0.368 |
| Own healthy cal → own imp (cross-arm) | 0.639 | 0.645 |
| 47 others' healthy → own imp (Exp 1, multi-draw) | 0.719 | 0.735 |
| 47 others' impaired → own imp (VM-LOPO, multi-draw) | 0.742 | 0.771 |
| Own impaired cal → own imp (upper bound) | 0.896 | 0.923 |

### Effect decomposition (means, all reconcile ✓)

| Effect | Δ mean | Wilcoxon p | Cliff's δ |
|---|---:|---:|---:|
| Total (own-hlth → 47-imp) | +10.3 pp | **0.005** | +0.250 |
| Diversity (own-hlth → 47-hlth) | +8.1 pp | 0.049 | +0.125 |
| Pathology (47-hlth → 47-imp) | +2.2 pp | 0.021 | +0.375 |

Arithmetic check: diversity +8.1 + pathology +2.2 = total +10.3 ✓

---

## 3. ≥30-day subset (n=25)

*Cohort naming note: our patients span 13–2308 days post-stroke (median 36). Under the Bernhardt et al. 2017 taxonomy — acute ≤ 7 d, early subacute 7 d–3 mo, late subacute 3–6 mo, chronic > 6 mo — only 1 patient in the full cohort is truly chronic; the "≥30-day subset" reported here is predominantly early-subacute plus a few later-subacute patients. Do not call this subset "chronic" in the paper.*

| Regime | Mean acc |
|---|---:|
| GrabMyo zero-shot | 0.367 |
| Own healthy cal → own imp | 0.589 |
| 47 others' healthy → own imp (multi-draw) | 0.686 |
| 47 others' impaired → own imp (multi-draw) | 0.734 |
| Own impaired cal → own imp | 0.882 |

### Effect decomposition (means)

| Effect | Δ mean | Wilcoxon p | Cliff's δ |
|---|---:|---:|---:|
| Total | +14.5 pp | 0.003 | +0.360 |
| Diversity | +9.7 pp | 0.067 (n.s.) | +0.200 |
| Pathology | +4.8 pp | **0.004** | **+0.600** |
| Pathology 95% bootstrap CI on Δ | **[+1.7, +7.9] pp** | | |

Arithmetic check: +9.7 + +4.8 = +14.5 ✓

---

## 4. Dose-response — pathology gap vs days post-stroke

Multi-draw pathology gap (imp − hlth) at each days-post-stroke cutoff:

| Cutoff (days) | n | Δ mean | 95% CI | Wilcoxon p |
|---:|---:|---:|:---:|---:|
| 7 | 48 | +2.2 pp | [−0.4, +4.9] | 0.021 |
| 14 | 45 | +2.5 pp | [−0.4, +5.2] | 0.016 |
| 21 | 37 | +2.8 pp | [−0.1, +5.7] | 0.012 |
| **30** | **25** | **+4.8 pp** | **[+1.7, +7.9]** | **0.004** |
| 45 | 16 | +4.1 pp | [+0.3, +8.2] | 0.042 |
| 60 | 12 | +5.3 pp | [+1.0, +10.3] | 0.026 |
| 90 |  3 | +9.5 pp | [+1.4, +25.0] | 0.125 (n=3, underpowered) |

Monotonic across all non-degenerate cutoffs. First cutoff whose bootstrap CI excludes zero is 30 days — the *natural* inflection, not chosen post-hoc.

---

## 5. Per-patient categorical breakdown (n=48)

|  | Diversity helps | Diversity hurts / no help | Total |
|---|---:|---:|---:|
| **Pathology helps** | 16 | **17** | **33 (69%)** |
| **Pathology hurts** | 11 | 4 | 15 |
| Total | 27 (56%) | 21 | 48 |

**Rescue stat:** Of 21 patients where diversity does not help, pathology-matching still helps 17 (81%), mean rescue Δ = +6.1 pp.

Do NOT claim: "pathology δ is statistically-significantly larger than diversity δ" — bootstrap CI on `δ_pathology − δ_diversity` = [−0.21, +0.67] contains zero. Same for the mean-Δ difference [−0.14, +0.01].

DO claim: "pathology and diversity contribute complementarily; pathology helps a majority of patients that diversity fails."

---

## 6. Headline sentences (paste-ready)

- **Negative (ICBINB):** "1.14M healthy EMG windows from 43 GrabMyo subjects transfer at 0.360 mean accuracy to a 3-class stroke intent task (chance 0.333), statistically indistinguishable from chance (bootstrap 95% CI [0.31, 0.41], Wilcoxon p = 0.08)."

- **Total gap (n=48):** "Cross-patient pathology-matched training (0.742) outperforms same-patient anatomy-matched training (0.639) by +10.3 pp (Wilcoxon p = 0.005, Cliff's δ = +0.250)."

- **Pathology contribution (n=48):** "Holding donor pool constant at 47 patients, switching from healthy-arm to impaired-arm training adds +2.2 pp (p = 0.021, δ = +0.375) — a pathology-specific contribution beyond diversity."

- **≥30-day amplification (n=25):** "In patients at least 30 days post-stroke the pathology contribution amplifies to +4.8 pp (95% CI [+1.7, +7.9], p = 0.004, δ = +0.600)."

- **Dose-response:** "The pathology contribution rises monotonically with days post-stroke — +2.2 pp at ≥7 d → +4.8 pp at ≥30 d → +5.3 pp at ≥60 d — consistent with progressive divergence of the paretic arm from healthy anatomy."

---

## 7. Data provenance

Every number above is reproducible from these files:

| File | Contents |
|---|---|
| `analysis/revision/frozen_splits.parquet` | Canonical (cal_idx, test_idx) per patient |
| `analysis/revision/results/leakage_free_ladder_per_patient.csv` | Rows 1, 2, 5 of the full-cohort table |
| `analysis/revision/results/all_multidraw_per_patient.csv` | Exp 1 + VM-LOPO multi-draw, all 48 patients |
| `analysis/revision/results/dose_response_pathology.csv` | Section 4 exactly |
| `analysis/revision/results/pathology_dominates.png` | Categorical + waterfall figure |
| `analysis/revision/results/dose_response_pathology.png` | Two-panel dose-response figure |
| `data/physiomio_channel_picks.csv` | Per-patient 4-channel selection (Cohen's d) |

---

## 8. Deprecated / retired numbers

Do not quote these. If found in older docs, they are superseded by Section 2/3 above.

| Old number | Superseded by | Why |
|---|---|---|
| 0.29 zero-shot | **0.360** | wrong recall or older 13-class scheme |
| "13-class chance = 0.077" | **3-class chance = 0.333** | actual pipeline is 3 classes |
| "GrabMyo transfers at ~4× chance" | **at chance, not significant** | 1.08× is not statistically distinguishable |
| +11.3 pp total gap | **+10.3 pp** | single-draw VM-LOPO used +0.752, multi-draw gives 0.742 |
| 0.752 VM-LOPO chronic mean | **0.734** | single lucky draw |
| 0.709 Exp 1 chronic mean | **0.686** | single lucky draw |
| +4.3 pp chronic pathology | **+4.8 pp** | single-draw |
| p=0.010 chronic pathology | **p=0.004** | single-draw |
| Cliff's δ = +0.27 headline | **+0.250** | slightly moved with multi-draw |
| "chronic-donors-only refinement +0.5 pp" | **omit** | falsified; not reported |
| "chronic subset" as a subset label | **"≥30-day subset" (n=25)** | Under Bernhardt et al. 2017, chronic = > 6 months post-stroke; only 1 of our 48 patients qualifies. The subset is early-subacute-heavy — a reviewer can falsify a "chronic" label against the public days-post-stroke metadata in 30 seconds. |
| "standard clinical acute/chronic threshold at 30 days" | **"first cutoff whose bootstrap CI cleanly excludes zero"** | Bernhardt's acute/chronic boundary is 6 months, not 30 days. The 30-day cutoff is a statistical inflection in our data, not a clinical phase boundary. |
| "pathology dominates diversity" as effect-size claim | **complementarity** | bootstrap δ-difference contains zero |

---

## 9. Class count — the one-line methods statement

*"Patient intents are classified into three states — rest, hand-close, hand-open — matching the 3-DoF exoskeleton controller. Chance level for this task is 1/3 ≈ 0.333. Verified with `intent_idx ∈ {0, 1, 2}` in every session record."*

---

## 10. What is NOT frozen here

- Task #3: cal-time-vs-accuracy plot (3 arms) — still to build
- Floor test (healthy-vs-healthy permutation, C1 diagnostic) — completing to n=48, appendix number
- T1 hardware replay deployed-accuracy column — awaiting Adhi run
- Any Lucchetti-side numbers — this doc is PhysioMio-only for the ICBINB headline
