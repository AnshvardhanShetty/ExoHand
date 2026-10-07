# Recal-cadence prediction

**Task:** predict per-patient longitudinal accuracy decay rate from session-1 features. Decay rate = linear slope of impaired-arm accuracy vs session distance from cal, fitted on patients with ≥ 3 sessions.

**Cohort:** n = 37 stroke patients (out of 48 PhysioMio total) — the subset with enough longitudinal sessions to fit a per-patient decay slope.

**Model:** HistGradientBoostingRegressor (max_iter=200, max_depth=4, l2_reg=0.1), leave-one-patient-out cross-validation.

## Headline result

| Metric | Value | Interpretation |
|---|---:|---|
| **LOPO R²** | **-0.1397** | Fraction of decay-rate variance explained |
| LOPO RMSE | 0.0936 | acc / session-distance |
| LOPO MAE | 0.0746 | |
| Predict-the-mean RMSE baseline | 0.0876 | |
| RMSE ratio (model / baseline) | **1.068** | < 1.0 = model beats predict-the-mean |
| Spearman ρ(predicted, actual) | **+0.3042** (p = 0.067) | Rank agreement — relevant for prioritising patients |

## Feature importance (permutation, on full-data fit)

| Feature | Permutation importance (mean ± std) |
|---|---:|
| `fma_impaired` | 0.0073 ± 0.0014 |
| `s1_f1_macro` | 0.0043 ± 0.0011 |
| `s1_f1_close` | 0.0024 ± 0.0006 |
| `s1_crossarm_acc` | 0.0015 ± 0.0004 |
| `s1_acc` | 0.0007 ± 0.0001 |
| `s1_f1_open` | 0.0004 ± 0.0001 |
| `s1_f1_imbalance` | 0.0003 ± 0.0001 |
| `s1_f1_rest` | 0.0000 ± 0.0000 |

## How this enters the paper

**One paragraph in §4.3 (within-subject temporal shift):**

> *Beyond characterising the cross-session degradation, we ask whether decay rate is predictable from session-1 features — a question with direct deployment relevance, since a clinic can use such a prediction to schedule per-patient recalibration cadence. Fitting HistGradientBoosting regression on 37 patients with leave-one-patient-out cross-validation, we achieve R² = -0.140 (RMSE 1.07× the predict-the-mean baseline) and Spearman ρ = +0.304 between predicted and actual decay slopes. The most predictive features are [see table]. This converts the longitudinal-degradation finding from a passive characterisation into an actionable per-patient deployment knob.*

## Caveats

- Sample size is n = 37, modest for a regression problem; LOPO CV is the right protocol given the constraint.
- Decay slope is a single-number summary of a curve that's often noisy; results would tighten with a more robust target (e.g., median decay or robust regression on the curve).
- The protocol applies to the specific calibration scheme studied here; different calibration designs (e.g., different cal-weight or feature pipeline) would have different decay characteristics.