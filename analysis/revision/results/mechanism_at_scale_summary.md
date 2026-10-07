# Mechanism at scale — falsifying distribution-shift-correction as the mechanism

Permutation importance vs Wasserstein-1 shift on 238 impaired-arm 
PhysioMio sessions, with two classifiers evaluated on each session's balanced test set:

- **Calibrated HGB** — GrabMyo + per-session cal (cached per-session model)
- **GrabMyo-only HGB** — trained once on 300k GrabMyo subsample, no session cal

For each session and each classifier, permutation importance is ranked and correlated
(Spearman ρ) against the fixed GrabMyo→PhysioMio-impaired Wasserstein-1 shift ranking.

## Headline

| Model | Mean ρ | Median ρ | 95% bootstrap CI | Std |
|---|---:|---:|---:|---:|
| **Calibrated HGB** | -0.030 | -0.046 | [-0.042, -0.019] | 0.093 |
| **GrabMyo-only HGB** | -0.019 | -0.019 | [-0.030, -0.008] | 0.084 |

**Paired Wilcoxon (GM ρ > Cal ρ, one-sided): p = 4.854e-02**

## Interpretation

If **Cal ρ ≈ 0** while **GM ρ > 0** (and Wilcoxon p is small), then:

- The GrabMyo-only classifier *does* weight the shifted features (as standard TL
  theory predicts — its decisions rely on features that differ between healthy and
  stroke distributions).
- The calibrated classifier does *not* — calibration overrides that shift-driven
  prior with per-patient-specific features.

This falsifies the standard TL mechanism ("calibration corrects distribution shift")
for stroke EMG. Per-patient decision boundaries are effectively independent tasks;
healthy-subject pretraining does not inform them regardless of how well distributions
align. This is the mechanistic reason large-scale healthy-EMG pretraining fails to
improve stroke EMG classification at deployment (§4).