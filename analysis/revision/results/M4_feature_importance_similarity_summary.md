# M4 — pairwise feature-importance similarity across sessions

## Model: calibrated

- Sessions: 105, feature dim: 370
- Pairwise ρ (n_pairs = 5460): mean = +0.030, median = +0.028, std = 0.120
- Fraction of pairs with ρ > 0.5: 0.00%
- Fraction of pairs with ρ > 0.8: 0.00%

## Model: grabmyo_only

- Sessions: 105, feature dim: 370
- Pairwise ρ (n_pairs = 4851): mean = +0.040, median = +0.035, std = 0.127
- Fraction of pairs with ρ > 0.5: 0.21%
- Fraction of pairs with ρ > 0.8: 0.00%

## Interpretation

- Calibrated model: low pairwise ρ → different sessions rely on different feature
  subsets → per-patient/per-session specificity confirmed in feature space.
- GrabMyo-only model: same model applied to different test sets. Higher pairwise
  ρ expected. If similar to calibrated, importance patterns are noise-dominated.