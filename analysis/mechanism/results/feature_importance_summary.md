# What the calibrated model relies on (and a null mechanism finding)

Permutation importance averaged across 10 representative impaired-arm PhysioMio sessions, 5 permutation repeats per feature, computed against each session's cached calibrated HGB and that session's balanced test set.

## Headline (two findings, one positive + one null)

**1. Positive: the top features are concentrated in a specific family.**

Of the top 15 features by permutation importance:

- **8 are waveform-length (WL) features** — `chN_wl`, `chN_wl_prev`, `chN_wl_roll5`, `chN_wl_sess_norm`, `chN_wl_prev2`. WL dominates over RMS, MAV, frequency-domain, and envelope features.
- **All 4 channels contribute**, with ch0 (FCR / forearm flexor) appearing most often (12/30 top features), then ch4 (ECR, 9/30), ch13 (EDC, 5/30), ch9 (FDS, 4/30).
- **Temporal derivatives are over-represented** (lag, rolling, session-normalised).

**2. Null: importance is NOT correlated with where the GrabMyo → PhysioMio distribution shift is largest.**

Spearman ρ(importance rank, Wasserstein-1 shift rank) = **+0.052, p = 0.32**.

This *falsifies* the simple hypothesis that calibration re-weights toward the most-shifted features. **Calibration's effect on feature use is diffuse, not localised to shifted features** — the calibrated HGB picks features predictive of the per-patient decision boundary regardless of source-target shift magnitude.

## How this enters the paper

One composite figure (panels a + b) plus a short paragraph framing both findings honestly:

> *Panel (a) shows the features the calibrated model relies on are not concentrated in the most distributionally-shifted regions of feature space (ρ = +0.05, ns). Panel (b) names the top features: temporal waveform-length variants across all four channels, with the proximal flexor channel weighted highest. The calibration mechanism is therefore not "re-weighting toward the shifted features" — it is diffuse, with the calibrated model using whichever engineered features are predictive of the per-patient decision boundary regardless of source-target shift magnitude.*

A negative result on a clean mechanism hypothesis is itself a contribution.
