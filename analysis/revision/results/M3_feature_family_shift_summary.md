# M3 — feature-family shift ranking

Wasserstein-1 shift from GrabMyo (healthy) to PhysioMio impaired-arm, grouped by feature family.
Total features with shift measured: 370.

## Mean W₁ shift per family

| Family | Mean shift | Median shift | Std | n |
|---|---:|---:|---:|---:|
| cross-channel | 0.346 | 0.324 | 0.094 | 148 |
| amplitude | 0.279 | 0.272 | 0.140 | 166 |
| other | 0.278 | 0.277 | 0.113 | 21 |
| degenerate | 0.051 | 0.046 | 0.029 | 35 |

## Top-10 most-shifted features (family composition)

| Family | Count in top-10 |
|---|---:|
| cross-channel | 7 |
| amplitude | 3 |

## Top-30 most-shifted features (family composition)

| Family | Count in top-30 |
|---|---:|
| amplitude | 15 |
| cross-channel | 14 |
| other | 1 |

## Top 10 features by name

- **ch4_ch13_env_rms_ratio_delta** (family: cross-channel, W₁ = 0.620)
- **ch0_ch9_env_rms_ratio_delta** (family: cross-channel, W₁ = 0.604)
- **ch4_ch13_rms_ratio_delta** (family: cross-channel, W₁ = 0.589)
- **ch0_ch13_env_rms_ratio_delta** (family: cross-channel, W₁ = 0.581)
- **ch4_ch13_mav_ratio_delta** (family: cross-channel, W₁ = 0.581)
- **ch13_env_rms_accel** (family: amplitude, W₁ = 0.581)
- **ch0_ch9_rms_ratio_delta** (family: cross-channel, W₁ = 0.573)
- **ch13_wl_accel** (family: amplitude, W₁ = 0.566)
- **ch0_ch9_mav_ratio_delta** (family: cross-channel, W₁ = 0.563)
- **ch4_env_rms_accel** (family: amplitude, W₁ = 0.555)

## Interpretation

- If amplitude-family features dominate both the shift ranking AND the deployment
  top-30 (from the feature audit), then the deployed signal path is exactly where
  the healthy→stroke distribution shift is largest.
- If crossing/frequency families dominate the shift, they're irrelevant at 20 Hz
  deployment because they're already dead there (F ≈ 2 in the feature audit).
- Ties into the K=30 finding: the features that matter at deployment are the same
  features whose distribution differs most.