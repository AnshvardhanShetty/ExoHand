# Stacked pathology + anatomy test

For 48 patients: compare (a) pathology-matched only, (b) anatomy-matched only, (c) stacked.
All volume-matched to each patient's cross-arm training size (mean n≈432).

| Arm | Mean acc | n windows |
|---|---:|---:|
| **Pathology-matched only** (other patients' impaired arms, subsampled) | **0.6820** | 432 |
| **Anatomy-matched only** (this patient's healthy arm) | **0.5486** | 432 |
| **Stacked** (both combined) | **0.6825** | 864 |

**Paired Wilcoxon:**
- Stacked > max(P, A): p = 9.2545e-01
- Stacked > P alone: p = 6.1605e-01
- Stacked > A alone: p = 2.6974e-08

## Interpretation

- Stacked > max(P, A) significantly → complementary. Combining is worth it.
  Would be a useful protocol finding: use both when available.
- Stacked ≈ P alone → anatomy adds nothing; pathology-matched is sufficient.
- Stacked ≈ A alone → pathology adds nothing; anatomy-matched is sufficient.
- Stacked < either alone → interference (unlikely but possible).