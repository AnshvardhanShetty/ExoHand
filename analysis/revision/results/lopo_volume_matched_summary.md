# Volume-matched LOPO — addressing the data-volume confound

For each of 48 PhysioMio patients: pool cal from the other 47 patients,
then subsample the pool to match THAT patient's cross-arm training-set size
(mean target n=432). Train HGB on the volume-matched
subsample, test on held-out patient's impaired-arm test set.

## Results

| Regime | Mean accuracy | Training size |
|---|---:|---:|
| Zero-shot (GrabMyo only) | 0.35 (reference) | 1.14M |
| **Cross-arm PO** (this patient's healthy-arm cal) | **0.5486** | ~432 |
| **Volume-matched LOPO** (47 other patients' cal, subsampled) | **0.6843** | ~432 |
| LOPO full pool (47 patients × all cal) | 0.6287 | ~20,000 |
| Impaired-arm own cal (baseline) | 0.8752 | ~432 |

**Paired Wilcoxon (volume-matched LOPO > cross-arm): p = 6.4994e-06**
Patients where VM-LOPO > cross-arm: 35 / 48

## Interpretation

- If VM-LOPO > cross-arm significantly: pathology-matched data really is more
  valuable per-window than anatomy-matched. The 'stroke EMG is a distinct
  distribution' claim survives, controlling for volume.
- If VM-LOPO ≈ cross-arm: the previous LOPO advantage was mostly data volume.
  Cross-arm finding shrinks to 'healthy arm doesn't transfer well' without the
  crisp 'beats LOPO' claim.
- Either way, cross-arm remains meaningfully below impaired-arm own cal (~0.87),
  so the within-arm cal advantage is not in question.