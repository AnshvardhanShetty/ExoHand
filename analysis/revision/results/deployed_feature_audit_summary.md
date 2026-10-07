# Deployed-pipeline feature audit — reviewer concern #6

Sampled 10 patients, 57 sessions, 53,352 20 Hz windows.

## Family-level F-statistics (mean across sessions)

- Amplitude features (rms, mav, var, wl, maxamp, iemg, env_*): mean F = **132.99**
- Degenerate features (zc, ssc, wamp, mean_freq, median_freq): mean F = **2.11**
- Features with F < 1 (essentially dead): **4 / 60**

## Accuracy vs feature count (top-K by F-stat)

| top-K | patient-session mean acc | sessions |
|---:|---:|---:|
| 5 | 0.8929 | 57 |
| 10 | 0.9625 | 57 |
| 20 | 0.9786 | 57 |
| 30 | 0.9820 | 57 |
| 40 | 0.9820 | 57 |
| 50 | 0.9830 | 57 |
| 60 | 0.9791 | 57 |

## Reading

If amplitude features have F much greater than degenerate features, and if
top-10 or top-20 accuracy matches top-60, the paper's '60-feature
deployment' claim is nominal not effective. Honest revision options:

  (a) Report the deployed feature set as the informative subset and re-run
      headline numbers to confirm they hold. Cleanest fix.
  (b) Keep 60 features but explicitly note that N carry no class signal
      at 20 Hz — a documented protocol limitation.

See `deployed_feature_audit_per_feature.csv` for the full ranked list.