# Montage-plumbing floor test — GrabMyo zero-shot on healthy arms

n = 48 patients with both healthy and impaired test sets.
Same classifier, same alignment, same features as headline zero-shot;
only test-target arm differs.

## The comparison

| Target arm | Zero-shot mean acc | Median | vs chance (0.333) |
|---|---:|---:|---:|
| Healthy arm | **0.2693** | 0.2662 | 0.81× |
| Impaired arm (headline) | 0.3602 | 0.3675 | 1.08× |
| Δ (healthy − impaired) | **-0.0909** | | |

- Bootstrap 95% CI on healthy-target mean: [0.2406, 0.2994]
- Paired Wilcoxon (H1: healthy > impaired): p = 0.9991

## Interpretation

- If healthy ~ 0.90: montage plumbing works, impaired 0.360 failure is pathology-specific
- If healthy ~ 0.36: montage is the bottleneck, transfer story reframes
- Observed healthy mean = 0.269

**Verdict: montage plumbing is largely broken.** Zero-shot healthy-target 
accuracy is near chance, meaning the alignment pipeline itself loses most of the signal. 
The 'healthy→impaired' framing needs to be replaced with 'we cannot cross-transfer 
between EMG montages at any distance.'