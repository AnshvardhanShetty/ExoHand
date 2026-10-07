# R-M1 — within-vs-across Wasserstein (leakage-free)

n = 48 patients. Leakage-free features via engineer_features_leakage_free.

## Side-by-side (leakage-free vs legacy)

| Metric | Leakage-free | Legacy | Δ |
|---|---:|---:|---:|
| d_within (own hlth ↔ own imp) | 0.4645 | 0.7363 | -0.2718 |
| d_across (own imp ↔ others imp) | 0.3265 | 0.3321 | -0.0056 |
| Ratio d_within / d_across | 1.423× | 2.217× | -0.794× |

**Paired Wilcoxon (d_within > d_across), leakage-free: p = 6.6054e-06**
**Cliff's δ: +0.458**
Patients where within > across: 35/48

Legacy: legacy d_within = 0.7363, d_across = 0.3321, ratio = 2.217×

## Gate decision (pre-registered)

- If ratio drops below ~1.5× OR loses significance → rewrite the mechanism section.
- If ratio and significance survive → geometry claim holds under clean features.