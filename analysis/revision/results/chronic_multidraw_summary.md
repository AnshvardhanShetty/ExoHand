# Chronic 2×2 — multi-draw donor sampling

n = 25 chronic patients (>30d).
Per-target: 5 independent 47-donor subsamples of 432 windows, avg per target.

| Cell | Chronic mean (multi-draw) | Prior single-draw |
|---|---:|---:|
| 47 others' impaired → chronic imp target | **0.7339** | 0.7525 |
| 47 others' healthy  → chronic imp target | **0.6862** | 0.7080 |
| Pathology gap                              | **+0.0477** | +0.0445 |

Statistics:
- Paired Wilcoxon (imp > hlth): p = 0.0037
- Bootstrap 95% CI for gap: [+0.0196, +0.0798]
- Per-patient draw std, mean: imp 0.065, hlth 0.063