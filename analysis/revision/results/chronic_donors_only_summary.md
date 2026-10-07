# Chronic-target × chronic-donor pool

n = 25 chronic patients (>30d post-stroke, both arms).
Donor pool restricted to 24 OTHER chronic patients (excludes acute donors).
Subsampled to 432 windows.

## Chronic 2×2 with clean donor pool

| Training source | 1 donor (own) | 24 chronic donors |
|---|---:|---:|
| healthy arm(s) | own healthy: 0.5894 | chronic-donors healthy: **0.7067** |
| impaired arm(s) | own imp cal: 0.8817 | chronic-donors impaired: **0.7118** |

## Improvement from filtering pool to chronic-only

- VM-LOPO (mixed 47-donor pool, chronic targets): 0.7525
- **Chronic-only donor pool (24 donors): 0.7118**
- Δ from filtering: **-0.0407**

## Pathology contribution (chronic-donors-impaired − chronic-donors-healthy)

- Chronic donors impaired: 0.7118
- Chronic donors healthy: 0.7067
- **Pathology contribution: +0.0051**
- Paired Wilcoxon: p = 0.4040
- Compare to mixed-pool pathology: +0.0431, p = 0.010