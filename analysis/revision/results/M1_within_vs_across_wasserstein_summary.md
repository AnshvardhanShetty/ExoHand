# M1 — within-patient vs across-patient Wasserstein distance

n = 48 patients with both healthy_01 and impaired_01 sessions.

For each patient:
- **d_within**: mean W₁(own_healthy features, own_impaired features) over 370 features
- **d_across**: mean W₁(own_impaired features, pooled_other_patients_impaired features)

## Headline

| Distance | Mean | Median |
|---|---:|---:|
| d_within (own healthy ↔ own impaired) | 0.7363 | 0.7387 |
| d_across (own impaired ↔ others' impaired) | 0.3321 | 0.2936 |

**Paired Wilcoxon (d_within > d_across): p = 3.2117e-12**
**Cliff's δ: +0.833**
Patients where within > across: 44 / 48

## Interpretation

If d_within > d_across (p<0.05, δ>0.2), the healthy-vs-impaired distance within
one person is larger than the impaired-vs-impaired distance across people. That's
the geometric explanation for why cross-arm PO underperforms LOPO: pathology
puts stroke EMG in a distinct region of feature space that healthy data doesn't
sample, and this region is more shared across patients than between arms of one.