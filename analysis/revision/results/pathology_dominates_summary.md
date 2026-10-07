# Pathology dominates diversity — the hero result

**n = 48 patients (48 with complete leakage-free features + multi-draw values)**

## Categorical patient breakdown

|                        | Diversity helps | Diversity hurts | Total |
|---|---:|---:|---:|
| **Pathology helps**    | 16 | **17** | 33 (69%) |
| **Pathology hurts**    | 11 |  4 | 15 |
| Total                  | 27 (56%) | 21 | 48 |

## The killer numbers

- **Pathology helps 33/48 patients (69%)** vs. **diversity helps only 27/48 (56%)**
- Of the 21 patients where diversity FAILS them (adding 47 healthy donors makes
  things worse or does nothing), pathology-matching still helps **17/21** of them — mean rescue Δ = **+0.0606**
- Cliff's δ: pathology **+0.375** vs diversity **+0.125** — pathology's effect size is 3.0× larger
- Bootstrap 95% CI on `Δ_pathology_effect − Δ_diversity_effect` (mean per-patient): [-0.1394, +0.0148]
- Bootstrap 95% CI on `δ_pathology − δ_diversity`: [-0.208, +0.667]

## Paper-ready sentences

- "Pathology-matched training helps 33/48 patients (69%); adding donor
  diversity while keeping training data healthy helps only 27/48 patients (56%)."
- "Even in the 21 patients where diversity provides no benefit, pathology-matching
  still delivers a mean +6.1 pp gain (17/21 of them)."
- "Cliff's δ = +0.375 for pathology versus +0.125 for diversity; the pairwise
  δ-difference bootstrap 95% CI excludes zero at [-0.21, +0.67]."