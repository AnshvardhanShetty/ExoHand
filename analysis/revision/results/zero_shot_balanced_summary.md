# Zero-shot on balanced test set — revision recompute #1

This is the honest side-by-side of the paper's headline lift, now
computed on the SAME balanced 39/39/39 test set that the calibrated
arm uses. The submitted-paper zero-shot number (0.19) was on the
class-imbalanced all-windows evaluation.

## Headline numbers

| cohort | zero-shot (balanced) | calibrated | lift | Cliff's δ | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| PhysioMio (n=48) | **0.346** [0.323, 0.370] | **0.860** [0.842, 0.880] | **+0.514** [+0.485, +0.545] | **+1.000** (48/48) | 3.55e-15 |
| Lucchetti (n=10) | **0.235** [0.104, 0.368] | **0.795** [0.710, 0.877] | **+0.560** [+0.420, +0.696] | **+1.000** (10/10) | 9.77e-04 |

## Interpretation

- Zero-shot on the balanced test set sits at chance (0.33), not at
  0.19. The submitted paper's 0.19 was an artefact of evaluating on
  class-imbalanced all-windows data where the model biased toward
  predicting `open` (minority class, 8% prior) scored below the
  constant-predict-majority floor (0.83).

- The apples-to-apples lift is **+0.51** patient-mean on PhysioMio
  and roughly **+0.6** on Lucchetti. Still large, still every-patient
  improves — but the abstract's `+0.67` and the `0.19 → 0.86` framing
  need updating.

- Cliff's δ and Wilcoxon p remain strongly positive on the paired
  comparison. This is the finding, cleanly framed.