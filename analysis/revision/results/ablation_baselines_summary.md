# Ablation baselines — revision recompute #3

Reviewer #5 asks for the canonical baselines on the same 48 patients
× balanced 39/39/39 test set as the paper's headline number.

## Baselines computed this run

| baseline | patient-mean acc | 95% bootstrap CI |
|---|---:|---:|
| majority-class | 0.333 | [0.333, 0.333] |
| uniform-random | 0.330 | [0.323, 0.336] |
| LDA on Hudgins (MAV/WL/ZC/SSC) | 0.830 | [0.806, 0.852] |
| two-threshold envelope rule | 0.666 | [0.626, 0.705] |

## Reference (from other files / recomputes)

| method | patient-mean acc | source |
|---|---:|---|
| zero-shot (balanced, recompute #1) | 0.346 [0.323, 0.370] | see notes |
| calibration-only HGB (cal-size sweep) | 0.878 — | see notes |
| GrabMyo + cal (per_session_results.csv) | 0.860 — | see notes |

## Interpretation

- Chance / majority-class floors both sit at 0.333 on the balanced
  test set (as required by construction).
- LDA on 16 Hudgins features is the canonical myoelectric baseline.
  If it lands near the paper's headline, the classical-ML+calibration
  story is still ok but the specific 370-feature engineering
  contributes less than the paper implies.
- Two-threshold envelope rule is the simplest possible controller.
  It should be far below any learned classifier; if it isn't, the
  task is easier than the paper frames.