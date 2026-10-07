# T1 hardware post-hoc deployed accuracy (Option 1)

n = 48 patients. Per-patient stratified 80/20 split on the 972 windows
of Teensy P-P envelopes; HGB with class_weight='balanced' on 20-dim features
(mean/std/min/max/last of 200ms aggregate + 4-channel raw P-P values).

## Headline

- Mean raw accuracy:         **0.9234** (95% CI [0.9175, 0.9293])
- Mean balanced accuracy:    **0.7508** (95% CI [0.7289, 0.7742])
- Mean majority-class baseline: 0.8359
- Mean lift over baseline:   +0.0875 pp

## Per-patient distribution

- Raw accuracy: min=0.876, median=0.923, max=0.979
- Balanced accuracy: min=0.598, median=0.738, max=0.917
- Patients with raw_acc > majority baseline: 48/48
- Patients with balanced_acc > 0.5 (above uniform 3-class chance 0.333): 48/48

## Interpretation

- Balanced accuracy above chance (0.333) confirms the Teensy hardware output carries discriminative signal for 3-class intent.
- Raw accuracy near majority baseline (0.83) is expected because the T1 gesture blocks are class-imbalanced (10 close-like : 1 rest : 1 open).
- Direct comparison to the paper's headline numbers (own-cal 0.896, etc.) is NOT valid — those use balanced 39/39/39 test sets, this uses the raw gesture-block distribution.
- This analysis establishes 'hardware works end-to-end on real stroke data', not 'hardware exactly reproduces paper accuracy'.