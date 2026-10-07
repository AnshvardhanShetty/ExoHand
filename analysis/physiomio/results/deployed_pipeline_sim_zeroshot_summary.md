# Deployed-pipeline simulation — zero-shot endpoint

GrabMyo training: 262,773 envelope windows from 129 participant-sessions
Patients evaluated: 48

## Result

- **Zero-shot at simulated 20 Hz P-P pipeline:** 0.2580 patient-mean
  (median 0.2521, std 0.1214)
- **Reference zero-shot at raw 2 kHz (zero_shot_per_session.csv):** 0.188 patient-mean
- **Δ:** +0.0700

## Per-class F1

- rest:  0.1023
- close: 0.2647
- open:  0.3630

## Implication

If this number lands near 0.19, the headline lift `0.19 → 0.86` is fully on the
deployed envelope pipeline. Both endpoints reported on the same regime → paper
describes one system, one pipeline, one set of numbers.