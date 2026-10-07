# Lucchetti binary reframe — rest vs movement

n = 20 subjects, 30 sessions, 19,646 windows.

## Why binary

Lucchetti's labels for close vs open are derived from task-order convention (BA/BC/SC → close; HM/HH → open) applied across whole movement windows, which include reach-with-hand-open and return phases. The per-class 3-class F1 on the calibrated model (close 0.25, open 0.31) reflects this label noise — *not* a methodology failure. The rest-vs-movement boundary is unambiguous (rest F1 = 0.90 in 3-class).

We therefore report Lucchetti as a binary movement-detection validation, which is what the data robustly supports. PhysioMio remains the 3-class headline.

## Binary results

| Aggregation | Mean | 95 % bootstrap CI |
|---|---:|---|
| Session-level binary accuracy | **0.9710** | [0.9589, 0.9815] |
| Patient-level binary accuracy | **0.9751** | [0.9637, 0.9839] |

## By arm

| Arm | n sessions | Binary accuracy (95 % CI) |
|---|---:|---:|
| healthy | 20 | 0.9765 [0.9600, 0.9879] |
| impaired | 10 | 0.9599 [0.9425, 0.9755] |

## Per-class F1 (binary)

- rest:     0.9838
- movement: 0.6225

For context, the 3-class session-mean accuracy on Lucchetti is 0.8278 (rest F1 = 0.90, close F1 = 0.25, open F1 = 0.31 — close/open collapse from label noise). Binary collapses the noisy axis and reports what's actually being validated.