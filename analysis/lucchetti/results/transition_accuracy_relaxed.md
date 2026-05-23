# Lucchetti transition accuracy (relaxed)

n = 28 sessions, 19 subjects, 67 transitions.
Reaction buffer = 500 ms, maint cap = 5500 ms, maint error tolerance = 10%.

## Headline

| | Mean | 95 % CI |
|---|---:|---|
| Patient-level transition acc | **0.6272** | [0.5438, 0.7127] |
| Session-level transition acc | 0.6310 | [0.5417, 0.7232] |
| Session-level raw acc (full stream) | 0.9593 | [0.9425, 0.9731] |

## By arm

| Arm | n | Transition acc (95 % CI) |
|---|---:|---|
| healthy | 18 | 0.6481 [0.5556, 0.7500] |
| impaired | 10 | 0.6000 [0.4250, 0.7750] |

## Per transition type

| from → to | n | Buffer acc | Maint acc | Combined |
|---|---:|---:|---:|---:|
| rest → close | 16 | 0.812 | 0.375 | **0.312** |
| rest → open | 17 | 0.941 | 0.412 | **0.353** |
| close → rest | 15 | 1.000 | 1.000 | **1.000** |
| close → open | 1 | 1.000 | 0.000 | **0.000** |
| open → rest | 18 | 0.944 | 0.944 | **0.944** |