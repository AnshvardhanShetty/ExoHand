# Lucchetti transition accuracy (strict)

n = 28 sessions, 19 subjects, 67 transitions.
Reaction buffer = 500 ms, maint cap = 5500 ms, maint error tolerance = 0%.

## Headline

| | Mean | 95 % CI |
|---|---:|---|
| Patient-level transition acc | **0.6009** | [0.4978, 0.6996] |
| Session-level transition acc | 0.5952 | [0.4881, 0.6964] |
| Session-level raw acc (full stream) | 0.9593 | [0.9425, 0.9731] |

## By arm

| Arm | n | Transition acc (95 % CI) |
|---|---:|---|
| healthy | 18 | 0.6343 [0.5323, 0.7454] |
| impaired | 10 | 0.5250 [0.3250, 0.7250] |

## Per transition type

| from → to | n | Buffer acc | Maint acc | Combined |
|---|---:|---:|---:|---:|
| rest → close | 16 | 0.812 | 0.375 | **0.312** |
| rest → open | 17 | 0.941 | 0.353 | **0.294** |
| close → rest | 15 | 1.000 | 0.867 | **0.867** |
| close → open | 1 | 1.000 | 0.000 | **0.000** |
| open → rest | 18 | 0.944 | 0.944 | **0.944** |