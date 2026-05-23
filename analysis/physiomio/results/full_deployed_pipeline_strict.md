# Full deployed pipeline — transition accuracy (strict (ReactEMG-faithful, 0% maint error))

Per-window probabilities (`per_window_probas.parquet`) passed through the full deployed runtime pipeline (EMA → argmax → stability → cooldown → hysteresis → confidence floor), exactly as in `runtime/run_exohand.py:_apply_stability + _smooth_proba`, with parameters from `runtime/assist_profile.py` for each of the five assist levels.

Common: reaction buffer = 10 win (500 ms), maint cap = 110 win (5500 ms), maint error tolerance = 0%.

## Sweep across deployed profiles

| Config | Level | Label | N | α | enter | exit | floor | cd(ms) | Patient transition acc (95 % CI) | Full-stream raw acc | +Latency |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| raw | 0 | Raw HGB (no pipeline) | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | **0.277** [0.223, 0.333] | 0.909 | +0 ms |
| L1 | 1 | Max Assist | 1 | 0.70 | 0.30 | 0.20 | 0.15 | 800 | **0.444** [0.391, 0.500] | 0.874 | +0 ms |
| L2 | 2 | High Assist | 1 | 0.60 | 0.40 | 0.25 | 0.25 | 600 | **0.405** [0.354, 0.463] | 0.876 | +0 ms |
| L3 ★ | 3 | Moderate Assist | 2 | 0.50 | 0.50 | 0.30 | 0.35 | 500 | **0.430** [0.372, 0.488] | 0.889 | +50 ms |
| L4 | 4 | Light Assist | 3 | 0.40 | 0.60 | 0.35 | 0.45 | 400 | **0.498** [0.445, 0.552] | 0.909 | +100 ms |
| L5 | 5 | Minimal Assist | 3 | 0.30 | 0.70 | 0.40 | 0.55 | 300 | **0.360** [0.307, 0.415] | 0.905 | +100 ms |

★ = deployed default (`runtime/run_exohand.py --assist-level 3`).

## Caveats

- Adaptive gain (per-channel signal-strength threshold scaling, `_apply_adaptive_gain`) is not replayed offline because we don't have the gain history. In live deployment, weak-signal patients (e.g. severely impaired) get hysteresis thresholds scaled by 0.4-0.7×, which would make transitions *easier* to accept and likely raise these numbers. The values here are therefore a slight under-estimate of true deployed transition accuracy on the most-impaired sub-cohort.
- Adaptive confidence floor reductions (same source) are similarly not replayed.
- All 5 profiles are evaluated; the deployed default in `run_exohand.py` is **Level 3** (Moderate Assist).