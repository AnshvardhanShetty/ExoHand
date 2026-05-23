# Deployed-configuration transition accuracy (strict (ReactEMG-faithful, 0% maint error))

Per-window predictions in `per_window_predictions.parquet` were passed through the runtime's N-window consistency filter (`runtime/run_exohand.py:_apply_stability`, `stability_required` parameter from `runtime/assist_profile.py`). The deployed assist profile uses **N=3** (Light / Minimal Assist), corresponding to 100 ms of added decision latency before a new motor command is issued.

Common parameters across all N: reaction buffer = 10 windows (500 ms), maintenance cap = 110 windows (5500 ms), maintenance error tolerance = 0%.

## Sweep results

| N | Profile | Patient-level transition acc (95 % CI) | Raw acc (full-stream, post-filter) | Added latency |
|---:|---|---|---:|---:|
| **1** | Max / High Assist (Levels 1-2) | **0.277** [0.223, 0.333] | 0.909 | +0 ms |
| **2** | Moderate Assist (Level 3) | **0.389** [0.341, 0.443] | 0.914 | +50 ms |
| **3** | Light / Minimal Assist (Levels 4-5) — deployed default | **0.464** [0.408, 0.523] | 0.917 | +100 ms |
| **5** | (sensitivity only — not in any profile) | **0.575** [0.520, 0.629] | 0.924 | +200 ms |
| **10** | (sensitivity only — not in any profile) | **0.686** [0.641, 0.729] | 0.925 | +450 ms |

## Comparison to raw classifier output (no stability filter)

Without the stability filter (N = 1, equivalent to `transition_accuracy.py` baseline): patient mean **0.277**.

With the deployed N = 3 stability filter: patient mean **0.464**, a **+18.6 pp** absolute improvement at the cost of +100 ms decision latency.

ReactEMG Stroke's reported transition accuracy (their best: 0.61 with LoRA full fine-tuning) is measured on their raw classifier output. The deployed configuration of our system applies additional stability filtering before motor command — a documented difference in evaluation framing.