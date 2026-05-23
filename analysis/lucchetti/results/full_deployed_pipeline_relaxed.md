# Lucchetti full deployed pipeline — relaxed

Strict = 0% maint error tolerance; relaxed = 10% maint error tolerance. Reaction buffer 500 ms, maint cap 5500 ms.

| Config | Level | Label | N | α | enter | exit | floor | cd(ms) | Patient transition acc | Raw full-stream | +Latency |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| raw | 0 | Raw HGB (no pipeline) | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | **0.627** [0.544, 0.713] | 0.959 | +0 ms |
| L1 | 1 | Max Assist | 1 | 0.70 | 0.30 | 0.20 | 0.15 | 800 | **0.542** [0.450, 0.634] | 0.955 | +0 ms |
| L2 | 2 | High Assist | 1 | 0.60 | 0.40 | 0.25 | 0.25 | 600 | **0.553** [0.441, 0.658] | 0.958 | +0 ms |
| L3 ★ | 3 | Moderate Assist | 2 | 0.50 | 0.50 | 0.30 | 0.35 | 500 | **0.605** [0.526, 0.684] | 0.958 | +50 ms |
| L4 | 4 | Light Assist | 3 | 0.40 | 0.60 | 0.35 | 0.45 | 400 | **0.588** [0.518, 0.660] | 0.960 | +100 ms |
| L5 | 5 | Minimal Assist | 3 | 0.30 | 0.70 | 0.40 | 0.55 | 300 | **0.588** [0.518, 0.660] | 0.960 | +100 ms |

★ = deployed default.