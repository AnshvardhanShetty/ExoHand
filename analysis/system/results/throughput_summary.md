# ExoHand throughput benchmark

50 repeats per batch size · single-thread sklearn predict · Mac CPU.

## Single-row latency (deployment-relevant)

| Model | Trees | Per-row latency | Sustained rate |
|---|---:|---:|---:|
| `heavy_grabmyo_base` | 6090 | **148.69 ms** | 7 rows/s |
| `fast_per_session` | 900 | **23.84 ms** | 42 rows/s |

**Interpretation.** The Teensy delivers one 4-channel EMG frame every 50 ms (20 Hz). The deployed loop can therefore call predict at most **20 rows/s** in steady state — both models can sustain this, but only the fast model has per-call latency comfortably below the 50 ms cycle budget. The heavy model's variance makes it occasionally exceed the cycle budget (see p95/p99 in `latency_breakdown.csv`), causing dropped frames during sustained use.

## Batched throughput (offline analysis)

| Model | Batch | Mean batch latency | Per-row | Rows/s |
|---|---:|---:|---:|---:|
| `heavy_grabmyo_base` | 1 | 148.69 ms | 148.688 ms | 7 |
| `heavy_grabmyo_base` | 32 | 176.80 ms | 5.525 ms | 181 |
| `heavy_grabmyo_base` | 256 | 209.53 ms | 0.818 ms | 1222 |
| `heavy_grabmyo_base` | 4096 | 820.00 ms | 0.200 ms | 4995 |
| `fast_per_session` | 1 | 23.84 ms | 23.838 ms | 42 |
| `fast_per_session` | 32 | 17.52 ms | 0.547 ms | 1827 |
| `fast_per_session` | 256 | 21.27 ms | 0.083 ms | 12035 |
| `fast_per_session` | 4096 | 79.54 ms | 0.019 ms | 51495 |

Batched predict amortises per-call Python overhead. For offline analysis (LOSO evaluation, longitudinal eval, post-hoc session scoring) the throughput scales near-linearly with batch size — a full PhysioMio session (~1 200 windows) predicts in well under a second.