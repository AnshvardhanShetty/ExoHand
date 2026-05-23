# ExoHand software-stage latency

**n = 1200 inference cycles · source: real recorded EMG (Adhi session, 2026-02-20_18-51)** · host: Mac (this benchmark machine)

## Per-stage latency (software only)

| Stage | Description | Latency |
|---|---|---|
| **A** | Bandpass + envelope (4-ch × 200 ms window) | 0.395 ms ± 0.018  (p50 0.392 / p95 0.420 / p99 0.456 / max 0.639) |
| **B** | 60 base features (15 × 4 channels) | 0.410 ms ± 0.016  (p50 0.409 / p95 0.434 / p99 0.473 / max 0.564) |
| **C** | StandardScaler.transform on 370-d row | 0.030 ms ± 0.002  (p50 0.030 / p95 0.032 / p99 0.038 / max 0.048) |
| **D-heavy** | HGB predict · shipped model (6090 trees, max_depth=18) | 124.168 ms ± 61.959  (p50 105.839 / p95 209.449 / p99 379.809 / max 867.320) |
| **D-fast** | HGB predict · per-session model (900 trees, max_depth=10) | 15.947 ms ± 6.460  (p50 13.852 / p95 33.604 / p99 41.830 / max 61.786) |
| **E** | Motor command serialization | 0.001 ms ± 0.000  (p50 0.001 / p95 0.001 / p99 0.001 / max 0.004) |
| **Total · heavy model** | A + B + C + D_heavy + E | **125.004 ms ± 61.959  (p50 106.668 / p95 210.279 / p99 380.638 / max 868.156)** |
| **Total · fast model** | A + B + C + D_fast + E | **16.783 ms ± 6.461  (p50 14.692 / p95 34.441 / p99 42.704 / max 62.624)** |

## Notes

- This benchmark covers the software path: from a complete 200 ms 4-channel raw EMG window to a serialized motor command byte string ready for serial write.
- **Stages NOT covered here** (because they need a connected Teensy + servo to measure):
  - EMG acquisition window on the Teensy (50 ms, hardcoded sample budget)
  - USB serial transfer Teensy → host at 115 200 baud (~1 ms per peak-to-peak frame)
  - Host → Teensy motor command transfer (~1 ms)
  - Servo response and tendon-driven actuation (~50–100 ms typical for hobby servos)
- See `HARDWARE_LATENCY.md` for the hardware-in-the-loop stage estimates.
- 370-feature engineering (per-participant z-score, temporal lags, cross-channel ratios) is not benchmarked per-cycle here because it is currently implemented as a batch pandas operation; the deployed runtime maintains incremental state for these features (see `runtime/run_exohand.py` `_init_adapted_state`) and the per-window cost is dominated by the same bandpass + envelope + base-feature pipeline benchmarked above.