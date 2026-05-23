# ExoHand hardware-in-the-loop latency

This document estimates the **hardware-bound** stages of the end-to-end ExoHand
control loop — the parts that require a connected Teensy + servo to measure
and are therefore *not* covered by the software microbenchmark in
`latency_benchmark.py`.

All numbers derive from inspection of `teensy_emg/teensy_emg.ino` and
`exohand_combined/exohand_combined.ino`, the 115 200 baud USB serial
parameters, and published servo timing specifications. They should be
replaced with direct hardware-in-the-loop measurements before final paper
submission.

## End-to-end pipeline

```
 ┌─────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌──────────┐
 │  EMG    │──> │ Teensy   │──> │ USB serial  │──> │ Host     │──> │ USB ser. │──> Servo
 │  cuff   │    │ sample   │    │ EMG frame   │    │ predict  │    │ A### cmd │
 └─────────┘    │  +p2p    │    │ → host      │    │ (sw)     │    │ → Teensy │
                └──────────┘    └─────────────┘    └──────────┘    └──────────┘
                   ~50 ms          ~1.5 ms          0.5–125 ms       ~1 ms

                                                                      ┌──────────┐
                                                              ──>     │ servo    │
                                                                      │ slew     │
                                                                      └──────────┘
                                                                       60–250 ms
```

## Per-stage breakdown

### Stage 0: Surface EMG → MyoWare 2.0 analog output

- **MyoWare 2.0 internal pipeline:** instrumentation amplifier → high-pass →
  rectifier → low-pass envelope. Group delay of the envelope filter is the
  dominant contribution.
- **Manufacturer-published envelope delay:** ≈ **30 ms** (from MyoWare 2.0
  datasheet, low-pass cutoff at ~10 Hz applied to rectified signal).
- This is *analog* latency before any sample is taken by the Teensy.

### Stage 1: Teensy 50 ms sampling window

- Hardcoded `while (millis() - start < 50)` loop in both firmware variants.
- Inside the window the Teensy runs `analogRead()` on 4 channels in a tight
  loop, tracking only per-channel max and min — a peak-to-peak amplitude per
  channel is emitted at the end of the window.
- The Teensy 4.0 `analogRead()` default conversion time is ~9.6 µs per
  channel; 4 channels × ~104 reads each over 50 ms ≈ 420 reads per channel,
  well within the budget.
- **Window latency contribution: 50 ms** by design. This is the dominant
  fixed cost of the current firmware: the host cannot make a new prediction
  more often than every 50 ms regardless of how fast the software pipeline
  runs.

### Stage 2: Teensy → host USB serial transfer

- Frame format: `"123\t456\t789\t012\n"` — peak-to-peak amplitudes per
  channel, tab-separated, newline-terminated. Max realistic size: 16 bytes
  (4 channels × 4-digit value + 3 tabs + newline = 20 bytes worst case).
- Wire rate: 115 200 baud, 8N1 → 11 520 bytes/s → **~1.7 ms per 20-byte
  frame**.
- Teensy 4.0 USB-CDC is a USB 2.0 full-speed device; in practice host-side
  read latency through `pyserial` adds another ~0.5 ms of OS scheduling
  jitter.
- **Per-frame transfer: ~1.5–2 ms.**

### Stage 3: Host software inference

- Measured in `latency_benchmark.py` (see `results/latency_summary.md`).
- Two model configurations are characterised:
  - **Heavy shipped GrabMyo model** (`improved_hgb_model.pkl`,
    `max_iter=2030`, `max_depth=18`, 6 090 trees): predict alone takes
    **mean 124 ms, p50 98 ms, p95 320 ms** on Mac CPU.
  - **Light per-session model** (`max_iter=300`, `max_depth=10`, the config
    used to obtain the PhysioMio 87.5 % per-session result): predict is
    expected to be **~10–20 ms** based on tree-count ratio (≈900 vs 6 090
    trees); benchmarked separately in `latency_breakdown.csv`.
- Stages A (bandpass + envelope) and B (60 base features) add a combined
  **~0.8 ms** when applied to a 200 ms / 2 kHz raw EMG window. With the
  *current* 20 Hz peak-to-peak firmware, the bandpass filter window
  collapses to 4 samples and these stages contribute well under 0.1 ms — they
  are essentially free.

### Stage 4: Host → Teensy motor command transfer

- Frame format: `"A###\n"` — 5 bytes.
- 115 200 baud → **~0.4 ms wire time**, plus ~0.5 ms OS scheduling jitter
  through `pyserial.write` → **~1 ms total**.

### Stage 5: Servo command parsing + actuation

- Firmware parses `A###\n` via byte-at-a-time `Serial.read()` in
  `parseSerial()`; parse itself is microseconds.
- `Servo.write(angle)` updates the PWM duty cycle on the next 20 ms PWM
  frame — so **PWM update latency is 0 – 20 ms** depending on phase.
- Servo mechanical slew rate (hobby servo, ~0.12 s/60°): for the
  rest → close transition (145° → 180° = 35°), slew time ≈ **70 ms** of
  motor travel.
- For rest → open (145° → 110° = 35°), slew time ≈ **70 ms**.
- Full open → full close (110° → 180° = 70°) is the worst case at **~140 ms**
  of motor travel — but this transition is rare in normal use because the
  state machine usually passes through rest.

## End-to-end latency budget

Per cycle (intent change → finger motion begins to be visible). The
deployed system applies an **N-window stability filter** (`runtime/run_exohand.py:_apply_stability`,
N = 3 at the deployed Light / Minimal Assist profile) before issuing a
new motor command. This absorbs Stage-1 maintenance flicker (see
`analysis/physiomio/REACTEMG_COMPARISON.md` §3) at the cost of (N − 1)
× 50 ms of decision latency on top of the raw classifier output.

| Stage | Light-model + L3 deployed default | Light-model + L4 Light Assist | Light-model + L1 Max Assist | Light-model + Stage 1 raw |
|---|---:|---:|---:|---:|
| MyoWare analog envelope delay | 30 ms | 30 ms | 30 ms | 30 ms |
| Teensy sampling window | 50 ms | 50 ms | 50 ms | 50 ms |
| Teensy → host serial | 2 ms | 2 ms | 2 ms | 2 ms |
| Host software pipeline (filter + features + predict + serialize) | ~12 ms | ~12 ms | ~12 ms | ~12 ms |
| **Stage 2 stability wait ((N-1) × 50 ms)** | **50 ms** | **100 ms** | **0 ms** | **0 ms** |
| Host → Teensy serial | 1 ms | 1 ms | 1 ms | 1 ms |
| PWM phase | 10 ms (avg) | 10 ms (avg) | 10 ms (avg) | 10 ms (avg) |
| Servo slew (rest → close or rest → open) | 70 ms | 70 ms | 70 ms | 70 ms |
| **End-to-end** | **~225 ms** | **~275 ms** | **~175 ms** | **~175 ms** |

The Stage 2 wait depends on which assist profile is selected. The
default (L3 Moderate Assist) adds 50 ms; the recommended-for-stroke L4
Light Assist adds 100 ms (and delivers higher transition accuracy — see
`analysis/physiomio/REACTEMG_COMPARISON.md` §3). L1 / L2 (Max / High
Assist for the most severely impaired patients) have N = 1, no stability
wait, and rely on EMA + hysteresis for flicker control. The Stage 1
column shows the latency budget if Stage 2 is entirely removed (matches
a hypothetical "raw classifier output" deployment, which our paper does
not recommend).

The light per-session model fits comfortably within a single 50 ms Teensy
cycle on the host side — the dominant latency is *physical* (sensor envelope
+ servo slew), not computational. The heavy GrabMyo base model exceeds the
50 ms cycle budget on Mac CPU; in deployment it would either need to be
recompiled to a faster runtime (LightGBM, treelite, ONNX), have its tree
count truncated, or be downgraded to the light configuration that we
already validated to >85 % accuracy on PhysioMio.

## What still needs hardware-in-the-loop measurement

- **Round-trip latency** measured by toggling a Teensy GPIO when a command
  is received and the same GPIO when the corresponding intent was emitted
  by the host — captured with a scope or logic analyser.
- **PWM-to-tendon-tension delay** (the elastic tendon system has some
  compliance that may add 20–50 ms before the finger begins to visibly
  move).
- **Jitter under load** (heavy model under sustained inference may cause
  cycle drops, audible as motor stutter).

These measurements need the actual hardware setup; they are recommended as a
short follow-up before paper submission but do not block the core latency
claims here, which are firmware-spec-derived and software-benchmarked.
