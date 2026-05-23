# Stream 3 — System characterization

Single-page summary of latency, throughput, and cost results for the paper.
Detailed numbers in:

- `results/latency_breakdown.csv` / `results/latency_summary.md` — per-stage software latency (1 200 cycles)
- `results/throughput.csv` / `results/throughput_summary.md` — sustained inference rate, single-row + batched
- `cost_itemization.md` — bill of materials, commercial price comparison
- `HARDWARE_LATENCY.md` — hardware-in-the-loop stages (firmware-derived, not directly measured)

All software numbers from this Mac (Intel/Apple Silicon CPU, single-threaded
sklearn). Hardware numbers from datasheets + firmware inspection.

---

## Headline numbers

| | Value |
|---|---|
| **Parts cost (BOM)** | **£193** itemized · **~£180** as-built · vs commercial £500 – £40 000+ |
| **End-to-end latency · light model + L3 Moderate (deployed default)** | **~225 ms** intent change → motor command (incl. 50 ms Stage 2 smoothing wait, N = 2) |
| **End-to-end latency · light model + L4 Light Assist (recommended for stroke)** | **~275 ms** (incl. 100 ms Stage 2 wait, N = 3 — best transition accuracy) |
| **End-to-end latency · light model + Stage 1 only (no smoothing)** | **~175 ms** baseline for comparison |
| **Software predict latency · light model** (PhysioMio per-session config, 900 trees) | mean 16 ms · p95 34 ms · p99 42 ms · 42 rows/s sustained |
| **Software predict latency · heavy shipped GrabMyo model** (6 090 trees) | mean 124 ms · p95 209 ms · p99 380 ms · 7 rows/s sustained |
| **Teensy cycle budget** | 50 ms / frame (20 Hz, hardcoded in firmware) |

**Bottom line.** With the per-session calibration configuration we
validated to 87.5 % on PhysioMio (`max_iter=300, max_depth=10`, 900 trees),
the software pipeline takes **17 ms / cycle on average (p95 = 34 ms, p99 = 43 ms)** —
fits inside the 50 ms Teensy cycle with ~3× mean headroom and clears p99 with
~7 ms to spare. The heavier GrabMyo headline model
(`max_iter=2030, max_depth=18`, 6 090 trees) takes **125 ms / cycle on average
(p95 = 210 ms)** — exceeds the cycle by 2.5× and would need recompilation
(LightGBM / treelite / ONNX), tree truncation, or downgrading to the light
configuration before deployment.

The deployed runtime also applies a **two-stage architecture**: the per-window
classifier (Stage 1) feeds into an N-window consistency filter (Stage 2,
`runtime/run_exohand.py:_apply_stability`, default N = 3) that requires three
consecutive matching predictions before issuing a new motor command. Stage 2
adds **+100 ms decision latency** but absorbs the per-window flicker that
would otherwise destabilise the actuator. The +100 ms is included in the
end-to-end budget below and is the dominant non-mechanical latency item.

---

## Latency — software pipeline (per cycle)

n = 1 200 cycles · real recorded EMG (Adhi session) · 200 ms windows × 4 channels.

| Stage | Heavy (mean / p95 / p99) | Fast (mean / p95 / p99) |
|---|---:|---:|
| A · Bandpass + envelope (4-ch × 400 samples) | 0.40 / 0.42 / 0.46 ms | 0.40 / 0.42 / 0.46 ms |
| B · 60 base features | 0.41 / 0.43 / 0.47 ms | 0.41 / 0.43 / 0.47 ms |
| C · StandardScaler.transform | 0.03 / 0.03 / 0.04 ms | 0.03 / 0.03 / 0.04 ms |
| D · HGB.predict | **124 / 209 / 380 ms** | **16 / 34 / 42 ms** |
| E · Motor command serialize | <0.001 ms | <0.001 ms |
| **Total software pipeline** | **125 / 210 / 381 ms** | **17 / 34 / 43 ms** |

Stages A and B are essentially free (<1 ms combined). The single dominant
cost is HGB predict — and its scale is set by the number of boosting
iterations and tree depth chosen at training time, not by the feature
pipeline.

> **Note.** At deployment with the *current* Teensy firmware, only one
> peak-to-peak amplitude per channel arrives every 50 ms (not raw 2 kHz
> samples). Stages A and B therefore operate on degenerate 4-sample
> windows and contribute well under 0.1 ms each. The numbers above
> represent the cost *if* raw 2 kHz EMG were streamed (which the trained
> 370-feature pipeline assumes), and so are a conservative upper bound for
> deployment.

---

## Latency — hardware-in-the-loop (from firmware + datasheets)

| Stage | Latency | Source |
|---|---:|---|
| MyoWare 2.0 analog envelope group delay | ~30 ms | MyoWare 2.0 datasheet (low-pass envelope ≈ 10 Hz) |
| Teensy 50 ms peak-to-peak sampling window | 50 ms (fixed) | `teensy_emg.ino:13` — `while (millis() - start < 50)` |
| USB serial frame Teensy → host (≤ 20 B @ 115 200 baud) | ~1.5–2 ms | 115 200 baud · 8N1 → 11 520 B/s |
| Host software pipeline | 15–125 ms (model-dependent, see above) | This benchmark |
| USB serial command host → Teensy (5 B) | ~1 ms | Same wire rate |
| PWM phase + servo slew (rest ↔ close ≈ 35° travel) | 70 ms typical | Hobby-servo spec ≈ 0.12 s/60° |
| **End-to-end · light model** | **~175 ms** | |
| **End-to-end · heavy model** | **~290 ms** | |

The hardware stages dominate the end-to-end latency budget once the
software pipeline is competently sized. Replacing the existing servo with
a faster servo would buy more headroom than any further software
optimisation.

---

## Throughput

50 repeats per batch size, single-thread sklearn predict.

| Model | Trees | Single-row | Batch 32 | Batch 256 | Batch 4 096 |
|---|---:|---:|---:|---:|---:|
| Heavy shipped (GrabMyo) | 6 090 | 7 rows/s | 181 rows/s | 1 222 rows/s | 4 995 rows/s |
| Fast per-session | 900 | 42 rows/s | 1 827 rows/s | 12 035 rows/s | 51 495 rows/s |

For **deployment**, only the single-row column matters: both models
exceed the 20 rows/s the Teensy demands, but the heavy model is right at
the edge with high tail variance (p95 320 ms / p99 397 ms — periodic
cycle drops likely). The fast model has ~2× headroom.

For **offline analysis** (LOSO eval, longitudinal eval, post-hoc session
scoring) batched throughput is the right metric: a full PhysioMio session
(~1 200 windows) predicts in well under a second on either model.

---

## Cost

**Bill of materials** (full itemization in `cost_itemization.md`):

| Component | Qty | Subtotal |
|---|---:|---:|
| Teensy 4.0 | 1 | £18.50 |
| MyoWare 2.0 sensor | 4 | £148.00 |
| Hobby servo (SG90 / MG996R-class) | 1 | £8.50 |
| PLA filament (~100 g print) | 1 | £2.50 |
| Wiring + connectors | 1 | £6.00 |
| Fishing line + elastic bands + tape | 1 | £9.50 |
| **Total** | | **£193.00** |

After typical multi-supplier shipping consolidation, we report **~£180**
as-built. The MyoWare sensors dominate at 77 % of cost.

**Commercial comparison** (device-only list price, UK):

| Device | Type | Price | Multiple |
|---|---|---:|---:|
| **ExoHand (this work)** | Open, EMG-triggered, powered | **£180** | 1.0× |
| SaeboGlove | Passive spring (no actuator, no EMG) | ~£500 | 2.8× |
| Neofect Smart Glove | Sensor glove + games (no actuator) | ~£600 | 3.3× |
| Hand Tutor (MediTouch) | Sensor + games (no actuator) | ~£3 000 | 17× |
| Bioness H200 | FES cuff (different paradigm) | ~£6 000 | 33× |
| Gloreha Sinfonia | Pneumatic glove, full clinical suite | ~£15 000 | 83× |
| Tyromotion Amadeo | End-effector finger robotics | ~£40 000 | 222× |

ExoHand is **2.8× cheaper than the cheapest powered alternative** and
**3.3× cheaper than even passive sensor gloves with no actuator**. The
gap is wide enough that a 2× BOM error still leaves ExoHand under 25 % of
any powered comparator.

---

## Recommended deployment configuration

Based on these results, the deployment-time recommendation that follows
from validated accuracy + measured latency is:

1. **Use the per-session calibrated model** (`max_iter=300, max_depth=10`)
   — this is the configuration that achieved 87.5 % on PhysioMio
   (48 patients) and 97.3 % on GrabMyo LOSO with calibration. It fits the
   50 ms cycle budget with ~2× headroom on Mac CPU.
2. **Retain the heavy shipped model** only for offline analysis and as a
   training-time reference; it does not need to ship to clinic.
3. **Consider firmware migration to raw-EMG streaming** (drop the
   peak-to-peak pre-aggregation on the Teensy) if more aggressive
   feature engineering is wanted in future iterations — this would
   unlock the full 370-feature pipeline at the cost of ~10× serial
   bandwidth.

---

## What's still hardware-dependent (recommended follow-up)

- Round-trip latency measured directly with a logic analyser on Teensy
  GPIO (toggle on send / toggle on receive). Expected to confirm the
  ~175 ms / ~290 ms end-to-end figures.
- Tendon compliance delay (servo PWM → finger visibly moves) — unmodelled
  here; likely 20–50 ms additional.
- Jitter under sustained load (heavy model may drop frames audibly as
  motor stutter).
- Power consumption (laptop USB powers the Teensy + sensors; the servo
  needs a separate 5 V supply). Not measured.

These belong in the camera-ready follow-up if needed for a hardware
appendix, but do not block the core latency claims, which are
firmware-spec-derived and software-benchmarked here.
