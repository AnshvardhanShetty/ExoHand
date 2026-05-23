# ExoHand bill of materials and cost comparison

The system claim is **~£180 total parts cost** (vs commercial hand-rehab
exoskeletons in the £1,200 – £40,000+ range). This document itemizes the
bill of materials with sources, totals it, and compares against representative
commercial alternatives.

> All prices are quoted in GBP and reflect typical UK online-retailer prices.
> Order any one of these components today and the price may differ ±20%
> depending on supplier, stock, and shipping. The BOM should be re-validated
> against current invoices before final paper submission.

## Bill of materials

| # | Component | Qty | Unit price (GBP) | Subtotal | Typical source |
|---|---|---:|---:|---:|---|
| 1 | **Teensy 4.0** microcontroller (600 MHz Cortex-M7, USB-native) | 1 | £18.50 | £18.50 | PJRC, Pimoroni, The Pi Hut |
| 2 | **MyoWare 2.0** muscle sensor (single-channel analog sEMG with on-board amplification, rectification, smoothing) | 4 | £37.00 | £148.00 | SparkFun, Mouser, RS Components |
| 3 | **SG90 / MG996R-class hobby servo** (180° rotation, ≥1.5 kg·cm torque sufficient for tendon-driven extension) | 1 | £8.50 | £8.50 | Amazon, ePos, hobby suppliers |
| 4 | **PLA filament** for 3D-printed finger segments, palm shell, cable routing (~100 g print) | 1 | £2.50 | £2.50 | Prusament, Bambu, eSun |
| 5 | **Hookup wire + JST connectors + protoboard** for sensor wiring | 1 | £6.00 | £6.00 | RS, ePos |
| 6 | **Fishing line** (braided, 20 lb test) for extension tendons | 1 | £3.00 | £3.00 | Decathlon, Amazon |
| 7 | **Elastic bands / silicone cord** for flexion (passive antagonist) | 1 | £2.50 | £2.50 | Generic |
| 8 | **3M micropore tape / dot electrodes** (consumable, ~30 sessions per pack) | 1 | £4.00 | £4.00 | Pharmacy |
| | **Total parts** | | | **£193.00** | |

After typical multi-component shipping consolidation and student discounts,
the as-built cost we report is **~£180** (rounded). The MyoWare sensors are
the dominant line item at **~77 % of the total** — switching to bare INA126
instrumentation amplifiers + custom analog front-end would cut the BOM to
under £80, at the cost of significantly more board-level work and worse
signal quality without further engineering.

### What this does *not* include

- **Host laptop / PC**: assumed to already be in clinical use. Inference runs
  on commodity CPU (Mac, Windows, Linux). No GPU required.
- **Servo power supply**: a 5 V USB power bank is sufficient for the
  prototype servo; a dedicated bench supply would add ~£15.
- **Therapist tablet / monitor**: the web dashboard runs in any modern browser
  on existing clinic hardware.
- **Labour / 3D printer time**: assumes access to a printer (FDM, 0.4 mm
  nozzle, ~6 h total print across all parts).

## Commercial alternative comparison

Representative powered hand-rehab devices currently marketed for stroke
recovery, with approximate device-only list prices (UK):

| System | Type | Approx. price (device only) | Notes |
|---|---|---:|---|
| **ExoHand (this work)** | Tendon-driven, passive flexion + active extension, EMG-triggered | **£180** | Open-source firmware + ML pipeline; assembled from off-the-shelf parts |
| Gloreha Sinfonia | Pneumatic glove, hand & finger therapy | ~£15,000 | Clinical-grade; includes therapy software suite |
| Tyromotion Amadeo | End-effector finger robotics | ~£40,000 | Hospital robotics; not portable |
| SaeboGlove | Passive spring-assisted glove (no actuator) | ~£500 | Mechanical only — no EMG, no powered assist |
| SaeboMAS | Mobile arm support with spring assistance | ~£1,500 | Arm-level, not hand-level |
| Hand Tutor (MediTouch) | Sensor glove + screen-based games | ~£3,000 | Sensing only; no powered assist |
| Neofect Smart Glove | Sensor glove + games | ~£600 | Sensing only; no powered assist |
| Bioness H200 | Functional electrical stimulation (FES) cuff | ~£6,000 | Stimulates muscles directly; different paradigm |

**Order of magnitude:** ExoHand is **2.5–220× cheaper** than the powered
commercial alternatives, and **3× cheaper than even the passive (non-powered)
SaeboGlove**. The cost gap is wide enough that even a 2× BOM error
(£180 → £360) still leaves ExoHand at <25 % of the cheapest powered option.

### Why so much cheaper?

1. **Off-the-shelf microcontroller + sensors**, not custom analog ASICs.
2. **Tendon-driven mechanism** with a single hobby servo replaces multi-axis
   robotic finger linkages.
3. **3D-printed enclosure** instead of injection-moulded medical plastics.
4. **No FDA/CE certification or clinical-trial overhead** is included — this
   is the BOM of a research prototype, not a regulated medical device. A
   certified version would carry significant additional manufacturing, QA,
   and regulatory costs that we do not include here.

The point of the cost comparison is **not** that ExoHand is
clinically equivalent to certified devices, but that the *underlying
hardware platform* for EMG-triggered hand assistance can be built at a
fraction of the cost typically assumed in this space — which is the
relevant comparison if the goal is access in low-resource settings.
