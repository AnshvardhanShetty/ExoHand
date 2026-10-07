# ExoHand

**Accepted to BrainBodyFM at NeurIPS 2026.**

EMG-controlled hand exoskeleton with real-time intent classification, adaptive motor assistance, and a full-stack rehabilitation platform.

[![Watch the demo](https://img.youtube.com/vi/RMq31iIWcPk/maxresdefault.jpg)](https://www.youtube.com/watch?v=RMq31iIWcPk)

**[Watch Demo](https://www.youtube.com/watch?v=RMq31iIWcPk)**

## Overview

ExoHand is a complete EMG-to-actuation system for hand rehabilitation. Surface EMG signals from the forearm are acquired via a Teensy 4.0 microcontroller, classified in real time using a gradient boosting model, and translated into servo commands that drive a 3D-printed exoskeleton hand. A therapist-facing web platform manages patients, tracks progress, and runs structured exercise sessions.

The system achieves **97.3% three-class accuracy** (close / open / rest) with a per-user calibration protocol, measured over 43 leave-one-subject-out folds on the GrabMyo dataset (95% bootstrap CI: [96.7%, 97.9%]).

## Research & Paper

The ExoHand paper has been accepted to **BrainBodyFM at NeurIPS 2026**. The repository includes the research analyses, evaluation results, figures, and supporting paper materials.

- [Paper materials](paper/README.md): research documentation, figures, appendix, and hardware verification handoffs.
- [Canonical results](paper/FINAL_NUMBERS.md): results and provenance for the paper.
- [Analysis and evaluation](analysis/README.md): scripts and results for calibration, cross-subject transfer, longitudinal evaluation, and deployment checks.

## System Architecture

```
EMG Sensors → Teensy 4.0 → Serial USB → Python Runtime → Motor Commands → Servo
                                              ↕
                                        Node.js Server ↔ React Dashboard
                                              ↕
                                        SQLite Database
```

**Real-time loop:** Read 4-channel EMG at 20 Hz → extract 370 features per window → classify intent → send single-character motor command (`c`/`o`/`r`) — all within 50ms.

## ML Pipeline

### Training Data
Trained on the [GrabMyo dataset](https://physionet.org/content/grabmyo/) — 43 participants, 1.14M samples at 2 kHz, reduced to 4 optimally selected channels targeting flexor and extensor digitorum muscles. Raw session data should be downloaded from PhysioNet and placed in `grabmyo/Session1/`, `Session2/`, `Session3/`.

### Feature Engineering (370 features)
- **Per-channel features** (6 × 4 channels): RMS, MAV, waveform length, zero crossings, slope sign changes, envelope RMS
- **Temporal features**: Lag values, deltas (velocity), acceleration, rolling means — captures how EMG signals evolve over time
- **Cross-channel interactions**: Flexor/extensor ratios, pairwise differences, and their temporal derivatives
- **Per-participant normalization**: Z-score normalization removes inter-subject amplitude variation

### Model
HistGradientBoostingClassifier (scikit-learn) with class balancing, participant-level train/test splits, and EMG-specific data augmentation (gain variation, bias shifts, channel dropout, noise injection).

### Accuracy

Full results from leave-one-subject-out evaluation (n=43, seed=42, 2000 bootstrap resamples):

| Configuration | Accuracy (mean, 95% CI) | Macro-F1 (mean, 95% CI) |
|---|---|---|
| Cross-subject baseline (no calibration) | 94.6% [93.3%, 95.8%] | 0.946 [0.932, 0.958] |
| + Per-user calibration (60 s, 1200 windows, 100× weight) | **97.3% [96.7%, 97.9%]** | **0.972 [0.965, 0.979]** |
| Δ from calibration (paired) | **+2.7% [+2.0%, +3.6%]** | — |

Cross-subject standard deviation reduces from **±4.2% [2.7%, 5.9%]** to **±2.1% [1.4%, 2.6%]** — a 2.05× collapse [1.58×, 2.51×]. Paired Wilcoxon signed-rank: **p ≈ 10⁻¹³**, Cliff's δ = +1.0 (every fold improves with calibration).

The mean improvement of +2.7% is distribution-dependent: easy subjects (no-cal > 97%) get near-zero improvement (already at ceiling), while the hardest fold (participant2, no-cal 77.4%) gains +13.7%. **Calibration consistently helps the subjects who need it most**, even when the average effect is modest.

Additional configurations not yet reproduced under this LOSO protocol: instantaneous-features-only baseline, temporal-features-only baseline, binary (movement vs rest) classifier, per-class precision/recall breakdown. Ablations are tracked under Stream 3 of the paper plan.

### Patient Calibration

Two protocols, used at different points in the patient's journey:

- **Initial calibration (~6 minutes, once per patient):** rest baseline, familiarization, sustained holds, quick contractions, and variable-effort phases, with onset trimming and outlier rejection. Run when a patient is registered.
- **Per-session re-cal (~30 seconds, every session):** the patient holds rest, close, and open for ~5 s each with ~5 s gaps. Refreshes the decision boundary against day-to-day electrode placement and skin-impedance drift. Without per-session re-cal, accuracy degrades from ~0.87 to ~0.50 within a few sessions on PhysioMio's longitudinal data.

The **LOSO evaluation protocol** that produced the 97.3% headline uses 1200 cued windows (~60 s at 50 ms stride) weighted 100× against the GrabMyo base — a deliberately larger calibration than the deployed 30 s re-cal, since the eval target is the largest accuracy lift that a per-user fine-tune produces in principle. Variance collapses 2.05× (±4.2% → ±2.1%) and mean accuracy rises 94.6% → 97.3%.

### Why the protocol assumes a population backbone

The cal=3 collapse is not a methodological footnote; it is the constraint the entire protocol is designed around. A 30-second per-session re-cal is only viable because the population backbone makes 25 windows/gesture sufficient. Without it, deployable cal sizes return classifiers that collapse to majority-class prediction, and the protocol reverts to a 6-minute enrollment per session — operationally non-viable for stroke rehabilitation. **The backbone is therefore not a performance enhancement at the operating point (where it ties patient-only on raw accuracy) but the precondition for the operational protocol to exist.**

We measured this with a cal-size sweep over all 48 PhysioMio patients × {3, 6, 12, 24, 36} cued windows/gesture × {patient-only HGB, GrabMyo+cal HGB}:

| windows/gesture | patient-only (same-session) | GrabMyo + cal (same-session) |
|---:|---:|---:|
| **3** | **0.333 (single-class collapse — 12/12 patients predict `close` for 100% of test windows)** | **0.716** |
| 6 | 0.774 | 0.768 |
| 12 | 0.837 | 0.827 |
| 24 | 0.870 | 0.870 |
| 36 | 0.878 | 0.884 |

The two ends of the curve carry the argument:

- **Left end (cal=3, 1.5 s/gesture).** Patient-only predicts a single class for every test window in every patient sampled. Mechanism: 3 windows × 12 trials × 75% overlap → ~1 independent sample per class, against 370 features — underdetermined by construction, confirmed empirically with single-column confusion matrices. GrabMyo + cal at the same budget reaches 0.72. The protocol assumes a backbone because at deployable cal sizes this is what happens without one.
- **Right end (cal=24–36, the protocol's operating budget).** Patient-only and GrabMyo+cal converge to ~0.87 — the prior has handed off to the patient data, exactly as the cal-weighting (×100 vs base ×1) is intended to produce. Steady-state accuracy at the operating point no longer requires the prior; the protocol that gets a 30 s re-cal to the operating point does.

Even the 6-minute initial calibration provides only ~30–45 independent samples per class (~10–15 distinct trials × 75%-overlapping windows), below the feature-count threshold. GrabMyo carries the cross-subject variability one patient's data cannot generate, which is what makes 30 s per-session re-cal a viable design rather than a 6-minute fresh enrollment every time the device is put on.

## Web Platform

### Server (Node.js + Express + TypeScript)
- REST API for patient management, session tracking, therapist dashboard
- WebSocket streaming for real-time EMG visualization
- Serial port bridge to Teensy hardware
- SQLite database for session history and patient profiles
- Calibration endpoint that triggers the Python calibration pipeline

### Client (React + Vite + TypeScript)
- Real-time exercise tracking with rep counting and assist-level control
- 3D hand model visualization (Three.js / React Three Fiber)
- Patient progress dashboard with session history
- Therapist management interface with outcome scoring

### Assist-as-Needed Profiles
Five graduated profiles for stroke rehabilitation, from maximum assistance (Level 1: low confidence threshold, high movement bias, long cooldowns) to minimal assistance (Level 5: standard thresholds, no bias). Each profile adjusts confidence floors, hysteresis, EMA smoothing, and adaptive gain.

## Hardware

### Mechanical Design

The exoskeleton uses a tendon-driven mechanism inspired by the human hand. Each finger has 3D-printed articulated segments that slip over the patient's fingers, connected by two opposing force systems:

- **Flexion (closing):** Elastic bands run along the palmar side of each finger, providing passive pull that curls the fingers closed — mimicking flexor tendons.
- **Extension (opening):** Fishing line routed along the dorsal side connects to a servo motor. When the motor pulls, the line straightens the fingers against the elastic tension — mimicking extensor tendons.

The balance between these two forces gives smooth, controlled movement. Cable routing channels are built into the 3D-printed finger segments to keep the fishing line aligned through each joint. The entire frame is lightweight and slips on like a glove.

The current prototype prioritises function over form — future revisions will focus on a sleeker form factor, cleaner wire management, and a more polished overall build.

### Electronics

- **Microcontroller**: Teensy 4.0
- **EMG sensors**: MyoWare 2.0 (4-channel analog, forearm placement)
- **Actuation**: Servo motor (110° open / 145° rest / 180° closed)
- **Protocol**: 115200 baud serial, tab-separated EMG values in, single-character commands out

Two firmware variants:
- `teensy_emg/` — EMG acquisition only (peak-to-peak amplitude, 50ms windows)
- `exohand_combined/` — Unified EMG + motor control on a single Teensy

## Tech Stack

| Layer | Technologies |
|---|---|
| ML / Signal Processing | Python, scikit-learn, NumPy, SciPy, joblib |
| Backend | Node.js, Express, TypeScript, WebSocket, better-sqlite3, serialport |
| Frontend | React 18, Vite, TypeScript, Three.js, React Three Fiber, Recharts, Tailwind CSS |
| Hardware | Teensy 4.0, MyoWare 2.0, Servo motor |
| Data | GrabMyo (PhysioNet), SQLite |

## Project Structure

```
ExoHand/
├── runtime/                     # Real-time inference & motor control
│   ├── run_exohand.py           # Main entry: free / exercise / web modes
│   ├── calibrate_patient.py     # Patient calibration protocol
│   ├── exercise.py              # Exercise state machine & rep tracking
│   └── assist_profile.py        # 5 graduated assist-as-needed profiles
├── ml/                          # Training & model adaptation
│   ├── train_hgb_v2.py          # Full training pipeline (GrabMyo)
│   ├── train_from_session.py    # Retrain from recorded session data
│   ├── adapt_model.py           # Fine-tune model for new users
│   └── preprocessing_grabmyo.py # GrabMyo WFDB preprocessing + feature extraction
├── data/                        # Data collection & labeling
│   ├── record_session.py        # Record labeled EMG sessions
│   └── label_session.py         # Post-hoc session labeling
├── exohand_model.pkl            # Base pre-trained model (LFS)
├── exohand_adapted_model.pkl    # Patient-adapted model (LFS)
├── server/                      # Node.js backend
│   └── src/
│       ├── index.ts             # Express + WebSocket server
│       ├── routes/              # Auth, patients, sessions, therapist, calibration
│       ├── emg/                 # EMG bridge + calibration logic
│       ├── motor/               # Serial communication + state machine
│       ├── exercise/            # Exercise tracking
│       ├── scoring/             # Outcome scoring
│       └── db/                  # SQLite schema + queries
├── client/                      # React frontend
│   └── src/
│       ├── pages/               # Dashboard, session, calibration views
│       ├── components/          # UI components + 3D hand model
│       └── hooks/               # WebSocket + data hooks
├── teensy_emg/                  # EMG-only firmware
│   └── teensy_emg.ino
├── exohand_combined/            # Combined EMG + motor firmware
│   └── exohand_combined.ino
├── grabmyo/                     # GrabMyo processed features + models (raw data from PhysioNet)
├── datasets/                    # Exercise protocol definitions (JSON)
├── analysis/                    # Research scripts, evaluation results & figures
├── paper/                       # BrainBodyFM paper materials & hardware handoffs
├── REPORT_EMG_Classification.md # Detailed classification report
└── report_figures/              # Result visualizations
```

## Setup

### Hardware
Flash `exohand_combined/exohand_combined.ino` to a Teensy 4.0 using the Arduino IDE with Teensyduino.

### Python Runtime
```bash
pip install numpy scipy scikit-learn joblib pyserial
python run_exohand.py --port /dev/tty.usbmodemXXXX --model exohand_model.pkl
```

### Web Platform
```bash
# Server
cd server && npm install && npm start    # localhost:3001

# Client
cd client && npm install && npm run dev  # localhost:5173
```

### Modes
- **Free mode** (default): Real-time EMG → motor passthrough
- **Exercise mode** (`--exercise`): Structured reps with state tracking, timeout warnings, and rep counting
- **Calibrate** (`--calibrate`): Run patient calibration — full 6-minute protocol for first-time setup, ~30-second per-session re-cal for returning patients


