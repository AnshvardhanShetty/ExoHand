# ExoHand

EMG-controlled hand exoskeleton with real-time intent classification, adaptive motor assistance, and a full-stack rehabilitation platform.

## Overview

ExoHand is a complete EMG-to-actuation system for hand rehabilitation. Surface EMG signals from the forearm are acquired via a Teensy 4.1 microcontroller, classified in real time using a gradient boosting model, and translated into servo commands that drive a 3D-printed exoskeleton hand. A therapist-facing web platform manages patients, tracks progress, and runs structured exercise sessions.

The system achieves **95.9% three-class accuracy** (close / open / rest) with a 30-second patient calibration protocol, and **98.8% binary movement detection accuracy**.

## System Architecture

```
EMG Sensors → Teensy 4.1 → Serial USB → Python Runtime → Motor Commands → Servo
                                              ↕
                                        Node.js Server ↔ React Dashboard
                                              ↕
                                        SQLite Database
```

**Real-time loop:** Read 4-channel EMG at 20 Hz → extract 140 features per window → classify intent → send single-character motor command (`c`/`o`/`r`) — all within 50ms.

## ML Pipeline

### Training Data
Trained on the [GrabMyo dataset](https://physionet.org/content/grabmyo/) — 43 participants, 1.14M samples at 2 kHz, reduced to 4 optimally selected channels targeting flexor and extensor digitorum muscles. Raw session data should be downloaded from PhysioNet and placed in `grabmyo/Session1/`, `Session2/`, `Session3/`.

### Feature Engineering (140 features)
- **Per-channel features** (6 × 4 channels): RMS, MAV, waveform length, zero crossings, slope sign changes, envelope RMS
- **Temporal features**: Lag values, deltas (velocity), acceleration, rolling means — captures how EMG signals evolve over time
- **Cross-channel interactions**: Flexor/extensor ratios, pairwise differences, and their temporal derivatives
- **Per-participant normalization**: Z-score normalization removes inter-subject amplitude variation

### Model
HistGradientBoostingClassifier (scikit-learn) with class balancing, participant-level train/test splits, and EMG-specific data augmentation (gain variation, bias shifts, channel dropout, noise injection).

### Accuracy

| Configuration | Accuracy |
|---|---|
| Baseline (instantaneous features only) | 70.2% |
| + Temporal features | 85.6% |
| + 30-second patient calibration | **95.9%** |
| Binary (movement vs rest) | **98.8%** |

Per-class breakdown at 95.9%:

| Intent | Precision | Recall | F1 |
|---|---|---|---|
| Close | 0.94 | 0.96 | 0.95 |
| Open | 0.97 | 0.95 | 0.96 |
| Rest | 0.96 | 0.99 | 0.98 |

### Patient Calibration
A 30-second protocol collects labeled EMG from a new user and retrains the model with their data weighted 10× against the base training set. This reduces cross-subject variance from ±8.6% to ±1.9%.

Full initial calibration (6 minutes) includes rest baseline, familiarization, sustained holds, quick contractions, and variable effort phases with onset trimming and outlier rejection.

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

- **Microcontroller**: Teensy 4.1
- **EMG sensors**: MyoWare 2.0 (4-channel analog, forearm placement)
- **Actuation**: Servo motor (110° open / 145° rest / 180° closed)
- **Frame**: 3D-printed exoskeleton hand
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
| Hardware | Teensy 4.1, MyoWare 2.0, Servo motor |
| Data | GrabMyo (PhysioNet), SQLite |

## Project Structure

```
ExoHand/
├── run_exohand.py               # Main entry: free / exercise / web modes
├── calibrate_patient.py         # Patient calibration protocol
├── adapt_model.py               # Fine-tune model for new users
├── train_hgb_v2.py              # Full training pipeline (GrabMyo)
├── train_from_session.py        # Retrain from recorded session data
├── preprocessing_grabmyo.py     # GrabMyo WFDB preprocessing + feature extraction
├── exercise.py                  # Exercise state machine & rep tracking
├── assist_profile.py            # 5 graduated assist-as-needed profiles
├── record_session.py            # Record labeled EMG sessions
├── label_session.py             # Post-hoc session labeling
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
├── REPORT_EMG_Classification.md # Detailed classification report
└── report_figures/              # Result visualizations
```

## Setup

### Hardware
Flash `exohand_combined/exohand_combined.ino` to a Teensy 4.1 using the Arduino IDE with Teensyduino.

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
- **Calibrate** (`--calibrate`): Run 30-second calibration for a new patient
