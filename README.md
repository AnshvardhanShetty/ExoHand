# ExoHand

**Accepted at BrainBodyFM (NeurIPS 2026).**

EMG-controlled hand exoskeleton with real-time intent classification, adaptive motor assistance, and a full-stack rehabilitation platform.

[![Watch the demo](https://img.youtube.com/vi/RMq31iIWcPk/maxresdefault.jpg)](https://www.youtube.com/watch?v=RMq31iIWcPk)

**[Watch Demo](https://www.youtube.com/watch?v=RMq31iIWcPk)**

## Overview

ExoHand is a closed-loop hand rehabilitation prototype that turns surface electromyography (sEMG) from the forearm into movement of a 3D-printed, tendon-driven hand exoskeleton. Four sensors capture the user's muscle activity; a locally trained classifier decodes **close / open / rest** and drives the device in real time. A therapist-facing web platform manages patients, tracks progress, and runs structured exercise sessions.

ExoHand uses a HistGradientBoosting classifier **trained from scratch on 22 seconds of the user's own signals**, with no pretrained model in the live loop. The guided calibration session takes about two minutes overall; **22 seconds refers to the labelled signal budget**, rather than the entire setup and cueing time. The hardware build costs **under £200** and runs with a Teensy 4.0, a laptop, four sEMG sensors, and two hobby servos.

## NeurIPS 2026

**Accepted at BrainBodyFM — Foundation Models for the Brain and Body — at NeurIPS 2026.**

**Title:** *ExoHand: A £200 Closed-loop Hand Exoskeleton, Calibrated in 22 Seconds*

**Authors:** Anshvardhan Shetty · Adhiraiyan Sasikumar

One participant wears the exoskeleton while a second wears the four-channel forearm sensor sleeve. After calibration, the second participant's hand intent drives the first participant's exoskeleton: closing their hand closes the device, and opening their hand opens it. Attendees swap roles to experience both decoding and assisted movement. This two-person arrangement makes the signal-to-movement path visible; the intended rehabilitation arrangement places sensing and assistance on the same patient.

Acquisition, 20 Hz envelope extraction, inference, and actuation run locally, using live signals. The interface displays the four EMG channels, decoded hand state, rep count, session accuracy, and stability.

## Research & Results

The research behind the demonstration tested whether a large healthy-population EMG corpus transfers to stroke intent decoding. GrabMyo contributes **1.14 million windows from 43 subjects**; the stroke evaluation covers **48 PhysioMio patients** and three intent classes.

| Offline stroke evaluation | Mean accuracy |
|---|---:|
| Three-class chance | 33.3% |
| GrabMyo → stroke, zero-shot | 36.0% |
| Patient-only classifier, 22 seconds of impaired-arm calibration | **89.6%** |

The zero-shot result was statistically indistinguishable from chance. Adding GrabMyo data at the tested weights did not improve on patient-only calibration. These findings motivated the demonstration's per-user training approach. The offline results use four channels selected per patient to match the live rig's channel count; **89.6% is an offline cohort result**, rather than a measured accuracy for every live attendee.

- [Paper materials](paper/README.md): research documentation, figures, appendix, and hardware verification handoffs.
- [Canonical research results](paper/FINAL_NUMBERS.md): cohort results and provenance.
- [Leakage-free calibration comparison](analysis/revision/results/C4_leakage_free_summary.md): patient-only training versus adding GrabMyo data.
- [Analysis and evaluation](analysis/README.md): experiment scripts and reproduction instructions.

## System Architecture

```
4 sEMG Sensors → Teensy 4.0 → Serial USB → Python Decoder → Motor Commands → 2 Servos
                                               ↕
                                         Node.js Server ↔ React Dashboard
                                               ↕
                                         SQLite Database
```

**Real-time loop:** Acquire four-channel EMG → extract envelope and window features → classify close / open / rest → issue motor commands. The acquisition stream runs at 20 Hz (one update every 50 ms); physical actuation and filtering add to the end-to-end response time.

## ML Pipeline

### Per-user Calibration

The demonstration fits a scikit-learn HistGradientBoostingClassifier to the user's own cued close, open, and rest signals. Feature scaling is fitted to calibration data. The repository includes a `paper22s` calibration protocol with 12 cued blocks of 1.8 seconds each (21.6 seconds of labelled signal), plus transitions, baseline collection, and instructions.

The runtime also retains longer calibration protocols: a full initial protocol of about six minutes and a quick abbreviated protocol of about 90 seconds. These are additional runtime options; ExoHand uses the short signal budget described above.

### Features

The pipeline uses 370 engineered features, including per-channel RMS, mean absolute value, waveform length, zero crossings, slope sign changes, envelope RMS, temporal lags and derivatives, rolling statistics, and cross-channel interactions.

### Earlier GrabMyo Benchmark

Earlier development evaluated a population model within GrabMyo using 43 leave-one-subject-out folds. These results describe the healthy-subject benchmark and its larger calibration budget; they are separate from the stroke evaluation and the live system.

| Configuration | Accuracy (mean, 95% CI) | Macro-F1 (mean, 95% CI) |
|---|---|---|
| Cross-subject baseline (no calibration) | 94.6% [93.3%, 95.8%] | 0.946 [0.932, 0.958] |
| + Per-user calibration (60 s, 1200 windows, 100× weight) | 97.3% [96.7%, 97.9%] | 0.972 [0.965, 0.979] |

The repository retains the earlier population-model training and adaptation tools for reproduction. Raw GrabMyo sessions can be downloaded from [PhysioNet](https://physionet.org/content/grabmyo/) and placed in `grabmyo/Session1/`, `Session2/`, and `Session3/`.

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

The demonstration uses two servos on a forearm cuff: one drives the four fingers and the other drives the thumb independently. Each servo is geared 2:1 to give its winches a full 360° of travel.

Each finger has its own winch, sized so its circumference matches the fingertip's travel between open and closed positions. This lets fingers of different lengths reach their target positions together. Opposing flexion and extension tendons share the winch: one runs over a dorsal plate to the fingertips, and the other follows the palmar side. Turning the winch reels in one tendon while releasing the other, maintaining controlled tension throughout movement.

### Electronics

- **Microcontroller:** Teensy 4.0
- **EMG sensors:** Four MyoWare 2.0 channels, placed over FCR, ECR, FDS, and EDC
- **Actuation:** Two hobby servos, with independent finger and thumb drive
- **Host:** Laptop running local decoding and the interface
- **Protocol:** 115200 baud serial, four-channel EMG input, and motor commands

Two firmware variants:
- `teensy_emg/` — EMG acquisition only (peak-to-peak amplitude, 50ms windows)
- `exohand_combined/` — Unified EMG + motor control on a single Teensy

## Tech Stack

| Layer | Technologies |
|---|---|
| ML / Signal Processing | Python, scikit-learn, NumPy, SciPy, joblib |
| Backend | Node.js, Express, TypeScript, WebSocket, better-sqlite3, serialport |
| Frontend | React 18, Vite, TypeScript, Three.js, React Three Fiber, Recharts, Tailwind CSS |
| Hardware | Teensy 4.0, four MyoWare 2.0 sensors, two hobby servos |
| Data | Per-user calibration, PhysioMio, GrabMyo (research), SQLite |

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
├── exohand_model.pkl            # Earlier shipped model artifact (LFS)
├── exohand_adapted_model.pkl    # Earlier adapted-model artifact (LFS)
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
pip install -r requirements.txt
python runtime/run_exohand.py --port /dev/tty.usbmodemXXXX --model /path/to/calibrated_model.pkl
```

Use a saved per-user model for live inference. The earlier population-model artifacts remain available for reproducing the historical benchmark. Calibration protocols are implemented in [`runtime/calibrate_patient.py`](runtime/calibrate_patient.py).

### Web Platform
```bash
# Server
cd server && npm install && npm run build && npm start  # localhost:3001

# Client
cd client && npm install && npm run dev  # localhost:5173
```

### Modes
- **Free mode** (default): Real-time EMG → motor passthrough
- **Exercise mode** (`--exercise`): Structured reps with state tracking, timeout warnings, and rep counting
- **Patient calibration** (`--patient-calibrate` / `--patient-recalibrate`): Full or abbreviated calibration through the runtime
- **Short cued calibration** (`paper22s` in the calibration module): ~22 seconds of labelled signals, with additional time for baseline collection, cueing, and transitions


