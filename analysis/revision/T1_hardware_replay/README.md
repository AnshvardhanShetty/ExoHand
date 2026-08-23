# T1 — Hardware-in-the-Loop Stroke Replay

Feed PhysioMio + Lucchetti raw stroke EMG through the actual deployed Teensy
firmware (in test mode) and log the deployed pipeline's per-window decisions.
Yields deployed-runtime accuracy on 58 stroke patients without recruiting anyone.

## Prerequisites

- Teensy 4.0 flashed with `teensy_emg/teensy_emg.ino` compiled with
  `#define TEST_MODE 1`
- Physical Teensy plugged into laptop via USB
- Serial port identified (look under `/dev/tty.usbmodem*` on macOS)
- PhysioMio raw parquet files present under `data/physiomio/data/patientN/impaired_arm/*.parquet`

## Verification before full run

Run once to confirm the firmware/host interface works:

```bash
python3 inject_and_capture.py --verify --port /dev/tty.usbmodem12345678
```

Expected: three test patterns pass (zeros → 0000, sinewave → known P-P, uniform → 0000).

## Full run

```bash
python3 run_T1_all_patients.py --port /dev/tty.usbmodem12345678
```

- Processes 48 PhysioMio + 10 Lucchetti patients sequentially
- ~2-3 minutes per patient
- Resumable: skips patients already in output CSV
- Writes:
  - `../results/T1_deployed_accuracy_per_patient.csv` — per-patient deployed accuracy
  - `../results/T1_deployed_stream_per_window.parquet` — per-window decisions for T4

## After T1 completes

Run `../recompute_T4_stream_metrics.py` to compute event-level F1, transition
latency, false-activation rate, flicker rate on the T1 output.

Run `../../plots/T1_figure2_decision_trace.py` to produce Figure 2 (decision
trace of deployed stack on a representative replayed session).

## Restoring production firmware

**When T1 is done, flash the Teensy back to production mode:**

Edit `teensy_emg/teensy_emg.ino`, set `#define TEST_MODE 0`, recompile, upload.

**Never leave TEST_MODE = 1 on a device that touches a real subject.**

## Interface contract (host ↔ Teensy)

**Host → Teensy per 50 ms window:**
- Exactly 800 bytes
- 100 samples × 4 channels × 2 bytes
- Order: `[s0c0, s0c1, s0c2, s0c3, s1c0, s1c1, s1c2, s1c3, ...]`
- Little-endian int16
- Values 0-4095

**Teensy → Host per 50 ms window:**
- One text line: `pp_ch0\tpp_ch1\tpp_ch2\tpp_ch3\n`
- Unsigned ints, 0-4095
