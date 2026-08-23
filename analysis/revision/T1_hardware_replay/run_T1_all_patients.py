"""
T1 driver — replay 58 stroke patients through the actual Teensy hardware.

For each PhysioMio + Lucchetti test patient:
  1. Load raw EMG (2 kHz per channel, 4 channels selected per Cohen's d picks)
  2. Filter + downsample to what the deployed pipeline sees (via bandpass_notch)
  3. Stream raw samples through the Teensy in TEST_MODE via inject_and_capture.py
  4. Teensy emits 20 Hz P-P envelope (its normal output)
  5. Run same host-side classification as runtime/run_exohand.py (feature extract +
     HGB predict + Stage 2 post-processing)
  6. Compute deployed accuracy on the balanced 39/39/39 test subset
  7. Log per-window decisions for T4 stream metrics later

Resumable: skips patients already in the output CSV.

Requires: physical Teensy plugged in, flashed with TEST_MODE = 1.
Verify with: python3 inject_and_capture.py --verify --port /dev/...
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from inject_and_capture import open_serial, replay_windows, SAMPLES_PER_WINDOW, N_CHANNELS
from analysis.seed import SEED, seed_everything
from analysis.physiomio.deployed_pipeline_sim import (
    PHYSIOMIO_ROOT, CHANNEL_PICKS, _bandpass_notch,
    TEENSY_PP_WINDOW_MS, INFERENCE_WINDOW_MS, INFERENCE_STRIDE_MS,
    TEENSY_OUTPUT_HZ, extract_session_features_20hz,
)
from analysis.physiomio.per_session_eval import SAMPLE_RATE_HZ

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "T1_deployed_accuracy_per_patient.csv"
OUT_PARQUET = OUT_DIR / "T1_deployed_stream_per_window.parquet"


def scale_to_adc(x, target_min=0, target_max=4095):
    """Scale filtered EMG to the 0-4095 range the Teensy ADC would see.

    We normalise each channel to its own dynamic range so the injected data
    plausibly resembles what the analog front-end would produce.
    """
    x = x.copy()
    for c in range(x.shape[0]):
        ch = x[c]
        lo, hi = np.percentile(ch, [1, 99])
        if hi - lo < 1e-6:
            x[c] = target_min
            continue
        # Clip 1-99 percentile range, linearly map to [target_min, target_max]
        ch_clip = np.clip(ch, lo, hi)
        x[c] = ((ch_clip - lo) / (hi - lo) * (target_max - target_min) + target_min)
    return np.round(x).astype(np.int16)


def replay_one_patient(ser, parquet_path, channels):
    """Full replay pipeline for one patient session.

    Returns:
      pp_envelope: (N_windows, N_CHANNELS) int32
      per_window_labels: (N_windows,) int8 — ground-truth intent per 50 ms window
      per_window_time_s: (N_windows,) float
    """
    # Reuse the existing extractor's data loading + filtering
    # (We ONLY use its raw-P-P computation for the label alignment.
    #  The actual P-P values come from the Teensy in this script.)
    df = pd.read_parquet(parquet_path)

    # Concatenate per-gesture segments so we have a continuous stream
    # (deployed_pipeline_sim.extract_session_features_20hz processes per-gesture;
    # here we do the same but stream to the Teensy)
    from analysis.physiomio.deployed_pipeline_sim import GESTURE_TO_INTENT, INTENT_TO_IDX
    pp_streams = []   # list of P-P frames from the Teensy
    labels = []       # per-window ground-truth intent

    for (gesture, gesture_df) in df.groupby("movement_type", sort=False):
        if gesture not in GESTURE_TO_INTENT:
            continue
        intent = GESTURE_TO_INTENT[gesture]
        if intent is None:
            continue
        intent_idx = INTENT_TO_IDX[intent]

        # Extract + filter each channel's raw waveform for this gesture segment
        raw = np.stack([
            _bandpass_notch(gesture_df[f"channel_{ch:02d}"].values)
            for ch in channels
        ], axis=0)   # (N_CHANNELS, T)

        raw_adc = scale_to_adc(raw)

        # Trim to a multiple of SAMPLES_PER_WINDOW
        n_full = (raw_adc.shape[1] // SAMPLES_PER_WINDOW) * SAMPLES_PER_WINDOW
        if n_full == 0:
            continue
        raw_adc = raw_adc[:, :n_full]

        # Stream through the Teensy — one 50 ms window at a time
        pp = replay_windows(ser, raw_adc)   # (n_windows, N_CHANNELS)

        pp_streams.append(pp)
        labels.extend([intent_idx] * pp.shape[0])

    pp_envelope = np.vstack(pp_streams) if pp_streams else np.zeros((0, N_CHANNELS))
    return pp_envelope, np.array(labels, dtype=np.int8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Teensy serial port")
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap #patients")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)

    picks = pd.read_csv(CHANNEL_PICKS)
    picks["chosen_channels"] = picks["chosen_channels"].apply(json.loads)
    pick_map = dict(zip(picks["patient"], picks["chosen_channels"]))

    patient_dirs = sorted(
        [p for p in PHYSIOMIO_ROOT.iterdir() if p.is_dir() and p.name.startswith("patient")],
        key=lambda p: int(p.name.replace("patient", "")),
    )
    if args.limit:
        patient_dirs = patient_dirs[:args.limit]

    # Resume support
    done = set()
    rows = []
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done = set(existing["patient"].tolist())
        print(f"Resume: {len(rows)} patients already done")

    print(f"Opening Teensy on {args.port} …")
    ser = open_serial(args.port)
    print("Teensy ready.")

    per_window_rows = []
    t0 = time.time()
    for pi, pdir in enumerate(patient_dirs, 1):
        patient = pdir.name
        if patient in done:
            continue
        channels = pick_map.get(patient)
        if channels is None:
            continue
        parquet = pdir / "impaired_arm" / "01.parquet"
        if not parquet.exists():
            continue

        t_p = time.time()
        try:
            pp, labels = replay_one_patient(ser, parquet, channels)
        except Exception as e:
            print(f"[{pi}] {patient}: replay FAILED ({e})", flush=True)
            continue

        # TODO: apply Stage 1 HGB + Stage 2 post-processing to pp_envelope
        # Reuse from runtime/run_exohand.py: extract features from 4-frame windows,
        # call hgb.predict_proba, apply _smooth_proba + _apply_stability + hysteresis
        # + confidence floor + cooldown → per-decision output
        # Then compute accuracy on the balanced 39/39/39 test subset.
        # (Left as an integration step — pattern is identical to the Python-only sim.)
        acc = float("nan")   # PLACEHOLDER — fill in after Stage 1/2 wiring

        rows.append({
            "patient": patient,
            "n_windows_replayed": int(pp.shape[0]),
            "deployed_acc": acc,
            "runtime_s": time.time() - t_p,
        })

        # Save per-window trace for T4 later
        for w in range(pp.shape[0]):
            per_window_rows.append({
                "patient": patient,
                "window_idx": w,
                "ground_truth_intent": int(labels[w]),
                "pp_ch0": int(pp[w, 0]),
                "pp_ch1": int(pp[w, 1]),
                "pp_ch2": int(pp[w, 2]),
                "pp_ch3": int(pp[w, 3]),
            })

        elapsed = time.time() - t0
        eta = elapsed / max(1, pi) * (len(patient_dirs) - pi)
        print(f"[{pi}/{len(patient_dirs)}] {patient}: {pp.shape[0]} windows replayed  "
              f"[{elapsed/60:.1f}min, eta {eta/60:.0f}min]", flush=True)

        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    if per_window_rows:
        existing_parquet = pd.DataFrame()
        if OUT_PARQUET.exists():
            existing_parquet = pd.read_parquet(OUT_PARQUET)
        combined = pd.concat([existing_parquet, pd.DataFrame(per_window_rows)], ignore_index=True)
        combined.to_parquet(OUT_PARQUET, index=False)

    ser.close()
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_PARQUET}")
    print(f"NOTE: deployed_acc column is currently NaN — wire up Stage 1/2 post-processing.")


if __name__ == "__main__":
    main()
