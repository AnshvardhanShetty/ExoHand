"""
T1 — inject stroke EMG samples into a Teensy in TEST_MODE, capture P-P envelopes.

Low-level primitive used by run_T1_all_patients.py. Also usable standalone for
verification via --verify.

Wire contract (host ↔ Teensy) — see README.md.

Requires:
    pip install pyserial numpy pandas
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import serial

SAMPLES_PER_WINDOW = 100     # matches TEST_SAMPLES_PER_WINDOW in firmware
N_CHANNELS = 4
BYTES_PER_WINDOW = SAMPLES_PER_WINDOW * N_CHANNELS * 2   # 800
BAUD = 115200


def open_serial(port, timeout=2.0):
    """Open the Teensy serial port, wait for it to be ready."""
    ser = serial.Serial(port, BAUD, timeout=timeout)
    # Teensy resets when the port is opened; give it a second to boot
    time.sleep(1.5)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser


def send_one_window(ser, samples):
    """Send one 50 ms window's worth of samples to the Teensy.

    Args:
      samples: numpy array shape (N_CHANNELS, SAMPLES_PER_WINDOW), int16 values 0-4095

    The Teensy is expecting per-sample-tuple bytes: (ch0_lo, ch0_hi, ch1_lo, ch1_hi, ...).
    """
    assert samples.shape == (N_CHANNELS, SAMPLES_PER_WINDOW), \
        f"expected shape ({N_CHANNELS}, {SAMPLES_PER_WINDOW}), got {samples.shape}"
    # Interleave: for each sample time i, all 4 channel values in order.
    packet = bytearray(BYTES_PER_WINDOW)
    for i in range(SAMPLES_PER_WINDOW):
        for c in range(N_CHANNELS):
            v = int(samples[c, i]) & 0xFFFF
            packet[(i * N_CHANNELS + c) * 2 + 0] = v & 0xFF
            packet[(i * N_CHANNELS + c) * 2 + 1] = (v >> 8) & 0xFF
    ser.write(bytes(packet))
    ser.flush()


def read_one_pp_frame(ser):
    """Read one P-P envelope line from Teensy: 'pp0\\tpp1\\tpp2\\tpp3\\n'."""
    line = ser.readline()
    if not line:
        raise TimeoutError("Teensy did not respond within serial timeout")
    parts = line.decode().strip().split("\t")
    if len(parts) != N_CHANNELS:
        raise ValueError(f"unexpected line format from Teensy: {line!r}")
    return np.array([int(p) for p in parts], dtype=np.int32)


def replay_windows(ser, raw_stream):
    """Replay a full stream of raw EMG through the Teensy.

    Args:
      raw_stream: numpy array shape (N_CHANNELS, N_samples_total). N_samples_total
                  must be a multiple of SAMPLES_PER_WINDOW.

    Returns:
      pp_envelope: numpy array shape (n_windows, N_CHANNELS) — Teensy's P-P output
                   per 50 ms window.
    """
    total_samples = raw_stream.shape[1]
    n_full_windows = total_samples // SAMPLES_PER_WINDOW
    pp = np.zeros((n_full_windows, N_CHANNELS), dtype=np.int32)
    for w in range(n_full_windows):
        window = raw_stream[:, w * SAMPLES_PER_WINDOW:(w + 1) * SAMPLES_PER_WINDOW]
        send_one_window(ser, window)
        pp[w] = read_one_pp_frame(ser)
    return pp


# ------------------------------------------------------------ verification
def verify_teensy(port):
    """Run three sanity patterns and report pass/fail."""
    print(f"Opening {port} …")
    ser = open_serial(port)

    print("\n[1/3] Sending 100 zero samples per channel …")
    zeros = np.zeros((N_CHANNELS, SAMPLES_PER_WINDOW), dtype=np.int16)
    send_one_window(ser, zeros)
    pp = read_one_pp_frame(ser)
    ok = np.all(pp == 0)
    print(f"     Teensy replied {pp.tolist()}  → {'PASS' if ok else 'FAIL (expected all zeros)'}")

    print("\n[2/3] Sending constant 2048 value on all channels …")
    const = np.full((N_CHANNELS, SAMPLES_PER_WINDOW), 2048, dtype=np.int16)
    send_one_window(ser, const)
    pp = read_one_pp_frame(ser)
    ok = np.all(pp == 0)
    print(f"     Teensy replied {pp.tolist()}  → {'PASS' if ok else 'FAIL (expected all zeros — constant has zero P-P)'}")

    print("\n[3/3] Sending 500-amplitude 100 Hz sinewave on all channels (2 kHz sample rate) …")
    t = np.arange(SAMPLES_PER_WINDOW)
    sinewave = (500 * np.sin(2 * np.pi * 100 * t / 2000) + 2048).astype(np.int16)
    stim = np.tile(sinewave, (N_CHANNELS, 1))
    send_one_window(ser, stim)
    pp = read_one_pp_frame(ser)
    ok = np.all(np.abs(pp - 1000) < 5)   # peak-to-peak of ±500 amplitude ≈ 1000
    print(f"     Teensy replied {pp.tolist()}  → {'PASS' if ok else 'FAIL (expected ~1000 per channel)'}")

    ser.close()
    print("\nAll three sanity patterns complete. If any FAILed, do not proceed to the full run.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/tty.usbmodem12345")
    parser.add_argument("--verify", action="store_true", help="Run 3 sanity patterns and exit")
    args = parser.parse_args()

    if args.verify:
        verify_teensy(args.port)
        return

    print("This module is meant to be imported. Use --verify or run "
          "run_T1_all_patients.py for the full sweep.")


if __name__ == "__main__":
    main()
