#!/usr/bin/env python3
"""
record_session.py — Replace CoolTerm with automated cued EMG recording.

Records one continuous session from Teensy (4-ch tab-separated EMG)
with timed audio cues via macOS `say`.

Usage:
    python record_session.py --port /dev/tty.usbmodemXXXX
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import serial


def say(text):
    """Non-blocking macOS speech."""
    subprocess.Popen(["say", text])


def say_sync(text):
    """Blocking macOS speech — waits until done."""
    subprocess.run(["say", text], check=False)


def build_protocol():
    """
    Build the cued recording protocol.
    Returns list of (duration_sec, cue_label, block_description) tuples.
    """
    protocol = []

    # Phase 1: Baseline rest — 10 s
    protocol.append((10, "rest", "baseline rest"))

    # Phase 2: CLOSE reps — 5 blocks × 6 reps × (3s gesture + 2s rest)
    close_blocks = [
        ("light effort, slow", "close block A – light effort"),
        ("medium effort", "close block B – medium effort"),
        ("strong effort, quick", "close block C – strong effort"),
        ("arm raised", "close block D – arm raised"),
        ("mixed natural", "close block E – mixed natural"),
    ]
    for voice_cue, desc in close_blocks:
        for rep in range(6):
            protocol.append((5, "close", f"{desc} rep {rep+1}"))
            protocol.append((4, "rest", f"{desc} rest after rep {rep+1}"))

    # 30-second break
    protocol.append((30, "rest", "break after close phase"))

    # Phase 3: OPEN reps — same block structure
    open_blocks = [
        ("light effort, slow", "open block A – light effort"),
        ("medium effort", "open block B – medium effort"),
        ("strong effort, quick", "open block C – strong effort"),
        ("arm raised", "open block D – arm raised"),
        ("mixed natural", "open block E – mixed natural"),
    ]
    for voice_cue, desc in open_blocks:
        for rep in range(6):
            protocol.append((5, "open", f"{desc} rep {rep+1}"))
            protocol.append((4, "rest", f"{desc} rest after rep {rep+1}"))

    # 30-second break
    protocol.append((30, "rest", "break after open phase"))

    # Phase 4: REST — 2 min in different positions
    protocol.append((30, "rest", "rest – arm on table"))
    protocol.append((30, "rest", "rest – arm hanging"))
    protocol.append((30, "rest", "rest – arm raised"))
    protocol.append((30, "rest", "rest – arm on table again"))

    return protocol


def parse_emg_line(line):
    """Parse a tab-separated line into 4 floats. Returns None on failure."""
    try:
        parts = line.strip().split()
        if len(parts) != 4:
            return None
        vals = [float(v) for v in parts]
        return vals
    except (ValueError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Record cued EMG session from Teensy")
    parser.add_argument("--port", required=True, help="Serial port (e.g. /dev/tty.usbmodem12345)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default 115200)")
    args = parser.parse_args()

    # Create session directory
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    session_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions", timestamp_str)
    os.makedirs(session_dir, exist_ok=True)

    raw_csv_path = os.path.join(session_dir, "raw_emg.csv")
    cues_csv_path = os.path.join(session_dir, "cues.csv")
    info_json_path = os.path.join(session_dir, "session_info.json")

    print(f"Session directory: {session_dir}")
    print(f"Connecting to {args.port} at {args.baud} baud...")

    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    time.sleep(2)  # Let Teensy settle

    # Drain any stale data
    ser.reset_input_buffer()

    print("\nTell your friend to relax.")
    input("Press Enter when ready to start recording...\n")

    protocol = build_protocol()
    total_duration = sum(d for d, _, _ in protocol)
    print(f"Protocol duration: ~{total_duration/60:.1f} minutes")
    print("Recording... (Ctrl+C to stop early)\n")

    samples = []
    cue_events = []
    t0 = time.perf_counter()
    sample_count = 0
    skipped = 0

    raw_f = open(raw_csv_path, "w", newline="")
    raw_writer = csv.writer(raw_f)
    raw_writer.writerow(["timestamp_sec", "ch0", "ch1", "ch2", "ch3"])

    cues_f = open(cues_csv_path, "w", newline="")
    cues_writer = csv.writer(cues_f)
    cues_writer.writerow(["timestamp_sec", "cue_label", "block_description"])

    try:
        for duration, cue_label, block_desc in protocol:
            cue_ts = time.perf_counter() - t0

            # Record cue event
            cues_writer.writerow([f"{cue_ts:.6f}", cue_label, block_desc])
            cue_events.append((cue_ts, cue_label, block_desc))

            # Speak the cue
            if cue_label == "rest":
                if "break" in block_desc:
                    say("Take a break. Relax.")
                    print(f"  [{cue_ts:7.1f}s] BREAK — 30 seconds")
                elif "baseline" in block_desc:
                    say("Stay relaxed. Recording baseline.")
                    print(f"  [{cue_ts:7.1f}s] BASELINE REST")
                elif "arm hanging" in block_desc:
                    say("Let your arm hang.")
                    print(f"  [{cue_ts:7.1f}s] REST — arm hanging")
                elif "arm raised" in block_desc:
                    say("Raise your arm.")
                    print(f"  [{cue_ts:7.1f}s] REST — arm raised")
                elif "rest after rep" in block_desc:
                    say("relax")
                    # Don't spam the console for every rest-between-reps
                else:
                    say("relax")
                    print(f"  [{cue_ts:7.1f}s] REST — {block_desc}")
            elif cue_label == "close":
                if "rep 1" in block_desc:
                    # Announce block
                    block_tag = block_desc.split(" rep")[0]
                    effort = block_tag.split("–")[-1].strip() if "–" in block_tag else ""
                    say(f"Close. {effort}")
                    print(f"  [{cue_ts:7.1f}s] CLOSE — {block_tag}")
                else:
                    say("close")
            elif cue_label == "open":
                if "rep 1" in block_desc:
                    block_tag = block_desc.split(" rep")[0]
                    effort = block_tag.split("–")[-1].strip() if "–" in block_tag else ""
                    say(f"Open. {effort}")
                    print(f"  [{cue_ts:7.1f}s] OPEN — {block_tag}")
                else:
                    say("open")

            # Collect data for this segment's duration
            segment_end = cue_ts + duration + t0  # absolute time
            while time.perf_counter() < segment_end:
                raw = ser.readline()
                if not raw:
                    continue
                try:
                    line = raw.decode("utf-8", errors="ignore").strip()
                except UnicodeDecodeError:
                    continue
                if not line:
                    continue

                vals = parse_emg_line(line)
                if vals is None:
                    skipped += 1
                    continue

                ts = time.perf_counter() - t0
                raw_writer.writerow([f"{ts:.6f}", f"{vals[0]:.4f}", f"{vals[1]:.4f}", f"{vals[2]:.4f}", f"{vals[3]:.4f}"])
                sample_count += 1

                if sample_count % 500 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = sample_count / elapsed if elapsed > 0 else 0
                    print(f"\r    Samples: {sample_count}  |  ~{rate:.0f} Hz  |  {elapsed:.0f}s elapsed", end="", flush=True)

        print(f"\n\nRecording complete!")

    except KeyboardInterrupt:
        print(f"\n\nRecording interrupted by user.")

    finally:
        raw_f.close()
        cues_f.close()
        ser.close()

        # Compute stats and save session info
        total_time = time.perf_counter() - t0
        approx_rate = sample_count / total_time if total_time > 0 else 0

        info = {
            "serial_port": args.port,
            "baud_rate": args.baud,
            "date": datetime.now().isoformat(),
            "num_samples": sample_count,
            "skipped_lines": skipped,
            "duration_sec": round(total_time, 2),
            "approx_sample_rate_hz": round(approx_rate, 1),
            "num_cue_events": len(cue_events),
        }

        with open(info_json_path, "w") as f:
            json.dump(info, f, indent=2)

        print(f"Saved {sample_count} samples to {raw_csv_path}")
        print(f"Saved {len(cue_events)} cue events to {cues_csv_path}")
        print(f"Session info: {info_json_path}")
        print(f"Approx sample rate: {approx_rate:.0f} Hz")
        print(f"Skipped lines: {skipped}")


if __name__ == "__main__":
    main()
