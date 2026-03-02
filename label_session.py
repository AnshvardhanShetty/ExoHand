#!/usr/bin/env python3
"""
label_session.py — Auto-detect gesture onsets and label every sample.

Post-processes a recorded session: loads raw EMG + cues, computes baseline,
finds onset/offset per cue window, produces labeled_data.npz + visualization.

Usage:
    python label_session.py --session sessions/2025-02-14_14-30/
"""

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


LABEL_MAP = {"close": 0, "open": 1, "rest": 2}
LABEL_NAMES = {0: "close", 1: "open", 2: "rest"}


def load_raw_emg(session_dir):
    """Load raw_emg.csv → (timestamps, data) both as numpy arrays."""
    path = os.path.join(session_dir, "raw_emg.csv")
    timestamps = []
    data = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            timestamps.append(float(row[0]))
            data.append([float(row[1]), float(row[2]), float(row[3]), float(row[4])])
    return np.array(timestamps), np.array(data)


def load_cues(session_dir):
    """Load cues.csv → list of (timestamp, label, description)."""
    path = os.path.join(session_dir, "cues.csv")
    cues = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            cues.append((float(row[0]), row[1], row[2]))
    return cues


def compute_baseline(timestamps, data, baseline_duration=10.0):
    """Compute per-channel mean and std from the first baseline_duration seconds."""
    mask = timestamps <= baseline_duration
    if mask.sum() < 10:
        print(f"WARNING: Only {mask.sum()} samples in baseline window. Using first 100 samples.")
        mask = np.zeros(len(timestamps), dtype=bool)
        mask[:min(100, len(timestamps))] = True

    baseline_data = data[mask]
    bl_mean = baseline_data.mean(axis=0)
    bl_std = baseline_data.std(axis=0)
    return bl_mean, bl_std


def find_onset_offset(timestamps, data, bl_mean, bl_std, sample_rate,
                      threshold_multiplier=3.0):
    """
    Find gesture onset and offset within a window using rolling RMS threshold.

    Returns (onset_idx, offset_idx) relative to the input arrays,
    or (None, None) if no onset detected.
    """
    # Per-sample amplitude: sum of absolute deviation from baseline across channels
    deviation = np.abs(data - bl_mean)
    amplitude = deviation.sum(axis=1)

    # Threshold
    threshold = threshold_multiplier * bl_std.sum()

    # 10ms rolling RMS
    rms_window = max(1, int(0.010 * sample_rate))
    if len(amplitude) < rms_window:
        return None, None

    # Compute rolling RMS
    rolling_rms = np.zeros(len(amplitude))
    amp_sq = amplitude ** 2
    cumsum = np.cumsum(amp_sq)
    cumsum = np.insert(cumsum, 0, 0)
    for i in range(len(amplitude)):
        start = max(0, i - rms_window + 1)
        n = i - start + 1
        rolling_rms[i] = np.sqrt((cumsum[i + 1] - cumsum[start]) / n)

    # Find onset: first sample where rolling RMS exceeds threshold
    above = rolling_rms > threshold
    onset_candidates = np.where(above)[0]
    if len(onset_candidates) == 0:
        return None, None

    onset_idx = onset_candidates[0]

    # Find offset: scan backwards, find last sample above threshold
    offset_idx = onset_candidates[-1]

    return onset_idx, offset_idx


def label_session(session_dir, threshold_multiplier=3.0):
    """Main labeling pipeline."""
    print(f"Loading session from {session_dir}")

    timestamps, data = load_raw_emg(session_dir)
    cues = load_cues(session_dir)

    print(f"  {len(timestamps)} samples, {len(cues)} cue events")

    # Load session info for sample rate
    info_path = os.path.join(session_dir, "session_info.json")
    with open(info_path) as f:
        info = json.load(f)
    sample_rate = info.get("approx_sample_rate_hz", 1000)
    print(f"  Approx sample rate: {sample_rate} Hz")

    # Compute baseline from first 10 seconds
    bl_mean, bl_std = compute_baseline(timestamps, data)
    print(f"  Baseline mean: [{bl_mean[0]:.2f}, {bl_mean[1]:.2f}, {bl_mean[2]:.2f}, {bl_mean[3]:.2f}]")
    print(f"  Baseline std:  [{bl_std[0]:.2f}, {bl_std[1]:.2f}, {bl_std[2]:.2f}, {bl_std[3]:.2f}]")

    # Initialize labels as -1 (unlabeled)
    labels = np.full(len(timestamps), -1, dtype=int)

    # Stats
    no_onset_count = 0
    onset_latencies = []
    cue_results = []

    # Process each cue window
    for i, (cue_ts, cue_label, cue_desc) in enumerate(cues):
        # Window end = next cue timestamp, or end of recording
        if i + 1 < len(cues):
            window_end = cues[i + 1][0]
        else:
            window_end = timestamps[-1] + 0.001

        # Find samples in this window
        mask = (timestamps >= cue_ts) & (timestamps < window_end)
        window_indices = np.where(mask)[0]

        if len(window_indices) == 0:
            continue

        if cue_label == "rest":
            labels[window_indices] = LABEL_MAP["rest"]
            cue_results.append((cue_ts, cue_label, cue_desc, "all_rest", len(window_indices), 0))

        elif cue_label in ("close", "open"):
            gesture_label = LABEL_MAP[cue_label]
            win_data = data[window_indices]
            win_ts = timestamps[window_indices]

            onset_idx, offset_idx = find_onset_offset(win_ts, win_data, bl_mean, bl_std, sample_rate,
                                                       threshold_multiplier)

            if onset_idx is None:
                # No onset detected — label entire window as rest, print warning
                labels[window_indices] = LABEL_MAP["rest"]
                no_onset_count += 1
                print(f"  WARNING: No onset for cue '{cue_label}' at {cue_ts:.1f}s ({cue_desc})")
                cue_results.append((cue_ts, cue_label, cue_desc, "no_onset", len(window_indices), 0))
            else:
                # Label: before onset → rest, onset-to-offset → gesture, after offset → rest
                labels[window_indices[:onset_idx]] = LABEL_MAP["rest"]
                labels[window_indices[onset_idx:offset_idx + 1]] = gesture_label
                labels[window_indices[offset_idx + 1:]] = LABEL_MAP["rest"]

                latency = win_ts[onset_idx] - cue_ts
                onset_latencies.append(latency)
                gesture_samples = offset_idx - onset_idx + 1
                cue_results.append((cue_ts, cue_label, cue_desc, "detected", gesture_samples, latency))

    # Any still-unlabeled samples get labeled as rest
    labels[labels == -1] = LABEL_MAP["rest"]

    # Class counts
    close_count = (labels == 0).sum()
    open_count = (labels == 1).sum()
    rest_count = (labels == 2).sum()

    print(f"\n  Label distribution:")
    print(f"    close: {close_count:>7d} samples ({100*close_count/len(labels):.1f}%)")
    print(f"    open:  {open_count:>7d} samples ({100*open_count/len(labels):.1f}%)")
    print(f"    rest:  {rest_count:>7d} samples ({100*rest_count/len(labels):.1f}%)")
    print(f"  No-onset cues: {no_onset_count}")

    if onset_latencies:
        lat = np.array(onset_latencies)
        print(f"  Onset latency: mean={lat.mean():.3f}s, std={lat.std():.3f}s, "
              f"min={lat.min():.3f}s, max={lat.max():.3f}s")

    # Save labeled data
    npz_path = os.path.join(session_dir, "labeled_data.npz")
    np.savez(npz_path,
             X=data,
             y=labels,
             timestamps=timestamps,
             baseline_mean=bl_mean,
             baseline_std=bl_std)
    print(f"\n  Saved: {npz_path}")

    # Save labeling report
    report_path = os.path.join(session_dir, "labeling_report.txt")
    with open(report_path, "w") as f:
        f.write("LABELING REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Session: {session_dir}\n")
        f.write(f"Total samples: {len(timestamps)}\n")
        f.write(f"Sample rate: ~{sample_rate} Hz\n")
        f.write(f"Duration: {timestamps[-1]:.1f}s\n\n")
        f.write(f"Baseline mean: {bl_mean}\n")
        f.write(f"Baseline std:  {bl_std}\n\n")
        f.write(f"Label distribution:\n")
        f.write(f"  close (0): {close_count} ({100*close_count/len(labels):.1f}%)\n")
        f.write(f"  open  (1): {open_count} ({100*open_count/len(labels):.1f}%)\n")
        f.write(f"  rest  (2): {rest_count} ({100*rest_count/len(labels):.1f}%)\n\n")
        f.write(f"Cues with no onset detected: {no_onset_count}\n\n")
        if onset_latencies:
            lat = np.array(onset_latencies)
            f.write(f"Onset latency stats:\n")
            f.write(f"  mean: {lat.mean():.4f}s\n")
            f.write(f"  std:  {lat.std():.4f}s\n")
            f.write(f"  min:  {lat.min():.4f}s\n")
            f.write(f"  max:  {lat.max():.4f}s\n\n")
        f.write("Per-cue results:\n")
        f.write(f"{'Time':>8s}  {'Label':>6s}  {'Status':>10s}  {'Samples':>8s}  {'Latency':>8s}  Description\n")
        f.write("-" * 80 + "\n")
        for ts, lbl, desc, status, n_samp, latency in cue_results:
            lat_str = f"{latency:.3f}s" if status == "detected" else "—"
            f.write(f"{ts:8.1f}  {lbl:>6s}  {status:>10s}  {n_samp:>8d}  {lat_str:>8s}  {desc}\n")

    print(f"  Saved: {report_path}")

    # Generate visualization
    generate_visualization(session_dir, timestamps, data, labels, cues, cue_results)

    return labels


def generate_visualization(session_dir, timestamps, data, labels, cues, cue_results):
    """Plot EMG channels with color-coded labels and onset/offset markers."""
    fig, axes = plt.subplots(5, 1, figsize=(20, 15), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.4]})

    ch_names = ["Ch0 (F1 Flexor)", "Ch1 (F5 Extensor)", "Ch2 (F10 Flexor)", "Ch3 (F14 Extensor)"]
    colors = {0: "#ff4444", 1: "#4488ff", 2: "#dddddd"}  # close=red, open=blue, rest=gray
    label_colors = {0: "red", 1: "blue", 2: "gray"}

    # Downsample for plotting if too many points
    max_points = 50000
    if len(timestamps) > max_points:
        step = len(timestamps) // max_points
    else:
        step = 1

    ts_plot = timestamps[::step]
    data_plot = data[::step]
    labels_plot = labels[::step]

    for ch in range(4):
        ax = axes[ch]
        ax.set_ylabel(ch_names[ch], fontsize=9)

        # Plot colored background spans
        # Find contiguous regions of each label
        current_label = labels_plot[0]
        start_idx = 0
        for j in range(1, len(labels_plot)):
            if labels_plot[j] != current_label or j == len(labels_plot) - 1:
                end_idx = j
                ax.axvspan(ts_plot[start_idx], ts_plot[min(end_idx, len(ts_plot)-1)],
                           alpha=0.15, color=colors[current_label], linewidth=0)
                current_label = labels_plot[j]
                start_idx = j

        # Plot signal
        ax.plot(ts_plot, data_plot[:, ch], linewidth=0.3, color="black", alpha=0.8)
        ax.tick_params(labelsize=8)

    # Bottom panel: label timeline
    ax = axes[4]
    ax.set_ylabel("Label", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["close", "open", "rest"], fontsize=8)

    # Plot label as colored segments
    current_label = labels_plot[0]
    start_idx = 0
    for j in range(1, len(labels_plot)):
        if labels_plot[j] != current_label or j == len(labels_plot) - 1:
            end_idx = j
            ax.axvspan(ts_plot[start_idx], ts_plot[min(end_idx, len(ts_plot)-1)],
                       alpha=0.5, color=colors[current_label], linewidth=0)
            current_label = labels_plot[j]
            start_idx = j

    # Mark onset points from cue_results
    for ts, lbl, desc, status, n_samp, latency in cue_results:
        if status == "detected":
            onset_time = ts + latency
            for ch_ax in axes[:4]:
                ch_ax.axvline(onset_time, color="green", linewidth=0.5, alpha=0.7)

    # Legend
    patches = [
        mpatches.Patch(color="#ff4444", alpha=0.4, label="close"),
        mpatches.Patch(color="#4488ff", alpha=0.4, label="open"),
        mpatches.Patch(color="#dddddd", alpha=0.4, label="rest"),
    ]
    axes[0].legend(handles=patches, loc="upper right", fontsize=8)

    plt.suptitle("EMG Labeling Verification", fontsize=14)
    plt.tight_layout()

    png_path = os.path.join(session_dir, "labeling_check.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"  Saved: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Label a recorded EMG session")
    parser.add_argument("--session", required=True, help="Path to session directory")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="Onset detection sensitivity: multiplier on baseline std "
                             "(default 3.0, lower = more sensitive to weak effort)")
    args = parser.parse_args()

    if not os.path.isdir(args.session):
        print(f"Error: {args.session} is not a directory")
        sys.exit(1)

    print(f"  Onset threshold: {args.threshold}x baseline std")
    label_session(args.session, threshold_multiplier=args.threshold)


if __name__ == "__main__":
    main()
