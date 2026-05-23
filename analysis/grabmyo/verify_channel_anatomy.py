"""
Empirically verify GrabMyo channel anatomy.

The pipeline (`ml/preprocessing_grabmyo.py:15`) uses channels [0, 4, 9, 13]
with an inline comment claiming "F1 = flexor, F5 = extensor, F10 = flexor,
F14 = extensor". This script tests that claim with data, not assumptions.

Procedure:
  1. Load `Hand Close` (gesture 16) and `Hand Open` (gesture 15) trials from
     the first N participants of Session 1.
  2. Per channel: bandpass 20-450 Hz, full-wave rectify, 50 ms envelope smooth.
  3. Compute mean envelope amplitude per channel per trial.
  4. Aggregate across trials → per-channel (close mean, open mean, differential).
  5. Differential > 0 → flexor-dominant; < 0 → extensor-dominant.
  6. Report the canonical [0, 4, 9, 13] channels' verdicts.

Outputs:
  - Console table
  - analysis/grabmyo/channel_anatomy.csv (paper-bound artifact)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import glob

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt

from analysis.seed import SEED, seed_everything

GRABMYO_SESSION1 = PROJECT_ROOT / "grabmyo" / "Session1"
CANONICAL_CHANNELS = [0, 4, 9, 13]  # what preprocessing_grabmyo.py uses
CLOSE_GESTURE = 16
OPEN_GESTURE = 15


def bandpass_filter(x: np.ndarray, fs: float, low: float = 20.0, high: float = 450.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, min(high / nyq, 0.99)], btype="band")
    return filtfilt(b, a, x)


def envelope_smooth(x: np.ndarray, fs: float, win_ms: float = 50.0) -> np.ndarray:
    win = max(1, int(win_ms / 1000.0 * fs))
    kernel = np.ones(win) / win
    return np.convolve(np.abs(x), kernel, mode="same")


def mean_envelope_per_channel(signal: np.ndarray, fs: float) -> np.ndarray:
    """signal shape (n_samples, n_channels). Returns mean envelope per channel."""
    n_channels = signal.shape[1]
    out = np.zeros(n_channels)
    for ch in range(n_channels):
        filtered = bandpass_filter(signal[:, ch] - np.mean(signal[:, ch]), fs)
        env = envelope_smooth(filtered, fs)
        out[ch] = float(np.mean(env))
    return out


def load_trial(dat_path: str) -> tuple:
    rec = wfdb.rdrecord(dat_path.replace(".dat", ""))
    return rec.p_signal, float(rec.fs)


def collect_trials(n_subjects: int) -> tuple:
    """Returns (close_envs, open_envs): each list of (n_channels,) arrays per trial."""
    close_envs = []
    open_envs = []
    for subj_idx in range(1, n_subjects + 1):
        subj_dir = GRABMYO_SESSION1 / f"session1_participant{subj_idx}"
        if not subj_dir.exists():
            continue
        for gesture, bucket in [(CLOSE_GESTURE, close_envs), (OPEN_GESTURE, open_envs)]:
            pattern = str(subj_dir / f"session1_participant{subj_idx}_gesture{gesture}_trial*.dat")
            trials = sorted(glob.glob(pattern))
            for trial_path in trials:
                try:
                    signal, fs = load_trial(trial_path)
                except Exception as e:
                    print(f"  WARN: {trial_path} failed to load: {e}", file=sys.stderr)
                    continue
                if signal.shape[1] != 32:
                    print(f"  WARN: {trial_path} has {signal.shape[1]} channels, expected 32 — skipping", file=sys.stderr)
                    continue
                env = mean_envelope_per_channel(signal, fs)
                bucket.append(env)
        print(f"  subj {subj_idx}: close trials so far={len(close_envs)}, open trials so far={len(open_envs)}")
    return close_envs, open_envs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-subjects", type=int, default=5,
                        help="Number of GrabMyo Session 1 participants to aggregate over.")
    parser.add_argument("--out", default="analysis/grabmyo/channel_anatomy.csv")
    args = parser.parse_args()

    seed_everything(SEED)
    print(f"Aggregating Hand Close (g16) + Hand Open (g15) over first {args.n_subjects} Session 1 participants...")
    close_envs, open_envs = collect_trials(args.n_subjects)
    print(f"\n  total close trials: {len(close_envs)}")
    print(f"  total open trials:  {len(open_envs)}")
    if not close_envs or not open_envs:
        print("ERROR: no trials loaded. Check GRABMYO_SESSION1 path.", file=sys.stderr)
        sys.exit(1)

    close_arr = np.array(close_envs)         # (n_close_trials, 32)
    open_arr = np.array(open_envs)           # (n_open_trials, 32)

    close_mean = close_arr.mean(axis=0)
    close_std = close_arr.std(axis=0, ddof=1)
    open_mean = open_arr.mean(axis=0)
    open_std = open_arr.std(axis=0, ddof=1)
    diff = close_mean - open_mean
    # Normalized differential (Cohen-d-style) — accounts for trial-to-trial spread
    pooled_std = np.sqrt(0.5 * (close_std ** 2 + open_std ** 2))
    d = diff / np.maximum(pooled_std, 1e-12)

    rows = []
    for ch in range(32):
        rows.append({
            "channel_0idx": ch,
            "label_1idx": f"F{ch + 1}",
            "close_mean_env": close_mean[ch],
            "open_mean_env": open_mean[ch],
            "diff_close_minus_open": diff[ch],
            "cohens_d": d[ch],
            "anatomy_call": "flexor" if diff[ch] > 0 else "extensor",
            "is_canonical": ch in CANONICAL_CHANNELS,
        })
    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Print table sorted by |d| descending — most discriminative first
    df_sorted = df.reindex(df["cohens_d"].abs().sort_values(ascending=False).index)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:+.4f}")
    print("\n=== ALL 32 CHANNELS, sorted by |Cohen's d| (most discriminative first) ===")
    print(df_sorted[["channel_0idx", "label_1idx", "close_mean_env", "open_mean_env",
                     "diff_close_minus_open", "cohens_d", "anatomy_call", "is_canonical"]]
          .to_string(index=False))

    print("\n=== CANONICAL [0, 4, 9, 13] (what preprocessing_grabmyo.py uses) ===")
    print("  Inline comment claims: F1=flexor, F5=extensor, F10=flexor, F14=extensor")
    print()
    canonical_claims = {0: "flexor", 4: "extensor", 9: "flexor", 13: "extensor"}
    all_correct = True
    for ch in CANONICAL_CHANNELS:
        row = df.iloc[ch]
        claim = canonical_claims[ch]
        verdict = "MATCH" if row["anatomy_call"] == claim else "MISMATCH"
        if verdict == "MISMATCH":
            all_correct = False
        print(f"  ch{ch} (F{ch+1}): comment says {claim:8s}  empirical says {row['anatomy_call']:8s}  "
              f"diff={row['diff_close_minus_open']:+.4f}  d={row['cohens_d']:+.4f}  → {verdict}")

    print("\n=== SUMMARY ===")
    if all_correct:
        print(f"  All 4 canonical channels match the inline-comment anatomy claims.")
    else:
        print(f"  ⚠  At least one canonical channel contradicts the inline comment. Pipeline assumption is wrong.")

    print(f"\n  Most flexor-dominant channels: {df.nlargest(4, 'diff_close_minus_open')['channel_0idx'].tolist()}")
    print(f"  Most extensor-dominant channels: {df.nsmallest(4, 'diff_close_minus_open')['channel_0idx'].tolist()}")
    print(f"\n  CSV written: {out_path}")


if __name__ == "__main__":
    main()
