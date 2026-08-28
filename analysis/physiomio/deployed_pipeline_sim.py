"""
Simulate the deployed (20 Hz P-P amplitude) pipeline on PhysioMio raw EMG.

Question: how much accuracy do we lose by giving up the raw 2 kHz EMG and using
only what the Teensy actually transmits (peak-to-peak amplitude per 50 ms window,
sampled at 20 Hz)?

Pipeline:
  1. Load raw PhysioMio EMG (2 kHz, 4 selected channels per patient).
  2. Apply the same bandpass + notch as the headline preprocessing.
  3. Compute 50 ms-window P-P amplitude per channel → 20 Hz stream.
     This is what the Teensy firmware does on-device.
  4. Window the 20 Hz P-P stream into 200 ms inference windows
     (4 P-P samples per window) at 50 ms stride (1 sample stride).
  5. Compute the same 60 base features per window using fs=20 Hz.
     Most "raw waveform" features (zc, ssc, mean_freq) will be degenerate
     on 4 samples; the amplitude features (rms, mav, env_*) remain meaningful.
  6. Engineer features (lags, deltas, cross-channel, per-participant z-score)
     → 370 features per window.
  7. Per-session split: first 36 windows/gesture cal, last 39 test (same as
     per_session_eval.py).
  8. Patient-only HGB fit, score, compare to baseline 0.878.

Output:
  analysis/physiomio/results/deployed_pipeline_sim.csv
  analysis/physiomio/results/deployed_pipeline_sim_summary.md
"""

import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from analysis.seed import SEED, seed_everything
from analysis.emgbench.feature_extraction import _features_for_one_window
from ml.train_hgb_v2 import engineer_features
from ml.preprocessing_physiomio import (
    SAMPLE_RATE_HZ, BANDPASS_ORDER, BANDPASS_LO_HZ, BANDPASS_HI_HZ,
    NOTCH_FREQ_HZ, NOTCH_Q, GESTURE_MAP, INTENT_TO_IDX,
)

# alias to local names this script already uses
GESTURE_TO_INTENT = GESTURE_MAP
BANDPASS_LOW_HZ = BANDPASS_LO_HZ
BANDPASS_HIGH_HZ = BANDPASS_HI_HZ

PHYSIOMIO_ROOT = PROJECT_ROOT / "data" / "physiomio" / "data"
CHANNEL_PICKS = PROJECT_ROOT / "data" / "physiomio_channel_picks.csv"

OUT_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "deployed_pipeline_sim.csv"
OUT_MD = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "deployed_pipeline_sim_summary.md"

CANONICAL_NAMES = [0, 4, 9, 13]   # GrabMyo channel naming convention

# Deployed pipeline parameters
TEENSY_PP_WINDOW_MS = 50       # P-P amplitude window size on Teensy
TEENSY_OUTPUT_HZ = 1000.0 / TEENSY_PP_WINDOW_MS   # 20 Hz output rate
INFERENCE_WINDOW_MS = 200       # ML inference window size (same as headline)
INFERENCE_STRIDE_MS = 50        # ML inference stride

# Eval-cal parameters (same as per_session_eval.py)
CAL_PER_GESTURE = 36
BUFFER = 3
TEST_PER_CLASS = 39
CLASSES = [0, 1, 2]


def _bandpass_notch(x):
    """Same bandpass + notch as preprocessing_physiomio."""
    nyq = 0.5 * SAMPLE_RATE_HZ
    b, a = butter(BANDPASS_ORDER, [BANDPASS_LOW_HZ / nyq, BANDPASS_HIGH_HZ / nyq], btype="band")
    nb, na = iirnotch(NOTCH_FREQ_HZ, NOTCH_Q, SAMPLE_RATE_HZ)
    y = filtfilt(b, a, x)
    y = filtfilt(nb, na, y)
    return y


def _pp_amplitude_downsample(filtered_raw, samples_per_pp):
    """Compute peak-to-peak amplitude per non-overlapping window — Teensy behaviour."""
    n_full = (len(filtered_raw) // samples_per_pp) * samples_per_pp
    chunks = filtered_raw[:n_full].reshape(-1, samples_per_pp)
    return chunks.max(axis=1) - chunks.min(axis=1)


def _windows_at_20hz(pp_stream, win_samples, stride_samples):
    """Slice a 20 Hz P-P stream into ML inference windows."""
    n = len(pp_stream)
    starts = list(range(0, n - win_samples + 1, stride_samples))
    return np.stack([pp_stream[s:s + win_samples] for s in starts]), starts


def _envelope_kernel_4samples(env_smooth_ms, fs):
    """Tiny moving-average kernel — at fs=20 Hz a typical 200ms smoothing = 4 samples."""
    win = max(1, int(env_smooth_ms / 1000.0 * fs))
    return np.ones(win) / win


def extract_session_features_20hz(parquet_path, channels):
    """Load one session's raw EMG and produce the 20 Hz-pipeline feature dataframe."""
    df = pd.read_parquet(parquet_path)

    pp_per_window = int(TEENSY_PP_WINDOW_MS / 1000.0 * SAMPLE_RATE_HZ)   # 100
    win_samples_20hz = int(INFERENCE_WINDOW_MS / 1000.0 * TEENSY_OUTPUT_HZ)  # 4
    stride_samples_20hz = int(INFERENCE_STRIDE_MS / 1000.0 * TEENSY_OUTPUT_HZ)  # 1
    env_kernel = _envelope_kernel_4samples(env_smooth_ms=200, fs=TEENSY_OUTPUT_HZ)

    feature_names_per_channel = [
        "rms", "mav", "var", "wl", "maxamp", "zc", "ssc", "wamp", "iemg",
        "mean_freq", "median_freq", "env_mean", "env_max", "env_std", "env_rms",
    ]

    rows = []
    trial_id = 0
    for (gesture, gesture_df) in df.groupby("movement_type", sort=False):
        if gesture not in GESTURE_TO_INTENT:
            continue
        intent = GESTURE_TO_INTENT[gesture]
        if intent is None:
            continue
        intent_idx = INTENT_TO_IDX[intent]
        trial_id += 1

        # Extract + filter each of 4 chosen channels for this gesture's segment
        pp_streams = []  # list of (n_pp,) arrays per channel
        for ch in channels:
            raw_ch = gesture_df[f"channel_{ch:02d}"].values
            filtered = _bandpass_notch(raw_ch)
            pp = _pp_amplitude_downsample(filtered, pp_per_window)
            pp_streams.append(pp)
        # Truncate all channels to common length
        min_n = min(len(s) for s in pp_streams)
        pp_4ch = np.stack([s[:min_n] for s in pp_streams], axis=1)   # (n_pp, 4)

        # Build envelope per channel (moving-average rectified) for env_* features
        env_4ch = np.stack([
            np.convolve(np.abs(pp_4ch[:, c]), env_kernel, mode="same")
            for c in range(4)
        ], axis=1)

        # Window into inference windows
        n_pp = pp_4ch.shape[0]
        starts = list(range(0, n_pp - win_samples_20hz + 1, stride_samples_20hz))
        for win_start in starts:
            win_end = win_start + win_samples_20hz
            row = {
                "participant": None,
                "session": None,
                "gesture_name": gesture,
                "trial": trial_id,
                "t_rel_s": (win_start + win_samples_20hz / 2) / TEENSY_OUTPUT_HZ,
                "intent": intent,
                "intent_idx": intent_idx,
            }
            for ci, canon in enumerate(CANONICAL_NAMES):
                w = pp_4ch[win_start:win_end, ci]
                env = env_4ch[win_start:win_end, ci]
                feats = _features_for_one_window(w, env, TEENSY_OUTPUT_HZ)
                for fname in feature_names_per_channel:
                    row[f"ch{canon}_{fname}"] = feats[fname]
            rows.append(row)
    return pd.DataFrame(rows)


def split_session(session_df, cal_per_gesture, test_per_class, rng):
    """Same per-gesture temporal split as per_session_eval.py."""
    cal_idx = []
    test_pool = {0: [], 1: [], 2: []}
    for _, g in session_df.groupby("trial", sort=True):
        cls = int(g["intent_idx"].iloc[0])
        sg = g.sort_values("t_rel_s")
        n = len(sg)
        if n >= cal_per_gesture + BUFFER + test_per_class:
            cal_idx.extend(sg.index[:cal_per_gesture].tolist())
            test_pool[cls].extend(
                sg.index[cal_per_gesture + BUFFER:cal_per_gesture + BUFFER + test_per_class].tolist()
            )
        else:
            cal_n = max(1, min(cal_per_gesture, n - BUFFER - 1))
            cal_idx.extend(sg.index[:cal_n].tolist())
            test_pool[cls].extend(sg.index[cal_n + BUFFER:].tolist())
    balanced_test = []
    for cls in CLASSES:
        pool = test_pool[cls]
        if len(pool) <= test_per_class:
            balanced_test.extend(pool)
        else:
            balanced_test.extend(rng.choice(pool, size=test_per_class, replace=False).tolist())
    return np.array(sorted(balanced_test)), np.array(sorted(cal_idx))


def make_clf(seed):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def main():
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)

    picks = pd.read_csv(CHANNEL_PICKS)
    picks["chosen_channels"] = picks["chosen_channels"].apply(json.loads)
    pick_map = dict(zip(picks["patient"], picks["chosen_channels"]))

    patients = sorted([p for p in PHYSIOMIO_ROOT.iterdir() if p.is_dir() and p.name.startswith("patient")],
                      key=lambda p: int(p.name.replace("patient", "")))

    print(f"Simulating deployed (20 Hz P-P) pipeline on impaired_01 for {len(patients)} patients...")
    print(f"  Teensy P-P window: {TEENSY_PP_WINDOW_MS} ms → output rate {TEENSY_OUTPUT_HZ:.0f} Hz")
    print(f"  Inference window:  {INFERENCE_WINDOW_MS} ms = {int(INFERENCE_WINDOW_MS / 1000 * TEENSY_OUTPUT_HZ)} P-P samples")
    print(f"  Inference stride:  {INFERENCE_STRIDE_MS} ms = {int(INFERENCE_STRIDE_MS / 1000 * TEENSY_OUTPUT_HZ)} P-P samples")
    print()

    t_start = time.time()
    all_session_dfs = []

    for pi, pdir in enumerate(patients, 1):
        patient = pdir.name
        impaired_01 = pdir / "impaired_arm" / "01.parquet"
        if not impaired_01.exists():
            print(f"  [skip] {patient}: no impaired_arm/01.parquet")
            continue
        channels = pick_map.get(patient)
        if channels is None:
            print(f"  [skip] {patient}: no channel pick")
            continue

        try:
            session_df = extract_session_features_20hz(impaired_01, channels)
        except Exception as e:
            print(f"  [err]  {patient}: {e}")
            continue
        session_df["participant"] = patient
        session_df["session"] = "impaired_01"
        all_session_dfs.append(session_df)
        print(f"  [{pi:>2d}/{len(patients)}] {patient}: {len(session_df)} windows ({time.time()-t_start:.0f}s elapsed)", flush=True)

    full_df = pd.concat(all_session_dfs, ignore_index=True)
    print(f"\nConcatenated: {len(full_df)} windows from {full_df.participant.nunique()} patients")

    # Engineer features (adds temporal lags, cross-channel ratios, per-participant z-score → 370)
    print("Engineering features (60 → 370)...")
    eng = engineer_features(full_df)
    print(f"  shape: {eng.shape}")

    # Identify feature columns (everything except meta)
    meta_cols = {"participant", "session", "gesture_name", "trial", "t_rel_s", "intent", "intent_idx"}
    feature_cols = [c for c in eng.columns if c not in meta_cols]
    print(f"  {len(feature_cols)} feature columns")

    # Per-session patient-only HGB fit
    rows = []
    for participant in sorted(eng.participant.unique()):
        s_data = eng[(eng.participant == participant) & (eng.session == "impaired_01")].copy()
        test_idx, cal_idx = split_session(s_data, CAL_PER_GESTURE, TEST_PER_CLASS, rng)
        if len(test_idx) == 0 or len(cal_idx) == 0:
            print(f"  [skip-eval] {participant}: empty split")
            continue
        X_cal = s_data.loc[cal_idx, feature_cols].values.astype(np.float32)
        y_cal = s_data.loc[cal_idx, "intent_idx"].values.astype(np.int64)
        X_test = s_data.loc[test_idx, feature_cols].values.astype(np.float32)
        y_test = s_data.loc[test_idx, "intent_idx"].values.astype(np.int64)
        if len(np.unique(y_cal)) < 2:
            print(f"  [skip-eval] {participant}: single class in cal")
            continue
        scaler = StandardScaler()
        X_cal_s = scaler.fit_transform(X_cal)
        X_test_s = scaler.transform(X_test).astype(np.float32)
        clf = make_clf(SEED)
        try:
            clf.fit(X_cal_s, y_cal)
        except Exception as e:
            print(f"  [skip-eval] {participant}: fit failed ({e})")
            continue
        preds = clf.predict(X_test_s)
        acc = accuracy_score(y_test, preds)
        f1m = f1_score(y_test, preds, average="macro", zero_division=0)
        cls_f1 = f1_score(y_test, preds, average=None, labels=CLASSES, zero_division=0)
        rows.append({
            "participant": participant,
            "n_cal": int(len(cal_idx)),
            "n_test": int(len(test_idx)),
            "acc": acc, "f1_macro": f1m,
            "f1_rest": cls_f1[0], "f1_close": cls_f1[1], "f1_open": cls_f1[2],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # Summary
    md = [
        "# Deployed-pipeline simulation on PhysioMio",
        "",
        "Simulates the deployment (20 Hz P-P amplitude) pipeline on PhysioMio raw EMG,",
        "to estimate the accuracy cost of giving up raw 2 kHz EMG.",
        "",
        f"- Patients evaluated: **{len(df)}**",
        f"- Same per-session cal protocol (first 36 windows/gesture, balanced 117-window test)",
        f"- Patient-only HGB (no GrabMyo), 370 features computed at fs={TEENSY_OUTPUT_HZ:.0f} Hz",
        "",
        "## Result",
        "",
        f"- **20 Hz P-P simulated pipeline:** {df.acc.mean():.4f} session-mean / patient-mean {df.acc.mean():.4f}",
        f"  (median {df.acc.median():.4f}, std {df.acc.std():.4f}, min {df.acc.min():.4f}, max {df.acc.max():.4f})",
        f"- **Baseline (raw 2 kHz, patient-only):** 0.8777 (from cal_size_sweep_v1.csv, n=48)",
        f"- **Δ (deployed sim − baseline):** {df.acc.mean() - 0.8777:+.4f}",
        "",
        "## Per-class F1",
        "",
        f"- rest:  {df.f1_rest.mean():.4f}",
        f"- close: {df.f1_close.mean():.4f}",
        f"- open:  {df.f1_open.mean():.4f}",
    ]
    OUT_MD.write_text("\n".join(md))

    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(f"\n=== RESULT ===")
    print(f"  Deployed-sim acc (patient-mean): {df.acc.mean():.4f}  (n={len(df)})")
    print(f"  Baseline raw 2 kHz patient-only:  0.8777")
    print(f"  Δ:                                 {df.acc.mean() - 0.8777:+.4f}")


if __name__ == "__main__":
    main()
