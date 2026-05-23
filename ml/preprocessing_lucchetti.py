"""
Lucchetti 2025 → GrabMyo-schema 60-feature parquet for cross-population calibration.

Source: data/lucchetti/{stroke,healthy}/{ST,HS}_NN/(ST|HS)NN.mat
Paper:  10.1038/s41597-025-06174-3 · CC-BY 4.0

What this script does:
  1. Load each subject's .mat (scipy.io)
  2. Select 4 EMG channels matching GrabMyo's flexor / extensor anatomy:
       channel 5 (FCR — wrist flexor)     → ch0  ←   flexor 1
       channel 7 (ECR — wrist extensor)   → ch1  ←   extensor 1
       channel 6 (FDS — finger flexor)    → ch2  ←   flexor 2  ← finger flexor!
       channel 8 (EDC — finger extensor)  → ch3  ←   extensor 2 ← finger extensor!
     This matches GrabMyo's CANONICAL_GRABMYO_NAMES = [0,4,9,13] interleaving.
  3. Upsample EMG from 1 kHz → 2 kHz (linear) so the existing 200 ms window =
     400 sample feature pipeline applies unchanged.
  4. Label 3-class intent based on Events.Start/End + task-order convention
     (paper order: BA, BC, SC, PS, HM, HH; tasks 0/1/2 grasps → close,
     task 3 prono-supination → rest, tasks 4/5 reach-with-extended-hand → open).
  5. Outside Event windows = rest.
  6. Extract 60 features (15 × 4 channels) per 200 ms window @ 50 ms stride.
  7. Save parquet with PhysioMio-compatible schema so downstream analyses
     (per_session_eval, severity, transition accuracy, ...) can target either.

What this script does NOT do:
  - Decode the MATLAB string fields (TaskCode, EmgVarName, AngleVarName) —
    those are stored in opaque cell-string format. We rely on the paper's
    documented muscle order and task order.
  - Use the synchronized motion-capture angles for kinematic-event labeling.
    The Events.Start/End markers are sufficient for movement/rest binary,
    and task-order disambiguates close-vs-open under our assumptions.

Output:
  data/lucchetti_features_60_per_subject.pkl
  data/lucchetti_label_log.txt   (per-subject labeling sanity log)
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch

from analysis.seed import SEED, seed_everything
from analysis.emgbench.feature_extraction import _features_for_one_window

# ── Constants ─────────────────────────────────────────────────────────

LUCCHETTI_DIR = PROJECT_ROOT / "data" / "lucchetti"
OUT_PKL = PROJECT_ROOT / "data" / "lucchetti_features_60_per_subject.pkl"
OUT_LOG = PROJECT_ROOT / "data" / "lucchetti_label_log.txt"

# Paper-documented Lucchetti EMG channel order (0-indexed):
#   0 Posterior Deltoid    1 Mediolateral Deltoid    2 Anterior Deltoid
#   3 Triceps              4 Biceps
#   5 Flexor Carpi Radialis (FCR)                    6 Flexor Superficialis Digitorum (FDS)
#   7 Extensor Carpi Radialis (ECR)                  8 Extensor Digitorum Communis (EDC)
#   9 Abductor Pollicis Brevis    10 Abductor Digiti Minimi    11 First Dorsal Interosseus
#
# To match GrabMyo's interleaved [flex, ext, flex, ext] convention:
PICKS_LUCCHETTI = [5, 7, 6, 8]   # FCR, ECR, FDS, EDC
PICK_NAMES = ["FCR", "ECR", "FDS", "EDC"]
# Match GrabMyo's canonical [0, 4, 9, 13] interleaved flex/ext/flex/ext so that
# train_hgb_v2.engineer_features (which references ch0/ch4/ch9/ch13 by name) works.
CANONICAL_GRABMYO_NAMES = [0, 4, 9, 13]

# Sampling
LUCCHETTI_EMG_FS = 1000
GRABMYO_EMG_FS = 2000           # our feature pipeline assumes 2 kHz
KIN_FS = 125                    # events are in kinematic frames

# Window / stride to match PhysioMio + GrabMyo
WINDOW_MS = 200
STRIDE_MS = 50
WIN_SAMPLES = int(WINDOW_MS / 1000.0 * GRABMYO_EMG_FS)   # 400
STRIDE_SAMPLES = int(STRIDE_MS / 1000.0 * GRABMYO_EMG_FS)   # 100

# Filtering (match preprocessing_grabmyo.py and the runtime)
LOW_HZ = 20.0
HIGH_HZ = 450.0
BP_ORDER = 4
NOTCH_HZ = 50.0
NOTCH_Q = 30.0
ENV_MS = 50.0

# Task-order convention (paper: BA, BC, SC, PS, HM, HH)
# Anchored to paper documentation; verified post-hoc via FDS amplitude check.
TASK_INTENT_MAP = {
    0: "close",   # BA — Grasp Ball
    1: "close",   # BC — Grasp 5cm³ Block
    2: "close",   # SC — Grasp 2.5cm³ Block
    3: "rest",    # PS — Prono-supination (wrist rotation; no hand close/open)
    4: "open",    # HM — Hand-to-Mouth (reach with extended hand)
    5: "open",    # HH — Hand-to-Head (reach with extended hand)
}
INTENT_TO_IDX = {"rest": 0, "close": 1, "open": 2}


def build_filters(fs: float):
    nyq = 0.5 * fs
    bp_b, bp_a = butter(BP_ORDER, [LOW_HZ / nyq, HIGH_HZ / nyq], btype="band")
    n_b, n_a = iirnotch(NOTCH_HZ, NOTCH_Q, fs)
    env_w = int(ENV_MS / 1000.0 * fs)
    env_kernel = np.ones(env_w) / env_w
    return bp_b, bp_a, n_b, n_a, env_kernel


def upsample_2x_linear(x: np.ndarray) -> np.ndarray:
    """Upsample 1D signal 1 kHz → 2 kHz via linear interpolation."""
    n_out = len(x) * 2
    return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x)


def label_stream_for_task(n_emg_samples: int, events_start_kin: np.ndarray,
                          events_end_kin: np.ndarray, task_idx: int) -> np.ndarray:
    """Build per-sample intent labels for one task's EMG stream.

    Args:
        n_emg_samples: length of the upsampled (2 kHz) EMG stream
        events_start_kin / events_end_kin: rep start/end frames in 125 Hz kinematic frames
        task_idx: 0..5, mapped via TASK_INTENT_MAP

    Returns:
        intent_idx array of shape (n_emg_samples,) with values in {0, 1, 2}.
    """
    # Kinematic frames at 125 Hz → 2 kHz EMG samples by factor 16
    factor = GRABMYO_EMG_FS // KIN_FS    # = 16
    labels = np.full(n_emg_samples, INTENT_TO_IDX["rest"], dtype=np.int8)
    movement_intent = TASK_INTENT_MAP.get(task_idx, "rest")
    movement_idx = INTENT_TO_IDX[movement_intent]
    for s_kin, e_kin in zip(events_start_kin, events_end_kin):
        s = int(s_kin * factor)
        e = int(e_kin * factor)
        s = max(0, min(s, n_emg_samples))
        e = max(s, min(e, n_emg_samples))
        if movement_intent != "rest":
            labels[s:e] = movement_idx
        # Else: leave as rest (for task 3 / PS — movement IS still "no hand intent")
    return labels


def extract_session_features(subject_id: str, session_id: str,
                             data_struct, log_lines: list) -> pd.DataFrame:
    """Extract 60-feature windows for one subject-arm-session.

    data_struct: numpy array of 6 task objects from s.DataULpleg / s.DataULnonpleg / s.DataULdom
    session_id: e.g. 'impaired_01' or 'healthy_01'
    """
    bp_b, bp_a, n_b, n_a, env_kernel = build_filters(GRABMYO_EMG_FS)
    rows = []
    trial_counter = 0

    fds_per_class = {0: [], 1: [], 2: []}   # for label-quality sanity check

    for task_idx in range(len(data_struct)):
        task = data_struct[task_idx]
        emg = task.EMG    # (12, N) at 1 kHz
        events = task.Events
        starts = np.atleast_1d(events.Start).astype(int) if hasattr(events, "Start") else np.array([], dtype=int)
        ends = np.atleast_1d(events.End).astype(int) if hasattr(events, "End") else np.array([], dtype=int)

        # Select 4 channels in interleaved [flex, ext, flex, ext] order
        emg_sel = emg[PICKS_LUCCHETTI, :].astype(np.float64)   # (4, N_1k)

        # Upsample 1 kHz → 2 kHz
        emg_us = np.stack([upsample_2x_linear(emg_sel[ch]) for ch in range(4)], axis=0)
        N = emg_us.shape[1]

        # Build per-sample intent labels for this task
        labels = label_stream_for_task(N, starts, ends, task_idx)

        # Filter (notch + bandpass + envelope per channel)
        filt = np.empty_like(emg_us)
        envs = np.empty_like(emg_us)
        for ch in range(4):
            x = emg_us[ch] - np.mean(emg_us[ch])
            x = filtfilt(n_b, n_a, x)
            x = filtfilt(bp_b, bp_a, x)
            filt[ch] = x
            envs[ch] = np.convolve(np.abs(x), env_kernel, mode="same")

        # Slide windows. For each window, take majority label and extract features.
        for w_start in range(0, N - WIN_SAMPLES + 1, STRIDE_SAMPLES):
            w_end = w_start + WIN_SAMPLES
            window_labels = labels[w_start:w_end]
            # Only keep pure-class windows (>= 95% one class) to avoid mixed-label noise
            unique, counts = np.unique(window_labels, return_counts=True)
            top_count = counts.max()
            if top_count / len(window_labels) < 0.95:
                continue
            intent_idx = int(unique[counts.argmax()])
            intent_name = {0: "rest", 1: "close", 2: "open"}[intent_idx]
            t_rel_s = w_start / GRABMYO_EMG_FS

            row = {
                "participant": subject_id,
                "session": session_id,
                "gesture": intent_idx,
                "gesture_name": intent_name,
                "trial": trial_counter,
                "t_rel_s": t_rel_s,
                "intent": intent_name,
                "intent_idx": intent_idx,
            }
            # 15 features per channel — write under GrabMyo canonical names so
            # engineer_features (which expects ch0/ch4/ch9/ch13) works
            for canon_ch, src_ch in zip(CANONICAL_GRABMYO_NAMES, range(4)):
                feats = _features_for_one_window(filt[src_ch, w_start:w_end],
                                                 envs[src_ch, w_start:w_end],
                                                 GRABMYO_EMG_FS)
                for k in ["rms", "mav", "var", "wl", "maxamp", "zc", "ssc", "wamp",
                          "iemg", "mean_freq", "median_freq",
                          "env_mean", "env_max", "env_std", "env_rms"]:
                    row[f"ch{canon_ch}_{k}"] = float(feats[k])
            # FDS amplitude check — FDS is local channel 2 (canonical ch9)
            fds_val = row.get("ch9_env_rms", float("nan"))
            if not np.isnan(fds_val):
                fds_per_class[intent_idx].append(fds_val)
            rows.append(row)
            trial_counter += 1   # one window = one "trial" id for engineer_features groupby

    df = pd.DataFrame(rows)

    # Trial fix: engineer_features expects trial = gesture index (multiple windows per
    # trial), not unique per window. Reassign trial = task_index_in_arm so all close
    # windows from one grasp task share a trial id.
    # We re-derive trial from t_rel_s + intent boundaries: each contiguous run of
    # same-intent windows = one trial.
    if len(df) > 0:
        df = df.sort_values("t_rel_s").reset_index(drop=True)
        diff = (df["intent_idx"].diff() != 0).cumsum()
        df["trial"] = diff.astype(np.int32).values
    # Sanity log
    for cls in [0, 1, 2]:
        if fds_per_class[cls]:
            mean_fds = np.mean(fds_per_class[cls])
            log_lines.append(f"    {subject_id} {session_id} class={cls} ({['rest','close','open'][cls]}) "
                             f"n_windows={len(fds_per_class[cls])} mean_FDS_env_rms={mean_fds:.4f}")
    return df


def process_subject(subject_id: str, kind: str, log_lines: list) -> pd.DataFrame:
    """Process one Lucchetti subject. Returns a DataFrame with all arms × all windows."""
    code = subject_id.replace("_", "")
    mat_path = LUCCHETTI_DIR / kind / subject_id / f"{code}.mat"
    if not mat_path.exists():
        log_lines.append(f"MISSING {mat_path}")
        return pd.DataFrame()
    d = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    s = d["s"]

    parts = []
    if kind == "stroke":
        # Both arms: impaired (ULpleg) and non-affected (ULnonpleg)
        for data_field, session in [("DataULpleg", "impaired_01"),
                                    ("DataULnonpleg", "healthy_01")]:
            arm_data = getattr(s, data_field, None)
            if arm_data is None:
                continue
            log_lines.append(f"  {subject_id} {session}: {len(arm_data)} tasks")
            df = extract_session_features(subject_id, session, arm_data, log_lines)
            if len(df) > 0:
                parts.append(df)
    elif kind == "healthy":
        # Only dominant arm
        arm_data = getattr(s, "DataULdom", None)
        if arm_data is not None:
            log_lines.append(f"  {subject_id} healthy_01: {len(arm_data)} tasks")
            df = extract_session_features(subject_id, "healthy_01", arm_data, log_lines)
            if len(df) > 0:
                parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    seed_everything(SEED)
    log_lines = [f"Lucchetti preprocessing log (seed={SEED})"]

    print("=" * 70)
    print("Lucchetti → 60-feature parquet")
    print("=" * 70)
    print(f"  Channel picks (interleaved flex/ext/flex/ext):")
    for i, (idx, name) in enumerate(zip(PICKS_LUCCHETTI, PICK_NAMES)):
        print(f"    ch{i} ← Lucchetti channel {idx} ({name})")
    print(f"  Task-intent map (paper order BA, BC, SC, PS, HM, HH):")
    for k, v in TASK_INTENT_MAP.items():
        print(f"    task {k} → movement = {v}")

    # Stroke subjects
    all_dfs = []
    t_start = time.time()
    print("\nProcessing stroke subjects (10):")
    for i in range(1, 11):
        sid = f"ST_{i:02d}"
        log_lines.append(f"\n--- {sid} ---")
        df = process_subject(sid, "stroke", log_lines)
        if len(df) > 0:
            all_dfs.append(df)
            print(f"  {sid}: {len(df):>6} windows  ({df['session'].nunique()} arms, "
                  f"per-class counts: {df['intent_idx'].value_counts().to_dict()})")

    print("\nProcessing healthy subjects (10):")
    for i in range(1, 11):
        sid = f"HS_{i:02d}"
        log_lines.append(f"\n--- {sid} ---")
        df = process_subject(sid, "healthy", log_lines)
        if len(df) > 0:
            all_dfs.append(df)
            print(f"  {sid}: {len(df):>6} windows  (per-class counts: "
                  f"{df['intent_idx'].value_counts().to_dict()})")

    final = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal: {len(final):,} windows  "
          f"({final['participant'].nunique()} subjects, "
          f"{final.groupby('participant')['session'].nunique().sum()} arms)")
    print(f"  per-class counts (whole dataset): {final['intent_idx'].value_counts().to_dict()}")
    final.to_pickle(OUT_PKL)
    OUT_LOG.write_text("\n".join(log_lines))
    print(f"\nWrote {OUT_PKL}")
    print(f"Wrote {OUT_LOG}")
    print(f"Wall: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
