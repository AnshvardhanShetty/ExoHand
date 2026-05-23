"""
preprocessing_physiomio.py — preprocess PhysioMio recordings into the
GrabMyo-schema feature dataframe that ml/train_hgb_v2.engineer_features expects.

Pipeline (per patient):
  1. Load patient's first healthy_arm session.
  2. CHANNEL SELECTION (per-patient empirical, locked decision from Task #36):
     - Candidate pool: all 64 channels EXCEPT channel_49 (paper flags as bad).
     - Score each candidate by Cohen's d on envelope amplitude during
       MassFlexion vs MassExtension windows.
     - Pick top 2 with most positive d  → patient's "flexor" channels
     - Pick top 2 with most negative d  → patient's "extensor" channels
     - These 4 channel indices are frozen for that patient across all sessions.
  3. GESTURE MAPPING (locked from Task #35, option 1):
     - close: MassFlexion, HookGrasp, ThumbAdduction, PinchGrasp,
              PinchGraspMiddle, PinchGraspRing, PinchGraspPinkie,
              DiameterGrasp, SphereGrasp, MassAdduction
     - open:  MassExtension
     - rest:  Rest
     - drop:  WristVolarFlexion, WristDorsiFlexion, ForearmPronation, ForearmSupination
  4. For each session (healthy + impaired) and each kept gesture:
     - Extract the 4 selected channels' raw samples
     - Bandpass 20-450 Hz (Butterworth 4) + 50 Hz notch (German mains)
     - Window into 200 ms windows at 50 ms stride (= 400 samples / 100 stride at 2000 Hz)
     - Compute 60 base features per window via analysis.emgbench.feature_extraction
  5. Build GrabMyo-schema rows: participant, session, gesture, gesture_name, trial,
     t_rel_s, intent, intent_idx, ch{a}_*, ch{b}_*, ch{c}_*, ch{d}_*
     where {a,b,c,d} are the GrabMyo canonical channel names [0, 4, 9, 13]
     (NOT the PhysioMio physical channel numbers — keeps the column schema
     aligned with the GrabMyo-trained model's expected feature names).
  6. Concatenate across all patients; write to a single CSV/pickle.
  7. Per-patient channel-pick metadata + selection rationale saved separately.

After this script, ml.train_hgb_v2.engineer_features() can be applied to expand
the 60 base features into the canonical 370.

Usage:
    # First-time run: select channels per patient and extract features for all
    python ml/preprocessing_physiomio.py

    # Limit to a few patients for smoke testing
    python ml/preprocessing_physiomio.py --patients patient1 patient2

    # Re-select channels without re-extracting features (e.g. after pool change)
    python ml/preprocessing_physiomio.py --reselect-only

Outputs:
    data/physiomio_features_60_per_patient.pkl  -- big concatenated dataframe
    data/physiomio_channel_picks.csv            -- per-patient channel selection
    data/physiomio_feature_log.txt              -- run log
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

from analysis.seed import SEED, seed_everything
from analysis.emgbench.feature_extraction import (
    _features_for_one_window,
)


# ── Paths ────────────────────────────────────────────────────────────────

PHYSIOMIO_ROOT = PROJECT_ROOT / "data" / "physiomio" / "data"
OUT_FEATURES = PROJECT_ROOT / "data" / "physiomio_features_60_per_patient.pkl"
OUT_CHANNEL_PICKS = PROJECT_ROOT / "data" / "physiomio_channel_picks.csv"
OUT_LOG = PROJECT_ROOT / "data" / "physiomio_feature_log.txt"


# ── Locked decisions from Task #35 + #36 ────────────────────────────────

GESTURE_MAP = {
    # close (10 gestures)
    "MassFlexion": "close",
    "HookGrasp": "close",
    "ThumbAdduction": "close",
    "PinchGrasp": "close",
    "PinchGraspMiddle": "close",
    "PinchGraspRing": "close",
    "PinchGraspPinkie": "close",
    "DiameterGrasp": "close",
    "SphereGrasp": "close",
    "MassAdduction": "close",
    # open (1 gesture)
    "MassExtension": "open",
    # rest (1 gesture)
    "Rest": "rest",
    # Wrist + forearm rotation gestures are dropped (not in this dict)
}
INTENT_TO_IDX = {"rest": 0, "close": 1, "open": 2}

# Channels excluded from selection pool (paper flags as bad)
EXCLUDED_CHANNELS = {49}

# Anatomical-label channels in the GrabMyo schema we'll emit.
# Per-patient channels get renamed to these slots so downstream
# train_hgb_v2.engineer_features works on the same column names.
CANONICAL_GRABMYO_NAMES = [0, 4, 9, 13]   # flexor, extensor, flexor, extensor


# ── Signal processing ───────────────────────────────────────────────────

SAMPLE_RATE_HZ = 2000.0     # confirmed empirically from time-column dt
NOTCH_FREQ_HZ = 50.0        # German mains; paper says no notch applied at acquisition
NOTCH_Q = 30.0
BANDPASS_LO_HZ = 20.0
BANDPASS_HI_HZ = 450.0
BANDPASS_ORDER = 4
ENV_SMOOTH_MS = 50.0

WINDOW_MS = 200.0
STRIDE_MS = 50.0
WINDOW_SAMPLES = int(WINDOW_MS / 1000.0 * SAMPLE_RATE_HZ)   # 400
STRIDE_SAMPLES = int(STRIDE_MS / 1000.0 * SAMPLE_RATE_HZ)   # 100


def _bandpass_coefs():
    nyq = 0.5 * SAMPLE_RATE_HZ
    return butter(BANDPASS_ORDER, [BANDPASS_LO_HZ / nyq, BANDPASS_HI_HZ / nyq], btype="band")


def _notch_coefs():
    return iirnotch(NOTCH_FREQ_HZ, NOTCH_Q, SAMPLE_RATE_HZ)


_BP_B, _BP_A = _bandpass_coefs()
_NOTCH_B, _NOTCH_A = _notch_coefs()


def _filter_channel(x: np.ndarray) -> np.ndarray:
    """Notch then bandpass on one 1-D channel."""
    x = x - np.mean(x)
    x = filtfilt(_NOTCH_B, _NOTCH_A, x)
    x = filtfilt(_BP_B, _BP_A, x)
    return x


def _envelope(filt: np.ndarray) -> np.ndarray:
    win = max(1, int(ENV_SMOOTH_MS / 1000.0 * SAMPLE_RATE_HZ))
    kernel = np.ones(win) / win
    return np.convolve(np.abs(filt), kernel, mode="same")


# ── Channel selection (per-patient empirical) ───────────────────────────

def select_channels_for_patient(session_df: pd.DataFrame, log_lines: list, patient: str) -> tuple:
    """Pick 4 channels (2 flexor + 2 extensor) by MassFlexion vs MassExtension Cohen's d.

    Returns:
        list of 4 channel indices (1-indexed PhysioMio names, e.g. [3, 17, 32, 50])
        sorted as [flexor_top1, flexor_top2, extensor_top1, extensor_top2]
        plus a dict of diagnostic info.
    """
    flex_mask = session_df["movement_type"] == "MassFlexion"
    ext_mask = session_df["movement_type"] == "MassExtension"
    if not flex_mask.any() or not ext_mask.any():
        raise ValueError(f"{patient}: missing MassFlexion or MassExtension in first session")

    channel_names = [c for c in session_df.columns if c.startswith("channel_")]
    candidates = [c for c in channel_names if int(c.split("_")[1]) not in EXCLUDED_CHANNELS]

    flex_block = session_df.loc[flex_mask, candidates].values     # (n_flex_samples, n_cand)
    ext_block = session_df.loc[ext_mask, candidates].values

    # Compute envelope per candidate channel on each block
    flex_env = np.array([np.mean(_envelope(_filter_channel(flex_block[:, i]))) for i in range(flex_block.shape[1])])
    ext_env = np.array([np.mean(_envelope(_filter_channel(ext_block[:, i]))) for i in range(ext_block.shape[1])])

    # Per-channel std on each block (for pooled std in Cohen's d)
    flex_std = np.array([np.std(_envelope(_filter_channel(flex_block[:, i])), ddof=1) for i in range(flex_block.shape[1])])
    ext_std = np.array([np.std(_envelope(_filter_channel(ext_block[:, i])), ddof=1) for i in range(ext_block.shape[1])])

    diff = flex_env - ext_env
    pooled = np.sqrt(0.5 * (flex_std ** 2 + ext_std ** 2)) + 1e-12
    d = diff / pooled

    # Pick top 2 most-positive (flexor) and top 2 most-negative (extensor)
    order_desc = np.argsort(-d)   # descending
    flex_picks = order_desc[:2]
    order_asc = np.argsort(d)     # ascending (most negative first)
    ext_picks = order_asc[:2]

    flex_chan_indices = [int(candidates[i].split("_")[1]) for i in flex_picks]
    ext_chan_indices = [int(candidates[i].split("_")[1]) for i in ext_picks]

    # CRITICAL: GrabMyo's CANONICAL_GRABMYO_NAMES = [0, 4, 9, 13] alternates
    # flexor/extensor/flexor/extensor (F1 flexor, F5 extensor, F10 flexor, F14 extensor).
    # Picks must be INTERLEAVED so slot 0 → flexor, slot 1 → extensor, slot 2 → flexor,
    # slot 3 → extensor. Grouping them as [flex1, flex2, ext1, ext2] causes a partial
    # channel swap that destroys the model's flexor_extensor_ratio feature.
    picks = [flex_chan_indices[0], ext_chan_indices[0], flex_chan_indices[1], ext_chan_indices[1]]
    info = {
        "flexor_picks": flex_chan_indices,
        "flexor_d_values": [float(d[i]) for i in flex_picks],
        "extensor_picks": ext_chan_indices,
        "extensor_d_values": [float(d[i]) for i in ext_picks],
    }

    log_lines.append(f"[{patient}] channel selection:")
    log_lines.append(f"  flexor:   channels {flex_chan_indices}  (Cohen's d = {info['flexor_d_values']})")
    log_lines.append(f"  extensor: channels {ext_chan_indices}  (Cohen's d = {info['extensor_d_values']})")

    return picks, info


# ── Feature extraction per session ──────────────────────────────────────

def extract_session_features(
    session_df: pd.DataFrame,
    channel_picks: list,
    patient: str,
    arm: str,
    session_idx: int,
) -> pd.DataFrame:
    """Extract 60-feature windows from one session, keeping only mapped gestures.

    Returns a dataframe with one row per window:
      participant, session, gesture, gesture_name, trial, t_rel_s,
      intent, intent_idx, ch0_*, ch4_*, ch9_*, ch13_*  (60 feature cols)
    """
    feature_names_per_channel = [
        "rms", "mav", "var", "wl", "maxamp", "zc", "ssc", "wamp", "iemg",
        "mean_freq", "median_freq", "env_mean", "env_max", "env_std", "env_rms",
    ]
    rows = []

    # GrabMyo convention: "trial" = one gesture execution (contains many windows
    # at increasing t_rel_s). engineer_features.add_temporal_features groupby
    # (participant, session, trial) and shifts within each trial — so a trial
    # MUST contain multiple windows for the _prev/_delta/_roll features to be
    # non-trivial. We assign trial = index of the gesture within the session
    # (0..11 after dropping 4 wrist/forearm gestures); each trial then contains
    # the ~78 windows extracted from that gesture's 4-second segment.
    for trial_idx, (movement_type, group) in enumerate(session_df.groupby("movement_type", sort=False)):
        intent = GESTURE_MAP.get(movement_type)
        if intent is None:
            continue   # dropped gesture
        intent_idx = INTENT_TO_IDX[intent]

        # Get raw signal per chosen channel for this gesture's segment
        raw = group[[f"channel_{ch:02d}" for ch in channel_picks]].values  # (n_samples, 4)

        # Pre-filter each channel once
        filtered = np.column_stack([_filter_channel(raw[:, i]) for i in range(4)])
        envelope = np.column_stack([_envelope(filtered[:, i]) for i in range(4)])

        n_samples = filtered.shape[0]
        for start in range(0, n_samples - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
            end = start + WINDOW_SAMPLES
            row = {
                "participant": patient,
                "session": f"{arm}_{session_idx:02d}",
                "gesture": intent_idx,
                "gesture_name": intent,
                "trial": trial_idx,
                "t_rel_s": (start + WINDOW_SAMPLES / 2) / SAMPLE_RATE_HZ,
                "intent": intent,
                "intent_idx": intent_idx,
            }
            for canon_ch, src_ch_idx in zip(CANONICAL_GRABMYO_NAMES, range(4)):
                feats = _features_for_one_window(
                    filtered[start:end, src_ch_idx],
                    envelope[start:end, src_ch_idx],
                    SAMPLE_RATE_HZ,
                )
                for fname in feature_names_per_channel:
                    row[f"ch{canon_ch}_{fname}"] = feats[fname]
            rows.append(row)

    return pd.DataFrame(rows)


# ── Main per-patient loop ───────────────────────────────────────────────

def process_patient(patient_dir: Path, log_lines: list) -> tuple:
    """Process all sessions of one patient. Returns (features_df, channel_pick_info)."""
    patient = patient_dir.name

    # First healthy_arm session = channel selection source
    healthy_sessions = sorted((patient_dir / "healthy_arm").glob("*.parquet"))
    impaired_sessions = sorted((patient_dir / "impaired_arm").glob("*.parquet"))

    if not healthy_sessions:
        log_lines.append(f"[{patient}] SKIP — no healthy_arm sessions")
        return None, None

    t0 = time.time()
    first_session = pd.read_parquet(healthy_sessions[0])
    channel_picks, pick_info = select_channels_for_patient(first_session, log_lines, patient)
    pick_info["patient"] = patient
    pick_info["selection_source"] = healthy_sessions[0].name
    pick_info["chosen_channels"] = channel_picks

    all_features = []

    for arm, sessions in [("healthy", healthy_sessions), ("impaired", impaired_sessions)]:
        for idx, session_path in enumerate(sessions, start=1):
            # Avoid re-loading first session
            if arm == "healthy" and idx == 1:
                df = first_session
            else:
                df = pd.read_parquet(session_path)
            feats = extract_session_features(df, channel_picks, patient, arm, idx)
            log_lines.append(f"[{patient}] {arm}_{idx:02d}: extracted {len(feats):>5} windows from {len(df):>7} raw samples")
            all_features.append(feats)

    out_df = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    elapsed = time.time() - t0
    log_lines.append(f"[{patient}] DONE  total_windows={len(out_df):,}  wall={elapsed:.1f}s")
    return out_df, pick_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", nargs="*", default=None,
                        help="Subset (e.g. patient1 patient2). Default = all.")
    parser.add_argument("--reselect-only", action="store_true",
                        help="Re-pick channels but skip feature extraction.")
    parser.add_argument("--out", default=str(OUT_FEATURES))
    args = parser.parse_args()

    seed_everything(SEED)

    all_patient_dirs = sorted(
        [p for p in PHYSIOMIO_ROOT.iterdir() if p.is_dir() and p.name.startswith("patient")],
        key=lambda p: int(p.name.replace("patient", "")),
    )
    if args.patients:
        all_patient_dirs = [p for p in all_patient_dirs if p.name in set(args.patients)]

    print(f"Processing {len(all_patient_dirs)} patient(s). Seed={SEED}.")
    print(f"  Sample rate: {SAMPLE_RATE_HZ:.0f} Hz  Window: {WINDOW_MS:.0f} ms / {STRIDE_MS:.0f} ms stride")
    print(f"  Gesture map: {len(GESTURE_MAP)} gestures kept, {16 - len(GESTURE_MAP)} dropped")
    print(f"  Excluded from channel pool: {sorted(EXCLUDED_CHANNELS)}")

    log_lines = [f"Seed: {SEED}", f"sample_rate_hz: {SAMPLE_RATE_HZ}",
                 f"window_ms: {WINDOW_MS}", f"stride_ms: {STRIDE_MS}",
                 f"gesture_map: {GESTURE_MAP}"]

    feature_pieces = []
    pick_records = []
    for i, p in enumerate(all_patient_dirs, 1):
        print(f"\n[{i}/{len(all_patient_dirs)}] {p.name}...")
        feats, info = process_patient(p, log_lines)
        if info is not None:
            pick_records.append(info)
        if not args.reselect_only and feats is not None and len(feats) > 0:
            feature_pieces.append(feats)

    # Write channel picks
    OUT_CHANNEL_PICKS.parent.mkdir(parents=True, exist_ok=True)
    picks_df = pd.DataFrame(pick_records)
    picks_df.to_csv(OUT_CHANNEL_PICKS, index=False)
    print(f"\nChannel picks written: {OUT_CHANNEL_PICKS}")

    # Write features
    if not args.reselect_only and feature_pieces:
        all_features = pd.concat(feature_pieces, ignore_index=True)
        print(f"Total feature rows: {len(all_features):,}")
        print(f"Total feature cols: {len(all_features.columns)}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        all_features.to_pickle(args.out)
        print(f"Features written:    {args.out}")

    # Write log
    OUT_LOG.write_text("\n".join(log_lines))
    print(f"Log written:         {OUT_LOG}")


if __name__ == "__main__":
    main()
