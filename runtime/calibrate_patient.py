"""
calibrate_patient.py — Patient-specific calibration system (6-minute protocol).

Guides the patient through structured EMG trials (close/open/rest), collects
labeled data, fine-tunes the model, and adapts gain/threshold parameters.

Protocol phases:
  1. Rest baseline       (10s)  — noise floor
  2. Familiarization     (~60s) — 2 blocked reps per gesture
  3. Sustained holds     (~3.5min) — 8 interleaved reps
  4. Quick contractions   (~90s) — rapid pulses (close/open only)
  5. Variable effort      (~60s) — light/medium/hard

Abbreviated recalibration (returning sessions): 10s rest + 3 reps interleaved = ~90s.

Robustness features:
  - Onset trimming: first 1s of each hold discarded (reaction time)
  - Outlier rejection: electrode pop detection per trial
  - Per-trial quality validation: minimum sample counts, empty trial handling
  - Per-class validation: minimum data requirements with clinician warnings
  - Calibration quality report: SNR, separability, class balance
  - Non-interactive mode for web/headless operation
  - Scaler refitting on patient data for session models
  - Audio pacing control to prevent TTS overlap
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class CalibrationTrial:
    """One trial in the calibration protocol."""
    gesture: str             # "close", "open", "rest"
    label: int               # 0=close, 1=open, 2=rest
    effort: str              # "normal", "light", "medium", "hard", "pulse"
    duration: float          # seconds to hold
    rest_period: float       # seconds of rest after
    phase: int               # protocol phase (1-5)
    samples: list = field(default_factory=list)  # collected EMG samples
    quality: str = "ok"      # "ok", "low_samples", "rejected", "no_response"


@dataclass
class CalibrationResult:
    """Output of a complete calibration."""
    patient_id: str
    timestamp: float
    calibration_type: str        # "full" or "abbreviated"
    rest_baseline: dict          # per-channel noise floor stats
    trials: List[CalibrationTrial]
    raw_samples: np.ndarray      # (N, 4) all EMG samples
    raw_labels: np.ndarray       # (N,) per-sample labels
    sample_rate: float
    per_class_stats: dict        # {label: {mean_amp, median_amp, std_amp, snr}}
    finetuned_model_data: dict   # model dict ready for RealtimePredictor
    calibration_params: dict     # gain/threshold overrides
    quality_report: dict = field(default_factory=dict)


LABEL_MAP = {"close": 0, "open": 1, "rest": 2}
GESTURE_NAMES = {0: "close", 1: "open", 2: "rest"}

# GrabMyo label order: {rest:0, close:1, open:2}
# Session label order: {close:0, open:1, rest:2}
GRABMYO_TO_SESSION_LABEL = {0: 2, 1: 0, 2: 1}  # rest→rest, close→close, open→open

# Onset trimming: discard first N seconds of each hold (reaction time delay).
# Stroke patients may need 0.5–2s to initiate movement after the cue.
ONSET_TRIM_S = 1.0
# Minimum trim for pulse (short) trials
PULSE_ONSET_TRIM_S = 0.2

# Minimum usable samples per trial (below this, flag as low quality)
MIN_TRIAL_SAMPLES = 5
# Minimum feature windows per class for training (below this, warn)
MIN_CLASS_WINDOWS = 20

# Artifact threshold: samples where any channel jumps more than this many
# standard deviations from the trial's running mean are rejected
ARTIFACT_SIGMA = 5.0


# ── Protocol builders ───────────────────────────────────────────────────────

def build_full_protocol() -> List[CalibrationTrial]:
    """Build the 5-phase, ~6-minute trial sequence."""
    trials = []

    # Phase 2: Familiarization — 2 blocked reps per gesture (5s hold, 5s rest)
    for gesture in ["close", "open", "rest"]:
        for _ in range(2):
            trials.append(CalibrationTrial(
                gesture=gesture, label=LABEL_MAP[gesture],
                effort="normal", duration=5.0, rest_period=5.0, phase=2,
            ))

    # Phase 3: Sustained holds — 8 reps x 3 classes, interleaved
    # 5s hold + 4s rest, with a 30s break after rep 4
    gestures_interleaved = ["close", "open", "rest"] * 8
    for i, gesture in enumerate(gestures_interleaved):
        rest = 4.0
        # 30s break midway (after 12 trials = 4 reps of each)
        if i == 12:
            rest = 30.0
        trials.append(CalibrationTrial(
            gesture=gesture, label=LABEL_MAP[gesture],
            effort="normal", duration=5.0, rest_period=rest, phase=3,
        ))

    # Phase 4: Quick contractions — 5 rapid pulses x 3 sets x close and open
    # (rest pulses are nonsensical — "quickly relax" has no EMG signature)
    for _ in range(3):
        for gesture in ["close", "open"]:
            for _ in range(5):
                trials.append(CalibrationTrial(
                    gesture=gesture, label=LABEL_MAP[gesture],
                    effort="pulse", duration=1.0, rest_period=1.0, phase=4,
                ))

    # Phase 5: Variable effort — light/medium/hard for close and open (3s each)
    for gesture in ["close", "open"]:
        for effort in ["light", "medium", "hard"]:
            trials.append(CalibrationTrial(
                gesture=gesture, label=LABEL_MAP[gesture],
                effort=effort, duration=3.0, rest_period=3.0, phase=5,
            ))

    return trials


def build_abbreviated_protocol() -> List[CalibrationTrial]:
    """Build the ~90-second recalibration protocol."""
    trials = []
    # 3 reps per class, interleaved (5s hold + 4s rest)
    for _ in range(3):
        for gesture in ["close", "open", "rest"]:
            trials.append(CalibrationTrial(
                gesture=gesture, label=LABEL_MAP[gesture],
                effort="normal", duration=5.0, rest_period=4.0, phase=3,
            ))
    return trials


# ── TTS cues with pacing control ───────────────────────────────────────────

# Track the last TTS process so we can wait for it before starting another
_last_tts_proc = None
# Suppress TTS in web mode (Node.js integration)
_tts_muted = False
# Emit live EMG readings as JSON to stdout (for CalibrationBridge)
_emit_emg = False
# Web-mode UI pause: instruction (3s) + countdown (3s)
_WEB_PAUSE_SEC = 6.0


def _say(text, blocking=False):
    """macOS TTS cue with pacing control.

    Waits for any in-flight TTS to finish before starting a new one
    to prevent overlapping audio during rapid trial sequences.
    """
    global _last_tts_proc

    if _tts_muted:
        return

    # Wait for previous TTS to finish (non-blocking poll, max 3s)
    if _last_tts_proc is not None:
        try:
            _last_tts_proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            _last_tts_proc.kill()
        _last_tts_proc = None

    try:
        p = subprocess.Popen(["say", text])
        _last_tts_proc = p
        if blocking:
            p.wait()
            _last_tts_proc = None
    except FileNotFoundError:
        pass


def _countdown(seconds=3):
    """Speak a countdown before a trial. Gives stroke patients time to prepare."""
    for i in range(seconds, 0, -1):
        _say(str(i), blocking=True)
        time.sleep(0.5)


def _announce_trial(trial, idx, total, with_countdown=True):
    """Speak appropriate cue for a trial with optional countdown."""
    effort_prefix = ""
    if trial.effort == "light":
        effort_prefix = "gently "
    elif trial.effort == "hard":
        effort_prefix = "strongly "
    elif trial.effort == "pulse":
        effort_prefix = "quickly "

    gesture_cue = {
        "close": "close your hand",
        "open": "open your hand",
        "rest": "relax",
    }[trial.gesture]

    _say(f"{effort_prefix}{gesture_cue}", blocking=True)

    # Countdown for non-pulse trials (pulses are too fast for countdown)
    if with_countdown and trial.effort != "pulse" and trial.duration >= 3.0:
        _countdown(3)


# ── EMG collection ──────────────────────────────────────────────────────────

def _collect_emg_segment(ser, duration_s, label):
    """Read serial EMG for a timed segment. Returns list of (vals, label) tuples.

    Same serial reading pattern as rest_calibrate() in run_exohand.py.
    """
    from run_exohand import parse_emg_line

    gesture_name = GESTURE_NAMES.get(label, "rest")
    samples = []
    ser.reset_input_buffer()
    t_start = time.perf_counter()

    while time.perf_counter() - t_start < duration_s:
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
            continue

        samples.append((vals, label))

        # Emit live EMG for web UI display
        if _emit_emg:
            print(json.dumps({"type": "emg", "emg": vals.tolist() if hasattr(vals, 'tolist') else list(vals), "gesture": gesture_name}), flush=True)

    return samples


# ── Trial data cleaning ────────────────────────────────────────────────────

def _trim_onset(samples, sample_rate, trial):
    """Remove the onset period from a trial's samples.

    Stroke patients have 0.5–2s reaction time after cue. The first portion
    of the hold period is likely still at rest, so labeling it as the target
    gesture would contaminate training data.

    Returns the trimmed sample list.
    """
    if not samples:
        return samples

    trim_s = PULSE_ONSET_TRIM_S if trial.effort == "pulse" else ONSET_TRIM_S
    n_trim = int(trim_s * sample_rate)

    if n_trim >= len(samples):
        # Trial is shorter than the trim window — keep at least half
        n_trim = max(0, len(samples) // 2)

    return samples[n_trim:]


def _reject_outliers(samples):
    """Remove electrode pop / cable movement artifacts from a trial's samples.

    Uses a robust z-score approach: samples where any channel deviates more
    than ARTIFACT_SIGMA standard deviations from the trial median are removed.

    Returns (cleaned_samples, n_rejected).
    """
    if len(samples) < 10:
        return samples, 0

    vals = np.array([s[0] for s in samples])
    labels = [s[1] for s in samples]

    # Robust center and spread (median + MAD instead of mean + std)
    median = np.median(vals, axis=0)
    mad = np.median(np.abs(vals - median), axis=0)
    # MAD-based std estimate (1.4826 is the consistency constant for normal dist)
    robust_std = mad * 1.4826
    robust_std = np.maximum(robust_std, 1e-6)  # avoid division by zero

    # Flag samples that deviate too far on any channel
    z_scores = np.abs(vals - median) / robust_std
    is_ok = np.all(z_scores < ARTIFACT_SIGMA, axis=1)

    cleaned = [(vals[i].tolist(), labels[i]) for i in range(len(samples)) if is_ok[i]]
    n_rejected = len(samples) - len(cleaned)

    return cleaned, n_rejected


def _validate_trial(trial, sample_rate):
    """Validate a completed trial. Sets trial.quality field.

    Returns True if the trial produced usable data.
    """
    n = len(trial.samples)

    if n == 0:
        trial.quality = "no_response"
        return False

    if n < MIN_TRIAL_SAMPLES:
        trial.quality = "low_samples"
        return True  # still usable, just flagged

    # Check if patient responded at all (for non-rest trials):
    # if all channels are near-zero, the patient didn't contract
    if trial.label != 2:  # not rest
        vals = np.array(trial.samples)
        rms = np.sqrt(np.mean(vals ** 2))
        if rms < 1e-4:
            trial.quality = "no_response"
            return False

    trial.quality = "ok"
    return True


# ── Calibration quality report ─────────────────────────────────────────────

def _compute_quality_report(raw_samples, raw_labels, per_class_stats, trials,
                            feature_labels, sample_rate):
    """Generate a calibration quality report for the clinician.

    Checks:
      - Per-class sample counts and balance
      - Class separability (inter-class vs intra-class variance)
      - Trial quality summary
      - Overall calibration grade
    """
    report = {
        "grade": "GOOD",
        "warnings": [],
        "class_counts": {},
        "trial_quality": {"ok": 0, "low_samples": 0, "no_response": 0, "rejected": 0},
        "separability": {},
    }

    X = np.array(raw_samples)
    y = np.array(raw_labels)

    # Per-class sample and window counts
    for label in [0, 1, 2]:
        name = GESTURE_NAMES[label]
        n_samples = int(np.sum(y == label))
        n_windows = int(np.sum(feature_labels == label)) if feature_labels is not None else 0
        report["class_counts"][name] = {
            "samples": n_samples,
            "windows": n_windows,
        }
        if n_windows < MIN_CLASS_WINDOWS:
            report["warnings"].append(
                f"'{name}' has only {n_windows} feature windows "
                f"(minimum {MIN_CLASS_WINDOWS}). Model may be unreliable for this gesture."
            )

    # Trial quality tally
    for trial in trials:
        q = trial.quality if trial.quality in report["trial_quality"] else "ok"
        report["trial_quality"][q] += 1

    n_bad = report["trial_quality"]["no_response"] + report["trial_quality"]["rejected"]
    if n_bad > len(trials) * 0.3:
        report["warnings"].append(
            f"{n_bad}/{len(trials)} trials had no response or were rejected. "
            "Check electrode placement and patient engagement."
        )

    # Class separability: ratio of between-class variance to within-class variance
    # (simplified Fisher's criterion on per-sample RMS)
    class_means = []
    class_vars = []
    for label in [0, 1, 2]:
        mask = y == label
        if mask.sum() < 2:
            continue
        amps = np.sqrt(np.mean(X[mask] ** 2, axis=1))
        class_means.append(np.mean(amps))
        class_vars.append(np.var(amps))

    if len(class_means) >= 2:
        overall_mean = np.mean(class_means)
        between_var = np.var(class_means) * len(class_means)
        within_var = np.mean(class_vars)
        fisher = between_var / (within_var + 1e-9)
        report["separability"]["fisher_ratio"] = round(float(fisher), 2)

        if fisher < 0.5:
            report["warnings"].append(
                f"Low class separability (Fisher={fisher:.2f}). "
                "The patient's close/open/rest signals are hard to distinguish. "
                "Consider electrode repositioning."
            )

    # Per-class SNR
    for label in [0, 1, 2]:
        name = GESTURE_NAMES[label]
        stats = per_class_stats.get(label, {})
        report["separability"][f"{name}_snr"] = round(stats.get("snr", 0.0), 2)

    # Overall grade
    if len(report["warnings"]) == 0:
        report["grade"] = "GOOD"
    elif len(report["warnings"]) <= 2:
        report["grade"] = "FAIR"
    else:
        report["grade"] = "POOR"

    return report


def _print_quality_report(report):
    """Print calibration quality report to console."""
    grade = report["grade"]
    grade_colors = {"GOOD": "", "FAIR": " (consider recalibrating)", "POOR": " (check electrodes)"}

    print(f"\n  Calibration Quality: {grade}{grade_colors.get(grade, '')}")
    print(f"  ─────────────────────────────")

    # Class counts
    for name, counts in report["class_counts"].items():
        print(f"    {name:6s}: {counts['samples']:5d} samples, {counts['windows']:4d} windows")

    # Trial quality
    tq = report["trial_quality"]
    print(f"    Trials : {tq['ok']} ok, {tq['low_samples']} low, "
          f"{tq['no_response']} no-response, {tq['rejected']} rejected")

    # Separability
    sep = report["separability"]
    if "fisher_ratio" in sep:
        print(f"    Fisher ratio: {sep['fisher_ratio']:.2f}")

    # Warnings
    if report["warnings"]:
        print(f"\n  Warnings:")
        for w in report["warnings"]:
            print(f"    ! {w}")


# ── Feature extraction bridge ───────────────────────────────────────────────

def _extract_calibration_features(raw_samples, raw_labels, sample_rate, model_data):
    """Extract features from calibration data, dispatching to the correct pipeline.

    Returns (features, labels) as numpy arrays ready for model training.
    """
    model_type = model_data.get("model_type", "session")

    if model_type == "adapted_hgb":
        return _extract_adapted_features(raw_samples, raw_labels, sample_rate, model_data)
    else:
        return _extract_session_features(raw_samples, raw_labels, sample_rate, model_data)


def _extract_session_features(raw_samples, raw_labels, sample_rate, model_data):
    """Session model path: extract_window_features + temporal features."""
    from train_from_session import extract_window_features
    from run_exohand import _add_temporal_features

    window_ms = model_data["window_ms"]
    stride_ms = model_data["stride_ms"]

    win_samples = max(1, int(window_ms / 1000.0 * sample_rate))
    stride_samples = max(1, int(stride_ms / 1000.0 * sample_rate))

    X = np.array(raw_samples)
    y = np.array(raw_labels)

    features = []
    labels = []

    for start in range(0, len(X) - win_samples + 1, stride_samples):
        end = start + win_samples
        window = X[start:end]
        feat = extract_window_features(window)
        features.append(feat)
        window_y = y[start:end]
        labels.append(np.bincount(window_y.astype(int), minlength=3).argmax())

    if len(features) == 0:
        # 36 features = 24 base + 12 temporal
        return np.array([]).reshape(0, 36), np.array([])

    features = np.array(features)
    labels = np.array(labels)

    # Add temporal features (same as training)
    features = _add_temporal_features(features)

    # Clean NaN/inf from edge cases (e.g. zero-variance windows)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features, labels


def _extract_adapted_features(raw_samples, raw_labels, sample_rate, model_data):
    """Adapted HGB path: bandpass -> envelope -> session features -> engineer.

    Critical label handling:
        raw_labels are in session order: {close:0, open:1, rest:2}
        extract_session_features() internally remaps to GrabMyo order via
        SESSION_TO_GRABMYO_LABEL, storing the result in df["intent_idx"].
        So df["intent_idx"] is already in GrabMyo order — do NOT remap again.

    Critical normalisation handling:
        engineer_features_for_saved_model() applies per-participant z-scoring
        using only the calibration data (since participant="patient" for all rows).
        But the saved model's real-time inference uses patient_norm_stats from
        the original adaptation run. We must apply the SAME normalisation here
        for train/serve consistency. If patient_norm_stats exists, we skip the
        built-in per-participant normalisation and apply the saved stats instead.
    """
    from adapt_model import (
        bandpass_filter, compute_envelope, extract_session_features,
    )
    from train_hgb_v2 import (
        add_temporal_features, add_cross_channel_features,
        add_temporal_on_interactions, add_rank_features,
        add_within_trial_position, add_per_session_normalisation,
        META_COLS,
    )

    X = np.array(raw_samples)
    y = np.array(raw_labels)

    # Filter and envelope
    filtered = bandpass_filter(X, sample_rate,
                               lowcut=model_data.get("bandpass_lowcut", 20.0),
                               highcut=model_data.get("bandpass_highcut", 450.0),
                               order=model_data.get("bandpass_order", 2))
    envelope = compute_envelope(filtered, sample_rate)

    # Channel map
    ch_map = model_data.get("channel_map", {0: 0, 1: 4, 2: 9, 3: 13})

    # Timestamps (synthetic, uniform spacing)
    timestamps = np.arange(len(X)) / sample_rate

    # Extract session-level features (returns DataFrame)
    # NOTE: extract_session_features already remaps labels to GrabMyo order
    # internally via SESSION_TO_GRABMYO_LABEL, so df["intent_idx"] is GrabMyo.
    df = extract_session_features(filtered, envelope, sample_rate, y, timestamps, ch_map)

    if len(df) == 0:
        feature_names = model_data["feature_names"]
        return np.array([]).reshape(0, len(feature_names)), np.array([])

    # Engineer features — but use saved patient_norm_stats if available
    # instead of the built-in per-participant z-scoring (which would compute
    # different stats from just the calibration data).
    patient_norm = model_data.get("patient_norm_stats")
    if patient_norm is not None:
        # Apply feature engineering WITHOUT per-participant normalisation,
        # then manually apply the saved normalisation stats.
        df = add_temporal_features(df)
        df = add_cross_channel_features(df)
        df = add_temporal_on_interactions(df)
        df = add_rank_features(df)
        df = add_within_trial_position(df)
        # Apply saved per-patient z-score normalisation
        feature_names = model_data["feature_names"]
        norm_mean = patient_norm["mean"]
        norm_std = patient_norm["std"]
        for col in feature_names:
            if col in df.columns:
                m = norm_mean.get(col, 0.0)
                s = norm_std.get(col, 1.0)
                df[col] = (df[col] - m) / s
        # Per-session normalisation on base features
        base_features = [c for c in df.columns if c not in META_COLS
                         and "_prev" not in c and "_delta" not in c
                         and "_roll" not in c and "_accel" not in c
                         and "_ratio" not in c and "_diff" not in c
                         and "_pctile" not in c and "_sess_norm" not in c
                         and c != "trial_pos"]
        df = add_per_session_normalisation(df, [c for c in base_features if c in df.columns])
    else:
        # Fallback: use the full pipeline (includes per-participant normalisation)
        from adapt_model import engineer_features_for_saved_model
        df = engineer_features_for_saved_model(df)

    # Get feature columns in model order
    feature_names = model_data["feature_names"]
    feat_cols = [c for c in feature_names if c in df.columns]
    missing = [c for c in feature_names if c not in df.columns]

    features = df[feat_cols].values
    # Add zero columns for any missing features
    if missing:
        zeros = np.zeros((len(df), len(missing)))
        features = np.hstack([features, zeros])
        # Reorder to match model feature order
        col_order = {name: i for i, name in enumerate(feat_cols + missing)}
        idx = [col_order[name] for name in feature_names]
        features = features[:, idx]

    # Clean NaN/inf from feature extraction edge cases
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Labels are already in GrabMyo order from extract_session_features —
    # do NOT remap again (that was the double-remapping bug).
    labels = df["intent_idx"].values.astype(int)

    return features, labels


# ── Model fine-tuning ───────────────────────────────────────────────────────

def finetune_model(model_data, cal_features, cal_labels):
    """Fine-tune the model on calibration data.

    For adapted HGB: subsample GrabMyo training concept, combine with
    calibration data (10x weight), retrain HGB with max_iter=200.

    For session model: refit scaler on patient data, train lightweight HGB.

    Returns updated model_data dict.
    """
    if len(cal_features) == 0 or len(cal_labels) == 0:
        print("  WARNING: No calibration features to train on. Keeping original model.")
        return model_data

    # Check class distribution
    unique_labels = np.unique(cal_labels)
    if len(unique_labels) < 2:
        print(f"  WARNING: Only {len(unique_labels)} class(es) in calibration data "
              f"(labels={unique_labels.tolist()}). Model will be degenerate. "
              "Keeping original model.")
        return model_data

    label_counts = {int(l): int(np.sum(cal_labels == l)) for l in unique_labels}
    print(f"  Training class distribution: {label_counts}")

    model_type = model_data.get("model_type", "session")

    if model_type == "adapted_hgb":
        return _finetune_adapted(model_data, cal_features, cal_labels)
    else:
        return _finetune_session(model_data, cal_features, cal_labels)


def _finetune_session(model_data, cal_features, cal_labels):
    """Session model: refit scaler on patient data, train new HGB.

    Refitting the scaler is critical: if the patient's signal range differs
    wildly from the original training data, features get clipped/compressed
    through the original scaler, and the model sees garbage.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    # Clean NaN/inf before fitting scaler
    cal_features_clean = np.nan_to_num(cal_features, nan=0.0, posinf=0.0, neginf=0.0)

    # Refit scaler on patient calibration data so features are properly
    # normalised for this patient's signal range
    patient_scaler = StandardScaler()
    X_cal = patient_scaler.fit_transform(cal_features_clean)

    print(f"  Scaler refit on {len(cal_features)} patient windows")

    model = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=50,
        max_depth=8,
        min_samples_leaf=min(10, max(1, len(X_cal) // 5)),
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_cal, cal_labels)

    cal_pred = model.predict(X_cal)
    cal_acc = np.mean(cal_pred == cal_labels)
    print(f"  Calibration accuracy (self): {cal_acc:.1%}")

    new_data = model_data.copy()
    new_data["model"] = model
    new_data["scaler"] = patient_scaler
    return new_data


def _finetune_adapted(model_data, cal_features, cal_labels):
    """Adapted HGB: combine subsampled GrabMyo data with calibration data."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    scaler = model_data["scaler"]

    # Clean NaN/inf BEFORE scaler — NaN * scale + bias = NaN
    cal_features_clean = np.nan_to_num(cal_features, nan=0.0, posinf=0.0, neginf=0.0)
    X_cal_s = scaler.transform(cal_features_clean)

    # Also clean after scaler in case of extreme values
    X_cal_s = np.nan_to_num(X_cal_s, nan=0.0, posinf=0.0, neginf=0.0)

    # Try to load original training data for combination
    grabmyo_path = model_data.get("grabmyo_features_path")
    X_train = None
    y_train = None

    if grabmyo_path and os.path.exists(grabmyo_path):
        try:
            saved = np.load(grabmyo_path)
            X_train = saved["X"]
            y_train = saved["y"]
            # Subsample to ~20k for speed
            if len(X_train) > 20000:
                idx = np.random.RandomState(42).choice(len(X_train), 20000, replace=False)
                X_train = X_train[idx]
                y_train = y_train[idx]
            X_train = scaler.transform(X_train)
        except Exception:
            X_train = None

    if X_train is not None:
        # Repeat calibration data 10x for weighting
        X_cal_rep = np.repeat(X_cal_s, 10, axis=0)
        y_cal_rep = np.repeat(cal_labels, 10, axis=0)

        X_combined = np.vstack([X_train, X_cal_rep])
        y_combined = np.concatenate([y_train, y_cal_rep])
        print(f"  Combined: {len(X_train)} GrabMyo + {len(X_cal_rep)} calibration (10x)")
    else:
        X_combined = X_cal_s
        y_combined = cal_labels
        print(f"  Training on calibration data only ({len(X_cal_s)} windows)")

    # Only use early stopping if we have enough data for a meaningful
    # validation split (at least 50 samples, to avoid degenerate splits)
    use_early_stopping = len(X_combined) >= 50

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_depth=6,
        min_samples_leaf=min(20, max(1, len(X_combined) // 5)),
        class_weight="balanced",
        random_state=42,
        early_stopping=use_early_stopping,
        n_iter_no_change=10 if use_early_stopping else None,
        validation_fraction=0.1 if use_early_stopping else None,
    )
    model.fit(X_combined, y_combined)

    cal_pred = model.predict(X_cal_s)
    cal_acc = np.mean(cal_pred == cal_labels)
    print(f"  Calibration accuracy (self): {cal_acc:.1%}")

    new_data = model_data.copy()
    new_data["model"] = model
    return new_data


# ── Gain / threshold tuning ────────────────────────────────────────────────

def compute_per_class_stats(raw_samples, raw_labels):
    """Compute per-class EMG amplitude statistics."""
    X = np.array(raw_samples)
    y = np.array(raw_labels)
    stats = {}

    for label in [0, 1, 2]:
        mask = y == label
        if mask.sum() == 0:
            stats[label] = {"mean_amp": 0.0, "median_amp": 0.0, "std_amp": 0.0, "snr": 0.0}
            continue

        class_data = X[mask]
        amplitudes = np.sqrt(np.mean(class_data ** 2, axis=1))  # per-sample RMS
        stats[label] = {
            "mean_amp": float(np.mean(amplitudes)),
            "median_amp": float(np.median(amplitudes)),
            "std_amp": float(np.std(amplitudes)),
            "snr": 0.0,  # computed after rest baseline is known
        }

    return stats


def apply_calibration(rest_baseline, per_class_stats, profile):
    """Compute gain/threshold overrides from calibration data.

    Returns dict with:
      - target_amplitude: median active EMG (replaces default 1.0)
      - noise_gate: from rest baseline
      - hysteresis_enter/exit: adjusted for patient's SNR
      - confidence_floor: adjusted for SNR
    """
    # Noise gate (same formula as RealtimePredictor)
    noise_mean = np.array(rest_baseline["mean"])
    noise_std = np.array(rest_baseline["std"])
    noise_max = np.array(rest_baseline["max"])
    k = 2.5
    noise_gate = np.maximum(
        noise_mean + k * noise_std,
        noise_max * 1.1,
    )

    # Active amplitude = median of close and open amplitudes
    close_amp = per_class_stats.get(0, {}).get("median_amp", 1.0)
    open_amp = per_class_stats.get(1, {}).get("median_amp", 1.0)
    active_amp = (close_amp + open_amp) / 2.0
    if active_amp < 1e-6:
        active_amp = 1.0

    # Rest amplitude
    rest_amp = per_class_stats.get(2, {}).get("median_amp", 0.0)
    if rest_amp < 1e-6:
        rest_amp = np.mean(noise_std)

    # Signal-to-noise ratio
    snr = active_amp / (rest_amp + 1e-9)

    # Update per_class_stats with SNR
    for label in per_class_stats:
        per_class_stats[label]["snr"] = float(snr)

    # SNR-based threshold scaling
    from assist_profile import adjust_profile_for_patient
    snr_scale = adjust_profile_for_patient(profile, snr)

    params = {
        "target_amplitude": float(active_amp),
        "noise_gate": noise_gate.tolist(),
        "hysteresis_enter": profile.hysteresis_enter * snr_scale,
        "hysteresis_exit": profile.hysteresis_exit * snr_scale,
        "confidence_floor": profile.confidence_floor * snr_scale,
        "snr": float(snr),
    }

    print(f"\n  Calibration parameters:")
    print(f"    Target amplitude : {params['target_amplitude']:.3f}")
    print(f"    SNR              : {params['snr']:.1f}")
    print(f"    Threshold scale  : {snr_scale:.2f}")
    print(f"    Hysteresis enter : {params['hysteresis_enter']:.3f}")
    print(f"    Confidence floor : {params['confidence_floor']:.3f}")

    return params


# ── Non-interactive rest calibration ───────────────────────────────────────

def _rest_calibrate_noninteractive(ser, duration=10):
    """Rest calibration without input() prompts — for web/headless mode.

    Same logic as run_exohand.rest_calibrate() but skips the blocking
    input("Press Enter when ready...") call that would hang a thread.
    """
    from run_exohand import parse_emg_line

    _say("Relax completely", blocking=True)
    time.sleep(1.0)

    print(f"  Rest calibrating ({duration}s)...")
    ser.reset_input_buffer()
    samples = []
    t_start = time.perf_counter()

    while time.perf_counter() - t_start < duration:
        raw = ser.readline()
        if not raw:
            continue
        try:
            line = raw.decode("utf-8", errors="ignore").strip()
        except UnicodeDecodeError:
            continue
        vals = parse_emg_line(line)
        if vals is None:
            continue
        samples.append(vals)

        # Emit live EMG for web UI display
        if _emit_emg:
            print(json.dumps({"type": "emg", "emg": vals.tolist() if hasattr(vals, 'tolist') else list(vals), "gesture": "rest"}), flush=True)

        count = len(samples)
        if count % 40 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"\r  {elapsed:.0f}s  |  {count} samples", end="", flush=True)

    if len(samples) < 10:
        print("\n  WARNING: Too few samples for rest calibration. Using defaults.")
        return None

    data = np.array(samples)
    baseline = {
        "mean": data.mean(axis=0),
        "std": data.std(axis=0),
        "max": data.max(axis=0),
        "p95": np.percentile(data, 95, axis=0),
    }

    print(f"\n  Rest baseline: {len(samples)} samples")
    for ch in range(4):
        print(f"    CH{ch+1}: mean={baseline['mean'][ch]:.1f}  "
              f"std={baseline['std'][ch]:.1f}  "
              f"max={baseline['max'][ch]:.1f}")

    _say("Calibration complete")
    return baseline


# ── Persistence ─────────────────────────────────────────────────────────────

def _calibration_dir(patient_id):
    """Return path to calibration directory for a patient."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrations", patient_id)
    os.makedirs(base, exist_ok=True)
    return base


def save_calibration(result, patient_id):
    """Save calibration data to calibrations/{patient_id}/."""
    cal_dir = _calibration_dir(patient_id)

    # Raw EMG + labels
    np.savez_compressed(
        os.path.join(cal_dir, "calibration_data.npz"),
        samples=result.raw_samples,
        labels=result.raw_labels,
        sample_rate=result.sample_rate,
    )

    # Metadata + per-class stats + quality report
    info = {
        "patient_id": result.patient_id,
        "timestamp": result.timestamp,
        "calibration_type": result.calibration_type,
        "sample_rate": result.sample_rate,
        "per_class_stats": {str(k): v for k, v in result.per_class_stats.items()},
        "calibration_params": result.calibration_params,
        "num_samples": len(result.raw_samples),
        "num_trials": len(result.trials),
        "quality_report": result.quality_report,
    }
    with open(os.path.join(cal_dir, "calibration_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # Rest baseline
    baseline_serializable = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in result.rest_baseline.items()
    }
    with open(os.path.join(cal_dir, "rest_baseline.json"), "w") as f:
        json.dump(baseline_serializable, f, indent=2)

    # Finetuned model
    joblib.dump(result.finetuned_model_data, os.path.join(cal_dir, "calibrated_model.pkl"))

    print(f"\n  Calibration saved to {cal_dir}/")


def load_calibration(patient_id):
    """Load previous calibration data. Returns dict or None."""
    cal_dir = _calibration_dir(patient_id)
    info_path = os.path.join(cal_dir, "calibration_info.json")
    data_path = os.path.join(cal_dir, "calibration_data.npz")

    if not os.path.exists(info_path) or not os.path.exists(data_path):
        return None

    with open(info_path) as f:
        info = json.load(f)

    data = np.load(data_path)

    with open(os.path.join(cal_dir, "rest_baseline.json")) as f:
        rest_baseline = json.load(f)
        for k in rest_baseline:
            if isinstance(rest_baseline[k], list):
                rest_baseline[k] = np.array(rest_baseline[k])

    return {
        "info": info,
        "samples": data["samples"],
        "labels": data["labels"],
        "sample_rate": float(data["sample_rate"]),
        "rest_baseline": rest_baseline,
    }


def load_calibrated_model(patient_id):
    """Load just the finetuned model dict."""
    cal_dir = _calibration_dir(patient_id)
    model_path = os.path.join(cal_dir, "calibrated_model.pkl")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


# ── Core collection loop (shared by both entry points) ─────────────────────

class CalibrationCancelled(Exception):
    """Raised when calibration is cancelled via stop_event."""
    pass


def _run_trials(ser, trials, sample_rate, progress_callback=None,
                interactive=True, stop_event=None):
    """Execute a list of calibration trials, collecting and cleaning EMG data.

    Returns (all_samples, all_labels) after onset trimming,
    outlier rejection, and quality validation.

    Raises CalibrationCancelled if stop_event is set.
    """
    total_trials = len(trials)
    all_samples = []
    all_labels = []
    current_phase = 0
    total_rejected = 0

    for idx, trial in enumerate(trials):
        # Check for cancellation between trials
        if stop_event is not None and stop_event.is_set():
            print("\n  Calibration cancelled by user.")
            raise CalibrationCancelled("Calibration cancelled")
        # Announce phase changes
        if trial.phase != current_phase:
            current_phase = trial.phase
            phase_names = {2: "Familiarization", 3: "Sustained holds",
                           4: "Quick contractions", 5: "Variable effort"}
            phase_name = phase_names.get(current_phase, "")
            if phase_name:
                _say(f"Phase {current_phase}: {phase_name}")
                time.sleep(1.5)

        # Progress callback — remaining shows data-collection time only (no UI pause overhead)
        remaining = sum(t.duration + t.rest_period for t in trials[idx:])
        pct = (idx / total_trials) * 100
        if progress_callback:
            progress_callback(trial.phase, idx, total_trials,
                              trial.gesture, remaining, pct)

        # Web mode: emit trial_start event and pause for UI transition
        if _emit_emg:
            print(json.dumps({
                "type": "trial_start",
                "gesture": trial.gesture,
                "duration": trial.duration,
                "trial_idx": idx,
                "total_trials": total_trials,
            }), flush=True)
            # Pause so the patient can see instruction + countdown
            # Serial buffer gets flushed by _collect_emg_segment's reset_input_buffer
            time.sleep(_WEB_PAUSE_SEC)
        else:
            # Announce trial with countdown for non-pulse trials
            _announce_trial(trial, idx, total_trials, with_countdown=(trial.effort != "pulse"))

        # Collect EMG during hold
        samples = _collect_emg_segment(ser, trial.duration, trial.label)

        # ── Robustness: onset trimming ──
        # Discard the initial reaction-time period where the patient
        # hasn't started the gesture yet
        samples = _trim_onset(samples, sample_rate, trial)

        # ── Robustness: outlier rejection ──
        # Remove electrode pops and cable movement artifacts
        samples, n_rejected = _reject_outliers(samples)
        total_rejected += n_rejected

        trial.samples = [s[0] for s in samples]
        _validate_trial(trial, sample_rate)

        # Only add data from trials that produced usable samples
        if trial.quality != "no_response":
            all_samples.extend([s[0] for s in samples])
            all_labels.extend([s[1] for s in samples])

        # Status
        quality_marker = ""
        if trial.quality == "no_response":
            quality_marker = " [NO RESPONSE]"
        elif trial.quality == "low_samples":
            quality_marker = " [LOW]"
        rejected_str = f" (-{n_rejected} outliers)" if n_rejected > 0 else ""

        print(f"\r  [{idx+1}/{total_trials}] {trial.gesture} ({trial.effort}) "
              f"— {len(trial.samples)} samples{rejected_str}{quality_marker}",
              end="", flush=True)

        # Rest period — collect as rest data
        if trial.rest_period > 0:
            if _emit_emg:
                print(json.dumps({
                    "type": "trial_rest",
                    "duration": trial.rest_period,
                    "trial_idx": idx,
                }), flush=True)
            elif trial.rest_period >= 10:
                _say("Take a break. Relax.")
            elif trial.effort != "pulse":
                _say("Relax")
            rest_samples = _collect_emg_segment(ser, trial.rest_period, 2)
            # No onset trimming for rest — patient is already relaxed
            rest_samples, _ = _reject_outliers(rest_samples)
            all_samples.extend([s[0] for s in rest_samples])
            all_labels.extend([s[1] for s in rest_samples])

    print()  # newline after progress

    if total_rejected > 0:
        print(f"  Outlier samples rejected: {total_rejected}")

    return all_samples, all_labels


# ── Entry points ────────────────────────────────────────────────────────────

def calibrate_patient(ser, model_data, sample_rate, patient_id="default",
                      progress_callback=None, assist_profile=None,
                      interactive=True, stop_event=None):
    """Full 6-minute patient calibration protocol.

    Args:
        ser: serial.Serial connection to Teensy
        model_data: loaded model dict
        sample_rate: estimated Hz
        patient_id: identifier for save/load
        progress_callback: optional fn(phase, trial_idx, total, gesture, time_remaining, pct)
        assist_profile: AssistProfile for threshold computation
        interactive: if False, skip input() prompts (for web mode)
        stop_event: threading.Event — if set, calibration will abort

    Returns:
        CalibrationResult with finetuned model and calibration params

    Raises:
        CalibrationCancelled if stop_event is set during execution
    """
    if assist_profile is None:
        from assist_profile import get_profile
        assist_profile = get_profile(3)

    print("\n" + "=" * 60)
    print("PATIENT CALIBRATION (6-minute protocol)")
    print("=" * 60)
    print(f"  Patient ID: {patient_id}")
    print()

    # Build protocol early so we can compute total remaining time
    trials = build_full_protocol()
    # Round to clean number for display
    total_remaining = round((10.0 + sum(t.duration + t.rest_period for t in trials)) / 10) * 10

    # Phase 1: Rest baseline
    if progress_callback:
        progress_callback(1, 0, 0, "rest", total_remaining, 0.0)
    _say("Phase 1: Rest baseline. Relax completely.", blocking=True)

    if interactive:
        from run_exohand import rest_calibrate
        rest_baseline = rest_calibrate(ser, duration=10)
    else:
        rest_baseline = _rest_calibrate_noninteractive(ser, duration=10)

    if rest_baseline is None:
        rest_baseline = {
            "mean": np.zeros(4), "std": np.ones(4),
            "max": np.zeros(4), "p95": np.zeros(4),
        }
    _say("Starting calibration trials. Follow the voice cues.", blocking=True)
    time.sleep(1.0)

    all_samples, all_labels = _run_trials(
        ser, trials, sample_rate, progress_callback, interactive, stop_event)

    if len(all_samples) == 0:
        print("\n  ERROR: No samples collected. Check serial connection.")
        # Return a result with the original model unchanged
        return CalibrationResult(
            patient_id=patient_id, timestamp=time.time(),
            calibration_type="full", rest_baseline=rest_baseline,
            trials=trials, raw_samples=np.array([]).reshape(0, 4),
            raw_labels=np.array([]), sample_rate=sample_rate,
            per_class_stats={}, finetuned_model_data=model_data,
            calibration_params={}, quality_report={"grade": "FAILED", "warnings": ["No samples collected"]},
        )

    raw_samples = np.array(all_samples)
    raw_labels = np.array(all_labels)

    print(f"\n  Total samples: {len(raw_samples)}")
    for label in [0, 1, 2]:
        n = np.sum(raw_labels == label)
        print(f"    {GESTURE_NAMES[label]}: {n} samples")

    # Compute per-class stats
    per_class_stats = compute_per_class_stats(raw_samples, raw_labels)

    # Extract features
    if progress_callback:
        progress_callback(0, len(trials), len(trials), "processing", 0.0, 90.0)
    print("\n  Extracting features...")
    features, feature_labels = _extract_calibration_features(
        raw_samples, raw_labels, sample_rate, model_data)
    print(f"  {len(features)} feature windows")

    # For quality report, we need labels in session order (matching raw_labels).
    # Adapted model features have labels in GrabMyo order — remap them.
    model_type = model_data.get("model_type", "session")
    if model_type == "adapted_hgb" and len(feature_labels) > 0:
        report_labels = np.array([GRABMYO_TO_SESSION_LABEL.get(int(l), int(l))
                                  for l in feature_labels])
    else:
        report_labels = feature_labels

    # Quality report (before training — so clinician sees it even if training fails)
    quality_report = _compute_quality_report(
        raw_samples, raw_labels, per_class_stats, trials, report_labels, sample_rate)
    _print_quality_report(quality_report)

    # Fine-tune model (uses feature_labels in the model's native label order)
    print("\n  Fine-tuning model...")
    finetuned = finetune_model(model_data, features, feature_labels)

    # Compute gain/threshold overrides
    cal_params = apply_calibration(rest_baseline, per_class_stats, assist_profile)

    result = CalibrationResult(
        patient_id=patient_id, timestamp=time.time(),
        calibration_type="full", rest_baseline=rest_baseline,
        trials=trials, raw_samples=raw_samples, raw_labels=raw_labels,
        sample_rate=sample_rate, per_class_stats=per_class_stats,
        finetuned_model_data=finetuned, calibration_params=cal_params,
        quality_report=quality_report,
    )

    save_calibration(result, patient_id)

    if progress_callback:
        progress_callback(0, len(trials), len(trials), "complete", 0.0, 100.0)
    _say("Calibration complete!")

    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)

    return result


def abbreviated_calibrate(ser, model_data, sample_rate, patient_id="default",
                          progress_callback=None, assist_profile=None,
                          interactive=True, stop_event=None):
    """Abbreviated 90-second recalibration for returning patients.

    Loads previous calibration, runs short protocol, merges data
    weighted toward recency.

    Raises:
        CalibrationCancelled if stop_event is set during execution
    """
    if assist_profile is None:
        from assist_profile import get_profile
        assist_profile = get_profile(3)

    print("\n" + "=" * 60)
    print("ABBREVIATED RECALIBRATION (90-second protocol)")
    print("=" * 60)
    print(f"  Patient ID: {patient_id}")

    # Load previous calibration
    prev = load_calibration(patient_id)
    if prev is not None:
        print(f"  Previous calibration found ({prev['info']['calibration_type']}, "
              f"{prev['info']['num_samples']} samples)")
    else:
        print("  No previous calibration found. Running abbreviated anyway.")

    # Phase 1: Quick rest baseline
    # Total time: 10s rest + 9 trials × (5s hold + 4s rest) = ~91s
    trials = build_abbreviated_protocol()
    # Round to clean number (91s → 90s = 1:30)
    total_remaining = round((10.0 + sum(t.duration + t.rest_period for t in trials)) / 10) * 10
    if progress_callback:
        progress_callback(1, 0, 0, "rest", total_remaining, 0.0)

    if _emit_emg:
        print(json.dumps({
            "type": "trial_start",
            "gesture": "rest",
            "duration": 10,
            "trial_idx": -1,
            "total_trials": 0,
        }), flush=True)
        time.sleep(6.0)  # Pause for instruction (3s) + countdown (3s)
    else:
        _say("Rest calibration. Relax completely.", blocking=True)

    if interactive:
        from run_exohand import rest_calibrate
        rest_baseline = rest_calibrate(ser, duration=10)
    else:
        rest_baseline = _rest_calibrate_noninteractive(ser, duration=10)

    if rest_baseline is None:
        rest_baseline = {
            "mean": np.zeros(4), "std": np.ones(4),
            "max": np.zeros(4), "p95": np.zeros(4),
        }

    # Short protocol (trials already built above for remaining-time calc)
    _say("Starting recalibration. Follow the cues.", blocking=True)
    time.sleep(1.0)

    all_samples, all_labels = _run_trials(
        ser, trials, sample_rate, progress_callback, interactive, stop_event)

    if len(all_samples) == 0:
        print("\n  ERROR: No samples collected.")
        return CalibrationResult(
            patient_id=patient_id, timestamp=time.time(),
            calibration_type="abbreviated", rest_baseline=rest_baseline,
            trials=trials, raw_samples=np.array([]).reshape(0, 4),
            raw_labels=np.array([]), sample_rate=sample_rate,
            per_class_stats={}, finetuned_model_data=model_data,
            calibration_params={}, quality_report={"grade": "FAILED", "warnings": ["No samples collected"]},
        )

    new_samples = np.array(all_samples)
    new_labels = np.array(all_labels)

    # Merge with previous data, weighted toward recency (2:1 new:old)
    if prev is not None:
        old_samples = prev["samples"]
        old_labels = prev["labels"]
        # Subsample old data to half the new data size
        n_old = min(len(old_samples), len(new_samples) // 2)
        if n_old > 0:
            idx_old = np.random.RandomState(42).choice(len(old_samples), n_old, replace=False)
            merged_samples = np.vstack([new_samples, old_samples[idx_old]])
            merged_labels = np.concatenate([new_labels, old_labels[idx_old]])
        else:
            merged_samples = new_samples
            merged_labels = new_labels
        print(f"\n  Merged: {len(new_samples)} new + {n_old} old = {len(merged_samples)} total")
    else:
        merged_samples = new_samples
        merged_labels = new_labels

    per_class_stats = compute_per_class_stats(merged_samples, merged_labels)

    # Extract features
    if progress_callback:
        progress_callback(0, len(trials), len(trials), "processing", 0.0, 90.0)
    print("\n  Extracting features...")
    features, feature_labels = _extract_calibration_features(
        merged_samples, merged_labels, sample_rate, model_data)
    print(f"  {len(features)} feature windows")

    # Remap labels for quality report (adapted labels are in GrabMyo order)
    model_type = model_data.get("model_type", "session")
    if model_type == "adapted_hgb" and len(feature_labels) > 0:
        report_labels = np.array([GRABMYO_TO_SESSION_LABEL.get(int(l), int(l))
                                  for l in feature_labels])
    else:
        report_labels = feature_labels

    # Quality report
    quality_report = _compute_quality_report(
        merged_samples, merged_labels, per_class_stats, trials, report_labels, sample_rate)
    _print_quality_report(quality_report)

    # Fine-tune (uses feature_labels in the model's native label order)
    print("\n  Fine-tuning model...")
    finetuned = finetune_model(model_data, features, feature_labels)

    # Gain/threshold overrides
    cal_params = apply_calibration(rest_baseline, per_class_stats, assist_profile)

    result = CalibrationResult(
        patient_id=patient_id, timestamp=time.time(),
        calibration_type="abbreviated", rest_baseline=rest_baseline,
        trials=trials, raw_samples=merged_samples, raw_labels=merged_labels,
        sample_rate=sample_rate, per_class_stats=per_class_stats,
        finetuned_model_data=finetuned, calibration_params=cal_params,
        quality_report=quality_report,
    )

    save_calibration(result, patient_id)

    if progress_callback:
        progress_callback(0, len(trials), len(trials), "complete", 0.0, 100.0)
    _say("Recalibration complete!")

    print("\n" + "=" * 60)
    print("RECALIBRATION COMPLETE")
    print("=" * 60)

    return result


def list_patients():
    """List all saved patient IDs."""
    cal_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrations")
    if not os.path.isdir(cal_base):
        return []
    return [d for d in os.listdir(cal_base)
            if os.path.isdir(os.path.join(cal_base, d))
            and os.path.exists(os.path.join(cal_base, d, "calibration_info.json"))]


# ── Web mode (--web-mode) CLI entry point ──────────────────────────────────

def _json_progress_callback(phase, trial_idx, total, gesture, remaining, pct):
    """Progress callback that outputs JSON lines for the Node.js CalibrationBridge."""
    print(json.dumps({
        "type": "progress",
        "phase": phase,
        "trial": trial_idx,
        "total": total,
        "gesture": gesture,
        "remaining": remaining,
        "percent": pct,
    }), flush=True)


def _run_web_mode(args):
    """Run calibration in web mode — JSON progress on stdout, no interactive prompts."""
    global _tts_muted, _emit_emg
    _tts_muted = True
    _emit_emg = True

    import serial
    import argparse

    port = args.port
    model_path = args.model
    patient_id = args.patient_id
    mode = args.mode
    assist_level = args.assist_level

    # Load model
    try:
        model_data = joblib.load(model_path)
        print(json.dumps({"type": "progress", "phase": 0, "trial": 0, "total": 0,
                          "gesture": "loading", "remaining": 0, "percent": 5}), flush=True)
    except Exception as e:
        print(json.dumps({"type": "error", "message": f"Failed to load model: {e}"}), flush=True)
        return

    # Open serial
    try:
        ser = serial.Serial(port, 115200, timeout=0.1)
        print(json.dumps({"type": "progress", "phase": 0, "trial": 0, "total": 0,
                          "gesture": "connecting", "remaining": 0, "percent": 10}), flush=True)
    except Exception as e:
        print(json.dumps({"type": "error", "message": f"Failed to open serial port {port}: {e}"}), flush=True)
        return

    # Estimate sample rate from first second
    from run_exohand import parse_emg_line
    ser.reset_input_buffer()
    count = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        raw = ser.readline()
        if raw and parse_emg_line(raw.decode("utf-8", errors="ignore").strip()):
            count += 1
    sample_rate = max(count, 50)  # fallback to 50 Hz minimum

    # Get assist profile
    from assist_profile import get_profile
    assist_profile = get_profile(assist_level)

    try:
        if mode == "full":
            result = calibrate_patient(
                ser, model_data, sample_rate,
                patient_id=patient_id,
                progress_callback=_json_progress_callback,
                assist_profile=assist_profile,
                interactive=False,
            )
        else:
            result = abbreviated_calibrate(
                ser, model_data, sample_rate,
                patient_id=patient_id,
                progress_callback=_json_progress_callback,
                assist_profile=assist_profile,
                interactive=False,
            )

        # Output completion event
        quality_report = result.quality_report if hasattr(result, "quality_report") else {}
        print(json.dumps({
            "type": "complete",
            "patient_id": patient_id,
            "calibration_type": result.calibration_type,
            "num_samples": len(result.raw_samples),
            "quality_report": quality_report,
        }), flush=True)

    except CalibrationCancelled:
        print(json.dumps({"type": "error", "message": "Calibration cancelled"}), flush=True)
    except Exception as e:
        print(json.dumps({"type": "error", "message": str(e)}), flush=True)
    finally:
        ser.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ExoHand patient calibration")
    parser.add_argument("--web-mode", action="store_true",
                        help="Run in web mode (JSON progress on stdout, non-interactive)")
    parser.add_argument("--port", type=str, default="/dev/cu.usbmodem176627901",
                        help="Serial port for Teensy")
    parser.add_argument("--model", type=str, default="exohand_model.pkl",
                        help="Path to model .pkl file")
    parser.add_argument("--patient-id", type=str, default="default",
                        help="Patient identifier")
    parser.add_argument("--mode", type=str, choices=["full", "quick"], default="quick",
                        help="Calibration mode: 'full' (6-min) or 'quick' (90s abbreviated)")
    parser.add_argument("--assist-level", type=int, default=3,
                        help="Assist level (1-5)")

    args = parser.parse_args()

    if args.web_mode:
        _run_web_mode(args)
    else:
        print("Run with --web-mode for Node.js integration, or import and call calibrate_patient() directly.")
