#!/usr/bin/env python3
"""
run_exohand.py — Real-time EMG inference and motor control.

Reads EMG from Teensy, predicts intent (close/open/rest), sends single-char
commands (c/o/r) back over serial. Includes optional calibration mode.

Modes:
    Free mode (default): raw EMG → motor passthrough
    Exercise mode (--exercise): structured rep-based exercises with state machine
    Web mode (--web): FastAPI server with browser UI at localhost:8000

Usage:
    python run_exohand.py --port /dev/tty.usbmodemXXXX --model exohand_model.pkl
    python run_exohand.py --port ... --model ... --exercise
    python run_exohand.py --port ... --model ... --web
"""

import argparse
import os
import sys
import time
from collections import deque

import joblib
import numpy as np
import serial
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.fft import rfft, rfftfreq

# Re-use feature extraction from training script
from train_from_session import extract_window_features
from assist_profile import get_profile, print_profile
from exercise import (
    Exercise, MotorCommand, ExerciseState, Event,
    ExerciseRunner, SessionRunner, default_programme,
    FINGER_SERIAL_CODES, ACTION_TO_SERIAL, INTENT_TO_ACTION,
)


LABEL_NAMES = ["close", "open", "rest"]
COMMAND_MAP = {0: "c", 1: "o", 2: "r"}  # close, open, rest

# Rest calibration duration
REST_CALIBRATION_SECONDS = 10


def parse_emg_line(line):
    """Parse tab-separated 4-channel EMG line. Returns list of 4 floats or None."""
    try:
        parts = line.strip().split()
        if len(parts) != 4:
            return None
        return [float(v) for v in parts]
    except (ValueError, IndexError):
        return None


def load_model(model_path):
    """Load the trained model package."""
    data = joblib.load(model_path)
    return data


def calibrate(ser, model_data, duration_per_gesture=30):
    """
    Calibration mode: collect 30s of each gesture from the user,
    fine-tune the model, save calibrated version.

    .. deprecated::
        Use ``calibrate_patient()`` from ``calibrate_patient.py`` for
        patient-specific calibration with structured protocol, adaptive
        gain tuning, and persistence.
    """
    import subprocess

    model = model_data["model"]
    scaler = model_data["scaler"]
    window_ms = model_data["window_ms"]
    stride_ms = model_data["stride_ms"]

    print("=" * 60)
    print("CALIBRATION MODE")
    print("=" * 60)
    print(f"Will collect {duration_per_gesture}s of each gesture.")
    print()

    all_samples = []
    all_labels = []

    for gesture_name, label in [("REST", 2), ("CLOSE (make a fist)", 0), ("OPEN (extend hand)", 1)]:
        input(f"Press Enter to start collecting '{gesture_name}' for {duration_per_gesture}s...")
        subprocess.Popen(["say", gesture_name.split("(")[0].strip()])
        print(f"Recording {gesture_name}...")

        samples = []
        ser.reset_input_buffer()
        t_start = time.perf_counter()

        while time.perf_counter() - t_start < duration_per_gesture:
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

            samples.append(vals)
            count = len(samples)
            if count % 200 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"\r  {elapsed:.0f}s  |  {count} samples", end="", flush=True)

        print(f"\n  Collected {len(samples)} samples for {gesture_name}")
        all_samples.extend(samples)
        all_labels.extend([label] * len(samples))

    X_cal = np.array(all_samples)
    y_cal = np.array(all_labels)

    print(f"\nExtracting calibration features...")

    # Estimate sample rate from collection
    total_samples = len(X_cal)
    total_time = duration_per_gesture * 3
    cal_sample_rate = total_samples / total_time

    # Extract features using the same windowing as training
    win_samples = max(1, int(window_ms / 1000.0 * cal_sample_rate))
    stride_samples = max(1, int(stride_ms / 1000.0 * cal_sample_rate))

    cal_features = []
    cal_labels = []

    for start in range(0, len(X_cal) - win_samples + 1, stride_samples):
        end = start + win_samples
        window = X_cal[start:end]
        feat = extract_window_features(window)
        cal_features.append(feat)
        # Majority label in window
        window_y = y_cal[start:end]
        cal_labels.append(np.bincount(window_y, minlength=3).argmax())

    cal_features = np.array(cal_features)
    cal_labels = np.array(cal_labels)

    # Add temporal features
    cal_features = _add_temporal_features(cal_features)

    print(f"  {len(cal_features)} calibration windows")

    # Fine-tune: combine subsampled original training concept with calibration data
    # Since we don't have the original training data, we train a new model
    # on just the calibration data (lightweight fine-tune)
    from sklearn.ensemble import HistGradientBoostingClassifier

    X_cal_s = scaler.transform(cal_features)

    calibrated_model = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=50,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
    )
    calibrated_model.fit(X_cal_s, cal_labels)

    # Evaluate on calibration data itself (just to sanity check)
    cal_pred = calibrated_model.predict(X_cal_s)
    cal_acc = np.mean(cal_pred == cal_labels)
    print(f"  Calibration accuracy (self): {cal_acc:.1%}")

    # Save calibrated model
    cal_model_data = model_data.copy()
    cal_model_data["model"] = calibrated_model
    cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exohand_model_calibrated.pkl")
    joblib.dump(cal_model_data, cal_path)
    print(f"  Calibrated model saved: {cal_path}")

    return cal_model_data


def _add_temporal_features(features):
    """Add _prev, _delta, _roll3 temporal features on env_rms columns."""
    from train_from_session import get_feature_names

    feat_names = get_feature_names()
    env_rms_indices = [i for i, name in enumerate(feat_names) if name.endswith("_env_rms")]

    temporal_cols = []
    for idx in env_rms_indices:
        col = features[:, idx]

        prev_col = np.zeros_like(col)
        prev_col[1:] = col[:-1]

        delta_col = col - prev_col

        roll3_col = np.zeros_like(col)
        for i in range(len(col)):
            start_r = max(0, i - 2)
            roll3_col[i] = col[start_r:i + 1].mean()

        temporal_cols.extend([prev_col, delta_col, roll3_col])

    if temporal_cols:
        temporal_array = np.column_stack(temporal_cols)
        return np.hstack([features, temporal_array])
    return features


class RealtimePredictor:
    """
    Buffers incoming samples, extracts features per stride,
    applies assist-as-needed mechanisms, outputs commands.

    Supports two model types:
      - "session" (default): 27-feature model from train_from_session.py
      - "adapted_hgb": 140-feature GrabMyo-adapted model from adapt_model.py

    Four mechanisms work together based on the assist profile:
      1. Adaptive gain normalization (before feature extraction)
      2. Movement bias (after predict_proba)
      3. Non-linear assist curve (confidence -> motor strength)
      4. Relaxed stability filter
    """

    def __init__(self, model_data, sample_rate, assist_profile,
                 rest_baseline=None, calibration_params=None):
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.window_ms = model_data["window_ms"]
        self.stride_ms = model_data["stride_ms"]
        self.profile = assist_profile
        self.sample_rate = sample_rate
        self.model_type = model_data.get("model_type", "session")

        self.win_samples = max(1, int(self.window_ms / 1000.0 * sample_rate))
        self.stride_samples = max(1, int(self.stride_ms / 1000.0 * sample_rate))

        # Sample buffer
        self.buffer = deque(maxlen=self.win_samples)

        # Temporal state for session model features
        self.prev_env_rms = np.zeros(4)
        self.env_rms_history = [deque(maxlen=3) for _ in range(4)]

        # Stability filter (driven by profile)
        self.stability_count = self.profile.stability_required
        self.recent_preds = deque(maxlen=self.stability_count)
        self.current_intent = 2  # start as rest

        # Anti-jitter: probability smoothing (EMA on raw predict_proba)
        self.smoothed_proba = np.array([0.0, 0.0, 1.0])  # start as rest
        self.proba_alpha = self.profile.proba_ema_alpha

        # Anti-jitter: cooldown timer
        self.last_transition_time = 0.0
        self.cooldown_s = self.profile.cooldown_ms / 1000.0

        # Counter for stride timing
        self.samples_since_last_pred = 0

        # Adaptive gain state: per-channel EMA of signal amplitude
        self.n_channels = 4
        self.channel_ema = np.ones(self.n_channels)  # starts at 1.0
        self.current_gains = np.ones(self.n_channels)
        self.ema_initialized = False

        # Target amplitude — the range healthy training data lives in.
        # Features like RMS/MAV on normalized MyoWare output sit around 1.0.
        self.target_amplitude = 1.0

        # ── Patient calibration overrides ─────────────────────────────
        self._cal_hysteresis_enter = None
        self._cal_hysteresis_exit = None
        self._cal_confidence_floor = None
        self._cal_noise_gate = None
        if calibration_params is not None:
            if "target_amplitude" in calibration_params:
                self.target_amplitude = calibration_params["target_amplitude"]
            self._cal_hysteresis_enter = calibration_params.get("hysteresis_enter")
            self._cal_hysteresis_exit = calibration_params.get("hysteresis_exit")
            self._cal_confidence_floor = calibration_params.get("confidence_floor")
            if "noise_gate" in calibration_params:
                self._cal_noise_gate = np.array(calibration_params["noise_gate"])

        # ── Stroke-robust: noise gate from rest calibration ───────────
        if rest_baseline is not None:
            self.noise_mean = np.array(rest_baseline["mean"])
            self.noise_std = np.array(rest_baseline["std"])
            self.noise_max = np.array(rest_baseline["max"])
            # Gate threshold: mean + k*std (k higher = more permissive)
            # Use max observed rest value as a floor
            k = 2.5
            self.noise_gate = np.maximum(
                self.noise_mean + k * self.noise_std,
                self.noise_max * 1.1,
            )
        else:
            self.noise_mean = np.zeros(4)
            self.noise_std = np.ones(4)
            self.noise_max = np.zeros(4)
            self.noise_gate = np.zeros(4)  # no gating

        # Override noise gate with calibration-computed value if available
        # (calibration noise gate uses the same formula but from the
        # calibration rest baseline, so it's more authoritative)
        if self._cal_noise_gate is not None:
            self.noise_gate = self._cal_noise_gate

        # ── Stroke-robust: artifact rejection ─────────────────────────
        # Reject samples where any channel jumps more than this factor
        # above the running EMA (electrode pop, cable movement)
        self.artifact_factor = 10.0
        self.prev_sample = np.zeros(4)
        self.artifact_count = 0

        # ── Stroke-robust: co-contraction detection ──────────────────
        # Flexor channels (ports 0, 2) and extensor channels (ports 1, 3)
        self.flexor_ports = [0, 2]
        self.extensor_ports = [1, 3]
        self.cocontraction_count = 0
        # If both groups are above this fraction of noise_gate simultaneously,
        # it's co-contraction
        self.cocontraction_threshold = 1.5  # relative to noise_gate

        # ── Stroke-robust: fatigue drift compensation ─────────────────
        # Long-term EMA of signal strength, separate from adaptive gain's
        # fast EMA. When this drops below a threshold, we boost gain further.
        self.fatigue_ema = np.ones(4)
        self.fatigue_baseline = np.ones(4)  # set after first few seconds
        self.fatigue_initialized = False
        self.fatigue_sample_count = 0
        self.fatigue_warmup = int(10 * sample_rate)  # 10s warmup
        self.fatigue_accumulator = np.zeros(4)

        # ── Signal quality tracking ───────────────────────────────────
        self.signal_quality = np.ones(4)  # 1.0 = good, 0.0 = dead
        self.quality_window = deque(maxlen=int(5 * sample_rate))  # 5s window

        # Adapted model state
        if self.model_type == "adapted_hgb":
            self._init_adapted_state(model_data)

    # ── Adapted model initialisation ─────────────────────────────────────

    def _init_adapted_state(self, model_data):
        """Initialise state for the 140-feature adapted_hgb pipeline."""
        fs = self.sample_rate

        # Persistent IIR bandpass filter (SOS form for numerical stability)
        nyq = 0.5 * fs
        low = model_data["bandpass_lowcut"] / nyq
        high = min(model_data["bandpass_highcut"] / nyq, 0.99)
        order = model_data["bandpass_order"]
        self.bp_sos = butter(order, [low, high], btype="band", output="sos")
        n_sections = self.bp_sos.shape[0]
        self.bp_state = [np.zeros((n_sections, 2)) for _ in range(4)]

        # Envelope: running mean of rectified signal
        self.env_win = max(1, int(model_data["env_smooth_ms"] / 1000.0 * fs))
        self.rect_bufs = [deque(maxlen=self.env_win) for _ in range(4)]

        # Separate buffers for filtered + envelope (both needed for features)
        self.filt_buf = deque(maxlen=self.win_samples)
        self.env_buf = deque(maxlen=self.win_samples)

        # Feature name ordering from saved model
        self.feature_names = model_data["feature_names"]
        self.feat_idx = {n: i for i, n in enumerate(self.feature_names)}

        # Key columns that get temporal features (sorted, matching train_hgb_v2)
        self.key_cols = sorted(
            f"ch{ch}_{sig}"
            for ch in [0, 4, 9, 13]
            for sig in ["env_rms", "mav", "rms", "wl"]
        )
        # History deques: 5 deep for roll5, keep last values for prev2/accel
        self.key_history = {col: deque(maxlen=5) for col in self.key_cols}
        self.key_prev_delta = {col: 0.0 for col in self.key_cols}

        # Patient normalisation stats
        self.patient_norm = model_data["patient_norm_stats"]

        # Channel mapping: session port index → GrabMyo channel name
        self.ch_map = model_data.get("channel_map", {0: 0, 1: 4, 2: 9, 3: 13})

    # ── Mechanism 1: Adaptive Gain Normalization ─────────────────────────

    def _apply_adaptive_gain(self, values):
        """Scale raw EMG values so weak signals fill the training-data range.

        Uses an asymmetric EMA: fast attack (quickly adopts new peaks) and
        slow decay (stable amplitude tracking).  Returns the gained values.
        """
        if not self.profile.adaptive_gain:
            return values

        vals = np.asarray(values, dtype=np.float64)
        amplitudes = np.abs(vals) + 1e-9  # avoid division by zero

        if not self.ema_initialized:
            self.channel_ema = amplitudes.copy()
            self.ema_initialized = True
        else:
            decay = self.profile.ema_decay
            # Asymmetric: fast attack (0.9) when signal is larger, slow decay otherwise
            for ch in range(self.n_channels):
                if amplitudes[ch] > self.channel_ema[ch]:
                    alpha = 1.0 - 0.9  # fast attack
                else:
                    alpha = 1.0 - decay  # slow decay
                self.channel_ema[ch] += alpha * (amplitudes[ch] - self.channel_ema[ch])

        # Compute gain: target / tracked_amplitude, clamped to [floor, ceiling]
        raw_gains = self.target_amplitude / (self.channel_ema + 1e-9)
        self.current_gains = np.clip(
            raw_gains, self.profile.gain_floor, self.profile.gain_ceiling
        )

        return vals * self.current_gains

    # ── Stroke-robust: Noise Gate (soft knee) ───────────────────────────

    def _apply_noise_gate(self, values):
        """Soft noise gate: gradually attenuates signals near the noise floor.

        Below noise_gate: heavy attenuation (but not zero — preserves weak intent)
        Above noise_gate: full pass-through
        The transition zone (knee) is 0.5× to 1.5× the gate threshold.

        This is critical for stroke patients whose real intent signals may be
        only slightly above the noise floor.
        """
        vals = np.asarray(values, dtype=np.float64)
        gated = np.zeros_like(vals)

        for ch in range(self.n_channels):
            gate = self.noise_gate[ch]
            if gate < 1e-6:
                gated[ch] = vals[ch]  # no gating if no calibration
                continue

            v = vals[ch]
            low = gate * 0.5    # below this: heavy attenuation
            high = gate * 1.5   # above this: full pass

            if v <= low:
                # Below knee: suppress to 10% (don't zero — preserve faint signals)
                gated[ch] = v * 0.1
            elif v >= high:
                # Above knee: full pass, subtract noise mean only
                gated[ch] = v - self.noise_mean[ch]
            else:
                # In the knee: smooth interpolation (cubic)
                t = (v - low) / (high - low)  # 0..1
                # Smooth step: 3t² - 2t³
                s = t * t * (3.0 - 2.0 * t)
                # Blend between attenuated and full-pass
                attenuated = v * 0.1
                full_pass = v - self.noise_mean[ch]
                gated[ch] = attenuated + s * (full_pass - attenuated)

        return gated

    # ── Stroke-robust: Artifact Rejection ─────────────────────────────

    def _is_artifact(self, values):
        """Detect electrode pops and cable movement artifacts.

        Returns True if any channel shows a sudden spike that's implausibly
        large compared to recent history.
        """
        vals = np.asarray(values, dtype=np.float64)
        if not self.ema_initialized:
            return False

        # Check for sudden jump: current >> running EMA
        ratios = vals / (self.channel_ema + 1e-9)
        if np.any(ratios > self.artifact_factor):
            self.artifact_count += 1
            return True

        # Check for sudden delta: |current - previous| >> EMA
        delta = np.abs(vals - self.prev_sample)
        delta_ratios = delta / (self.channel_ema + 1e-9)
        if np.any(delta_ratios > self.artifact_factor * 0.5):
            self.artifact_count += 1
            return True

        return False

    # ── Stroke-robust: Co-contraction Detection ──────────────────────

    def _detect_cocontraction(self, values):
        """Detect when flexors and extensors fire simultaneously.

        Common in stroke patients. Returns True if co-contraction detected.
        When detected, the system should default to rest rather than
        guessing which intent the patient means.
        """
        vals = np.asarray(values, dtype=np.float64)
        gate = self.noise_gate

        # Skip co-contraction detection if no noise gate is set (no rest
        # baseline available), since we can't meaningfully determine what
        # constitutes "above noise" — a zero gate makes every sample active.
        if np.all(gate < 1e-6):
            return False

        # Check if both muscle groups are active above threshold
        flexor_active = any(
            vals[p] > gate[p] * self.cocontraction_threshold
            for p in self.flexor_ports
        )
        extensor_active = any(
            vals[p] > gate[p] * self.cocontraction_threshold
            for p in self.extensor_ports
        )

        if flexor_active and extensor_active:
            # Both groups firing — check if they're roughly balanced
            flexor_sum = sum(vals[p] for p in self.flexor_ports)
            extensor_sum = sum(vals[p] for p in self.extensor_ports)
            ratio = min(flexor_sum, extensor_sum) / (max(flexor_sum, extensor_sum) + 1e-9)
            # If the weaker side is at least 40% of the stronger side, it's co-contraction
            if ratio > 0.4:
                self.cocontraction_count += 1
                return True

        return False

    # ── Stroke-robust: Fatigue Drift Compensation ────────────────────

    def _update_fatigue_tracking(self, values):
        """Track long-term signal strength decline and boost gain to compensate.

        Uses a very slow EMA (over minutes) to detect gradual fatigue-related
        signal amplitude drop, then increases the adaptive gain target.
        """
        vals = np.asarray(values, dtype=np.float64)
        self.fatigue_sample_count += 1

        # Warmup: accumulate baseline strength over first 10s
        if not self.fatigue_initialized:
            self.fatigue_accumulator += np.abs(vals)
            if self.fatigue_sample_count >= self.fatigue_warmup:
                self.fatigue_baseline = self.fatigue_accumulator / self.fatigue_warmup
                self.fatigue_ema = self.fatigue_baseline.copy()
                self.fatigue_initialized = True
            return

        # Very slow EMA (decay ~0.9999 → half-life ~7000 samples = ~350s at 20Hz)
        alpha = 0.0001
        amplitude = np.abs(vals)
        self.fatigue_ema = (1 - alpha) * self.fatigue_ema + alpha * amplitude

        # Compute fatigue ratio: how much has signal dropped?
        fatigue_ratio = self.fatigue_ema / (self.fatigue_baseline + 1e-9)

        # If signal has dropped below 60% of baseline, boost the target amplitude
        # proportionally (up to 3x boost)
        for ch in range(self.n_channels):
            if fatigue_ratio[ch] < 0.6:
                boost = min(3.0, 1.0 / (fatigue_ratio[ch] + 1e-9))
                # Gradually increase target (don't jump)
                self.target_amplitude = max(self.target_amplitude, boost)

    # ── Stroke-robust: Signal Quality Monitor ────────────────────────

    def _update_signal_quality(self, values):
        """Track per-channel signal quality over a rolling window.

        Quality drops when a channel is flatlined, saturated, or excessively
        noisy (high variance with no structure).
        """
        vals = np.asarray(values, dtype=np.float64)
        self.quality_window.append(vals.copy())

        if len(self.quality_window) < 20:
            return

        recent = np.array(list(self.quality_window))  # (N, 4)
        for ch in range(self.n_channels):
            ch_data = recent[:, ch]
            ch_std = np.std(ch_data)
            ch_range = np.max(ch_data) - np.min(ch_data)

            # Flatlined: std near zero
            if ch_std < 0.01:
                self.signal_quality[ch] = 0.1
            # Saturated: always at max
            elif ch_range < 0.5 and np.mean(ch_data) > 900:
                self.signal_quality[ch] = 0.2
            else:
                # Normal: quality based on signal-to-noise ratio
                self.signal_quality[ch] = min(1.0, ch_std / (self.noise_std[ch] + 1e-9) * 0.3)
                self.signal_quality[ch] = np.clip(self.signal_quality[ch], 0.0, 1.0)

    # ── Mechanism 2: Movement Bias ───────────────────────────────────────

    def _apply_movement_bias(self, proba):
        """Shift probability mass from rest toward close/open.

        proba: array of shape (3,) — [close, open, rest] (session order)
        Returns adjusted probability array (sums to ~1).
        """
        bias = self.profile.movement_bias
        if bias == 0.0:
            return proba

        adjusted = proba.copy()
        # Only boost the dominant movement class — prevents stray cross-class predictions
        if adjusted[0] >= adjusted[1]:
            adjusted[0] += bias
        else:
            adjusted[1] += bias
        adjusted[2] -= bias  # rest loses what movement gains (1x, not 2x)

        # Clamp to [0, 1] and re-normalize
        adjusted = np.clip(adjusted, 0.0, None)
        total = adjusted.sum()
        if total > 0:
            adjusted /= total

        return adjusted

    # ── Mechanism 3: Non-linear Assist Curve ─────────────────────────────

    def _compute_assist_strength(self, confidence):
        """Map classifier confidence to motor assist strength via power curve.

        assist_strength = confidence ^ exponent
        exponent < 1 → boost (low confidence maps to higher motor drive)
        exponent = 1 → linear (no change)
        """
        return confidence ** self.profile.assist_exponent

    # ── Anti-jitter: Probability Smoothing ────────────────────────────

    def _smooth_proba(self, proba):
        """Apply EMA smoothing to probability outputs.

        Prevents single-frame spikes from flipping the prediction.
        alpha=0 means no smoothing, alpha=0.9 means heavy smoothing.
        """
        a = self.proba_alpha
        self.smoothed_proba = a * self.smoothed_proba + (1.0 - a) * proba
        # Re-normalize (EMA can drift slightly)
        total = self.smoothed_proba.sum()
        if total > 0:
            self.smoothed_proba /= total
        return self.smoothed_proba.copy()

    # ── Anti-jitter: Stability + Hysteresis + Cooldown ────────────────

    def _apply_stability(self, pred, confidence):
        """Combined stability filter with hysteresis, cooldown, and
        signal-strength-adaptive thresholds.

        When adaptive gain is high (= weak signals), confidence thresholds
        are automatically lowered so stroke patients can still trigger
        movements. The logic: if the system has to amplify 20x to see
        anything, the patient is working hard — lower the bar.

        Returns (effective_intent, changed).
        """
        profile = self.profile
        now = time.perf_counter()

        # Adaptive threshold scaling based on current gain level.
        # Higher gain = weaker patient signals = lower thresholds.
        # gain_factor: 1.0 at gain=1x (no reduction), up to 0.4 at gain=50x
        if profile.adaptive_gain and np.max(self.current_gains) > 1.0:
            max_gain = np.max(self.current_gains)
            # log scale: gain=1→1.0, gain=5→0.7, gain=20→0.5, gain=50→0.4
            gain_factor = max(0.4, 1.0 - 0.15 * np.log2(max_gain))
        else:
            gain_factor = 1.0

        # Use patient calibration overrides when available, else profile defaults
        base_enter = self._cal_hysteresis_enter if self._cal_hysteresis_enter is not None else profile.hysteresis_enter
        base_exit = self._cal_hysteresis_exit if self._cal_hysteresis_exit is not None else profile.hysteresis_exit
        base_floor = self._cal_confidence_floor if self._cal_confidence_floor is not None else profile.confidence_floor

        enter_threshold = base_enter * gain_factor
        exit_threshold = base_exit * gain_factor
        conf_floor = base_floor * gain_factor

        # Stability filter: require N consecutive same predictions
        self.recent_preds.append(pred)
        stable = (len(self.recent_preds) >= self.stability_count
                  and all(p == pred for p in self.recent_preds))

        changed = False
        if stable and pred != self.current_intent:
            # Cooldown check: don't switch if we just switched
            time_since_last = now - self.last_transition_time
            if time_since_last < self.cooldown_s:
                pass  # too soon, hold current state
            else:
                # Hysteresis check (with adaptive thresholds)
                if self.current_intent == 2:
                    # Leaving rest → movement
                    if confidence >= enter_threshold:
                        self.current_intent = pred
                        self.last_transition_time = now
                        changed = True
                else:
                    if pred == 2:
                        # Going to rest
                        if confidence >= exit_threshold:
                            self.current_intent = pred
                            self.last_transition_time = now
                            changed = True
                    else:
                        # Switching between close/open
                        if confidence >= enter_threshold:
                            self.current_intent = pred
                            self.last_transition_time = now
                            changed = True

        # Confidence floor (adaptive): fall back to rest if below
        effective_intent = self.current_intent
        if effective_intent != 2 and confidence < conf_floor:
            effective_intent = 2

        return effective_intent, changed

    # ── Adapted model: per-sample filter + feature extraction ────────────

    def _filter_sample(self, gained):
        """Apply causal IIR bandpass + update envelope for one sample."""
        filt_vals = np.empty(4)
        env_vals = np.empty(4)
        for ch in range(4):
            y, self.bp_state[ch] = sosfilt(
                self.bp_sos, np.array([gained[ch]]), zi=self.bp_state[ch]
            )
            filt_vals[ch] = y[0]
            self.rect_bufs[ch].append(abs(y[0]))
            env_vals[ch] = np.mean(self.rect_bufs[ch])
        return filt_vals, env_vals

    def _extract_adapted_features(self):
        """Extract the full feature vector from current window buffers."""
        filt_arr = np.array(self.filt_buf)   # (win, 4)
        env_arr = np.array(self.env_buf)     # (win, 4)
        fs = self.sample_rate
        feat = {}

        # --- 15 base features per channel ---
        for sess_ch, grab_ch in self.ch_map.items():
            w = filt_arr[:, sess_ch]
            env = env_arr[:, sess_ch]
            Nw = len(w)
            pref = f"ch{grab_ch}_"

            rms = np.sqrt(np.mean(w ** 2))
            mav = np.mean(np.abs(w))
            var = np.var(w)
            wl = np.sum(np.abs(np.diff(w)))
            maxamp = np.max(np.abs(w))
            thr = 0.01 * maxamp if maxamp > 0 else 0.0

            prod = w[:-1] * w[1:]
            zc = float(np.sum((prod < 0) & (np.abs(w[:-1] - w[1:]) > thr)))
            d1 = w[1:-1] - w[:-2]
            d2 = w[1:-1] - w[2:]
            ssc = float(np.sum((d1 * d2 > 0) & (np.abs(d1) > thr) & (np.abs(d2) > thr)))
            wamp = float(np.sum(np.abs(np.diff(w)) > thr))
            iemg = np.sum(np.abs(w))

            win_han = np.hanning(Nw)
            fft_vals = np.abs(rfft(w * win_han))
            freqs = rfftfreq(Nw, 1.0 / fs)
            psd = fft_vals ** 2
            psd_sum = psd.sum()
            if psd_sum > 0:
                mean_f = np.sum(freqs * psd) / psd_sum
                cum = np.cumsum(psd)
                median_f = freqs[np.searchsorted(cum, 0.5 * cum[-1])]
            else:
                mean_f = median_f = 0.0

            env_mean = np.mean(env)
            env_max = np.max(env)
            env_std = np.std(env)
            env_rms = np.sqrt(np.mean(env ** 2))

            feat[pref + "rms"] = rms
            feat[pref + "mav"] = mav
            feat[pref + "var"] = var
            feat[pref + "wl"] = wl
            feat[pref + "maxamp"] = maxamp
            feat[pref + "zc"] = zc
            feat[pref + "ssc"] = ssc
            feat[pref + "wamp"] = wamp
            feat[pref + "iemg"] = iemg
            feat[pref + "mean_freq"] = mean_f
            feat[pref + "median_freq"] = median_f
            feat[pref + "env_mean"] = env_mean
            feat[pref + "env_max"] = env_max
            feat[pref + "env_std"] = env_std
            feat[pref + "env_rms"] = env_rms

        # --- Temporal features (prev, prev2, delta, accel, roll3, roll5) ---
        for col in self.key_cols:
            cur = feat[col]
            hist = self.key_history[col]
            n = len(hist)

            prev = hist[-1] if n >= 1 else 0.0
            prev2 = hist[-2] if n >= 2 else 0.0
            delta = cur - prev
            prev_d = self.key_prev_delta[col]
            accel = delta - prev_d

            vals_for_roll = list(hist) + [cur]
            roll3 = np.mean(vals_for_roll[-3:]) if len(vals_for_roll) >= 1 else cur
            roll5 = np.mean(vals_for_roll[-5:]) if len(vals_for_roll) >= 1 else cur

            feat[f"{col}_prev"] = prev
            feat[f"{col}_prev2"] = prev2
            feat[f"{col}_delta"] = delta
            feat[f"{col}_accel"] = accel
            feat[f"{col}_roll3"] = roll3
            feat[f"{col}_roll5"] = roll5

            hist.append(cur)
            self.key_prev_delta[col] = delta

        # --- Cross-channel features ---
        channels = [0, 4, 9, 13]
        for sig in ["rms", "mav", "env_rms"]:
            for i in range(len(channels)):
                for j in range(i + 1, len(channels)):
                    ci, cj = channels[i], channels[j]
                    vi = feat[f"ch{ci}_{sig}"]
                    vj = feat[f"ch{cj}_{sig}"]
                    feat[f"ch{ci}_ch{cj}_{sig}_ratio"] = vi / (vj + 1e-8)
                    feat[f"ch{ci}_ch{cj}_{sig}_diff"] = vi - vj

        feat["rest_activity"] = feat["ch0_env_rms"] + feat["ch4_env_rms"] + feat["ch9_env_rms"] + feat["ch13_env_rms"]
        feat["flexor_activity"] = feat["ch0_rms"] + feat["ch9_rms"]
        feat["extensor_activity"] = feat["ch4_rms"] + feat["ch13_rms"]
        feat["flexor_extensor_ratio"] = feat["flexor_activity"] / (feat["extensor_activity"] + 1e-8)

        # --- trial_pos (fixed at 0.5 for real-time) ---
        feat["trial_pos"] = 0.5

        # --- Per-patient z-score normalisation ---
        norm_mean = self.patient_norm["mean"]
        norm_std = self.patient_norm["std"]
        for name in self.feature_names:
            feat[name] = (feat.get(name, 0.0) - norm_mean.get(name, 0.0)) / norm_std.get(name, 1.0)

        # --- Assemble into array in correct order ---
        return np.array([feat.get(n, 0.0) for n in self.feature_names])

    # ── Main sample-processing pipeline ──────────────────────────────────

    def add_sample(self, values):
        """Add a new 4-channel sample.

        Returns (intent, confidence, changed, assist_strength) or None if
        not enough data yet.
        """
        if self.model_type == "adapted_hgb":
            return self._add_sample_adapted(values)
        return self._add_sample_session(values)

    def _add_sample_session(self, values):
        """Session model path with stroke-robust preprocessing."""
        # 0. Stroke-robust preprocessing
        self._update_fatigue_tracking(values)
        self._update_signal_quality(values)

        if self._is_artifact(values):
            self.prev_sample = np.asarray(values, dtype=np.float64).copy()
            return None  # drop artifact samples entirely
        self.prev_sample = np.asarray(values, dtype=np.float64).copy()

        gated = self._apply_noise_gate(values)
        is_cocontraction = self._detect_cocontraction(values)

        # 1. Adaptive gain normalization (before buffering)
        gained_values = self._apply_adaptive_gain(gated)
        self.buffer.append(gained_values)
        self.samples_since_last_pred += 1

        if len(self.buffer) < self.win_samples:
            return None

        if self.samples_since_last_pred < self.stride_samples:
            return None

        self.samples_since_last_pred = 0

        # Extract features from current window
        window = np.array(list(self.buffer))
        feat = extract_window_features(window)

        # Add temporal features inline
        env_rms_vals = []
        from train_from_session import get_feature_names
        feat_names = get_feature_names()
        for i, name in enumerate(feat_names):
            if name.endswith("_env_rms"):
                env_rms_vals.append(feat[i])

        temporal = []
        for ch_idx, val in enumerate(env_rms_vals):
            prev = self.prev_env_rms[ch_idx]
            self.env_rms_history[ch_idx].append(val)

            temporal.append(prev)  # _prev
            temporal.append(val - prev)  # _delta
            temporal.append(np.mean(list(self.env_rms_history[ch_idx])))  # _roll3

            self.prev_env_rms[ch_idx] = val

        full_feat = np.concatenate([feat, temporal])
        X = full_feat.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Predict (raw probabilities)
        proba = self.model.predict_proba(X_scaled)[0]

        # 2. Co-contraction override: force rest if detected
        if is_cocontraction:
            proba = np.array([0.0, 0.0, 1.0])  # force rest

        # 3. Movement bias
        proba = self._apply_movement_bias(proba)

        # 4. Probability smoothing (EMA)
        proba = self._smooth_proba(proba)

        pred = int(np.argmax(proba))
        confidence = proba[pred]

        # 5. Assist strength (power-curve mapping)
        assist_strength = self._compute_assist_strength(confidence)

        # 6. Stability filter + hysteresis + cooldown
        effective_intent, changed = self._apply_stability(pred, confidence)

        return (effective_intent, confidence, changed, assist_strength)

    def _add_sample_adapted(self, values):
        """Adapted model path with stroke-robust preprocessing."""
        # 0. Stroke-robust preprocessing
        self._update_fatigue_tracking(values)
        self._update_signal_quality(values)

        if self._is_artifact(values):
            self.prev_sample = np.asarray(values, dtype=np.float64).copy()
            return None  # drop artifact samples entirely
        self.prev_sample = np.asarray(values, dtype=np.float64).copy()

        gated = self._apply_noise_gate(values)
        is_cocontraction = self._detect_cocontraction(values)

        gained_values = self._apply_adaptive_gain(gated)

        # Causal IIR bandpass + envelope (per-sample, persistent state)
        filt_vals, env_vals = self._filter_sample(gained_values)
        self.filt_buf.append(filt_vals)
        self.env_buf.append(env_vals)
        self.samples_since_last_pred += 1

        if len(self.filt_buf) < self.win_samples:
            return None
        if self.samples_since_last_pred < self.stride_samples:
            return None
        self.samples_since_last_pred = 0

        # Extract full feature vector
        feat_vec = self._extract_adapted_features()
        X_scaled = self.scaler.transform(feat_vec.reshape(1, -1))

        # Predict — GrabMyo order: [rest=0, close=1, open=2]
        proba_grabmyo = self.model.predict_proba(X_scaled)[0]

        # Remap to session order: [close=0, open=1, rest=2]
        proba = np.array([
            proba_grabmyo[1],  # close
            proba_grabmyo[2],  # open
            proba_grabmyo[0],  # rest
        ])

        # Co-contraction override: force rest
        if is_cocontraction:
            proba = np.array([0.0, 0.0, 1.0])

        # Movement bias (operates in session order)
        proba = self._apply_movement_bias(proba)

        # Probability smoothing (EMA)
        proba = self._smooth_proba(proba)

        pred = int(np.argmax(proba))
        confidence = proba[pred]
        assist_strength = self._compute_assist_strength(confidence)

        # Stability filter + hysteresis + cooldown
        effective_intent, changed = self._apply_stability(pred, confidence)

        return (effective_intent, confidence, changed, assist_strength)


def rest_calibrate(ser, duration=REST_CALIBRATION_SECONDS):
    """Collect EMG during rest to establish per-channel noise floor.

    The patient must be completely relaxed during this period.
    Returns dict with per-channel mean, std, max, and p95 of rest signal.
    """
    import subprocess

    print("\n" + "=" * 60)
    print("REST CALIBRATION")
    print("=" * 60)
    print(f"Patient must be completely relaxed for {duration} seconds.")
    print("Arm in the position they'll use during the session.")
    input("Press Enter when ready...")

    try:
        subprocess.Popen(["say", "Relax completely"])
    except FileNotFoundError:
        pass

    print(f"Calibrating ({duration}s)...")
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

        count = len(samples)
        if count % 40 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"\r  {elapsed:.0f}s  |  {count} samples", end="", flush=True)

    if len(samples) < 10:
        print("\n  WARNING: Too few samples for calibration. Using defaults.")
        return None

    data = np.array(samples)  # (N, 4)
    baseline = {
        "mean": data.mean(axis=0),
        "std": data.std(axis=0),
        "max": data.max(axis=0),
        "p95": np.percentile(data, 95, axis=0),
    }

    print(f"\n\n  Samples: {len(samples)}")
    for ch in range(4):
        print(f"  CH{ch+1}: mean={baseline['mean'][ch]:.1f}  "
              f"std={baseline['std'][ch]:.1f}  "
              f"max={baseline['max'][ch]:.1f}  "
              f"p95={baseline['p95'][ch]:.1f}")

    # Noise gate will be set at mean + 2.5*std or 1.1*max (whichever is higher)
    gate = np.maximum(
        baseline["mean"] + 2.5 * baseline["std"],
        baseline["max"] * 1.1,
    )
    print(f"\n  Noise gate: [{gate[0]:.1f}, {gate[1]:.1f}, {gate[2]:.1f}, {gate[3]:.1f}]")
    print("  Signals below this will be treated as noise.\n")

    try:
        subprocess.Popen(["say", "Calibration complete"])
    except FileNotFoundError:
        pass

    return baseline


def _drain_serial(ser):
    """Read all available lines from serial buffer, return the latest valid one.

    This prevents the input buffer from growing unboundedly when inference
    takes longer than one sample period (~50 ms at 20 Hz).  We feed every
    sample into the predictor (so the sliding window stays correct), but
    the caller gets back a list of *all* parsed samples so it can decide
    which ones to process.
    """
    lines = []
    while ser.in_waiting:
        raw = ser.readline()
        if not raw:
            break
        try:
            line = raw.decode("utf-8", errors="ignore").strip()
        except UnicodeDecodeError:
            continue
        if line:
            lines.append(line)
    return lines


def run_loop(ser, predictor):
    """Main real-time inference loop."""
    profile = predictor.profile
    print("\n" + "=" * 60)
    print(f"REAL-TIME MODE — Assist Level {profile.level} ({profile.label})")
    print("Sending commands to Teensy  |  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    t0 = time.perf_counter()
    sample_count = 0
    cmd_count = 0
    last_result = None

    try:
        while True:
            # Drain all buffered lines to prevent serial buffer buildup
            buffered = _drain_serial(ser)

            if not buffered:
                # Nothing buffered — do a blocking read for the next line
                raw = ser.readline()
                if not raw:
                    continue
                try:
                    line = raw.decode("utf-8", errors="ignore").strip()
                except UnicodeDecodeError:
                    continue
                if line:
                    buffered = [line]
                else:
                    continue

            # Feed every sample into the predictor (keeps sliding window
            # up-to-date) but only act on the last prediction result
            for line in buffered:
                vals = parse_emg_line(line)
                if vals is None:
                    continue

                sample_count += 1
                result = predictor.add_sample(vals)
                if result is not None:
                    last_result = (result, vals)

            if last_result is None:
                continue

            (intent, confidence, changed, assist_strength), vals = last_result
            last_result = None

            intent_name = LABEL_NAMES[intent].upper()
            cmd = COMMAND_MAP[intent]

            # Send command (non-blocking: write_timeout prevents stalls)
            try:
                ser.write(cmd.encode())
            except serial.SerialTimeoutException:
                pass  # drop command rather than freeze
            cmd_count += 1

            ts = time.perf_counter() - t0
            marker = " *" if changed else ""

            # Build gain string
            gains = predictor.current_gains
            if profile.adaptive_gain:
                gain_str = (f" gain=[{gains[0]:4.1f}x {gains[1]:4.1f}x {gains[2]:4.1f}x {gains[3]:4.1f}x]")
            else:
                gain_str = ""

            print(f"\r  [{ts:7.1f}s] ch1={vals[0]:7.1f} ch2={vals[1]:7.1f} ch3={vals[2]:7.1f} ch4={vals[3]:7.1f}"
                  f" -> {intent_name:5s} (conf: {confidence:.0%} "
                  f"assist: {assist_strength:.0%}){gain_str}{marker}   ",
                  end="", flush=True)

    except KeyboardInterrupt:
        elapsed = time.perf_counter() - t0
        print(f"\n\nStopped. {sample_count} samples, {cmd_count} commands in {elapsed:.1f}s")


def run_loop_json(ser, predictor):
    """JSON output mode for Node.js PythonBridge integration.

    Outputs one JSON object per prediction to stdout, parsed by bridge.ts.
    Also relays motor commands from stdin to serial via a background thread.
    """
    import json
    import threading

    # Background thread relays motor commands from stdin to serial.
    # Using a thread instead of select.select() because macOS pipes
    # from Node.js child processes don't always work with select().
    def stdin_relay():
        try:
            for line in sys.stdin:
                if line.strip():
                    try:
                        ser.write(line.encode())
                    except Exception:
                        pass  # drop command rather than crash
        except Exception:
            pass

    relay_thread = threading.Thread(target=stdin_relay, daemon=True)
    relay_thread.start()

    # EMA smoothing for display EMG — reduces single-sample noise
    emg_ema = np.zeros(4)
    emg_ema_alpha = 0.08  # slow EMA (~12 sample window) — smooth display matching state transitions
    emg_ema_initialized = False

    try:
        while True:
            buffered = _drain_serial(ser)
            if not buffered:
                raw = ser.readline()
                if not raw:
                    continue
                try:
                    line = raw.decode("utf-8", errors="ignore").strip()
                except UnicodeDecodeError:
                    continue
                if line:
                    buffered = [line]
                else:
                    continue

            last_result = None
            last_vals = None
            for line in buffered:
                vals = parse_emg_line(line)
                if vals is None:
                    continue
                # Update EMG display EMA on every sample (not just predictions)
                arr = np.array(vals, dtype=np.float64)
                if not emg_ema_initialized:
                    emg_ema[:] = np.abs(arr)
                    emg_ema_initialized = True
                else:
                    emg_ema[:] = emg_ema_alpha * np.abs(arr) + (1 - emg_ema_alpha) * emg_ema
                result = predictor.add_sample(vals)
                if result is not None:
                    last_result = result
                    last_vals = vals

            if last_result is None:
                continue

            intent, confidence, changed, assist_strength = last_result
            print(json.dumps({
                "emg": [round(float(v), 2) for v in emg_ema],
                "intent": LABEL_NAMES[intent],
                "confidence": round(confidence, 3),
                "assist_strength": round(assist_strength, 3),
            }), flush=True)

    except KeyboardInterrupt:
        pass


def send_motor_command(ser, motor_cmd, current_finger_code=[None]):
    """Translate a MotorCommand to serial bytes and send to Teensy.

    Sends finger selection code only when it changes.
    """
    finger_code = FINGER_SERIAL_CODES.get(motor_cmd.finger, "A")
    if finger_code != current_finger_code[0]:
        ser.write(finger_code.encode())
        current_finger_code[0] = finger_code

    action_char = ACTION_TO_SERIAL.get(motor_cmd.action, "r")
    ser.write(action_char.encode())


def run_exercise_loop(ser, predictor, exercises):
    """Exercise mode: structured rep-based exercises with state machine."""
    profile = predictor.profile
    last_assist = [0.0]

    def get_assist():
        return last_assist[0]

    session = SessionRunner(exercises, assist_strength_fn=get_assist)

    print("\n" + "=" * 60)
    print(f"EXERCISE MODE — Assist Level {profile.level} ({profile.label})")
    print(f"{len(exercises)} exercises in programme")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    for i, ex in enumerate(exercises):
        target = "close" if ex.target_intent == 0 else "open"
        print(f"  {i+1}. {ex.name} ({ex.finger}, {target}) x{ex.reps}")
    print()

    t0 = time.perf_counter()
    finger_state = [None]

    try:
        while not session.is_completed:
            # Drain serial buffer to prevent buildup
            buffered = _drain_serial(ser)
            if not buffered:
                raw = ser.readline()
                if not raw:
                    continue
                try:
                    line = raw.decode("utf-8", errors="ignore").strip()
                except UnicodeDecodeError:
                    continue
                if line:
                    buffered = [line]
                else:
                    continue

            # Feed all samples, keep last prediction result
            latest_result = None
            for line in buffered:
                vals = parse_emg_line(line)
                if vals is None:
                    continue
                result = predictor.add_sample(vals)
                if result is not None:
                    latest_result = result

            if latest_result is None:
                continue

            intent, confidence, changed, assist_strength = latest_result
            last_assist[0] = assist_strength

            events = session.update(intent, confidence)
            motor_cmd = session.get_motor_command()

            # Send finger selection if changed
            new_finger = session.finger_changed()
            if new_finger is not None:
                code = FINGER_SERIAL_CODES.get(new_finger, "A")
                try:
                    ser.write(code.encode())
                except serial.SerialTimeoutException:
                    pass

            # Send motor command
            action_char = ACTION_TO_SERIAL.get(motor_cmd.action, "r")
            try:
                ser.write(action_char.encode())
            except serial.SerialTimeoutException:
                pass

            # Handle events
            for event in events:
                runner = session.current_runner
                ex = session.current_exercise
                if event == Event.EFFORT_DETECTED:
                    print(f"\n  Effort detected!")
                elif event == Event.REP_COMPLETED:
                    prev_idx = session.current_index
                    prev_runner = runner
                    # If exercise just completed, results are already recorded
                    if session.results:
                        last_result = session.results[-1]
                        print(f"\n  Rep {last_result.reps_completed} complete")
                    elif runner:
                        print(f"\n  Rep {runner.reps_completed} complete")
                elif event == Event.EXERCISE_COMPLETED:
                    print(f"\n  Exercise complete!")
                elif event == Event.TIMEOUT_WARNING:
                    ex_name = ex.name if ex else "finger"
                    print(f"\n  Try to move your {ex_name.split()[0].lower()}...")
                elif event == Event.TIMEOUT_PROMPT:
                    print(f"\n  No effort detected. Press 's' in terminal to skip.")

            # Status line
            ex = session.current_exercise
            runner = session.current_runner
            if ex and runner:
                ex_idx = session.current_index + 1
                ex_total = len(session.exercises)
                state_name = runner.state.name
                state_t = runner.state_elapsed

                gains = predictor.current_gains
                if profile.adaptive_gain:
                    gain_str = f" gain=[{gains[0]:4.1f}x {gains[1]:4.1f}x {gains[2]:4.1f}x {gains[3]:4.1f}x]"
                else:
                    gain_str = ""

                print(f"\r  [Ex {ex_idx}/{ex_total}] {ex.name} | "
                      f"Rep {runner.reps_completed + 1}/{ex.reps} | "
                      f"{state_name} ({state_t:.1f}s) | "
                      f"conf: {confidence:.0%} assist: {assist_strength:.0%}"
                      f"{gain_str}   ",
                      end="", flush=True)

    except KeyboardInterrupt:
        session.stop()
        # Send rest command
        ser.write(b"Ar")
        print("\n\nSession stopped by user.")

    # Print summary
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    duration = session.session_duration
    print(f"  Duration: {duration:.0f}s ({duration/60:.1f} min)")
    print()
    for r in session.get_summary():
        status = "SKIPPED" if r.skipped else f"{r.reps_completed}/{r.reps_target}"
        print(f"  {r.name:40s}  {status}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Real-time ExoHand EMG control")
    parser.add_argument("--port", required=True, help="Serial port")
    parser.add_argument("--model", required=True, help="Path to exohand_model.pkl")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default 115200)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibration before starting")
    parser.add_argument("--assist-level", type=int, default=3, choices=[1, 2, 3, 4, 5],
                        help="Assist level 1-5 (1=max assist for early rehab, "
                             "5=minimal assist for near-healthy). Default: 3")
    parser.add_argument("--exercise", action="store_true",
                        help="Enable exercise mode (structured rep-based exercises)")
    parser.add_argument("--web", action="store_true",
                        help="Start FastAPI web server at localhost:8000")
    parser.add_argument("--node", action="store_true",
                        help="Node.js bridge mode: JSON lines on stdout for PythonBridge")
    parser.add_argument("--skip-rest-cal", action="store_true",
                        help="Skip rest calibration (not recommended for stroke patients)")
    parser.add_argument("--patient-calibrate", action="store_true",
                        help="Run full 6-min patient calibration protocol")
    parser.add_argument("--patient-recalibrate", action="store_true",
                        help="Run abbreviated 90-second patient recalibration")
    parser.add_argument("--patient-id", type=str, default="default",
                        help="Patient identifier for calibration save/load (default: 'default')")
    args = parser.parse_args()

    # Web mode: delegate to server.py which manages its own loop
    if args.web:
        from server import start_server
        start_server(args)
        return

    profile = get_profile(args.assist_level)

    print(f"Loading model from {args.model}...")
    model_data = load_model(args.model)
    mtype = model_data.get("model_type", "session")
    print(f"  Model type: {mtype}")
    print(f"  Window: {model_data['window_ms']}ms, Stride: {model_data['stride_ms']}ms")
    print(f"  Features: {len(model_data['feature_names'])}")
    if mtype == "adapted_hgb":
        chm = model_data.get("channel_map", {0: 0, 1: 4, 2: 9, 3: 13})
        print(f"  Channel map: port0→ch{chm[0]}  port1→ch{chm[1]}  port2→ch{chm[2]}  port3→ch{chm[3]}")
        print(f"  Patient accuracy: {model_data.get('patient_accuracy', '?'):.1%}")
        print(f"  GrabMyo accuracy: {model_data.get('grabmyo_accuracy', '?'):.1%}")

    print()
    print_profile(profile)

    print(f"\nConnecting to {args.port} at {args.baud}...")
    ser = serial.Serial(args.port, args.baud, timeout=0.1, write_timeout=0.05)
    time.sleep(2)
    ser.reset_input_buffer()

    if args.calibrate:
        model_data = calibrate(ser, model_data)

    # Estimate sample rate from a quick burst
    print("\nEstimating sample rate (2s)...")
    ser.reset_input_buffer()
    count = 0
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < 2.0:
        raw = ser.readline()
        if raw:
            try:
                line = raw.decode("utf-8", errors="ignore").strip()
                if parse_emg_line(line) is not None:
                    count += 1
            except UnicodeDecodeError:
                pass
    sample_rate = count / 2.0
    print(f"  Estimated: ~{sample_rate:.0f} Hz")

    if sample_rate < 10:
        print("WARNING: Very low sample rate. Check serial connection.")

    # Patient calibration (replaces standard rest calibration when used)
    rest_baseline = None
    calibration_params = None

    if args.patient_calibrate:
        from calibrate_patient import calibrate_patient
        cal_result = calibrate_patient(
            ser, model_data, sample_rate,
            patient_id=args.patient_id, assist_profile=profile)
        model_data = cal_result.finetuned_model_data
        rest_baseline = cal_result.rest_baseline
        calibration_params = cal_result.calibration_params
    elif args.patient_recalibrate:
        from calibrate_patient import abbreviated_calibrate, load_calibrated_model
        # Load previous calibrated model if it exists
        prev_model = load_calibrated_model(args.patient_id)
        if prev_model is not None:
            print(f"  Loaded previous calibrated model for '{args.patient_id}'")
            model_data = prev_model
        cal_result = abbreviated_calibrate(
            ser, model_data, sample_rate,
            patient_id=args.patient_id, assist_profile=profile)
        model_data = cal_result.finetuned_model_data
        rest_baseline = cal_result.rest_baseline
        calibration_params = cal_result.calibration_params
    elif not args.skip_rest_cal:
        rest_baseline = rest_calibrate(ser, REST_CALIBRATION_SECONDS)
    else:
        print("\n  Skipping rest calibration (--skip-rest-cal)")

    predictor = RealtimePredictor(model_data, sample_rate, assist_profile=profile,
                                  rest_baseline=rest_baseline,
                                  calibration_params=calibration_params)

    if not args.node:
        print(f"  Window: {predictor.win_samples} samples, Stride: {predictor.stride_samples} samples")

    ser.reset_input_buffer()

    if args.node:
        # Signal to Node.js PythonBridge that model is loaded and ready
        import json
        print(json.dumps({"type": "ready"}), flush=True)
        run_loop_json(ser, predictor)
    elif args.exercise:
        run_exercise_loop(ser, predictor, default_programme())
    else:
        run_loop(ser, predictor)

    ser.close()


if __name__ == "__main__":
    main()
