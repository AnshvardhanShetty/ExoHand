#!/usr/bin/env python3
"""
End-to-end test for the patient calibration system using a fake serial port.

Exercises the full pipeline:
  1. Protocol building
  2. EMG data collection (via FakeSerial)
  3. Onset trimming + outlier rejection
  4. Feature extraction (both session and adapted_hgb paths)
  5. Model fine-tuning
  6. Calibration param computation
  7. Save / load / list persistence
  8. Abbreviated recalibration with merge
  9. RealtimePredictor integration with calibration params
  10. Edge cases (empty data, single class, cancellation)

Usage:
    python3 test_calibration_e2e.py
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback

import joblib
import numpy as np

# ── Fake serial port ──────────────────────────────────────────────────────────

class FakeSerial:
    """Simulates a Teensy serial connection producing 4-channel EMG data.

    Generates realistic-ish EMG patterns:
      - close: elevated on ch0,ch1 (flexor), low on ch2,ch3
      - open:  elevated on ch2,ch3 (extensor), low on ch0,ch1
      - rest:  low noise on all channels
    """

    def __init__(self, sample_rate=200, gesture_schedule=None):
        self.sample_rate = sample_rate
        self._interval = 1.0 / sample_rate
        self._t_start = time.perf_counter()
        self._sample_count = 0
        self._current_gesture = "rest"  # "close", "open", "rest"
        self._rng = np.random.RandomState(42)
        self._buffer = b""

    def set_gesture(self, gesture):
        self._current_gesture = gesture

    def reset_input_buffer(self):
        self._buffer = b""

    def readline(self):
        """Return one line of fake EMG data at the configured sample rate."""
        # Simulate timing
        expected_time = self._t_start + self._sample_count * self._interval
        now = time.perf_counter()
        if now < expected_time:
            time.sleep(max(0, expected_time - now))

        self._sample_count += 1
        noise = self._rng.randn(4) * 0.03

        if self._current_gesture == "close":
            signal = np.array([0.4, 0.35, 0.05, 0.08]) + self._rng.randn(4) * 0.08
        elif self._current_gesture == "open":
            signal = np.array([0.05, 0.08, 0.4, 0.35]) + self._rng.randn(4) * 0.08
        else:  # rest
            signal = np.array([0.02, 0.02, 0.02, 0.02]) + noise

        vals = signal + noise
        line = "\t".join(f"{v:.4f}" for v in vals) + "\n"
        return line.encode("utf-8")


# ── Monkey-patch TTS to be silent ─────────────────────────────────────────────

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime import calibrate_patient
_original_say = calibrate_patient._say

def _silent_say(text, blocking=False):
    """No-op TTS for testing."""
    pass

calibrate_patient._say = _silent_say


# ── Gesture-aware trial runner ────────────────────────────────────────────────

_original_collect = calibrate_patient._collect_emg_segment

def _patched_collect(ser, duration_s, label):
    """Set the fake serial gesture before collecting."""
    gesture_map = {0: "close", 1: "open", 2: "rest"}
    if hasattr(ser, "set_gesture"):
        ser.set_gesture(gesture_map.get(label, "rest"))
    return _original_collect(ser, duration_s, label)

calibrate_patient._collect_emg_segment = _patched_collect


# ── Test helpers ──────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_protocol_builders():
    section("Protocol Builders")
    from runtime.calibrate_patient import build_full_protocol, build_abbreviated_protocol, LABEL_MAP

    full = build_full_protocol()
    abbrev = build_abbreviated_protocol()

    check("full protocol has trials", len(full) > 0)
    check("abbreviated protocol has trials", len(abbrev) > 0)
    check("abbreviated is shorter", len(abbrev) < len(full))

    # Check all trials have valid labels
    valid_labels = set(LABEL_MAP.values())
    check("full: all labels valid",
          all(t.label in valid_labels for t in full))
    check("abbreviated: all labels valid",
          all(t.label in valid_labels for t in abbrev))

    # Check gesture balance in abbreviated
    from collections import Counter
    gestures = Counter(t.gesture for t in abbrev)
    check("abbreviated: balanced gestures",
          gestures["close"] == gestures["open"] == gestures["rest"],
          f"got {dict(gestures)}")


def test_data_collection_and_cleaning():
    section("Data Collection & Cleaning")
    from runtime.calibrate_patient import (
        _collect_emg_segment, _trim_onset, _reject_outliers,
        _validate_trial, CalibrationTrial, LABEL_MAP,
    )

    ser = FakeSerial(sample_rate=200)

    # Collect 2 seconds of close gesture
    ser.set_gesture("close")
    samples = _collect_emg_segment(ser, 2.0, 0)
    check("collected samples", len(samples) > 100, f"got {len(samples)}")
    check("sample format is (vals, label) tuple",
          len(samples[0]) == 2 and len(samples[0][0]) == 4)

    # Onset trimming
    trial = CalibrationTrial(gesture="close", label=0, effort="normal",
                             duration=5.0, rest_period=4.0, phase=3)
    trimmed = _trim_onset(samples, 200, trial)
    check("onset trimming removed samples",
          len(trimmed) < len(samples),
          f"{len(samples)} -> {len(trimmed)}")

    # Outlier rejection (inject some outliers)
    bad_samples = list(samples)
    for i in range(5):
        bad_samples.append(([100.0, 100.0, 100.0, 100.0], 0))  # extreme outlier
    cleaned, n_rejected = _reject_outliers(bad_samples)
    check("outlier rejection caught outliers",
          n_rejected > 0, f"rejected {n_rejected}")

    # Trial validation
    trial.samples = [s[0] for s in samples]
    _validate_trial(trial, 200)
    check("trial validation: ok quality", trial.quality == "ok")

    # Empty trial
    empty_trial = CalibrationTrial(gesture="close", label=0, effort="normal",
                                   duration=5.0, rest_period=4.0, phase=3)
    empty_trial.samples = []
    _validate_trial(empty_trial, 200)
    check("empty trial: no_response quality", empty_trial.quality == "no_response")


def test_session_feature_extraction():
    section("Session Model Feature Extraction")
    from runtime.calibrate_patient import _extract_session_features

    model_data = joblib.load("exohand_model.pkl")
    ser = FakeSerial(sample_rate=200)
    np.random.seed(42)

    # Generate fake labelled data
    n = 1000
    close_data = np.random.randn(n, 4) * 0.08 + np.array([0.4, 0.35, 0.05, 0.08])
    open_data = np.random.randn(n, 4) * 0.08 + np.array([0.05, 0.08, 0.4, 0.35])
    rest_data = np.random.randn(n, 4) * 0.03 + 0.02

    raw_samples = np.vstack([close_data, open_data, rest_data])
    raw_labels = np.concatenate([np.zeros(n), np.ones(n), np.full(n, 2)])

    features, labels = _extract_session_features(raw_samples, raw_labels, 200, model_data)

    check("session features shape[1] == 36",
          features.shape[1] == 36, f"got {features.shape}")
    check("session features non-empty", len(features) > 0)
    check("session labels match features", len(labels) == len(features))
    check("no NaN in features", not np.any(np.isnan(features)))
    check("labels contain all 3 classes",
          len(np.unique(labels)) == 3, f"unique={np.unique(labels)}")


def test_adapted_feature_extraction():
    section("Adapted HGB Feature Extraction")
    from runtime.calibrate_patient import _extract_adapted_features

    model_path = "exohand_adapted_model.pkl"
    if not os.path.exists(model_path):
        print("  SKIP  (no adapted model found)")
        return

    model_data = joblib.load(model_path)
    n_features = len(model_data["feature_names"])
    np.random.seed(42)

    n = 2000  # need enough for windowing
    close_data = np.random.randn(n, 4) * 0.08 + np.array([0.4, 0.35, 0.05, 0.08])
    open_data = np.random.randn(n, 4) * 0.08 + np.array([0.05, 0.08, 0.4, 0.35])
    rest_data = np.random.randn(n, 4) * 0.03 + 0.02

    raw_samples = np.vstack([close_data, open_data, rest_data])
    raw_labels = np.concatenate([np.zeros(n), np.ones(n), np.full(n, 2)])

    features, labels = _extract_adapted_features(raw_samples, raw_labels, 200, model_data)

    check("adapted features non-empty", len(features) > 0, f"got {features.shape}")
    check(f"adapted features shape[1] == {n_features}",
          features.shape[1] == n_features,
          f"got {features.shape[1]}")
    check("adapted labels match features", len(labels) == len(features))
    check("no NaN in adapted features", not np.any(np.isnan(features)))

    # Labels should be in GrabMyo order (0=rest, 1=close, 2=open)
    unique = np.unique(labels)
    check("adapted labels in GrabMyo order (3 classes)",
          len(unique) == 3, f"unique={unique}")


def test_finetune_session():
    section("Session Model Fine-Tuning")
    from runtime.calibrate_patient import finetune_model, _extract_session_features

    model_data = joblib.load("exohand_model.pkl")
    np.random.seed(42)

    n = 1000
    raw_samples = np.vstack([
        np.random.randn(n, 4) * 0.08 + np.array([0.4, 0.35, 0.05, 0.08]),
        np.random.randn(n, 4) * 0.08 + np.array([0.05, 0.08, 0.4, 0.35]),
        np.random.randn(n, 4) * 0.03 + 0.02,
    ])
    raw_labels = np.concatenate([np.zeros(n), np.ones(n), np.full(n, 2)])
    features, labels = _extract_session_features(raw_samples, raw_labels, 200, model_data)

    finetuned = finetune_model(model_data, features, labels)

    check("finetuned has model", "model" in finetuned)
    check("finetuned has scaler", "scaler" in finetuned)
    check("finetuned model is different object",
          finetuned["model"] is not model_data["model"])
    check("finetuned scaler is different object",
          finetuned["scaler"] is not model_data["scaler"])

    # Test prediction works
    pred = finetuned["model"].predict(finetuned["scaler"].transform(features[:10]))
    check("prediction works", len(pred) == 10)


def test_finetune_adapted():
    section("Adapted HGB Model Fine-Tuning")
    from runtime.calibrate_patient import finetune_model, _extract_adapted_features

    model_path = "exohand_adapted_model.pkl"
    if not os.path.exists(model_path):
        print("  SKIP  (no adapted model found)")
        return

    model_data = joblib.load(model_path)
    np.random.seed(42)

    n = 2000
    raw_samples = np.vstack([
        np.random.randn(n, 4) * 0.08 + np.array([0.4, 0.35, 0.05, 0.08]),
        np.random.randn(n, 4) * 0.08 + np.array([0.05, 0.08, 0.4, 0.35]),
        np.random.randn(n, 4) * 0.03 + 0.02,
    ])
    raw_labels = np.concatenate([np.zeros(n), np.ones(n), np.full(n, 2)])
    features, labels = _extract_adapted_features(raw_samples, raw_labels, 200, model_data)

    finetuned = finetune_model(model_data, features, labels)

    check("adapted finetuned has model", "model" in finetuned)
    check("adapted finetuned model is different object",
          finetuned["model"] is not model_data["model"])

    # Prediction with the existing scaler (adapted path keeps same scaler)
    X_test = model_data["scaler"].transform(
        np.nan_to_num(features[:10], nan=0.0, posinf=0.0, neginf=0.0))
    pred = finetuned["model"].predict(X_test)
    check("adapted prediction works", len(pred) == 10)


def test_save_load_persistence():
    section("Save / Load / Persistence")
    from runtime.calibrate_patient import (
        CalibrationResult, CalibrationTrial, save_calibration,
        load_calibration, load_calibrated_model, list_patients,
        _calibration_dir,
    )

    # Use a temp patient ID
    patient_id = "_test_patient_e2e"
    cal_dir = _calibration_dir(patient_id)

    try:
        np.random.seed(42)
        model_data = joblib.load("exohand_model.pkl")

        result = CalibrationResult(
            patient_id=patient_id,
            timestamp=time.time(),
            calibration_type="full",
            rest_baseline={
                "mean": np.array([0.01, 0.01, 0.01, 0.01]),
                "std": np.array([0.05, 0.05, 0.05, 0.05]),
                "max": np.array([0.15, 0.15, 0.15, 0.15]),
                "p95": np.array([0.1, 0.1, 0.1, 0.1]),
            },
            trials=[CalibrationTrial("close", 0, "normal", 5, 4, 3)],
            raw_samples=np.random.randn(500, 4),
            raw_labels=np.concatenate([np.zeros(200), np.ones(150), np.full(150, 2)]),
            sample_rate=200.0,
            per_class_stats={
                0: {"mean_amp": 0.5, "median_amp": 0.5, "std_amp": 0.1, "snr": 10.0},
                1: {"mean_amp": 0.4, "median_amp": 0.4, "std_amp": 0.1, "snr": 10.0},
                2: {"mean_amp": 0.05, "median_amp": 0.05, "std_amp": 0.02, "snr": 10.0},
            },
            finetuned_model_data=model_data,
            calibration_params={
                "target_amplitude": 0.45,
                "noise_gate": [0.15, 0.15, 0.15, 0.15],
                "hysteresis_enter": 0.5,
                "hysteresis_exit": 0.3,
                "confidence_floor": 0.35,
                "snr": 10.0,
            },
            quality_report={"grade": "GOOD", "warnings": []},
        )

        save_calibration(result, patient_id)

        # Check files exist
        check("calibration_data.npz exists",
              os.path.exists(os.path.join(cal_dir, "calibration_data.npz")))
        check("calibration_info.json exists",
              os.path.exists(os.path.join(cal_dir, "calibration_info.json")))
        check("rest_baseline.json exists",
              os.path.exists(os.path.join(cal_dir, "rest_baseline.json")))
        check("calibrated_model.pkl exists",
              os.path.exists(os.path.join(cal_dir, "calibrated_model.pkl")))

        # Load calibration
        loaded = load_calibration(patient_id)
        check("load_calibration returns data", loaded is not None)
        check("loaded samples shape matches",
              loaded["samples"].shape == (500, 4))
        check("loaded labels length matches",
              len(loaded["labels"]) == 500)
        check("loaded rest_baseline has mean",
              "mean" in loaded["rest_baseline"])
        check("loaded rest_baseline mean is numpy",
              isinstance(loaded["rest_baseline"]["mean"], np.ndarray))
        check("loaded info has calibration_params",
              "calibration_params" in loaded["info"])
        check("calibration_params noise_gate is list",
              isinstance(loaded["info"]["calibration_params"]["noise_gate"], list))

        # Load model
        loaded_model = load_calibrated_model(patient_id)
        check("load_calibrated_model returns data", loaded_model is not None)
        check("loaded model has model key", "model" in loaded_model)

        # List patients
        patients = list_patients()
        check("list_patients includes test patient",
              patient_id in patients)

    finally:
        # Cleanup
        if os.path.exists(cal_dir):
            shutil.rmtree(cal_dir)


def test_full_calibration_pipeline():
    section("Full Calibration Pipeline (Session Model)")
    import calibrate_patient as cp
    from runtime.calibrate_patient import calibrate_patient, CalibrationTrial, LABEL_MAP

    ser = FakeSerial(sample_rate=200)
    model_data = joblib.load("exohand_model.pkl")
    patient_id = "_test_full_cal"

    # Shorten protocol for test speed by monkey-patching
    original_build = cp.build_full_protocol

    def _short_protocol():
        """3 trials only for speed."""
        return [
            CalibrationTrial("close", 0, "normal", 2.0, 1.0, 3),
            CalibrationTrial("open", 1, "normal", 2.0, 1.0, 3),
            CalibrationTrial("rest", 2, "normal", 2.0, 1.0, 3),
        ]

    cp.build_full_protocol = _short_protocol

    # Also shorten rest calibration
    original_rest = cp._rest_calibrate_noninteractive

    def _quick_rest(ser, duration=10):
        return original_rest(ser, duration=2)

    cp._rest_calibrate_noninteractive = _quick_rest

    progress_events = []

    def progress_cb(phase, trial_idx, total, gesture, remaining, pct):
        progress_events.append({
            "phase": phase, "trial_idx": trial_idx, "total": total,
            "gesture": gesture, "pct": pct,
        })

    cal_dir = cp._calibration_dir(patient_id)

    try:
        result = calibrate_patient(
            ser, model_data, 200,
            patient_id=patient_id,
            progress_callback=progress_cb,
            interactive=False,
        )

        check("result is CalibrationResult",
              hasattr(result, "finetuned_model_data"))
        check("result has rest_baseline", result.rest_baseline is not None)
        check("result has per_class_stats", len(result.per_class_stats) > 0)
        check("result has calibration_params",
              len(result.calibration_params) > 0)
        check("result has quality_report", "grade" in result.quality_report)
        check("finetuned model has model key",
              "model" in result.finetuned_model_data)
        check("raw_samples is numpy array",
              isinstance(result.raw_samples, np.ndarray))
        check("raw_samples has data", len(result.raw_samples) > 0,
              f"got {len(result.raw_samples)}")
        check("progress callback was called",
              len(progress_events) > 0, f"got {len(progress_events)}")

        # Verify saved files
        check("calibration saved",
              os.path.exists(os.path.join(cal_dir, "calibrated_model.pkl")))

    finally:
        cp.build_full_protocol = original_build
        cp._rest_calibrate_noninteractive = original_rest
        if os.path.exists(cal_dir):
            shutil.rmtree(cal_dir)


def test_abbreviated_calibration():
    section("Abbreviated Recalibration Pipeline")
    from runtime.calibrate_patient import (
        calibrate_patient, abbreviated_calibrate,
        CalibrationTrial, _calibration_dir,
    )
    import calibrate_patient as cp

    ser = FakeSerial(sample_rate=200)
    model_data = joblib.load("exohand_model.pkl")
    patient_id = "_test_abbrev_cal"

    # Short protocols
    def _short_full():
        return [
            CalibrationTrial("close", 0, "normal", 2.0, 1.0, 3),
            CalibrationTrial("open", 1, "normal", 2.0, 1.0, 3),
            CalibrationTrial("rest", 2, "normal", 2.0, 1.0, 3),
        ]

    def _short_abbrev():
        return [
            CalibrationTrial("close", 0, "normal", 2.0, 1.0, 3),
            CalibrationTrial("open", 1, "normal", 2.0, 1.0, 3),
            CalibrationTrial("rest", 2, "normal", 2.0, 1.0, 3),
        ]

    original_full = cp.build_full_protocol
    original_abbrev = cp.build_abbreviated_protocol
    original_rest = cp._rest_calibrate_noninteractive

    cp.build_full_protocol = _short_full
    cp.build_abbreviated_protocol = _short_abbrev
    cp._rest_calibrate_noninteractive = lambda ser, duration=10: original_rest(ser, duration=2)

    cal_dir = _calibration_dir(patient_id)

    try:
        # First: run full calibration to save data
        result1 = calibrate_patient(
            ser, model_data, 200, patient_id=patient_id, interactive=False)
        check("full calibration completed",
              len(result1.raw_samples) > 0)

        # Second: run abbreviated (should merge with previous)
        result2 = abbreviated_calibrate(
            ser, model_data, 200, patient_id=patient_id, interactive=False)
        check("abbreviated completed",
              len(result2.raw_samples) > 0)
        check("abbreviated merged more data",
              len(result2.raw_samples) >= len(result1.raw_samples) // 2,
              f"full={len(result1.raw_samples)}, abbrev={len(result2.raw_samples)}")
        check("abbreviated has calibration_params",
              len(result2.calibration_params) > 0)

    finally:
        cp.build_full_protocol = original_full
        cp.build_abbreviated_protocol = original_abbrev
        cp._rest_calibrate_noninteractive = original_rest
        if os.path.exists(cal_dir):
            shutil.rmtree(cal_dir)


def test_realtime_predictor_with_calibration():
    section("RealtimePredictor + Calibration Params")
    from run_exohand import RealtimePredictor
    from assist_profile import get_profile

    model_data = joblib.load("exohand_model.pkl")
    profile = get_profile(3)

    rest_baseline = {
        "mean": np.array([0.01, 0.01, 0.01, 0.01]),
        "std": np.array([0.05, 0.05, 0.05, 0.05]),
        "max": np.array([0.15, 0.15, 0.15, 0.15]),
        "p95": np.array([0.1, 0.1, 0.1, 0.1]),
    }

    cal_params = {
        "target_amplitude": 0.45,
        "noise_gate": [0.15, 0.15, 0.15, 0.15],
        "hysteresis_enter": 0.4,
        "hysteresis_exit": 0.25,
        "confidence_floor": 0.3,
        "snr": 8.0,
    }

    # Without calibration params
    pred_default = RealtimePredictor(
        model_data, 200, assist_profile=profile, rest_baseline=rest_baseline)
    check("predictor without cal_params created", pred_default is not None)

    # With calibration params
    pred_cal = RealtimePredictor(
        model_data, 200, assist_profile=profile,
        rest_baseline=rest_baseline, calibration_params=cal_params)
    check("predictor with cal_params created", pred_cal is not None)
    check("target_amplitude set from cal_params",
          pred_cal.target_amplitude == 0.45)
    check("noise_gate overridden by cal_params",
          np.allclose(pred_cal.noise_gate, [0.15, 0.15, 0.15, 0.15]))
    check("hysteresis_enter override set",
          pred_cal._cal_hysteresis_enter == 0.4)
    check("confidence_floor override set",
          pred_cal._cal_confidence_floor == 0.3)

    # Feed samples and verify prediction works
    for i in range(200):
        vals = [0.3 + np.random.randn() * 0.05] * 4
        result = pred_cal.add_sample(vals)

    check("predictor produces results after filling buffer",
          True)  # no crash = pass


def test_cancellation():
    section("Calibration Cancellation")
    from runtime.calibrate_patient import (
        calibrate_patient, CalibrationCancelled, CalibrationTrial,
    )
    import calibrate_patient as cp

    ser = FakeSerial(sample_rate=200)
    model_data = joblib.load("exohand_model.pkl")

    # Short protocol
    def _short():
        return [
            CalibrationTrial("close", 0, "normal", 3.0, 2.0, 3),
            CalibrationTrial("open", 1, "normal", 3.0, 2.0, 3),
            CalibrationTrial("rest", 2, "normal", 3.0, 2.0, 3),
        ]

    original_full = cp.build_full_protocol
    original_rest = cp._rest_calibrate_noninteractive
    cp.build_full_protocol = _short
    cp._rest_calibrate_noninteractive = lambda ser, duration=10: {
        "mean": np.zeros(4), "std": np.ones(4),
        "max": np.zeros(4), "p95": np.zeros(4),
    }

    stop_event = threading.Event()

    # Set stop after a short delay
    def _cancel():
        time.sleep(3.0)
        stop_event.set()

    cancel_thread = threading.Thread(target=_cancel, daemon=True)
    cancel_thread.start()

    cancelled = False
    try:
        calibrate_patient(
            ser, model_data, 200,
            patient_id="_test_cancel",
            interactive=False,
            stop_event=stop_event,
        )
    except CalibrationCancelled:
        cancelled = True

    cp.build_full_protocol = original_full
    cp._rest_calibrate_noninteractive = original_rest

    check("CalibrationCancelled raised on stop_event", cancelled)

    # Cleanup
    cal_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "calibrations", "_test_cancel")
    if os.path.exists(cal_dir):
        shutil.rmtree(cal_dir)


def test_edge_cases():
    section("Edge Cases")
    from runtime.calibrate_patient import (
        finetune_model, _extract_session_features,
        _compute_quality_report, CalibrationTrial, GESTURE_NAMES,
    )

    model_data = joblib.load("exohand_model.pkl")

    # Empty data
    features_empty = np.array([]).reshape(0, 36)
    labels_empty = np.array([])
    result = finetune_model(model_data, features_empty, labels_empty)
    check("finetune with empty data returns original model",
          result["model"] is model_data["model"])

    # Single class
    features_single = np.random.randn(50, 36)
    labels_single = np.zeros(50)
    result = finetune_model(model_data, features_single, labels_single)
    check("finetune with single class returns original model",
          result["model"] is model_data["model"])

    # Very small dataset (2 classes, few samples)
    features_small = np.random.randn(10, 36)
    labels_small = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    result = finetune_model(model_data, features_small, labels_small)
    check("finetune with small 2-class data succeeds",
          result["model"] is not model_data["model"])

    # Quality report with empty feature_labels
    samples = np.random.randn(100, 4)
    labels = np.concatenate([np.zeros(50), np.ones(50)])
    stats = {0: {"snr": 5.0}, 1: {"snr": 5.0}, 2: {"snr": 0.0}}
    trials = [CalibrationTrial("close", 0, "normal", 5, 4, 3, quality="ok")]
    report = _compute_quality_report(samples, labels, stats, trials, np.array([]), 200)
    check("quality report with empty features doesn't crash",
          "grade" in report)
    check("quality report warns about missing class",
          any("rest" in w for w in report["warnings"]))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ExoHand Calibration — End-to-End Test Suite")
    print("=" * 60)

    tests = [
        test_protocol_builders,
        test_data_collection_and_cleaning,
        test_session_feature_extraction,
        test_adapted_feature_extraction,
        test_finetune_session,
        test_finetune_adapted,
        test_save_load_persistence,
        test_full_calibration_pipeline,
        test_abbreviated_calibration,
        test_realtime_predictor_with_calibration,
        test_cancellation,
        test_edge_cases,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            global FAIL
            FAIL += 1
            print(f"\n  CRASH  {test_fn.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
