#!/usr/bin/env python3
"""
evaluate_calibrations.py — compare 22-second vs 6-minute calibration.

Runs both calibrations on the same participant in one sitting, then a
fresh held-out test that neither model has seen, then scores both models
side by side.

Usage:
    python -m runtime.evaluate_calibrations
    python -m runtime.evaluate_calibrations --port COM5

Resume after a crash:
    python -m runtime.evaluate_calibrations --skip-22s
    python -m runtime.evaluate_calibrations --skip-22s --skip-full
    python -m runtime.evaluate_calibrations --skip-22s --skip-full --skip-holdout
        (last one just re-scores from saved data)

Everything runs on the deployed calibration pipeline (feature extraction,
finetuning, save format) — only the cueing is different, so results
reflect what a real user would get.
"""

import os
import sys

# Put project root and runtime/ on sys.path so the deployed pipeline
# imports work either way this script is invoked.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json
import platform
import subprocess
import time
from typing import Optional

import joblib
import numpy as np
import serial
import serial.tools.list_ports

# Deployed calibration pipeline — used verbatim
import calibrate_patient as cp
from calibrate_patient import (
    CalibrationResult,
    LABEL_MAP,
    GESTURE_NAMES,
    GRABMYO_TO_SESSION_LABEL,
    _collect_emg_segment,
    _extract_calibration_features,
    _reject_outliers,
    _trim_onset,
    apply_calibration,
    build_full_protocol,
    build_paper22s_protocol,
    compute_per_class_stats,
    finetune_model,
    load_calibration,
    load_calibrated_model,
    save_calibration,
)
from run_exohand import parse_emg_line
from assist_profile import get_profile


IS_MAC = platform.system() == "Darwin"
IS_WIN = platform.system() == "Windows"

# Patient IDs used for saved eval artifacts. Living under runtime/calibrations/.
EVAL_22S_ID = "eval_22s"
EVAL_FULL_ID = "eval_full"
EVAL_HOLDOUT_ID = "eval_holdout"

# Held-out test protocol: 3 reps of [close, open, rest], each held for 3s.
# 3s hold is deliberately different from 22s cal (1.8s) and full cal (5s).
HOLDOUT_HOLD_S = 3.0
HOLDOUT_REST_S = 2.0
HOLDOUT_REPS = 3

# Prep seconds shown before every trial. 6s matches the deployed web UI
# (3s instruction + 3s countdown).
DEFAULT_PREP_SECONDS = 6

# Default baseline model — the same one the deployed system loads for
# healthy users.
DEFAULT_BASELINE = os.path.join(_PROJECT_ROOT, "exohand_adapted_model.pkl")

# Results output
RESULTS_JSON = os.path.join(_PROJECT_ROOT, "runtime", "eval_results.json")


# ─────────────────────────────────────────────────────────────────────
# Terminal utilities
# ─────────────────────────────────────────────────────────────────────

BAR = "=" * 72
BARL = "-" * 72


def banner(text):
    print()
    print(BAR)
    print(f"  {text}")
    print(BAR, flush=True)


def sub(text):
    print(f"\n{BARL}")
    print(f"  {text}")
    print(BARL, flush=True)


def bell():
    """Terminal bell (works on macOS Terminal, most Windows terminals)."""
    print("\a", end="", flush=True)


# ─────────────────────────────────────────────────────────────────────
# Cross-platform TTS
# ─────────────────────────────────────────────────────────────────────

def speak(text, blocking=False):
    """Say `text` out loud. Non-blocking by default so we can print at
    the same time. Falls back silently if no TTS engine is available.
    """
    try:
        if IS_MAC:
            cmd = ["say", text]
        elif IS_WIN:
            safe = text.replace("'", "''")
            ps = (
                "Add-Type -AssemblyName System.Speech; "
                f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')"
            )
            cmd = ["powershell", "-NoProfile", "-Command", ps]
        else:
            cmd = ["espeak", text]

        if blocking:
            subprocess.run(cmd, check=False, capture_output=True, timeout=15)
        else:
            subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    except Exception:
        pass  # silent fallback


def wait_enter(prompt):
    """Blocking prompt with speech."""
    speak(prompt, blocking=False)
    try:
        input(f"\n  [Press ENTER to continue]  ")
    except EOFError:
        # Non-interactive; just continue
        pass


# ─────────────────────────────────────────────────────────────────────
# Gesture cueing
# ─────────────────────────────────────────────────────────────────────

_GESTURE_CUE = {
    "close": "close your hand",
    "open":  "open your hand",
    "rest":  "relax",
}

_EFFORT_PREFIX = {
    "light": "gently ",
    "hard":  "strongly ",
    "pulse": "quickly ",
}


def cue_gesture(gesture, trial_idx, total, prep_seconds, effort="normal"):
    """Show instruction + countdown before a hold.

    Layout: half of prep_seconds is instruction display, other half is
    a spoken 3-2-1 countdown. Matches the deployed web UI timing.
    The effort modifier (gently/strongly/quickly) is spoken and displayed
    BEFORE the countdown, so the participant knows how to modulate the
    hold when GO fires.
    """
    prefix = _EFFORT_PREFIX.get(effort, "")
    spoken = (prefix + _GESTURE_CUE[gesture]).strip().capitalize()
    display = spoken

    banner(f">>  TRIAL {trial_idx + 1} / {total}   |   {display.upper()}")

    # Speak the instruction, print big
    speak(spoken, blocking=False)
    print(f"\n     {display}", flush=True)

    # Half of prep is instruction display (no countdown yet)
    inst_time = max(1.0, prep_seconds / 2.0)
    time.sleep(inst_time)

    # Remaining prep = countdown
    countdown_from = min(3, max(1, int(prep_seconds - inst_time)))
    print()
    for i in range(countdown_from, 0, -1):
        print(f"       {i}", flush=True)
        speak(str(i), blocking=False)
        bell()
        time.sleep(1.0)

    print(f"     >>>  GO — {gesture_display}  <<<", flush=True)
    bell()


def cue_rest_short():
    print("     relax", flush=True)


# ─────────────────────────────────────────────────────────────────────
# Serial port discovery
# ─────────────────────────────────────────────────────────────────────

def autodetect_port() -> Optional[str]:
    ports = list(serial.tools.list_ports.comports())
    if IS_MAC:
        candidates = [p for p in ports if "usbmodem" in p.device.lower()]
    elif IS_WIN:
        candidates = [p for p in ports if p.device.upper().startswith("COM")]
    else:
        candidates = [
            p for p in ports if "ttyACM" in p.device or "ttyUSB" in p.device
        ]
    if len(candidates) == 1:
        return candidates[0].device
    return None


def resolve_port(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    detected = autodetect_port()
    if detected:
        print(f"  Auto-detected serial port: {detected}")
        return detected
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise SystemExit(
            "\nERROR: No serial ports found.\n"
            "  Is the Teensy plugged in and powered?\n"
            "  On macOS check /dev/cu.usbmodem*, on Windows check Device Manager."
        )
    print("\nMultiple serial ports found. Pass --port to pick one:")
    for p in ports:
        print(f"  {p.device}    {p.description}")
    raise SystemExit("\nRerun with --port <name>")


def measure_sample_rate(ser, duration_s=2.0, min_hz=10) -> int:
    """Measure incoming EMG stream rate from the Teensy, then apply the
    same 50 Hz floor the deployed calibration pipeline uses (see
    calibrate_patient.py line 1531). The Teensy sends a 20 Hz peak-to-peak
    envelope by design, so raw counts of ~20 are healthy; the floor keeps
    feature window sizes aligned with what the models were trained on.
    """
    ser.reset_input_buffer()
    count = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        if parse_emg_line(line) is not None:
            count += 1
    measured = int(round(count / duration_s))
    if measured < min_hz:
        raise SystemExit(
            f"\nERROR: EMG stream rate too low ({measured} Hz). "
            f"Expected at least {min_hz} Hz from the Teensy.\n"
            "  Is the sleeve on and pressed against skin?\n"
            "  Is the Teensy streaming EMG (not idle)?"
        )
    # Match deployed pipeline's max(count, 50) floor. The Teensy's actual
    # 20 Hz envelope becomes an effective 50 Hz for feature-window sizing
    # inside _extract_calibration_features — same convention the shipping
    # 22s cal uses, so the two calibrations we compare match what a real
    # patient would get.
    effective = max(measured, 50)
    if effective != measured:
        print(f"  Measured stream rate: {measured} Hz "
              f"(deployed pipeline floors this to {effective} Hz internally)")
    return effective


# ─────────────────────────────────────────────────────────────────────
# Rest baseline
# ─────────────────────────────────────────────────────────────────────

def collect_rest_baseline(ser, duration=10):
    banner(f"REST BASELINE  ({duration}s)")
    print(
        "  Sit still and relax your arm completely.\n"
        "  Do not tense any muscles."
    )
    speak(
        f"Rest baseline. Sit still and relax completely for {duration} seconds.",
        blocking=False,
    )
    time.sleep(1.5)
    bell()
    print("  Recording ...", flush=True)

    ser.reset_input_buffer()
    samples = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        vals = parse_emg_line(line)
        if vals is not None:
            samples.append(vals)

    if len(samples) < 20:
        raise SystemExit(
            f"\nERROR: Only {len(samples)} rest-baseline samples collected.\n"
            "  Sleeve may be disconnected. Check contact and rerun."
        )

    data = np.array(samples)
    baseline = {
        "mean": data.mean(axis=0),
        "std": data.std(axis=0),
        "max": np.abs(data).max(axis=0),
        "p95": np.percentile(np.abs(data), 95, axis=0),
    }
    print(f"  {len(samples)} samples collected.")
    speak("Baseline done.", blocking=False)
    return baseline


# ─────────────────────────────────────────────────────────────────────
# Eval-specific trial loop
# ─────────────────────────────────────────────────────────────────────

def _phase_name(phase):
    return {
        1: "Rest baseline",
        2: "Familiarization",
        3: "Sustained holds",
        4: "Quick contractions",
        5: "Variable effort",
    }.get(phase, "")


def _run_eval_trials(ser, trials, sample_rate, prep_seconds):
    """Trial loop with clear terminal + TTS cues.

    Mirrors calibrate_patient._run_trials in every way that affects the
    saved model — same per-trial data-collection duration, same onset
    trim, same outlier rejection. Only the cueing is different.
    """
    all_samples = []
    all_labels = []
    current_phase = None
    total = len(trials)

    for idx, trial in enumerate(trials):
        # Phase-change announce (full-cal only; paper22s is single-phase)
        if trial.phase != current_phase:
            current_phase = trial.phase
            phase_name = _phase_name(current_phase)
            if phase_name and current_phase > 1:
                sub(f"Phase {current_phase}: {phase_name}")
                speak(f"Phase {current_phase}: {phase_name}", blocking=False)
                time.sleep(1.8)

        # Pulse trials get shorter prep; hold trials get full prep.
        this_prep = 2.0 if trial.effort == "pulse" else prep_seconds
        cue_gesture(trial.gesture, idx, total,
                    prep_seconds=this_prep, effort=trial.effort)

        # Hold — collect data
        samples = _collect_emg_segment(ser, trial.duration, trial.label)

        # Deployed cleaning: onset trim then outlier reject
        samples = _trim_onset(samples, sample_rate, trial)
        samples, n_rejected = _reject_outliers(samples)

        if len(samples) < 5:
            print(f"     WARNING: only {len(samples)} usable samples in this trial", flush=True)
        else:
            all_samples.extend([s[0] for s in samples])
            all_labels.extend([s[1] for s in samples])

        rej_msg = f"  (-{n_rejected} outliers)" if n_rejected > 0 else ""
        print(f"     {len(samples)} samples kept{rej_msg}", flush=True)

        # Rest period between trials
        if trial.rest_period > 0:
            if trial.rest_period >= 10:
                speak("Take a break. Relax.", blocking=False)
                print(f"     [ break: {trial.rest_period:.0f}s ]", flush=True)
            else:
                cue_rest_short()
                speak("relax", blocking=False)
            rest = _collect_emg_segment(ser, trial.rest_period, LABEL_MAP["rest"])
            rest, _ = _reject_outliers(rest)
            all_samples.extend([s[0] for s in rest])
            all_labels.extend([s[1] for s in rest])

    return np.array(all_samples), np.array(all_labels)


# ─────────────────────────────────────────────────────────────────────
# Full calibration + save via deployed pipeline
# ─────────────────────────────────────────────────────────────────────

def run_and_save_calibration(
    ser,
    protocol_builder,
    display_label,
    saved_calibration_type,
    baseline_model,
    sample_rate,
    rest_baseline,
    patient_id,
    prep_seconds,
):
    """Run a calibration protocol end-to-end, using the deployed pipeline
    for feature extraction, finetuning, and persistence.

    display_label: human-friendly name shown in banner and spoken aloud.
    saved_calibration_type: the `calibration_type` field on the saved
        CalibrationResult (matches deployment's convention: "abbreviated"
        for the 22s protocol, "full" for the 6-min protocol).
    """
    trials = protocol_builder()
    n_trials = len(trials)
    est_data_s = sum(t.duration + t.rest_period for t in trials)
    banner(
        f"CALIBRATION — {display_label}  "
        f"({n_trials} trials, ~{int(est_data_s)}s of data + prep time)"
    )
    speak(f"Starting {display_label}.", blocking=False)
    time.sleep(1.5)

    raw_samples, raw_labels = _run_eval_trials(
        ser, trials, sample_rate, prep_seconds=prep_seconds
    )

    if len(raw_samples) == 0:
        raise SystemExit(
            f"ERROR: no samples collected during {calibration_type_label} calibration."
        )

    # Deployed pipeline from here on
    print(f"\n  Total samples collected: {len(raw_samples)}")
    for lbl in [0, 1, 2]:
        n = int(np.sum(raw_labels == lbl))
        print(f"    {GESTURE_NAMES[lbl]}: {n}")

    per_class_stats = compute_per_class_stats(raw_samples, raw_labels)

    # Load a fresh baseline for this cal (finetune mutates, so both cals
    # must start from an unmutated baseline)
    model_data = joblib.load(baseline_model)

    print("\n  Extracting features ...")
    features, feature_labels = _extract_calibration_features(
        raw_samples, raw_labels, sample_rate, model_data
    )
    print(f"  {len(features)} feature windows")

    if len(features) == 0:
        raise SystemExit(f"ERROR: no feature windows from {calibration_type_label} data.")

    print("\n  Fine-tuning model ...")
    finetuned = finetune_model(model_data, features, feature_labels)

    assist_profile = get_profile(3)
    cal_params = apply_calibration(rest_baseline, per_class_stats, assist_profile)

    result = CalibrationResult(
        patient_id=patient_id,
        timestamp=time.time(),
        calibration_type=saved_calibration_type,
        rest_baseline=rest_baseline,
        trials=trials,
        raw_samples=raw_samples,
        raw_labels=raw_labels,
        sample_rate=sample_rate,
        per_class_stats=per_class_stats,
        finetuned_model_data=finetuned,
        calibration_params=cal_params,
        quality_report={},
    )
    save_calibration(result, patient_id)
    speak(f"{display_label} complete.", blocking=False)
    return finetuned


# ─────────────────────────────────────────────────────────────────────
# Held-out test
# ─────────────────────────────────────────────────────────────────────

def _build_holdout_protocol():
    """Build a held-out test protocol. Deliberately uses different hold
    durations from either cal so neither model has seen this exact rhythm.
    """
    from calibrate_patient import CalibrationTrial

    trials = []
    for _ in range(HOLDOUT_REPS):
        for gesture in ["close", "open", "rest"]:
            trials.append(CalibrationTrial(
                gesture=gesture,
                label=LABEL_MAP[gesture],
                effort="normal",
                duration=HOLDOUT_HOLD_S,
                rest_period=HOLDOUT_REST_S,
                phase=3,
            ))
    return trials


def run_holdout_test(ser, sample_rate, rest_baseline, prep_seconds):
    banner(
        f"HELD-OUT TEST  "
        f"({HOLDOUT_REPS} reps x 3 gestures x {HOLDOUT_HOLD_S:.0f}s hold)"
    )
    print(
        "  Fresh gestures for scoring both models. Neither model has\n"
        "  seen this data — this is the honest accuracy number.\n"
    )
    speak(
        "Held out test. Follow the voice cues, same as before.",
        blocking=False,
    )
    time.sleep(2.0)

    trials = _build_holdout_protocol()
    raw_samples, raw_labels = _run_eval_trials(
        ser, trials, sample_rate, prep_seconds=prep_seconds
    )

    if len(raw_samples) == 0:
        raise SystemExit("ERROR: no samples collected during held-out test.")

    # Save the held-out recording so re-scoring can skip the collection
    # step. Use the deployed save_calibration to keep the format consistent.
    per_class_stats = compute_per_class_stats(raw_samples, raw_labels)
    holdout_result = CalibrationResult(
        patient_id=EVAL_HOLDOUT_ID,
        timestamp=time.time(),
        calibration_type="holdout",
        rest_baseline=rest_baseline,
        trials=trials,
        raw_samples=raw_samples,
        raw_labels=raw_labels,
        sample_rate=sample_rate,
        per_class_stats=per_class_stats,
        # Scoring doesn't need the finetuned model — store a stub
        finetuned_model_data={"model_type": "holdout_stub"},
        calibration_params={},
        quality_report={},
    )
    save_calibration(holdout_result, EVAL_HOLDOUT_ID)
    speak("Held out test complete.", blocking=False)

    return raw_samples, raw_labels


# ─────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────

def score_model(model_data, raw_samples, raw_labels, sample_rate, label):
    """Score one saved model on held-out raw EMG.

    Uses the model's own feature-extraction path (via
    _extract_calibration_features which dispatches by model_type), then
    predicts, then remaps to session-order labels for display.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix,
    )

    features, feat_labels = _extract_calibration_features(
        raw_samples, raw_labels, sample_rate, model_data
    )
    if len(features) == 0:
        return {"error": f"{label}: no feature windows extracted"}

    model = model_data["model"]
    scaler = model_data.get("scaler")

    if scaler is not None:
        X = scaler.transform(features)
    else:
        X = features

    preds = model.predict(X)

    model_type = model_data.get("model_type", "session")

    # Both preds and feat_labels are in the model's native label order.
    # For session models that's session order; for adapted_hgb it's
    # GrabMyo order. Remap both to session order for display.
    if model_type == "adapted_hgb":
        preds_s = np.array([GRABMYO_TO_SESSION_LABEL[int(p)] for p in preds])
        labels_s = np.array([GRABMYO_TO_SESSION_LABEL[int(l)] for l in feat_labels])
    else:
        preds_s = preds.astype(int)
        labels_s = feat_labels.astype(int)

    classes = [0, 1, 2]  # close, open, rest in session order
    acc = accuracy_score(labels_s, preds_s)
    f1_each = f1_score(labels_s, preds_s, labels=classes,
                       average=None, zero_division=0)
    macro_f1 = f1_score(labels_s, preds_s, labels=classes,
                        average="macro", zero_division=0)
    cm = confusion_matrix(labels_s, preds_s, labels=classes)

    # Balanced accuracy = mean of per-class recall — class-count-independent.
    # Useful because held-out has more rest windows than close/open.
    per_class_recall = []
    for c in classes:
        mask = labels_s == c
        if mask.sum() > 0:
            per_class_recall.append((preds_s[mask] == c).mean())
    balanced_acc = float(np.mean(per_class_recall)) if per_class_recall else 0.0

    return {
        "label": label,
        "model_type": model_type,
        "n_windows": int(len(features)),
        "windows_per_class": {
            GESTURE_NAMES[i]: int(np.sum(labels_s == i)) for i in classes
        },
        "accuracy": float(acc),
        "balanced_accuracy": balanced_acc,
        "macro_f1": float(macro_f1),
        "f1_close": float(f1_each[0]),
        "f1_open": float(f1_each[1]),
        "f1_rest": float(f1_each[2]),
        "confusion_matrix": cm.tolist(),
    }


def print_comparison(r22, rfull):
    banner("EVALUATION RESULTS  —  HELD-OUT TEST")
    if "error" in r22 or "error" in rfull:
        print("  ERROR during scoring:")
        for name, r in [("22s", r22), ("full", rfull)]:
            if "error" in r:
                print(f"    {name}: {r['error']}")
        return

    def pp(x, y):
        return f"{(y - x) * 100:+.1f} pp"

    def d(x, y):
        return f"{y - x:+.3f}"

    print()
    print(f"  Windows scored:  {r22['n_windows']} (22s cal), "
          f"{rfull['n_windows']} (full cal)")
    wpc = r22["windows_per_class"]
    print(f"  True class counts:  close={wpc['close']}  "
          f"open={wpc['open']}  rest={wpc['rest']}")

    print()
    print(f"                       {'22s cal':>10}   {'Full 6-min':>12}   {'Delta':>10}")
    print(f"  Accuracy         {r22['accuracy']*100:>10.1f}%   "
          f"{rfull['accuracy']*100:>10.1f}%   "
          f"{pp(r22['accuracy'], rfull['accuracy']):>10}")
    print(f"  Balanced acc.    {r22['balanced_accuracy']*100:>10.1f}%   "
          f"{rfull['balanced_accuracy']*100:>10.1f}%   "
          f"{pp(r22['balanced_accuracy'], rfull['balanced_accuracy']):>10}")
    print(f"  Macro F1         {r22['macro_f1']:>11.3f}   "
          f"{rfull['macro_f1']:>12.3f}   "
          f"{d(r22['macro_f1'], rfull['macro_f1']):>10}")
    print(f"  Close F1         {r22['f1_close']:>11.3f}   "
          f"{rfull['f1_close']:>12.3f}   "
          f"{d(r22['f1_close'], rfull['f1_close']):>10}")
    print(f"  Open F1          {r22['f1_open']:>11.3f}   "
          f"{rfull['f1_open']:>12.3f}   "
          f"{d(r22['f1_open'], rfull['f1_open']):>10}")
    print(f"  Rest F1          {r22['f1_rest']:>11.3f}   "
          f"{rfull['f1_rest']:>12.3f}   "
          f"{d(r22['f1_rest'], rfull['f1_rest']):>10}")

    print()
    print("  Confusion matrices  (rows = true label, cols = predicted):")
    print("    Classes in both matrices: close, open, rest")
    print()
    print("               22s cal                        Full 6-min cal")
    print("             close  open  rest              close  open  rest")
    class_names = ["close", "open ", "rest "]
    for i, name in enumerate(class_names):
        c22 = r22["confusion_matrix"][i]
        cf = rfull["confusion_matrix"][i]
        print(f"    {name}   {c22[0]:>5} {c22[1]:>5} {c22[2]:>5}    "
              f"       {name}   {cf[0]:>5} {cf[1]:>5} {cf[2]:>5}")

    # Use balanced accuracy for the verdict since holdout rest is oversampled
    delta = (rfull["balanced_accuracy"] - r22["balanced_accuracy"]) * 100
    print()
    print("  Note: balanced accuracy is the fair headline (class-count-independent).")
    if abs(delta) <= 3.0:
        verdict = f"22s essentially matches full ({delta:+.1f} pp balanced). Post the video."
    elif delta > 3.0:
        verdict = (f"22s trails full by {delta:.1f} pp (balanced). "
                   "Show UX in video, do not cite accuracy comparison.")
    else:
        verdict = (f"22s beats full by {-delta:.1f} pp (balanced) on this run. "
                   "Suspicious — rerun before citing.")
    print(f"  Verdict: {verdict}")


def save_results_json(r22, rfull, sample_rate, args):
    payload = {
        "timestamp": time.time(),
        "sample_rate_hz": sample_rate,
        "baseline_model": args.baseline,
        "prep_seconds": args.prep_seconds,
        "twenty_two_second": r22,
        "full_six_minute": rfull,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results saved to {RESULTS_JSON}")


# ─────────────────────────────────────────────────────────────────────
# Rest breaks
# ─────────────────────────────────────────────────────────────────────

def rest_break(seconds, reason):
    banner(f"REST BREAK  ({seconds}s)")
    print(f"  {reason}")
    print("  Keep the sleeve on. Sit still.")
    speak(
        f"Take a {seconds} second break. Do not remove the sleeve.",
        blocking=False,
    )
    end = time.perf_counter() + seconds
    while True:
        remaining = int(round(end - time.perf_counter()))
        if remaining <= 0:
            break
        print(f"    {remaining}s remaining ...", flush=True)
        # Sleep in 5s chunks so counter updates are visible
        time.sleep(min(5.0, remaining))
    speak("Break's over.", blocking=False)


# ─────────────────────────────────────────────────────────────────────
# Load held-out data on resume
# ─────────────────────────────────────────────────────────────────────

def load_holdout():
    cal = load_calibration(EVAL_HOLDOUT_ID)
    if cal is None:
        raise SystemExit(
            "ERROR: --skip-holdout set, but no saved held-out data at "
            f"runtime/calibrations/{EVAL_HOLDOUT_ID}/"
        )
    return cal["samples"], cal["labels"], cal["sample_rate"], cal["rest_baseline"]


def clear_eval_dir(patient_id):
    """Delete an eval calibration directory so a fresh run doesn't mix
    with stale data from a previous run.
    """
    import shutil
    d = os.path.join(_PROJECT_ROOT, "runtime", "calibrations", patient_id)
    if os.path.exists(d):
        shutil.rmtree(d)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Compare 22s vs 6-min calibration on the same participant."
    )
    ap.add_argument("--port", type=str, default=None,
                    help="Serial port for the Teensy (autodetect if omitted).")
    ap.add_argument("--baseline", type=str, default=DEFAULT_BASELINE,
                    help=f"Baseline model .pkl to fine-tune from "
                         f"(default: {DEFAULT_BASELINE}).")
    ap.add_argument("--prep-seconds", type=int, default=DEFAULT_PREP_SECONDS,
                    help=f"Seconds of instruction+countdown before each hold "
                         f"(default: {DEFAULT_PREP_SECONDS}, matches deployed UI).")
    ap.add_argument("--skip-22s", action="store_true",
                    help="Use existing eval_22s calibration instead of running it.")
    ap.add_argument("--skip-full", action="store_true",
                    help="Use existing eval_full calibration instead of running it.")
    ap.add_argument("--skip-holdout", action="store_true",
                    help="Use existing held-out recording instead of running it.")
    args = ap.parse_args()

    banner("EXOHAND CALIBRATION EVALUATION")
    print(
        "  This will run:\n"
        "    1. Rest baseline           (~10s)\n"
        "    2. 22-second calibration   (~1.5 min wall-clock)\n"
        "    3. Rest break              (30s)\n"
        "    4. Full 6-min calibration  (~13 min wall-clock)\n"
        "    5. Rest break              (30s)\n"
        "    6. Held-out test           (~1.5 min wall-clock)\n"
        "    7. Score both models       (few seconds)\n"
        "\n"
        "  Total time: about 16 minutes.\n"
        "  Wear the EMG sleeve for the whole thing. Glove off.\n"
    )

    # Verify baseline model
    if not os.path.exists(args.baseline):
        raise SystemExit(
            f"\nERROR: baseline model not found at:\n  {args.baseline}\n"
            "  Pass --baseline <path> if it lives somewhere else."
        )

    if not (args.skip_22s and args.skip_full and args.skip_holdout):
        wait_enter("Ready when you are.")

    # If everything is skipped, just re-score from saved data
    if args.skip_22s and args.skip_full and args.skip_holdout:
        print("\n  Re-scoring from saved data ...")
        holdout_samples, holdout_labels, sample_rate, _ = load_holdout()
        model_22s = load_calibrated_model(EVAL_22S_ID)
        model_full = load_calibrated_model(EVAL_FULL_ID)
        if model_22s is None or model_full is None:
            raise SystemExit(
                "ERROR: cannot re-score — saved models missing. "
                "Rerun without all --skip-* flags."
            )
        r22 = score_model(model_22s, holdout_samples, holdout_labels,
                          sample_rate, "22s cal")
        rfull = score_model(model_full, holdout_samples, holdout_labels,
                            sample_rate, "full cal")
        print_comparison(r22, rfull)
        save_results_json(r22, rfull, sample_rate, args)
        return

    # Open serial and measure sample rate
    port = resolve_port(args.port)
    try:
        ser = serial.Serial(port, 115200, timeout=0.1)
    except serial.SerialException as e:
        raise SystemExit(f"\nERROR: failed to open serial port {port}:\n  {e}")

    print(f"\n  Serial port opened: {port}")
    print("  Measuring sample rate (1s) ...")
    sample_rate = measure_sample_rate(ser)
    print(f"  Sample rate: {sample_rate} Hz")

    # Rest baseline (or reuse from a saved cal if all cals are being skipped)
    if args.skip_22s and args.skip_full:
        # We still need rest_baseline for the held-out save
        prior = load_calibration(EVAL_22S_ID)
        rest_baseline = prior["rest_baseline"] if prior else \
            collect_rest_baseline(ser)
    else:
        rest_baseline = collect_rest_baseline(ser)

    try:
        # ─── 22-second calibration ──────────────────────────────────
        if args.skip_22s:
            print(f"\n  Skipping 22s cal — using saved {EVAL_22S_ID}/")
        else:
            clear_eval_dir(EVAL_22S_ID)
            wait_enter("Ready to start the 22-second calibration.")
            run_and_save_calibration(
                ser,
                protocol_builder=build_paper22s_protocol,
                display_label="22-second calibration",
                saved_calibration_type="abbreviated",
                baseline_model=args.baseline,
                sample_rate=sample_rate,
                rest_baseline=rest_baseline,
                patient_id=EVAL_22S_ID,
                prep_seconds=args.prep_seconds,
            )
            rest_break(30, "Between calibrations. Arm relaxed.")

        # ─── Full 6-min calibration ─────────────────────────────────
        if args.skip_full:
            print(f"\n  Skipping full cal — using saved {EVAL_FULL_ID}/")
        else:
            clear_eval_dir(EVAL_FULL_ID)
            wait_enter("Ready to start the full 6-minute calibration. "
                       "This one is long — take your time.")
            run_and_save_calibration(
                ser,
                protocol_builder=build_full_protocol,
                display_label="full 6-minute calibration",
                saved_calibration_type="full",
                baseline_model=args.baseline,
                sample_rate=sample_rate,
                rest_baseline=rest_baseline,
                patient_id=EVAL_FULL_ID,
                prep_seconds=args.prep_seconds,
            )
            rest_break(30, "Before the held-out test.")

        # ─── Held-out test ──────────────────────────────────────────
        if args.skip_holdout:
            print(f"\n  Skipping held-out test — using saved {EVAL_HOLDOUT_ID}/")
            holdout_samples, holdout_labels, _sr_prev, _ = load_holdout()
        else:
            clear_eval_dir(EVAL_HOLDOUT_ID)
            wait_enter("Ready for the held-out test. This is what we score on.")
            holdout_samples, holdout_labels = run_holdout_test(
                ser, sample_rate, rest_baseline,
                prep_seconds=args.prep_seconds,
            )
    finally:
        try:
            ser.close()
        except Exception:
            pass

    # ─── Load models and score ──────────────────────────────────────
    banner("SCORING BOTH MODELS ON HELD-OUT DATA")
    model_22s = load_calibrated_model(EVAL_22S_ID)
    model_full = load_calibrated_model(EVAL_FULL_ID)
    if model_22s is None:
        raise SystemExit(f"ERROR: model missing at eval_22s/")
    if model_full is None:
        raise SystemExit(f"ERROR: model missing at eval_full/")

    r22 = score_model(model_22s, holdout_samples, holdout_labels,
                      sample_rate, "22s cal")
    rfull = score_model(model_full, holdout_samples, holdout_labels,
                        sample_rate, "full cal")

    print_comparison(r22, rfull)
    save_results_json(r22, rfull, sample_rate, args)

    banner("DONE")
    speak("Evaluation complete. Results are on the screen.", blocking=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        banner("INTERRUPTED")
        print(
            "  Eval stopped by user. Any completed phases have been saved.\n"
            "  Resume with the appropriate --skip-* flags. For example:\n"
            "    --skip-22s              (22s cal already done)\n"
            "    --skip-22s --skip-full  (both cals done)\n"
            "    --skip-22s --skip-full --skip-holdout   (just re-score)"
        )
        sys.exit(130)
