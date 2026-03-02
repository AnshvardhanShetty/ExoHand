"""
server.py — FastAPI web server for ExoHand browser-based UI.

Serves a single-page app at localhost:8000. Streams real-time EMG/exercise
data over WebSocket, exposes REST endpoints for session control.

Architecture:
    Browser (localhost:8000) <-WebSocket-> FastAPI <-serial-> Teensy

Usage:
    python server.py --port /dev/tty.usbmodemXXXX --model exohand_model.pkl
    # or via run_exohand.py:
    python run_exohand.py --port ... --model ... --web
"""

import argparse
import asyncio
import json
import math
import os
import random
import threading
import time
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime.exercise import (
    Exercise, MotorCommand, ExerciseState, Event, SessionRunner,
    default_programme, FINGER_SERIAL_CODES, ACTION_TO_SERIAL,
)

# Lazy imports — only needed when NOT in demo mode
serial = None
LABEL_NAMES = ["close", "open", "rest"]

_demo_mode = False

# ── Globals shared between background thread and API ─────────────────────────

app = FastAPI(title="ExoHand Controller")

# State shared between the inference thread and the web handlers
_state = {
    "running": False,
    "mode": "idle",            # "idle" / "free" / "exercise"
    "intent": "rest",
    "confidence": 0.0,
    "assist_strength": 0.0,
    "gains": [1.0, 1.0, 1.0, 1.0],
    "channels": [0.0, 0.0, 0.0, 0.0],
    "exercise": None,          # {name, index, total}
    "rep": None,               # {current, target}
    "state": "idle",
    "state_time": 0.0,
    "events": [],
    "session_summary": None,
    "calibration": None,       # {phase, trial_idx, total_trials, gesture, time_remaining, progress_pct}
}

_lock = threading.Lock()
_ws_clients: List[WebSocket] = []
_ws_lock = threading.Lock()

# Handles set by start_server
_ser = None       # serial.Serial (set in non-demo mode)
_predictor = None  # RealtimePredictor (set in non-demo mode)
_session: Optional[SessionRunner] = None
_inference_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_args = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_calibration_thread: Optional[threading.Thread] = None
_calibration_stop = threading.Event()
_calibration_active = threading.Event()   # set while calibration owns the serial port
_model_data = None
_sample_rate = None
_rest_baseline = None
_calibration_params = None


# ── Inference thread (runs in background) ────────────────────────────────────

def _inference_loop():
    """Background thread: reads serial, runs prediction, pushes state."""
    global _session

    last_assist = [0.0]

    def get_assist():
        return last_assist[0]

    while not _stop_event.is_set():
        with _lock:
            mode = _state["mode"]
            session = _session

        if mode == "idle":
            time.sleep(0.05)
            continue

        # Yield serial port while calibration is running
        if _calibration_active.is_set():
            time.sleep(0.1)
            continue

        if _ser is None or _predictor is None:
            time.sleep(0.05)
            continue

        try:
            raw = _ser.readline()
        except (serial.SerialException, OSError):
            time.sleep(0.01)
            continue

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

        result = _predictor.add_sample(vals)
        if result is None:
            continue

        intent, confidence, changed, assist_strength = result
        last_assist[0] = assist_strength

        events = []
        exercise_info = None
        rep_info = None
        state_name = "free"
        state_time = 0.0

        if mode == "exercise" and session is not None:
            ex_events = session.update(intent, confidence)
            events = [e.name.lower() for e in ex_events]

            motor_cmd = session.get_motor_command()

            # Send finger selection if changed
            new_finger = session.finger_changed()
            if new_finger is not None:
                code = FINGER_SERIAL_CODES.get(new_finger, "A")
                _ser.write(code.encode())

            # Send motor command
            action_char = ACTION_TO_SERIAL.get(motor_cmd.action, "r")
            _ser.write(action_char.encode())

            ex = session.current_exercise
            runner = session.current_runner
            if ex and runner:
                exercise_info = {
                    "name": ex.name,
                    "index": session.current_index + 1,
                    "total": len(session.exercises),
                }
                rep_info = {
                    "current": runner.reps_completed + 1,
                    "target": ex.reps,
                }
                state_name = runner.state.name.lower()
                state_time = runner.state_elapsed

            if session.is_completed:
                with _lock:
                    _state["mode"] = "idle"
                    _state["running"] = False
                    _state["session_summary"] = [
                        {
                            "name": r.name,
                            "finger": r.finger,
                            "reps_target": r.reps_target,
                            "reps_completed": r.reps_completed,
                            "skipped": r.skipped,
                        }
                        for r in session.get_summary()
                    ]
        elif mode == "free":
            # Free mode: direct EMG -> motor
            from run_exohand import COMMAND_MAP
            cmd = COMMAND_MAP[intent]
            _ser.write(cmd.encode())
            state_name = "free"

        gains = _predictor.current_gains.tolist()

        update = {
            "type": "prediction",
            "intent": LABEL_NAMES[intent],
            "confidence": round(confidence, 3),
            "assist_strength": round(assist_strength, 3),
            "gains": [round(g, 2) for g in gains],
            "channels": [round(v, 4) for v in vals],
            "exercise": exercise_info,
            "rep": rep_info,
            "state": state_name,
            "state_time": round(state_time, 2),
            "events": events,
        }

        with _lock:
            _state.update({
                "intent": LABEL_NAMES[intent],
                "confidence": round(confidence, 3),
                "assist_strength": round(assist_strength, 3),
                "gains": [round(g, 2) for g in gains],
                "channels": [round(v, 4) for v in vals],
                "exercise": exercise_info,
                "rep": rep_info,
                "state": state_name,
                "state_time": round(state_time, 2),
                "events": events,
            })

        # Push to WebSocket clients
        _broadcast(json.dumps(update))


# ── Demo inference loop (fake data, no hardware) ─────────────────────────────

# Demo state machine timings (seconds per state)
_DEMO_STATE_DURATIONS = {
    "waiting": 2.5,
    "assisting": 0.5,
    "holding": 2.0,
    "returning": 0.5,
    "pause": 2.0,
}

_DEMO_STATES_ORDER = ["waiting", "assisting", "holding", "returning", "pause"]


def _demo_loop():
    """Background thread: generates fake EMG / exercise data for UI preview."""
    exercises = default_programme()
    ex_idx = 0
    rep = 0
    state_idx = 0
    state_entered = time.perf_counter()
    t0 = time.perf_counter()
    completed = False

    while not _stop_event.is_set():
        with _lock:
            mode = _state["mode"]

        if mode == "idle":
            time.sleep(0.05)
            continue

        now = time.perf_counter()
        t = now - t0

        # Simulated EMG channels (smooth sine waves + noise)
        ch1 = 0.3 * math.sin(2 * math.pi * 1.2 * t) + 0.05 * random.gauss(0, 1)
        ch2 = 0.25 * math.sin(2 * math.pi * 0.8 * t + 1.0) + 0.05 * random.gauss(0, 1)
        ch3 = 0.2 * math.sin(2 * math.pi * 1.0 * t + 2.0) + 0.05 * random.gauss(0, 1)
        ch4 = 0.15 * math.sin(2 * math.pi * 0.6 * t + 3.0) + 0.05 * random.gauss(0, 1)

        events = []
        exercise_info = None
        rep_info = None
        state_name = "free"
        state_time = 0.0

        if mode == "exercise" and not completed:
            ex = exercises[ex_idx]
            cur_state = _DEMO_STATES_ORDER[state_idx]
            state_elapsed = now - state_entered
            state_time = round(state_elapsed, 2)
            state_name = cur_state

            exercise_info = {
                "name": ex.name,
                "index": ex_idx + 1,
                "total": len(exercises),
            }
            rep_info = {
                "current": rep + 1,
                "target": ex.reps,
            }

            # Simulate effort ramp during assisting/holding
            if cur_state in ("assisting", "holding"):
                assist = 0.5 + 0.4 * math.sin(2 * math.pi * 0.3 * t)
                ch1 = 0.6 + 0.1 * random.gauss(0, 1)
                ch3 = 0.5 + 0.1 * random.gauss(0, 1)
            else:
                assist = 0.05 + 0.05 * abs(random.gauss(0, 1))

            # Advance state machine
            duration = _DEMO_STATE_DURATIONS[cur_state]
            if state_elapsed >= duration:
                if cur_state == "waiting":
                    events.append("effort_detected")
                elif cur_state == "returning":
                    rep += 1
                    events.append("rep_completed")
                    if rep >= ex.reps:
                        events.append("exercise_completed")
                        ex_idx += 1
                        rep = 0
                        if ex_idx >= len(exercises):
                            completed = True
                            with _lock:
                                _state["mode"] = "idle"
                                _state["running"] = False
                                _state["session_summary"] = [
                                    {
                                        "name": e.name,
                                        "finger": e.finger,
                                        "reps_target": e.reps,
                                        "reps_completed": e.reps,
                                        "skipped": False,
                                    }
                                    for e in exercises
                                ]
                            state_name = "idle"

                # Advance to next state
                if not completed:
                    state_idx = (state_idx + 1) % len(_DEMO_STATES_ORDER)
                    # After a rep_completed that also completed the exercise,
                    # reset to waiting for the new exercise
                    if "exercise_completed" in events and not completed:
                        state_idx = 0
                    state_entered = now
        elif mode == "free":
            state_name = "free"
            assist = 0.3 + 0.3 * math.sin(2 * math.pi * 0.5 * t)
        else:
            assist = 0.0

        if completed and mode == "idle":
            # Already sent summary, just idle
            time.sleep(0.05)
            continue

        intent_idx = 0 if (math.sin(2 * math.pi * 0.2 * t) > 0) else 1
        confidence = 0.7 + 0.25 * abs(math.sin(2 * math.pi * 0.15 * t))

        update = {
            "type": "prediction",
            "intent": LABEL_NAMES[intent_idx],
            "confidence": round(confidence, 3),
            "assist_strength": round(assist, 3),
            "gains": [1.0, 1.0, 1.0, 1.0],
            "channels": [round(ch1, 4), round(ch2, 4), round(ch3, 4), round(ch4, 4)],
            "exercise": exercise_info,
            "rep": rep_info,
            "state": state_name,
            "state_time": round(state_time, 2),
            "events": events,
        }

        with _lock:
            _state.update({
                "intent": LABEL_NAMES[intent_idx],
                "confidence": round(confidence, 3),
                "assist_strength": round(assist, 3),
                "gains": [1.0, 1.0, 1.0, 1.0],
                "channels": [round(ch1, 4), round(ch2, 4), round(ch3, 4), round(ch4, 4)],
                "exercise": exercise_info,
                "rep": rep_info,
                "state": state_name,
                "state_time": round(state_time, 2),
                "events": events,
            })

        _broadcast(json.dumps(update))

        # ~30 Hz update rate
        time.sleep(0.033)


def _broadcast(message: str):
    """Send message to all connected WebSocket clients."""
    loop = _event_loop
    if loop is None:
        return

    with _ws_lock:
        clients = list(_ws_clients)

    for ws in clients:
        try:
            asyncio.run_coroutine_threadsafe(ws.send_text(message), loop)
        except Exception:
            pass


# ── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    with _ws_lock:
        _ws_clients.append(ws)
    try:
        while True:
            # Keep connection alive; client may send pings
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        with _ws_lock:
            if ws in _ws_clients:
                _ws_clients.remove(ws)


# ── REST endpoints ───────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    with _lock:
        return JSONResponse(dict(_state))


@app.post("/api/session/start")
async def start_session(body: dict = None):
    global _session

    if body is None:
        body = {}

    mode = body.get("mode", "exercise")

    with _lock:
        if _state["running"]:
            return JSONResponse({"error": "Session already running"}, status_code=400)

    if mode == "exercise":
        last_assist = [0.0]

        def get_assist():
            with _lock:
                return _state.get("assist_strength", 0.0)

        exercises = default_programme()
        session = SessionRunner(exercises, assist_strength_fn=get_assist)

        with _lock:
            _session = session
            _state["mode"] = "exercise"
            _state["running"] = True
            _state["session_summary"] = None
    else:
        with _lock:
            _session = None
            _state["mode"] = "free"
            _state["running"] = True
            _state["session_summary"] = None

    return JSONResponse({"status": "started", "mode": mode})


@app.post("/api/exercise/skip")
async def skip_exercise():
    with _lock:
        session = _session

    if session is None:
        return JSONResponse({"error": "No active session"}, status_code=400)

    session.skip_exercise()
    return JSONResponse({"status": "skipped"})


@app.post("/api/exercise/select-finger/{finger}")
async def select_finger(finger: str):
    if finger not in FINGER_SERIAL_CODES:
        return JSONResponse({"error": f"Unknown finger: {finger}"}, status_code=400)

    if _ser is not None:
        code = FINGER_SERIAL_CODES[finger]
        _ser.write(code.encode())

    return JSONResponse({"status": "ok", "finger": finger})


@app.post("/api/session/stop")
async def stop_session():
    global _session

    with _lock:
        session = _session

    if session is not None:
        session.stop()
        with _lock:
            _state["session_summary"] = [
                {
                    "name": r.name,
                    "finger": r.finger,
                    "reps_target": r.reps_target,
                    "reps_completed": r.reps_completed,
                    "skipped": r.skipped,
                }
                for r in session.get_summary()
            ]

    # Send rest command to Teensy
    if _ser is not None:
        _ser.write(b"Ar")

    with _lock:
        _state["mode"] = "idle"
        _state["running"] = False
        _session = None

    return JSONResponse({"status": "stopped"})


@app.get("/api/session/summary")
async def get_summary():
    with _lock:
        summary = _state.get("session_summary")

    if summary is None:
        return JSONResponse({"error": "No summary available"}, status_code=404)

    return JSONResponse({"summary": summary})


@app.post("/api/settings")
async def update_settings(body: dict):
    # Allow updating assist level at runtime
    level = body.get("assist_level")
    if level is not None and _predictor is not None and not _demo_mode:
        try:
            from assist_profile import get_profile
            profile = get_profile(int(level))
            _predictor.profile = profile
            _predictor.stability_count = profile.stability_required
            _predictor.recent_preds = __import__("collections").deque(
                maxlen=profile.stability_required
            )
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    return JSONResponse({"status": "ok"})


# ── Calibration endpoints ────────────────────────────────────────────────

def _calibration_progress_callback(phase, trial_idx, total_trials, gesture,
                                   time_remaining, progress_pct):
    """Update state and broadcast calibration progress via WebSocket."""
    cal_state = {
        "phase": phase,
        "trial_idx": trial_idx,
        "total_trials": total_trials,
        "gesture": gesture,
        "time_remaining": round(time_remaining, 1),
        "progress_pct": round(progress_pct, 1),
    }
    with _lock:
        _state["calibration"] = cal_state

    msg = json.dumps({"type": "calibration_progress", **cal_state})
    _broadcast(msg)


def _run_calibration(cal_type, patient_id):
    """Background thread that runs patient calibration."""
    global _predictor, _model_data, _rest_baseline, _calibration_params

    try:
        from calibrate_patient import (calibrate_patient, abbreviated_calibrate,
                                       CalibrationCancelled)
        from run_exohand import RealtimePredictor
        from assist_profile import get_profile

        level = getattr(_args, "assist_level", 3)
        profile = get_profile(level)

        # Claim the serial port — inference loop will yield
        _calibration_active.set()
        time.sleep(0.2)  # allow inference loop to release serial port

        if cal_type == "full":
            cal_result = calibrate_patient(
                _ser, _model_data, _sample_rate,
                patient_id=patient_id,
                progress_callback=_calibration_progress_callback,
                assist_profile=profile,
                interactive=False,
                stop_event=_calibration_stop,
            )
        else:
            from calibrate_patient import load_calibrated_model
            prev_model = load_calibrated_model(patient_id)
            cal_model = prev_model if prev_model is not None else _model_data
            cal_result = abbreviated_calibrate(
                _ser, cal_model, _sample_rate,
                patient_id=patient_id,
                progress_callback=_calibration_progress_callback,
                assist_profile=profile,
                interactive=False,
                stop_event=_calibration_stop,
            )

        _model_data = cal_result.finetuned_model_data
        _rest_baseline = cal_result.rest_baseline
        _calibration_params = cal_result.calibration_params

        # Recreate predictor with new model + calibration params
        _predictor = RealtimePredictor(
            _model_data, _sample_rate, assist_profile=profile,
            rest_baseline=_rest_baseline,
            calibration_params=_calibration_params,
        )

        with _lock:
            _state["calibration"] = None

        _broadcast(json.dumps({
            "type": "calibration_complete",
            "patient_id": patient_id,
            "calibration_type": cal_type,
            "success": True,
        }))

    except CalibrationCancelled:
        with _lock:
            _state["calibration"] = None
        _broadcast(json.dumps({
            "type": "calibration_complete",
            "success": False,
            "error": "Calibration cancelled",
        }))
    except Exception as e:
        with _lock:
            _state["calibration"] = None
        _broadcast(json.dumps({
            "type": "calibration_complete",
            "success": False,
            "error": str(e),
        }))
    finally:
        # Release serial port back to inference loop
        _calibration_active.clear()


@app.post("/api/calibration/start")
async def start_calibration(body: dict = None):
    global _calibration_thread
    if body is None:
        body = {}

    if _demo_mode:
        return JSONResponse({"error": "Calibration not available in demo mode"}, status_code=400)

    with _lock:
        if _state["calibration"] is not None:
            return JSONResponse({"error": "Calibration already running"}, status_code=400)
        if _state["running"]:
            return JSONResponse({"error": "Session running, stop it first"}, status_code=400)

    cal_type = body.get("type", "full")
    patient_id = body.get("patient_id", "default")

    _calibration_stop.clear()
    _calibration_thread = threading.Thread(
        target=_run_calibration, args=(cal_type, patient_id), daemon=True)
    _calibration_thread.start()

    return JSONResponse({"status": "started", "type": cal_type, "patient_id": patient_id})


@app.post("/api/calibration/stop")
async def stop_calibration():
    _calibration_stop.set()
    with _lock:
        _state["calibration"] = None
    return JSONResponse({"status": "stopped"})


@app.get("/api/calibration/status")
async def get_calibration_status():
    with _lock:
        cal = _state.get("calibration")
    return JSONResponse({"calibration": cal})


@app.get("/api/patients")
async def list_patients():
    from calibrate_patient import list_patients as _list
    return JSONResponse({"patients": _list()})


@app.post("/api/patients/{patient_id}/load")
async def load_patient(patient_id: str):
    global _predictor, _model_data, _rest_baseline, _calibration_params

    if _demo_mode:
        return JSONResponse({"error": "Not available in demo mode"}, status_code=400)

    from calibrate_patient import load_calibrated_model, load_calibration
    from run_exohand import RealtimePredictor
    from assist_profile import get_profile

    model = load_calibrated_model(patient_id)
    if model is None:
        return JSONResponse({"error": f"No calibration found for '{patient_id}'"}, status_code=404)

    cal_data = load_calibration(patient_id)
    level = getattr(_args, "assist_level", 3)
    profile = get_profile(level)

    _model_data = model
    _rest_baseline = cal_data["rest_baseline"] if cal_data else None
    _calibration_params = cal_data["info"].get("calibration_params") if cal_data else None

    _predictor = RealtimePredictor(
        _model_data, _sample_rate, assist_profile=profile,
        rest_baseline=_rest_baseline,
        calibration_params=_calibration_params,
    )

    return JSONResponse({"status": "loaded", "patient_id": patient_id})


# ── Static files ─────────────────────────────────────────────────────────────

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


# ── Server startup ───────────────────────────────────────────────────────────

def start_server(args):
    """Called from run_exohand.py --web or directly."""
    global _ser, _predictor, _inference_thread, _args, _event_loop, _demo_mode
    global _model_data, _sample_rate, _rest_baseline

    _args = args
    _demo_mode = getattr(args, "demo", False)

    if _demo_mode:
        print("Starting in DEMO mode (no hardware required)")
        print("  Simulated EMG data will be generated\n")

        # Start demo inference thread
        _inference_thread = threading.Thread(target=_demo_loop, daemon=True)
        _inference_thread.start()
    else:
        import serial as _serial_mod
        global serial
        serial = _serial_mod
        from run_exohand import parse_emg_line, load_model, RealtimePredictor
        from assist_profile import get_profile, print_profile

        profile = get_profile(args.assist_level)

        print(f"Loading model from {args.model}...")
        model_data = load_model(args.model)
        _model_data = model_data
        print(f"  Window: {model_data['window_ms']}ms, Stride: {model_data['stride_ms']}ms")

        print()
        print_profile(profile)

        print(f"\nConnecting to {args.port} at {args.baud}...")
        _ser = serial.Serial(args.port, args.baud, timeout=0.1)
        time.sleep(2)
        _ser.reset_input_buffer()

        # Estimate sample rate
        print("Estimating sample rate (2s)...")
        _ser.reset_input_buffer()
        count = 0
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < 2.0:
            raw = _ser.readline()
            if raw:
                try:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if parse_emg_line(line) is not None:
                        count += 1
                except UnicodeDecodeError:
                    pass
        sample_rate = count / 2.0
        _sample_rate = sample_rate
        print(f"  Estimated: ~{sample_rate:.0f} Hz")

        # Rest calibration for web mode (use non-interactive version)
        skip_rest = getattr(args, "skip_rest_cal", False)
        if not skip_rest:
            from calibrate_patient import _rest_calibrate_noninteractive
            _rest_baseline = _rest_calibrate_noninteractive(_ser, duration=10)
        else:
            _rest_baseline = None

        _predictor = RealtimePredictor(model_data, sample_rate, assist_profile=profile,
                                       rest_baseline=_rest_baseline)
        print(f"  Window: {_predictor.win_samples} samples, Stride: {_predictor.stride_samples} samples")

        _ser.reset_input_buffer()

        # Start inference thread
        _inference_thread = threading.Thread(target=_inference_loop, daemon=True)
        _inference_thread.start()

    # Capture the event loop for cross-thread WebSocket broadcasts
    @app.on_event("startup")
    async def _capture_loop():
        global _event_loop
        _event_loop = asyncio.get_running_loop()

    print(f"Starting web server at http://localhost:8000")
    print("Open Chrome and navigate to http://localhost:8000")
    print("Press Ctrl+C to stop\n")

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        _stop_event.set()
        if _ser:
            _ser.write(b"Ar")
            _ser.close()


# ── Direct invocation ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExoHand Web Server")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode with simulated data (no hardware needed)")
    parser.add_argument("--port", help="Serial port (required unless --demo)")
    parser.add_argument("--model", help="Path to exohand_model.pkl (required unless --demo)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--assist-level", type=int, default=3, choices=[1, 2, 3, 4, 5],
                        help="Assist level 1-5 (default: 3)")
    args = parser.parse_args()

    if not args.demo and (not args.port or not args.model):
        parser.error("--port and --model are required unless --demo is set")

    start_server(args)
