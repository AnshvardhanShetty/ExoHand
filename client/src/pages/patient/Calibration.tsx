import React, { useEffect, useState, useRef, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button } from "../../components/ui/Button";
import { ProgressBar } from "../../components/ui/ProgressBar";
import { LoadingScreen } from "../../components/ui/LoadingScreen";
import { CountdownScreen } from "../../components/ui/CountdownScreen";
import { useWebSocket } from "../../hooks/useWebSocket";
import { api } from "../../lib/api";

interface CalibrationStatus {
  active: boolean;
  completed: boolean;
  mode: string;
  modelLoaded: boolean;
  phaseIndex: number;
  trialIndex: number;
  totalPhases: number;
  phaseName: string | null;
  phaseInstruction: string | null;
  phaseTargetAngle: number;
  phaseDurationSec: number;
  phaseTrials: number;
  phaseElapsedSec: number;
  phaseProgress: number;
  overallProgress: number;
  remainingSec: number;
  phaseWaiting: boolean;
  error: string | null;
}

type CalibStep =
  | "landing"
  | "choose"
  | "loading_model"
  | "loading"
  | "pre_phase"
  | "countdown"
  | "running"
  | "done";

interface CalibrationProps {
  patientId?: number;
  therapistMode?: boolean;
  onComplete?: () => void;
  onCancel?: () => void;
}

export function Calibration({ patientId, therapistMode, onComplete, onCancel }: CalibrationProps = {}) {
  const navigate = useNavigate();
  const location = useLocation();
  const isFullMode = therapistMode || new URLSearchParams(location.search).get("mode") === "full";
  const { connected, frame } = useWebSocket();
  const [step, setStep] = useState<CalibStep>(isFullMode ? "choose" : "landing");
  const [status, setStatus] = useState<CalibrationStatus | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [lastPhase, setLastPhase] = useState(-1);

  // Phase countdown timer — driven by local clock, stops at 0
  const [localTimer, setLocalTimer] = useState(0);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const clockAnchorRef = useRef<number>(0);
  const phaseDurationRef = useRef<number>(5);

  // Track the phase gesture for display during rest periods
  const phaseGestureRef = useRef<string>("REST");

  const startPolling = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await api.getCalibrationStatus();
        setStatus(s);
      } catch {
        // ignore
      }
    }, 500);
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  // Start phase countdown clock — only tracks phase timer, stops at 0
  const startClock = useCallback((phaseDuration: number) => {
    if (tickRef.current) clearInterval(tickRef.current);
    phaseDurationRef.current = phaseDuration;
    clockAnchorRef.current = Date.now();
    setLocalTimer(Math.ceil(phaseDuration));

    tickRef.current = setInterval(() => {
      const elapsed = (Date.now() - clockAnchorRef.current) / 1000;
      const remaining = Math.max(0, Math.ceil(phaseDurationRef.current - elapsed));
      setLocalTimer(remaining);
      // Auto-stop when timer reaches 0 — don't keep ticking during rest period
      if (remaining <= 0 && tickRef.current) {
        clearInterval(tickRef.current);
        tickRef.current = null;
      }
    }, 100);
  }, []);

  const stopClock = useCallback(() => {
    if (tickRef.current) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      stopPolling();
      stopClock();
    };
  }, [stopPolling, stopClock]);

  /* ── Polling-driven step transitions ── */
  useEffect(() => {
    if (!status) return;

    // Error from Python (e.g. serial port not found)
    if (status.error && !status.active) {
      stopPolling();
      return; // stay on loading_model which will show the error
    }

    if (status.completed) {
      stopPolling();
      stopClock();
      setStep("done");
      return;
    }

    // Model loaded → transition from loading_model to loading (brief) then pre_phase
    if (status.active && status.modelLoaded && step === "loading_model") {
      setStep("loading");
      return;
    }

    // Server is waiting for us (phaseWaiting=true) → show pre_phase
    if (status.active && status.phaseWaiting && step === "loading") {
      setLastPhase(status.phaseIndex);
      setStep("pre_phase");
      return;
    }

    // Trial changed while running → server signals new gesture, show pre_phase
    if (
      (step === "running" || step === "pre_phase" || step === "countdown") &&
      status.phaseWaiting &&
      status.phaseIndex !== lastPhase
    ) {
      stopClock();
      setLastPhase(status.phaseIndex);
      setStep("pre_phase");
      return;
    }
  }, [status, step, lastPhase, stopPolling, stopClock]);

  /* ── Navigate after done ── */
  useEffect(() => {
    if (step === "done") {
      if (therapistMode && onComplete) {
        const timer = setTimeout(() => onComplete(), 2000);
        return () => clearTimeout(timer);
      } else if (!therapistMode) {
        const timer = setTimeout(() => {
          navigate("/patient/session/new");
        }, 2000);
        return () => clearTimeout(timer);
      }
    }
  }, [step, navigate, therapistMode, onComplete]);

  /* ── Auto-advance pre_phase → countdown after 3s ── */
  useEffect(() => {
    if (step === "pre_phase") {
      const timer = setTimeout(() => setStep("countdown"), 3000);
      return () => clearTimeout(timer);
    }
  }, [step, lastPhase]);

  const handleStart = async (mode: "full" | "quick") => {
    setStep("loading_model");
    if (!therapistMode) {
      // Signal layout to hide sidebar during active calibration
      navigate("/patient/calibration?active=1", { replace: true });
    }
    await api.startCalibration(mode, patientId);
    startPolling();
  };

  const handleCancel = async () => {
    await api.stopCalibration();
    stopPolling();
    stopClock();
    if (therapistMode && onCancel) {
      onCancel();
      return;
    }
    if (!therapistMode) {
      // Restore sidebar by removing active param
      navigate("/patient/calibration", { replace: true });
    }
    setStep(isFullMode ? "choose" : "landing");
    setStatus(null);
    setLastPhase(-1);
  };

  const formatTime = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  // Smooth EMG display — only update while phase timer is active (localTimer > 0).
  // This freezes the bars during rest periods between trials so users don't see
  // rest-level EMG while the header still shows "CLOSE" or "OPEN".
  // EMA buffer is reset in countdown→running transition (onComplete) for clean starts.
  const emgSmoothedRef = useRef([0, 0, 0, 0]);
  const rawEmg = frame?.emg ?? [0, 0, 0, 0];
  const phaseActive = step === "running" && localTimer > 0;
  if (phaseActive) {
    const EMG_SMOOTH = 0.4; // Fast alpha — 95% in ~0.3s at 20Hz
    emgSmoothedRef.current = emgSmoothedRef.current.map(
      (prev, i) => prev + (Math.abs(rawEmg[i] ?? 0) - prev) * EMG_SMOOTH
    );
  }
  const emgChannels = emgSmoothedRef.current;

  // Remaining time: use server's authoritative remainingSec from Python progress callbacks
  const serverRemaining = status?.remainingSec ?? 0;

  /* ── Choose mode (full calibration: ?mode=full, therapist-initiated) ── */
  if (step === "choose") {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="max-w-md w-full space-y-6 p-6">
          <div className="text-center">
            <h2 className="text-h2 font-bold text-text font-mono">Calibration</h2>
            <p className="text-body text-muted mt-2">
              {therapistMode ? "Calibrate EMG sensors for this patient." : "Full calibration for EMG sensor setup."}
            </p>
          </div>

          <button
            onClick={() => handleStart("full")}
            disabled={!connected}
            className="w-full px-8 py-3 rounded-lg border border-white/20 text-text font-mono text-body tracking-wide hover:bg-white/[0.06] transition-colors disabled:opacity-40"
          >
            Start Full Calibration
          </button>

          {therapistMode && (
            <button
              onClick={() => handleStart("quick")}
              disabled={!connected}
              className="w-full px-8 py-3 rounded-lg border border-white/10 text-muted font-mono text-body tracking-wide hover:bg-white/[0.06] transition-colors disabled:opacity-40"
            >
              Start Quick Calibration
            </button>
          )}

          {!therapistMode && (
            <button
              onClick={() => navigate("/patient/session/new")}
              className="w-full px-4 py-2 text-small text-muted hover:text-text transition-colors"
            >
              Skip calibration
            </button>
          )}

          {!connected && (
            <p className="text-small text-warn text-center">
              Waiting for connection...
            </p>
          )}
        </div>
      </div>
    );
  }

  /* ── Landing screen (default session flow — quick only) ── */
  if (step === "landing") {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="max-w-md w-full space-y-6 p-6">
          <div className="text-center">
            <h2 className="text-h2 font-bold text-text font-mono">Session Calibration</h2>
            <p className="text-body text-muted mt-2">
              A quick calibration to prepare your sensors for this session.
            </p>
          </div>

          <button
            onClick={() => handleStart("quick")}
            disabled={!connected}
            className="w-full px-8 py-3 rounded-lg border border-white/20 text-text font-mono text-body tracking-wide hover:bg-white/[0.06] transition-colors disabled:opacity-40"
          >
            Start Calibration
          </button>

          <button
            onClick={() => navigate("/patient/session/new")}
            className="w-full px-4 py-2 text-small text-muted hover:text-text transition-colors"
          >
            Skip calibration
          </button>

          {!connected && (
            <p className="text-small text-warn text-center">
              Waiting for connection...
            </p>
          )}
        </div>
      </div>
    );
  }

  /* ── Loading model ── */
  if (step === "loading_model") {
    if (status?.error) {
      return (
        <div className="h-full flex items-center justify-center">
          <div className="text-center space-y-4 max-w-md px-6">
            <p className="text-h3 font-semibold text-text">Failed to load model</p>
            <p className="text-small text-muted">{status.error}</p>
            <Button variant="danger" onClick={handleCancel}>
              Go back
            </Button>
          </div>
        </div>
      );
    }
    return (
      <LoadingScreen
        message="Loading model..."
        submessage="Initializing EMG calibration engine. This may take a few seconds."
      />
    );
  }

  /* ── Loading ── */
  if (step === "loading") {
    return (
      <LoadingScreen
        message="Preparing calibration..."
        submessage="Model loaded. Starting calibration."
      />
    );
  }

  /* ── Done ── */
  if (step === "done") {
    return (
      <LoadingScreen
        message="Calibration complete"
        submessage={therapistMode ? "Model saved for this patient." : "Choose your exercise."}
      />
    );
  }

  /* ── Pre-phase instruction card ── */
  if (step === "pre_phase") {
    const phaseName = status?.phaseName ?? "NEXT";
    const instruction = status?.phaseInstruction ?? "Follow the instructions";
    // Remember this phase's gesture for display during rest periods
    phaseGestureRef.current = phaseName;
    return (
      <div className="h-full flex items-center justify-center bg-bg">
        <div className="text-center space-y-4 max-w-md px-6">
          <p className="text-small text-muted uppercase tracking-wider">
            Up next
          </p>
          <h2 className="text-[48px] font-bold text-text leading-none">{phaseName}</h2>
          <p className="text-body text-muted">{instruction}</p>
          <p className="text-small text-muted mt-2">
            Hold for {status?.phaseDurationSec ?? 5}s
          </p>
        </div>
      </div>
    );
  }

  /* ── Countdown 3-2-1 ── */
  if (step === "countdown") {
    const phaseName = status?.phaseName ?? "NEXT";
    const instruction = status?.phaseInstruction ?? "";
    return (
      <CountdownScreen
        title={`Calibrating: ${phaseName}`}
        subtitle={instruction}
        onComplete={() => {
          api.calibrationPhaseReady();
          emgSmoothedRef.current = [0, 0, 0, 0]; // Reset EMA for clean phase start
          setStep("running");
          startClock(status?.phaseDurationSec ?? 5);
        }}
      />
    );
  }

  /* ── Running (calibration phase active) ── */
  // Use the gesture name from when this phase started (not from server
  // which may have moved to REST between trials)
  const displayName = phaseGestureRef.current;
  const displayInstruction = status?.phaseInstruction ?? "Follow the instructions";
  const phaseComplete = localTimer <= 0;

  return (
    <div className="h-full flex">
      {/* Left — large timer display */}
      <div className="flex-[2] flex items-center justify-center bg-bg">
        <div className="text-center space-y-6">
          <p className="text-small text-muted uppercase tracking-wider">
            {displayName}
          </p>
          <p className="text-[96px] font-bold font-mono text-text leading-none">
            {phaseComplete ? "0s" : `${localTimer}s`}
          </p>
          <p className="text-body text-muted">
            {phaseComplete ? "Done — next phase starting..." : displayInstruction}
          </p>
        </div>
      </div>

      {/* Right — Calibration info panel */}
      <div className="w-[340px] bg-panel border-l border-border p-5 flex flex-col gap-4 overflow-auto">
        {/* Phase label */}
        <div className="text-center py-3 border-b border-border">
          <p className="text-small text-muted uppercase tracking-wider">
            {displayName}
          </p>
          <p className="text-h3 font-bold text-text mt-1">
            {phaseComplete ? "Done — next phase starting..." : displayInstruction}
          </p>
        </div>

        {/* Phase timer */}
        <div className="text-center">
          <p className="text-h1 font-bold font-mono text-text">
            {phaseComplete ? "0s" : `${localTimer}s`}
          </p>
          <p className="text-small text-muted">
            {phaseComplete ? "Preparing next phase..." : "Hold steady"}
          </p>
        </div>

        {/* Overall progress */}
        <ProgressBar
          label="Overall Progress"
          value={(status?.overallProgress ?? 0) * 100}
        />

        {/* Total remaining — from server's Python progress callback */}
        <div className="flex items-center justify-between">
          <span className="text-small text-muted">Time remaining</span>
          <span className="font-mono text-body text-text">
            {formatTime(serverRemaining)}
          </span>
        </div>

        {/* Live EMG bars — frozen during rest periods */}
        <div className="space-y-2 pt-2 border-t border-border">
          <p className="text-small text-muted">
            Live EMG{localTimer <= 0 ? " (paused)" : ""}
          </p>
          <div className="space-y-1.5">
            {emgChannels.slice(0, 4).map((val: number, i: number) => {
              const normalized = Math.min(100, (val / 100) * 100);
              return (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-muted w-6">
                    CH{i + 1}
                  </span>
                  <div className="flex-1 h-2 bg-white/[0.06] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-150 ${
                        localTimer > 0 ? "bg-white" : "bg-white/30"
                      }`}
                      style={{ width: `${normalized}%` }}
                    />
                  </div>
                  <span className="text-[10px] font-mono text-muted w-10 text-right">
                    {val.toFixed(0)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Cancel button */}
        <div className="mt-auto pt-4">
          <Button variant="danger" onClick={handleCancel} className="w-full">
            Cancel Calibration
          </Button>
        </div>
      </div>
    </div>
  );
}
