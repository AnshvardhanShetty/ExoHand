import React, { useEffect, useState, useRef, useCallback, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "../../components/ui/Button";
import { Badge } from "../../components/ui/Badge";
import { ProgressBar } from "../../components/ui/ProgressBar";
import { HandViewer } from "../../components/HandViewer";
import { LoadingScreen } from "../../components/ui/LoadingScreen";
import { CountdownScreen } from "../../components/ui/CountdownScreen";
import type { DriverMode, LiveState, ExerciseInfo } from "../../components/HandViewer";
import { useWebSocket } from "../../hooks/useWebSocket";
import { api } from "../../lib/api";

interface SessionProps {
  patientId: number;
}

type SessionStep = "ready" | "loading_model" | "loading" | "demo" | "countdown" | "active";

export function PatientSession({ patientId }: SessionProps) {
  const navigate = useNavigate();
  const { connected, frame, stale, send } = useWebSocket();
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const elapsedRef = useRef(0);
  const [repCount, setRepCount] = useState(0);
  const [negativeRepCount, setNegativeRepCount] = useState(0);
  const [patient, setPatient] = useState<any>(null);
  const [sessionStep, setSessionStep] = useState<SessionStep>("ready");
  const [bridgeError, setBridgeError] = useState<string | null>(null);
  const [demoCountdown, setDemoCountdown] = useState(10);
  const [holdSeconds, setHoldSeconds] = useState(0);
  const holdStartRef = useRef<number | null>(null);
  const holdTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const demoTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const bridgePollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const activeExercise = useMemo<ExerciseInfo | null>(() => {
    try {
      const raw = sessionStorage.getItem("activeExercise");
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }, []);

  useEffect(() => {
    api.getPatient(patientId).then(setPatient);
  }, [patientId]);

  const beginRecording = useCallback(async () => {
    setSessionStep("active");
    startTimeRef.current = Date.now();
    send({ type: "startSession" });

    timerRef.current = setInterval(() => {
      if (startTimeRef.current) {
        const e = Math.floor((Date.now() - startTimeRef.current) / 1000);
        setElapsed(e);
        elapsedRef.current = e;
      }
    }, 1000);
  }, [send]);

  const startDemo = useCallback(() => {
    setSessionStep("demo");
    setDemoCountdown(10);

    demoTimerRef.current = setInterval(() => {
      setDemoCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(demoTimerRef.current!);
          setSessionStep("countdown");
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  }, []);

  const startSession = useCallback(async () => {
    // Start session (spawns PythonBridge) and show loading screen
    setSessionStep("loading_model");
    setBridgeError(null);
    const session = await api.startSession(patientId, activeExercise ?? undefined);
    setSessionId(session.id);

    // Poll bridge status until model is ready or errors out
    bridgePollRef.current = setInterval(async () => {
      try {
        const status = await api.getBridgeStatus();
        if (status.ready) {
          if (bridgePollRef.current) clearInterval(bridgePollRef.current);
          bridgePollRef.current = null;
          startDemo();
        } else if (status.error) {
          if (bridgePollRef.current) clearInterval(bridgePollRef.current);
          bridgePollRef.current = null;
          setBridgeError(status.error);
        }
      } catch {
        // ignore poll errors
      }
    }, 500);
  }, [patientId, startDemo]);

  const endSession = useCallback(async () => {
    if (!sessionId) return;
    send({ type: "endSession" });
    if (timerRef.current) clearInterval(timerRef.current);
    await api.endSession(sessionId, elapsedRef.current);
    navigate(`/patient/summary/${sessionId}`);
  }, [sessionId, send, navigate]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (demoTimerRef.current) clearInterval(demoTimerRef.current);
      if (bridgePollRef.current) clearInterval(bridgePollRef.current);
      if (holdTimerRef.current) clearInterval(holdTimerRef.current);
    };
  }, []);

  useEffect(() => {
    if (frame?.repCount !== undefined) {
      setRepCount(frame.repCount);
    }
    if (frame?.negativeRepCount !== undefined) {
      setNegativeRepCount(frame.negativeRepCount);
    }
  }, [frame?.repCount, frame?.negativeRepCount]);

  // Track hold duration for countdown display
  // Only show "Hold steady..." when the state matches the exercise's target direction
  const category = activeExercise?.category;
  const isHolding = (() => {
    const cmd = frame?.stateCmd;
    if (!cmd) return false;
    if (cmd === "HOLDING") return true; // always a hold
    if (category === "close" && cmd === "CLOSED") return true;
    if (category === "open" && cmd === "OPEN") return true;
    if (category === "combined" && (cmd === "CLOSED" || cmd === "OPEN")) return true;
    return false;
  })();
  useEffect(() => {
    if (isHolding && sessionStep === "active") {
      if (!holdStartRef.current) {
        holdStartRef.current = Date.now();
        setHoldSeconds(0);
        holdTimerRef.current = setInterval(() => {
          if (holdStartRef.current) {
            setHoldSeconds(Math.floor((Date.now() - holdStartRef.current) / 1000) + 1);
          }
        }, 200);
      }
    } else {
      if (holdStartRef.current) {
        holdStartRef.current = null;
        setHoldSeconds(0);
        if (holdTimerRef.current) {
          clearInterval(holdTimerRef.current);
          holdTimerRef.current = null;
        }
      }
    }
    return () => {
      if (holdTimerRef.current) clearInterval(holdTimerRef.current);
    };
  }, [isHolding, sessionStep]);

  const totalReps = activeExercise?.reps ?? patient?.rep_count ?? 10;

  // Auto-end session when all reps are completed
  useEffect(() => {
    if (sessionStep === "active" && repCount >= totalReps && repCount > 0 && sessionId) {
      const timer = setTimeout(() => {
        endSession();
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [repCount, totalReps, sessionStep, sessionId, endSession]);

  // M1: Auto-end on too many missed reps (sessionFailed from server)
  useEffect(() => {
    if (sessionStep === "active" && (frame as any)?.sessionFailed && sessionId) {
      const timer = setTimeout(() => {
        endSession();
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [(frame as any)?.sessionFailed, sessionStep, sessionId, endSession]);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };
  const repCircles = Array.from({ length: totalReps }, (_, i) => i < repCount);

  // Use stateCmd from motor state machine (REST/MOVING/CLOSED/OPEN — stable, no jitter)
  const displayState = frame?.stateCmd || "WAITING";

  const stateBadgeVariant = (() => {
    const s = displayState;
    if (!s || s === "REST" || s === "WAITING") return "default";
    if (category === "close" && s === "OPEN") return "danger";
    if (category === "open" && s === "CLOSED") return "danger";
    if (s === "CLOSED" || s === "OPEN") return "success";
    if (s === "MOVING") return "warning";
    return "default";
  })();

  const demoPhase = sessionStep === "demo";
  const isActive = sessionStep === "demo" || sessionStep === "active";

  // Determine hand viewer mode + live state
  const handMode: DriverMode = (sessionStep === "active") ? "LIVE" : "DEMO";
  const liveState: LiveState | null = frame
    ? {
        stateCmd: frame.stateCmd,
        motionLocked: frame.motionLocked,
        lockRemainingMs: frame.lockRemainingMs,
        targetAngle: frame.targetAngle,
      }
    : null;

  /* -- Ready screen -- */
  if (sessionStep === "ready") {
    return (
      <div className="h-full flex flex-col">
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-4">
            <h3 className="text-h2 font-semibold text-text">Ready...</h3>
            <p className="text-body text-muted">
              {activeExercise?.reps ?? patient?.rep_count ?? 10} reps
            </p>
            <p className="text-small text-muted">
              A 10-second demo will play first
            </p>
            <Button size="lg" onClick={startSession} disabled={!connected}>
              {connected ? "Begin Session" : "Connecting..."}
            </Button>
          </div>
        </div>
      </div>
    );
  }

  /* -- Model loading screen -- */
  if (sessionStep === "loading_model") {
    if (bridgeError) {
      return (
        <div className="h-full flex items-center justify-center">
          <div className="text-center space-y-4 max-w-md px-6">
            <p className="text-h3 font-semibold text-text">Failed to load model</p>
            <p className="text-small text-muted">{bridgeError}</p>
            <Button variant="danger" onClick={() => navigate("/patient/session/new")}>
              Go back
            </Button>
          </div>
        </div>
      );
    }
    return (
      <LoadingScreen
        message="Loading model..."
        submessage="Initializing EMG inference engine. This may take a few seconds."
      />
    );
  }

  /* -- Loading screen -- */
  if (sessionStep === "loading") {
    return (
      <LoadingScreen
        message="Preparing session..."
        submessage="Setting up exercise tracking."
      />
    );
  }

  /* -- Pre-session countdown 3-2-1 -- */
  if (sessionStep === "countdown") {
    return (
      <CountdownScreen
        title="Get Ready"
        subtitle="Exercise begins now."
        onComplete={beginRecording}
      />
    );
  }

  /* -- Demo + Active session -- */
  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 flex">
        {/* Left — 3D Hand */}
        <div className="flex-[2] relative">
          <HandViewer
            mode={handMode}
            liveState={liveState}
            exercise={demoPhase ? activeExercise : null}
            className="w-full h-full"
          />
          {/* Demo countdown banner at top */}
          {demoPhase && (
            <div className="absolute top-0 left-0 right-0 flex justify-center pt-4">
              <div className="bg-black/60 backdrop-blur-sm rounded-xl px-6 py-3 flex items-center gap-4">
                <div className="text-center">
                  <p className="text-small text-muted">Watch the motion</p>
                  {activeExercise && (
                    <p className="text-body font-semibold text-text">{activeExercise.name}</p>
                  )}
                </div>
                <div className="text-h1 font-bold font-mono text-text">
                  {demoCountdown}
                </div>
                <button
                  onClick={() => {
                    if (demoTimerRef.current) clearInterval(demoTimerRef.current);
                    setSessionStep("countdown");
                  }}
                  className="ml-2 px-3 py-1 text-small text-muted hover:text-text border border-white/20 rounded-lg transition-colors"
                >
                  Skip
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Right — Info panel */}
        <div className="w-[300px] bg-panel border-l border-border p-5 flex flex-col gap-4 overflow-auto">
          <div className="text-center py-3 border-b border-border">
            <p className="text-h1 font-bold font-mono text-text">
              {demoPhase ? "--:--" : formatTime(elapsed)}
            </p>
            <p className="text-small text-muted mt-1">
              {demoPhase ? "Demo mode" : "Session time"}
            </p>
          </div>

          {!demoPhase && (
            <>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-small text-muted">State</span>
                  <Badge variant={stateBadgeVariant}>
                    {displayState}
                  </Badge>
                </div>
                {(activeExercise?.holdSeconds ?? 0) > 0 && isHolding && (
                  <p className="text-small text-text">
                    Hold steady{holdSeconds > 0 ? ` ... ${holdSeconds}` : ""}
                  </p>
                )}
              </div>

              <div>
                <p className="text-small text-muted mb-2">
                  Reps <span className="font-mono">{repCount}/{totalReps}</span>
                </p>
                <div className="flex gap-1.5 flex-wrap">
                  {repCircles.map((done, i) => (
                    <div
                      key={i}
                      className={`w-6 h-6 rounded flex items-center justify-center text-[10px] font-mono font-medium ${
                        done
                          ? "bg-purple text-white"
                          : "bg-white/[0.06] text-muted"
                      }`}
                    >
                      {i + 1}
                    </div>
                  ))}
                </div>
              </div>

              {/* Scoring */}
              <div className="flex items-center justify-between">
                <span className="text-small text-muted">Accuracy</span>
                <span className="font-mono text-body text-text">
                  {(() => {
                    const totalAttempts = repCount + negativeRepCount;
                    if (totalAttempts === 0) return "—";
                    return `${Math.round((repCount / totalAttempts) * 100)}%`;
                  })()}
                </span>
              </div>

              {negativeRepCount > 0 && (
                <div className="flex items-center justify-between">
                  <span className="text-small text-muted">Missed</span>
                  <span className="font-mono text-body text-danger">
                    {negativeRepCount}
                  </span>
                </div>
              )}

              <ProgressBar
                label="Stability"
                value={
                  frame?.classifierConfidence
                    ? frame.classifierConfidence * 100
                    : 0
                }
              />

              {/* Live EMG bars */}
              <div className="space-y-2 pt-2 border-t border-border">
                <p className="text-small text-muted">Live EMG</p>
                <div className="space-y-1.5">
                  {(frame?.emg ?? [0, 0, 0, 0]).slice(0, 4).map((val: number, i: number) => {
                    const normalized = Math.min(100, (Math.abs(val) / 100) * 100);
                    return (
                      <div key={i} className="flex items-center gap-2">
                        <span className="text-[10px] font-mono text-muted w-6">
                          CH{i + 1}
                        </span>
                        <div className="flex-1 h-2 bg-white/[0.06] rounded-full overflow-hidden">
                          <div
                            className="h-full bg-white rounded-full transition-all duration-150"
                            style={{ width: `${normalized}%` }}
                          />
                        </div>
                        <span className="text-[10px] font-mono text-muted w-10 text-right">
                          {(val ?? 0).toFixed(0)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="mt-auto pt-4">
                <Button variant="danger" onClick={endSession} className="w-full">
                  End Session
                </Button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
