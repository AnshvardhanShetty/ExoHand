/**
 * mockApi.ts
 *
 * Hardcoded responses for every endpoint in api.ts. Used when
 * VITE_DEMO_MODE=true so the frontend can run without a backend.
 *
 * Data is deliberately simple — one demo patient, one demo therapist,
 * two prior sessions, a small library of exercises. Any writes are
 * accepted no-op-style and returned as if they persisted.
 */

const DEMO_PATIENT = {
  id: 1,
  name: "Demo Patient",
  description: "Right-hemiparesis, chronic phase, first-year post-stroke.",
  assist_level: 3,
  pin: "0000",
  dob: "1962-03-14",
  hospital: "Demo Rehabilitation Centre",
  createdAt: "2026-01-08T09:00:00Z",
};

const DEMO_THERAPIST = {
  id: 100,
  name: "Dr. Demo",
  role: "therapist" as const,
};

const DEMO_EXERCISES = [
  {
    id: 1,
    name: "Close · 6 reps · light effort",
    description: "Six close-hold reps at light effort. Rest 5 s between reps.",
    reps: 6,
    hold_ms: 2000,
    rest_ms: 5000,
    class: "close",
    effort: "light",
  },
  {
    id: 2,
    name: "Open · 6 reps · light effort",
    description: "Six open-hold reps at light effort.",
    reps: 6,
    hold_ms: 2000,
    rest_ms: 5000,
    class: "open",
    effort: "light",
  },
  {
    id: 3,
    name: "Alternating · 8 reps",
    description: "Alternate close and open, 8 reps total.",
    reps: 8,
    hold_ms: 2000,
    rest_ms: 4000,
    class: "alternating",
    effort: "medium",
  },
];

const DEMO_SESSIONS = [
  {
    id: 101,
    patient_id: 1,
    date: "2026-06-27T10:15:00Z",
    exercise: "Close · 6 reps · light effort",
    duration_sec: 380,
    score: 82,
    reps: 6,
    completion: 0.95,
    stability: 0.87,
  },
  {
    id: 102,
    patient_id: 1,
    date: "2026-06-30T10:20:00Z",
    exercise: "Alternating · 8 reps",
    duration_sec: 420,
    score: 88,
    reps: 8,
    completion: 1.0,
    stability: 0.90,
  },
];

const DEMO_PROGRESS = {
  totalSessions: 12,
  totalReps: 84,
  averageScore: 84,
  trends: {
    score: { current: 88, previous: 82, delta: +6 },
    stability: { current: 0.90, previous: 0.85, delta: +0.05 },
    completion: { current: 1.0, previous: 0.95, delta: +0.05 },
  },
  weekly: [
    { week: "2026-06-01", sessions: 3, avgScore: 76 },
    { week: "2026-06-08", sessions: 4, avgScore: 79 },
    { week: "2026-06-15", sessions: 3, avgScore: 82 },
    { week: "2026-06-22", sessions: 2, avgScore: 85 },
  ],
};

const DEMO_RECOMMENDATIONS = [
  {
    id: 1,
    patient_id: 1,
    message: "Consider raising assist level from 3 to 4 based on the last two sessions.",
    created_at: "2026-07-01T08:00:00Z",
    approved: null,
  },
];

// Track a mutable "live" session id created by startSession().
let liveSessionId = 200;

// ── Calibration state machine (matches the client's expected shape) ───────────
//
// The Calibration.tsx page polls getCalibrationStatus() every 500 ms and drives
// the UI off the returned shape (loading_model → pre_phase → countdown →
// running → next phase → done). We simulate the same server-side state machine
// here so the demo runs through a full mock calibration in ~30 seconds without
// a backend or real hardware.

interface CalibPhase {
  name: string;
  instruction: string;
  durationSec: number;
  targetAngle: number;
}

const CALIB_PHASES: CalibPhase[] = [
  { name: "REST", instruction: "Relax your hand and rest.", durationSec: 5, targetAngle: 145 },
  { name: "CLOSE", instruction: "Close your hand and hold.", durationSec: 5, targetAngle: 180 },
  { name: "OPEN", instruction: "Open your hand and hold.", durationSec: 5, targetAngle: 110 },
];
const MODEL_LOAD_MS = 1200;

interface CalibState {
  running: boolean;
  completed: boolean;
  startedAt: number;
  currentPhaseIdx: number;
  phaseStartedAt: number; // set when calibrationPhaseReady() is called
  phaseReady: boolean;
  error: string | null;
}

let calibState: CalibState = {
  running: false,
  completed: false,
  startedAt: 0,
  currentPhaseIdx: 0,
  phaseStartedAt: 0,
  phaseReady: false,
  error: null,
};

function calibStatusSnapshot() {
  const totalPhaseDurations = CALIB_PHASES.reduce((s, p) => s + p.durationSec, 0);

  // Not running (idle) — return an inactive shape.
  if (!calibState.running && !calibState.completed) {
    return {
      active: false,
      completed: false,
      mode: "quick",
      modelLoaded: false,
      phaseIndex: 0,
      trialIndex: 0,
      totalPhases: CALIB_PHASES.length,
      phaseName: null,
      phaseInstruction: null,
      phaseTargetAngle: 0,
      phaseDurationSec: 0,
      phaseTrials: 1,
      phaseElapsedSec: 0,
      phaseProgress: 0,
      overallProgress: 0,
      remainingSec: totalPhaseDurations,
      phaseWaiting: false,
      error: null,
    };
  }

  // Completed — hold the "done" state briefly so the UI can react.
  if (calibState.completed) {
    return {
      active: false,
      completed: true,
      mode: "quick",
      modelLoaded: true,
      phaseIndex: CALIB_PHASES.length,
      trialIndex: 0,
      totalPhases: CALIB_PHASES.length,
      phaseName: null,
      phaseInstruction: null,
      phaseTargetAngle: 0,
      phaseDurationSec: 0,
      phaseTrials: 1,
      phaseElapsedSec: 0,
      phaseProgress: 1,
      overallProgress: 1,
      remainingSec: 0,
      phaseWaiting: false,
      error: null,
    };
  }

  const now = Date.now();
  const elapsedSinceStartMs = now - calibState.startedAt;

  // Simulate the model-loading window (~1.2 s) — modelLoaded stays false.
  if (elapsedSinceStartMs < MODEL_LOAD_MS) {
    return {
      active: true,
      completed: false,
      mode: "quick",
      modelLoaded: false,
      phaseIndex: 0,
      trialIndex: 0,
      totalPhases: CALIB_PHASES.length,
      phaseName: null,
      phaseInstruction: null,
      phaseTargetAngle: 0,
      phaseDurationSec: 0,
      phaseTrials: 1,
      phaseElapsedSec: 0,
      phaseProgress: 0,
      overallProgress: 0,
      remainingSec: totalPhaseDurations,
      phaseWaiting: false,
      error: null,
    };
  }

  // Advance the phase state machine if the current phase is finished.
  while (
    calibState.phaseReady &&
    calibState.currentPhaseIdx < CALIB_PHASES.length &&
    (now - calibState.phaseStartedAt) / 1000 >= CALIB_PHASES[calibState.currentPhaseIdx].durationSec
  ) {
    calibState.currentPhaseIdx += 1;
    calibState.phaseReady = false;
    calibState.phaseStartedAt = 0;
  }

  // All phases done → mark completed.
  if (calibState.currentPhaseIdx >= CALIB_PHASES.length) {
    calibState.completed = true;
    calibState.running = false;
    return calibStatusSnapshot();
  }

  const phase = CALIB_PHASES[calibState.currentPhaseIdx];
  const remainingBeforeThisPhase = CALIB_PHASES
    .slice(calibState.currentPhaseIdx)
    .reduce((s, p) => s + p.durationSec, 0);

  // Waiting for the client to acknowledge (pre_phase → countdown → calibrationPhaseReady).
  if (!calibState.phaseReady) {
    return {
      active: true,
      completed: false,
      mode: "quick",
      modelLoaded: true,
      phaseIndex: calibState.currentPhaseIdx,
      trialIndex: 0,
      totalPhases: CALIB_PHASES.length,
      phaseName: phase.name,
      phaseInstruction: phase.instruction,
      phaseTargetAngle: phase.targetAngle,
      phaseDurationSec: phase.durationSec,
      phaseTrials: 1,
      phaseElapsedSec: 0,
      phaseProgress: 0,
      overallProgress: calibState.currentPhaseIdx / CALIB_PHASES.length,
      remainingSec: remainingBeforeThisPhase,
      phaseWaiting: true,
      error: null,
    };
  }

  // Phase is actively running.
  const phaseElapsed = Math.min(
    phase.durationSec,
    (now - calibState.phaseStartedAt) / 1000
  );
  const phaseProgress = phaseElapsed / phase.durationSec;

  return {
    active: true,
    completed: false,
    mode: "quick",
    modelLoaded: true,
    phaseIndex: calibState.currentPhaseIdx,
    trialIndex: 0,
    totalPhases: CALIB_PHASES.length,
    phaseName: phase.name,
    phaseInstruction: phase.instruction,
    phaseTargetAngle: phase.targetAngle,
    phaseDurationSec: phase.durationSec,
    phaseTrials: 1,
    phaseElapsedSec: phaseElapsed,
    phaseProgress,
    overallProgress: (calibState.currentPhaseIdx + phaseProgress) / CALIB_PHASES.length,
    remainingSec: Math.max(0, remainingBeforeThisPhase - phaseElapsed),
    phaseWaiting: false,
    error: null,
  };
}

async function ok<T>(value: T, delayMs = 60): Promise<T> {
  // A slight delay so the UI's loading states are visible.
  await new Promise((r) => setTimeout(r, delayMs));
  return value;
}

export const mockApi = {
  login: async (pin: string) => {
    if (pin === "0000") return ok({ role: "patient", id: DEMO_PATIENT.id, name: DEMO_PATIENT.name });
    if (pin === "9999") return ok({ role: "therapist", id: DEMO_THERAPIST.id, name: DEMO_THERAPIST.name });
    throw new Error("Invalid PIN. Try 0000 (patient) or 9999 (therapist).");
  },

  startSimulation: async () =>
    ok({ role: "patient", id: DEMO_PATIENT.id, name: DEMO_PATIENT.name }),

  getPatient: async (_id: number) => ok(DEMO_PATIENT),
  updatePatient: async (_id: number, data: any) => ok({ ...DEMO_PATIENT, ...data }),
  getPatientSessions: async (_id: number) => ok(DEMO_SESSIONS),
  getPatientProgress: async (_id: number) => ok(DEMO_PROGRESS),

  startSession: async (_patientId: number, exercise?: any) => {
    const id = ++liveSessionId;
    return ok({ id, session_id: id, exercise });
  },
  endSession: async (sessionId: number, exerciseDuration?: number) =>
    ok({ id: sessionId, duration_sec: exerciseDuration ?? 0, saved: true }),
  getSessionSummary: async (sessionId: number) =>
    ok({
      id: sessionId,
      patient_id: DEMO_PATIENT.id,
      date: new Date().toISOString(),
      exercise: "Alternating · 8 reps",
      duration_sec: 420,
      score: 87,
      reps: 8,
      negativeReps: 8,
      completion: 1.0,
      stability: 0.89,
      averageConfidence: 0.88,
      classDistribution: { rest: 0.52, close: 0.24, open: 0.24 },
    }),
  recordRep: async (_sessionId: number, _rep: any) => ok({ ok: true }),

  getTherapistPatients: async () => ok([DEMO_PATIENT]),
  getTherapistPatientDetail: async (_id: number) =>
    ok({ ...DEMO_PATIENT, sessions: DEMO_SESSIONS, progress: DEMO_PROGRESS, recommendations: DEMO_RECOMMENDATIONS }),
  updatePatientSettings: async (_id: number, settings: any) => ok({ ...DEMO_PATIENT, ...settings }),
  approveRecommendation: async (_patientId: number, recommendationId: number, approved: boolean) =>
    ok({ id: recommendationId, approved }),

  getExercises: async (_patientId: number) => ok(DEMO_EXERCISES),
  saveExercises: async (_patientId: number, exercises: any[]) => ok({ saved: true, count: exercises.length }),

  createPatient: async (data: any) => ok({ id: 999, ...data }),
  deletePatient: async (_id: number) => ok({ ok: true }),

  startCalibration: async (mode: "full" | "quick", _patientId?: number) => {
    calibState = {
      running: true,
      completed: false,
      startedAt: Date.now(),
      currentPhaseIdx: 0,
      phaseStartedAt: 0,
      phaseReady: false,
      error: null,
    };
    return ok({ session_id: 1, mode }, 20);
  },
  stopCalibration: async () => {
    calibState = {
      running: false,
      completed: false,
      startedAt: 0,
      currentPhaseIdx: 0,
      phaseStartedAt: 0,
      phaseReady: false,
      error: null,
    };
    return ok({ ok: true }, 20);
  },
  calibrationPhaseReady: async () => {
    calibState.phaseReady = true;
    calibState.phaseStartedAt = Date.now();
    return ok({ ok: true }, 20);
  },
  // No artificial delay on status polls — the polling loop runs at 500 ms.
  getCalibrationStatus: async () => ok(calibStatusSnapshot(), 0),

  getBridgeStatus: async () => ok({ running: true, ready: true, error: null }),
};
