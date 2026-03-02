import express from "express";
import cors from "cors";
import path from "path";
import { createServer } from "http";
import { getDb } from "./db";
import authRouter from "./routes/auth";
import patientsRouter from "./routes/patients";
import sessionsRouter from "./routes/sessions";
import therapistRouter from "./routes/therapist";
import calibrationRouter from "./routes/calibration";
import { stateMachine, wsManager, bridge, calibBridge, serial, getExerciseTracker } from "./shared";

const PORT = process.env.PORT ? parseInt(process.env.PORT) : 3001;

const app = express();
app.use(cors());
app.use(express.json());

// Initialize database
getDb();

// Health check
app.get("/health", (_req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

// Routes
app.use("/auth", authRouter);
app.use("/patients", patientsRouter);
app.use("/sessions", sessionsRouter);
app.use("/therapist", therapistRouter);
app.use("/calibration", calibrationRouter);

// Serial connection to Teensy (skipped unless SERIAL_PORT is explicitly set)
const SERIAL_PORT = process.env.SERIAL_PORT || "";

stateMachine.on("serial", (cmd: string) => {
  if (bridge.isRunning()) {
    bridge.sendCommand(cmd);
  } else {
    serial.sendMotorCommand(cmd);
  }
  console.log(`[MOTOR] ${cmd.trim()}`);
});

serial.on("connected", () => console.log(`[SERIAL] Connected to ${SERIAL_PORT}`));
serial.on("error", (err: Error) => console.error(`[SERIAL] Error: ${err.message}`));
serial.on("data", (line: string) => console.log(`[TEENSY] ${line}`));
serial.on("mock", (msg: string) => console.warn(`[SERIAL] ${msg}`));

if (SERIAL_PORT) {
  serial.connect(SERIAL_PORT);
} else {
  console.log("[SERIAL] No SERIAL_PORT set — running without hardware (simulation mode)");
}

// ── Debug harness ──
let debugTickCount = 0;
let lastIntent = "rest";

function debugLog(data: {
  intent: string;
  repCount: number;
  negativeRepCount: number;
  sessionFailed: boolean;
  stateCmd: string;
  confidence: number;
  trackerActive: boolean;
}) {
  debugTickCount++;
  const intentChanged = data.intent !== lastIntent;
  if (intentChanged) {
    lastIntent = data.intent;
  }
  // Log every 20th tick (~5s at 4Hz) or on intent changes
  if (intentChanged || debugTickCount % 20 === 0) {
    console.log(
      `[DEBUG] tick=${debugTickCount} intent=${data.intent} state=${data.stateCmd} ` +
      `reps=${data.repCount} missed=${data.negativeRepCount} failed=${data.sessionFailed} ` +
      `conf=${data.confidence.toFixed(2)} tracker=${data.trackerActive}`
    );
  }
}

// Reset tracker when client sends startSession (R0: prevent phantom reps during demo)
wsManager.onSessionStart(() => {
  const tracker = getExerciseTracker();
  if (tracker) {
    tracker.reset();
    console.log("[EXERCISE] Tracker reset on session start (R0)");
  }
  debugTickCount = 0;
  lastIntent = "rest";
});

// Wire PythonBridge EMG readings → WebSocket frames + exercise tracker
// Raw intent passes straight through — state machine has its own rate limiting
// (1 change/sec, 700ms motion, 800ms cooldown) for stability
bridge.on("reading", (reading) => {
  // Only feed tracker when state machine is running (R0: prevents phantom reps during demo)
  const tracker = getExerciseTracker();
  const trackerActive = !!tracker && stateMachine.isRunning();

  // Clamp impossible intents: close exercise never produces "open", etc.
  const intent = trackerActive
    ? tracker!.filterIntent(reading.intent)
    : reading.intent;

  const repCount = trackerActive
    ? tracker!.onFrame(intent, reading.confidence)
    : 0;

  // Map intent to grip value (0=open, 0.5=rest, 1=close)
  const gripFromIntent =
    intent === "close" ? 1.0
    : intent === "open" ? 0.0
    : 0.5;

  wsManager.onEmgFrame({
    emg: reading.emg,
    classifierConfidence: reading.confidence,
    assistStrength: reading.assistStrength,
    repCount,
    negativeRepCount: trackerActive ? tracker!.getNegativeRepCount() : 0,
    sessionFailed: trackerActive ? tracker!.isFailed() : false,
    grip: gripFromIntent,
    intent,
  });

  // Debug harness logging
  const motorState = stateMachine.getState();
  debugLog({
    intent,
    repCount,
    negativeRepCount: trackerActive ? tracker!.getNegativeRepCount() : 0,
    sessionFailed: trackerActive ? tracker!.isFailed() : false,
    stateCmd: motorState.stateCmd,
    confidence: reading.confidence,
    trackerActive,
  });
});

bridge.on("log", (msg) => console.log(`[PYTHON] ${msg}`));
bridge.on("exit", (code) => {
  console.log(`[PYTHON] Process exited: ${code}`);
  // Reclaim serial port after session ends (only if hardware is configured)
  if (SERIAL_PORT && !serial.isConnected()) {
    console.log("[SERIAL] Reclaiming port after session");
    serial.connect(SERIAL_PORT);
    // Send return-to-rest after serial reconnects (motor safety)
    serial.once("connected", () => {
      serial.sendMotorCommand("A145\n");
      console.log("[MOTOR] Return to rest (post-session)");
    });
  }
});

// Wire CalibrationBridge EMG readings → WebSocket frames (live display during calibration)
calibBridge.on("reading", (reading) => {
  wsManager.onEmgFrame({
    emg: reading.emg,
    classifierConfidence: 0,
    assistStrength: 0,
    repCount: 0,
    negativeRepCount: 0,
    sessionFailed: false,
    grip: 0,
    intent: reading.intent,
  });
});

calibBridge.on("log", (msg) => console.log(`[CALIBRATION] ${msg}`));
calibBridge.on("progress", (p) => console.log(`[CALIBRATION] Phase ${p.phase} — ${p.gesture} (${p.percent.toFixed(0)}%)`));
calibBridge.on("complete", (data) => console.log(`[CALIBRATION] Complete: ${data.num_samples} samples`));
calibBridge.on("error", (data) => console.error(`[CALIBRATION] Error: ${data.message}`));
calibBridge.on("exit", (code) => {
  console.log(`[CALIBRATION] Process exited: ${code}`);
  // Reclaim serial port after calibration ends (only if hardware is configured)
  if (SERIAL_PORT && !serial.isConnected()) {
    console.log("[SERIAL] Reclaiming port after calibration");
    serial.connect(SERIAL_PORT);
  }
});

// When all browser tabs close, stop Python processes and motor state machine
wsManager.onAllDisconnected(() => {
  console.log("[WS] All clients disconnected — stopping bridges");
  bridge.stop();
  calibBridge.stop();
  stateMachine.stop();
});

// Create HTTP server and attach WebSocket
const server = createServer(app);
wsManager.attach(server);

// Start
server.listen(PORT, () => {
  console.log(`ExoHand server running on http://localhost:${PORT}`);
  console.log(`WebSocket available on ws://localhost:${PORT}`);
  console.log(`Serial port: ${SERIAL_PORT}`);
});

export { app, stateMachine, wsManager, bridge, calibBridge, SERIAL_PORT };
