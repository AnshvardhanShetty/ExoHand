import { Router, Request, Response } from "express";
import path from "path";
import fs from "fs";
import { getDb } from "../db";
import { computeSessionScore } from "../scoring";
import { bridge, stateMachine, setExerciseTracker, serial } from "../shared";
import { ExerciseTracker } from "../exercise/tracker";

const router = Router();

// GET /sessions/bridge-status — must be before /:id routes
router.get("/bridge-status", (_req: Request, res: Response) => {
  res.json({
    running: bridge.isRunning(),
    ready: bridge.isReady(),
    error: bridge.getError(),
  });
});

// POST /sessions/start
router.post("/start", (req: Request, res: Response) => {
  const { patient_id, exercise } = req.body;
  if (!patient_id) {
    res.status(400).json({ error: "patient_id required" });
    return;
  }

  const db = getDb();

  // Look up patient for assist_level
  const patient = db.prepare("SELECT * FROM patients WHERE id = ?").get(patient_id) as any;

  const result = db
    .prepare("INSERT INTO sessions (patient_id) VALUES (?)")
    .run(patient_id);
  const session = db
    .prepare("SELECT * FROM sessions WHERE id = ?")
    .get(result.lastInsertRowid) as any;

  // Save exercise type
  if (exercise?.id) {
    db.prepare("UPDATE sessions SET exercise_type = ? WHERE id = ?")
      .run(exercise.id, session.id);
  }

  // Set session ID for safety event logging
  stateMachine.setSessionId(session.id);

  // Create exercise tracker if exercise info was provided
  if (exercise && exercise.id && exercise.category && exercise.reps) {
    const tracker = new ExerciseTracker(exercise, session.id);
    tracker.on("rep", (rep: any) => {
      console.log(
        `[EXERCISE] Rep ${rep.repNumber}/${rep.totalReps} — ` +
        `accuracy: ${rep.accuracy.toFixed(0)}%, stability: ${rep.stability.toFixed(0)}%, ` +
        `time: ${rep.timeToTarget.toFixed(1)}s, success: ${rep.success}`
      );
    });
    tracker.on("complete", () => {
      console.log("[EXERCISE] All reps completed!");
    });
    setExerciseTracker(tracker);
  }

  // Write exercise type + session mode to temp files for simulator auto-detection
  const exerciseType = exercise?.id || "close";
  try {
    fs.writeFileSync("/tmp/exohand_exercise.txt", exerciseType);
    fs.writeFileSync("/tmp/exohand_gesture.txt", "session");
  } catch {
    // Non-critical — only needed when using simulator
  }

  // Start PythonBridge for real-time EMG inference
  const assistLevel = patient?.assist_level ?? 3;
  const projectRoot = path.resolve(__dirname, "..", "..", "..");
  const calModelPath = path.join(projectRoot, "calibrations", String(patient_id), "calibrated_model.pkl");
  const modelPath = fs.existsSync(calModelPath)
    ? calModelPath
    : (process.env.MODEL_PATH || path.join(projectRoot, "exohand_model.pkl"));

  // Release serial port so Python can own it during the session.
  // Close unconditionally — port may be mid-connect from previous session cleanup.
  serial.close();
  console.log("[SERIAL] Releasing port for session");

  bridge.start({
    port: process.env.EMG_PORT || process.env.SERIAL_PORT || "/dev/cu.usbmodem176627901",
    model: modelPath,
    assistLevel,
    patientId: String(patient_id),
  });

  res.json(session);
});

// POST /sessions/:id/end
router.post("/:id/end", (req: Request, res: Response) => {
  const db = getDb();
  const session = db
    .prepare("SELECT * FROM sessions WHERE id = ?")
    .get(req.params.id) as any;

  if (!session) {
    res.status(404).json({ error: "Session not found" });
    return;
  }

  // Stop PythonBridge, exercise tracker, and motor state machine
  bridge.stop();
  setExerciseTracker(null);
  stateMachine.setSessionId(null);
  stateMachine.stop();

  // Compute scoring from reps
  const reps = db
    .prepare("SELECT * FROM reps WHERE session_id = ? ORDER BY rep_number")
    .all(req.params.id) as any[];

  const scores = computeSessionScore(reps);

  const exerciseDuration = req.body?.exercise_duration ?? null;

  db.prepare(
    `UPDATE sessions
     SET ended_at = datetime('now'),
         overall_score = ?,
         completion_rate = ?,
         avg_stability = ?,
         avg_accuracy = ?,
         exercise_duration = ?
     WHERE id = ?`
  ).run(
    scores.overallScore,
    scores.completionRate,
    scores.avgStability,
    scores.avgAccuracy,
    exerciseDuration,
    req.params.id
  );

  const updated = db
    .prepare("SELECT * FROM sessions WHERE id = ?")
    .get(req.params.id);
  res.json(updated);
});

// GET /sessions/:id/summary
router.get("/:id/summary", (req: Request, res: Response) => {
  const db = getDb();
  const session = db
    .prepare("SELECT * FROM sessions WHERE id = ?")
    .get(req.params.id) as any;

  if (!session) {
    res.status(404).json({ error: "Session not found" });
    return;
  }

  const reps = db
    .prepare("SELECT * FROM reps WHERE session_id = ? ORDER BY rep_number")
    .all(req.params.id);

  // Get previous session for comparison
  const prevSession = db
    .prepare(
      `SELECT * FROM sessions
       WHERE patient_id = ? AND id < ? AND ended_at IS NOT NULL
       ORDER BY id DESC LIMIT 1`
    )
    .get(session.patient_id, session.id) as any;

  // Get pending recommendation
  const recommendation = db
    .prepare(
      `SELECT * FROM recommendations
       WHERE patient_id = ? AND approved = 0
       ORDER BY created_at DESC LIMIT 1`
    )
    .get(session.patient_id) as any;

  res.json({
    session,
    reps,
    previous: prevSession || null,
    recommendation: recommendation || null,
  });
});

// POST /sessions/:id/reps — record a rep
router.post("/:id/reps", (req: Request, res: Response) => {
  const { rep_number, accuracy, stability, time_to_target, success } = req.body;
  const db = getDb();

  db.prepare(
    `INSERT INTO reps (session_id, rep_number, accuracy, stability, time_to_target, success)
     VALUES (?, ?, ?, ?, ?, ?)`
  ).run(req.params.id, rep_number, accuracy, stability, time_to_target, success ? 1 : 0);

  res.json({ ok: true });
});

export default router;
