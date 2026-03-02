import { Router, Request, Response } from "express";
import { getDb } from "../db";

const router = Router();

// GET /therapist/patients
router.get("/patients", (_req: Request, res: Response) => {
  const db = getDb();
  const patients = db
    .prepare(
      `SELECT p.*,
        (SELECT COUNT(*) FROM sessions s WHERE s.patient_id = p.id AND s.ended_at IS NOT NULL) as session_count,
        (SELECT overall_score FROM sessions s WHERE s.patient_id = p.id AND s.ended_at IS NOT NULL ORDER BY s.started_at DESC LIMIT 1) as last_score,
        (SELECT avg_stability FROM sessions s WHERE s.patient_id = p.id AND s.ended_at IS NOT NULL ORDER BY s.started_at DESC LIMIT 1) as last_stability
       FROM patients p
       WHERE p.pin != '__demo__'`
    )
    .all() as any[];

  // Compute alert badges
  const enriched = patients.map((p) => {
    const alerts: string[] = [];

    // Check for progression readiness
    const recentSessions = db
      .prepare(
        `SELECT completion_rate, avg_stability FROM sessions
         WHERE patient_id = ? AND ended_at IS NOT NULL
         ORDER BY started_at DESC LIMIT 3`
      )
      .all(p.id) as any[];

    if (recentSessions.length >= 3) {
      const allHighCompletion = recentSessions.every(
        (s: any) => s.completion_rate >= 90
      );
      const allHighStability = recentSessions.every(
        (s: any) => s.avg_stability >= 80
      );
      if (allHighCompletion && allHighStability) {
        alerts.push("Ready to progress");
      }
    }

    // Check for declining stability
    if (recentSessions.length >= 2) {
      const [latest, prev] = recentSessions;
      if (
        latest.avg_stability != null &&
        prev.avg_stability != null &&
        latest.avg_stability < prev.avg_stability - 15
      ) {
        alerts.push("Declining stability");
      }
    }

    // Check for pending recommendations
    const pendingRec = db
      .prepare(
        "SELECT COUNT(*) as cnt FROM recommendations WHERE patient_id = ? AND approved = 0"
      )
      .get(p.id) as any;
    if (pendingRec.cnt > 0) {
      alerts.push("Pending recommendation");
    }

    return { ...p, alerts };
  });

  res.json(enriched);
});

// PUT /therapist/patients/:id/settings
router.put("/patients/:id/settings", (req: Request, res: Response) => {
  const db = getDb();
  const { assist_level, target_closure, hold_duration, rep_count } = req.body;

  const updates: string[] = [];
  const values: any[] = [];

  if (assist_level !== undefined) {
    updates.push("assist_level = ?");
    values.push(Math.max(1, Math.min(5, Number(assist_level))));
  }
  if (target_closure !== undefined) {
    updates.push("target_closure = ?");
    values.push(Math.max(0, Math.min(100, Number(target_closure))));
  }
  if (hold_duration !== undefined) {
    updates.push("hold_duration = ?");
    values.push(Number(hold_duration));
  }
  if (rep_count !== undefined) {
    updates.push("rep_count = ?");
    values.push(Number(rep_count));
  }

  if (updates.length === 0) {
    res.status(400).json({ error: "No settings to update" });
    return;
  }

  values.push(req.params.id);
  db.prepare(`UPDATE patients SET ${updates.join(", ")} WHERE id = ?`).run(
    ...values
  );

  const patient = db
    .prepare("SELECT * FROM patients WHERE id = ?")
    .get(req.params.id);
  res.json(patient);
});

// POST /therapist/patients/:id/approve-recommendation
router.post(
  "/patients/:id/approve-recommendation",
  (req: Request, res: Response) => {
    const db = getDb();
    const { recommendation_id, approved } = req.body;

    if (recommendation_id == null) {
      res.status(400).json({ error: "recommendation_id required" });
      return;
    }

    const rec = db
      .prepare(
        "SELECT * FROM recommendations WHERE id = ? AND patient_id = ?"
      )
      .get(recommendation_id, req.params.id) as any;

    if (!rec) {
      res.status(404).json({ error: "Recommendation not found" });
      return;
    }

    db.prepare("UPDATE recommendations SET approved = ? WHERE id = ?").run(
      approved ? 1 : -1,
      recommendation_id
    );

    // If approved, apply the recommendation
    if (approved && rec.type === "reduce_assist") {
      const patient = db
        .prepare("SELECT assist_level FROM patients WHERE id = ?")
        .get(req.params.id) as any;
      if (patient && patient.assist_level > 1) {
        db.prepare(
          "UPDATE patients SET assist_level = ? WHERE id = ?"
        ).run(patient.assist_level - 1, req.params.id);
      }
    } else if (approved && rec.type === "increase_target") {
      const patient = db
        .prepare("SELECT target_closure FROM patients WHERE id = ?")
        .get(req.params.id) as any;
      if (patient && patient.target_closure < 100) {
        db.prepare(
          "UPDATE patients SET target_closure = ? WHERE id = ?"
        ).run(Math.min(100, patient.target_closure + 10), req.params.id);
      }
    }

    res.json({ ok: true });
  }
);

// GET /therapist/patients/:id — detailed patient view with history
router.get("/patients/:id", (req: Request, res: Response) => {
  const db = getDb();
  const patient = db
    .prepare("SELECT * FROM patients WHERE id = ?")
    .get(req.params.id);

  if (!patient) {
    res.status(404).json({ error: "Patient not found" });
    return;
  }

  const sessions = db
    .prepare(
      `SELECT * FROM sessions WHERE patient_id = ? AND ended_at IS NOT NULL
       ORDER BY started_at DESC LIMIT 50`
    )
    .all(req.params.id);

  const safetyEvents = db
    .prepare(
      `SELECT se.* FROM safety_events se
       JOIN sessions s ON se.session_id = s.id
       WHERE s.patient_id = ?
       ORDER BY se.timestamp DESC LIMIT 100`
    )
    .all(req.params.id);

  const recommendations = db
    .prepare(
      "SELECT * FROM recommendations WHERE patient_id = ? ORDER BY created_at DESC"
    )
    .all(req.params.id);

  res.json({ patient, sessions, safetyEvents, recommendations });
});

// GET /therapist/patients/:id/exercises
router.get("/patients/:id/exercises", (req: Request, res: Response) => {
  const db = getDb();

  // Ensure table exists
  db.exec(
    `CREATE TABLE IF NOT EXISTS exercise_programmes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id INTEGER NOT NULL REFERENCES patients(id),
      exercises TEXT NOT NULL DEFAULT '[]',
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    )`
  );

  const row = db
    .prepare(
      "SELECT exercises FROM exercise_programmes WHERE patient_id = ? ORDER BY updated_at DESC LIMIT 1"
    )
    .get(req.params.id) as any;

  if (!row) {
    res.json([]);
    return;
  }

  try {
    res.json(JSON.parse(row.exercises));
  } catch {
    res.json([]);
  }
});

// PUT /therapist/patients/:id/exercises
router.put("/patients/:id/exercises", (req: Request, res: Response) => {
  const db = getDb();
  const { exercises } = req.body;

  if (!Array.isArray(exercises)) {
    res.status(400).json({ error: "exercises must be an array" });
    return;
  }

  // Ensure table exists
  db.exec(
    `CREATE TABLE IF NOT EXISTS exercise_programmes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id INTEGER NOT NULL REFERENCES patients(id),
      exercises TEXT NOT NULL DEFAULT '[]',
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    )`
  );

  const existing = db
    .prepare(
      "SELECT id FROM exercise_programmes WHERE patient_id = ?"
    )
    .get(req.params.id) as any;

  const exercisesJson = JSON.stringify(exercises);

  if (existing) {
    db.prepare(
      "UPDATE exercise_programmes SET exercises = ?, updated_at = datetime('now') WHERE patient_id = ?"
    ).run(exercisesJson, req.params.id);
  } else {
    db.prepare(
      "INSERT INTO exercise_programmes (patient_id, exercises) VALUES (?, ?)"
    ).run(req.params.id, exercisesJson);
  }

  res.json({ ok: true });
});

// DELETE /therapist/patients/:id — remove a patient and all related data
router.delete("/patients/:id", (req: Request, res: Response) => {
  const db = getDb();
  const patient = db
    .prepare("SELECT * FROM patients WHERE id = ?")
    .get(req.params.id);

  if (!patient) {
    res.status(404).json({ error: "Patient not found" });
    return;
  }

  // Delete child records (FK order)
  db.prepare(
    `DELETE FROM safety_events WHERE session_id IN
     (SELECT id FROM sessions WHERE patient_id = ?)`
  ).run(req.params.id);
  db.prepare(
    `DELETE FROM metrics WHERE session_id IN
     (SELECT id FROM sessions WHERE patient_id = ?)`
  ).run(req.params.id);
  db.prepare(
    `DELETE FROM reps WHERE session_id IN
     (SELECT id FROM sessions WHERE patient_id = ?)`
  ).run(req.params.id);
  db.prepare("DELETE FROM sessions WHERE patient_id = ?").run(req.params.id);
  db.prepare("DELETE FROM recommendations WHERE patient_id = ?").run(req.params.id);
  db.prepare("DELETE FROM exercise_programmes WHERE patient_id = ?").run(req.params.id);
  db.prepare("DELETE FROM patients WHERE id = ?").run(req.params.id);

  res.json({ ok: true });
});

// POST /therapist/patients — create a new patient
router.post("/patients", (req: Request, res: Response) => {
  const db = getDb();
  const { name, pin, description, assist_level, dob, hospital } = req.body;

  if (!name || !pin) {
    res.status(400).json({ error: "name and pin are required" });
    return;
  }

  // Check for duplicate pin
  const existing = db
    .prepare("SELECT id FROM patients WHERE pin = ?")
    .get(pin) as any;
  if (existing) {
    res.status(409).json({ error: "A patient with this PIN already exists" });
    return;
  }

  const level = Math.max(1, Math.min(5, Number(assist_level) || 3));

  const result = db
    .prepare(
      "INSERT INTO patients (name, pin, assist_level, description, dob, hospital) VALUES (?, ?, ?, ?, ?, ?)"
    )
    .run(name, pin, level, description || "", dob || "", hospital || "");

  const patient = db
    .prepare("SELECT * FROM patients WHERE id = ?")
    .get(result.lastInsertRowid);

  res.json(patient);
});

export default router;
