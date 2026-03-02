import { Router, Request, Response } from "express";
import { getDb } from "../db";

const router = Router();

// GET /patients/:id
router.get("/:id", (req: Request, res: Response) => {
  const db = getDb();
  const patient = db
    .prepare("SELECT * FROM patients WHERE id = ?")
    .get(req.params.id);
  if (!patient) {
    res.status(404).json({ error: "Patient not found" });
    return;
  }
  res.json(patient);
});

// PUT /patients/:id
router.put("/:id", (req: Request, res: Response) => {
  const db = getDb();
  const { name, assist_level, target_closure, hold_duration, rep_count } =
    req.body;
  const updates: string[] = [];
  const values: any[] = [];

  if (name !== undefined) {
    updates.push("name = ?");
    values.push(name);
  }
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
    res.status(400).json({ error: "No fields to update" });
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

// GET /patients/:id/sessions
router.get("/:id/sessions", (req: Request, res: Response) => {
  const db = getDb();
  const sessions = db
    .prepare(
      "SELECT * FROM sessions WHERE patient_id = ? ORDER BY started_at DESC"
    )
    .all(req.params.id);
  res.json(sessions);
});

// GET /patients/:id/progress
router.get("/:id/progress", (req: Request, res: Response) => {
  const db = getDb();
  const sessions = db
    .prepare(
      `SELECT id, started_at, overall_score, completion_rate, avg_stability, avg_accuracy, exercise_type
       FROM sessions WHERE patient_id = ? AND ended_at IS NOT NULL
       ORDER BY started_at ASC`
    )
    .all(req.params.id) as any[];

  // Compute trends
  const recent = sessions.slice(-5);
  const previous = sessions.slice(-10, -5);

  const avg = (arr: any[], field: string) => {
    const vals = arr.map((s) => s[field]).filter((v: any) => v != null);
    return vals.length ? vals.reduce((a: number, b: number) => a + b, 0) / vals.length : null;
  };

  res.json({
    sessions,
    trends: {
      score: {
        current: avg(recent, "overall_score"),
        previous: avg(previous, "overall_score"),
      },
      stability: {
        current: avg(recent, "avg_stability"),
        previous: avg(previous, "avg_stability"),
      },
      completion: {
        current: avg(recent, "completion_rate"),
        previous: avg(previous, "completion_rate"),
      },
    },
    totalSessions: sessions.length,
    thisWeek: sessions.filter((s) => {
      const d = new Date(s.started_at);
      const now = new Date();
      const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      return d >= weekAgo;
    }).length,
  });
});

export default router;
