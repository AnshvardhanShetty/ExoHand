import { getDb } from "../db";

interface SessionData {
  id: number;
  completion_rate: number;
  avg_stability: number;
}

// Check if a patient qualifies for a progression recommendation
export function checkForRecommendation(patientId: number): void {
  const db = getDb();

  // Get last 3 completed sessions
  const sessions = db
    .prepare(
      `SELECT id, completion_rate, avg_stability FROM sessions
       WHERE patient_id = ? AND ended_at IS NOT NULL
       ORDER BY started_at DESC LIMIT 3`
    )
    .all(patientId) as SessionData[];

  if (sessions.length < 3) return;

  // Check safety events in these sessions
  const sessionIds = sessions.map((s) => s.id);
  const safetyCount = db
    .prepare(
      `SELECT COUNT(*) as cnt FROM safety_events
       WHERE session_id IN (${sessionIds.map(() => "?").join(",")})
       AND event_type = 'fail_safe_open'`
    )
    .get(...sessionIds) as { cnt: number };

  if (safetyCount.cnt > 0) return;

  // Check qualification: completion >= 90%, stability >= 80 for all 3
  const allQualify = sessions.every(
    (s) => s.completion_rate >= 90 && s.avg_stability >= 80
  );

  if (!allQualify) return;

  // Check if there's already a pending recommendation
  const existing = db
    .prepare(
      "SELECT id FROM recommendations WHERE patient_id = ? AND approved = 0"
    )
    .get(patientId);

  if (existing) return;

  // Get current patient settings to decide recommendation type
  const patient = db
    .prepare("SELECT assist_level, target_closure FROM patients WHERE id = ?")
    .get(patientId) as { assist_level: number; target_closure: number };

  if (!patient) return;

  let type: string;
  let message: string;

  if (patient.assist_level > 1) {
    type = "reduce_assist";
    message = `Great progress! Your last 3 sessions show strong performance. Consider reducing assist level from ${patient.assist_level} to ${patient.assist_level - 1}.`;
  } else if (patient.target_closure < 100) {
    type = "increase_target";
    const newTarget = Math.min(100, patient.target_closure + 10);
    message = `Excellent work! You're performing well at current settings. Consider increasing target closure from ${patient.target_closure}% to ${newTarget}%.`;
  } else {
    // Already at max difficulty
    return;
  }

  db.prepare(
    "INSERT INTO recommendations (patient_id, type, message) VALUES (?, ?, ?)"
  ).run(patientId, type, message);
}
