import { Router, Request, Response } from "express";
import { getDb } from "../db";

const router = Router();

router.post("/login", (req: Request, res: Response) => {
  const { pin } = req.body;
  if (!pin || typeof pin !== "string") {
    res.status(400).json({ error: "PIN required" });
    return;
  }

  const db = getDb();

  // Check patients first
  const patient = db
    .prepare("SELECT id, name FROM patients WHERE pin = ?")
    .get(pin) as { id: number; name: string } | undefined;

  if (patient) {
    res.json({ role: "patient", id: patient.id, name: patient.name });
    return;
  }

  // Check therapists
  const therapist = db
    .prepare("SELECT id, name FROM therapists WHERE pin = ?")
    .get(pin) as { id: number; name: string } | undefined;

  if (therapist) {
    res.json({ role: "therapist", id: therapist.id, name: therapist.name });
    return;
  }

  res.status(401).json({ error: "Invalid PIN" });
});

// POST /auth/simulation — get or create the dedicated demo patient
router.post("/simulation", (_req: Request, res: Response) => {
  const db = getDb();
  const DEMO_PIN = "__demo__";

  let demo = db
    .prepare("SELECT id, name FROM patients WHERE pin = ?")
    .get(DEMO_PIN) as { id: number; name: string } | undefined;

  if (!demo) {
    const result = db
      .prepare(
        "INSERT INTO patients (name, pin, assist_level, description) VALUES (?, ?, ?, ?)"
      )
      .run("Demo Patient", DEMO_PIN, 3, "Simulation demo patient.");
    demo = db
      .prepare("SELECT id, name FROM patients WHERE id = ?")
      .get(result.lastInsertRowid) as { id: number; name: string };
  }

  res.json({ role: "patient", id: demo.id, name: demo.name });
});

export default router;
