import { Router, Request, Response } from "express";
import path from "path";
import { getDb } from "../db";
import { calibBridge, serial } from "../shared";

const router = Router();

/* ── Routes ── */

router.post("/start", (req: Request, res: Response) => {
  const mode = req.body.mode === "full" ? "full" : "quick" as const;
  const patientId = req.body.patient_id || "default";

  // Look up patient assist_level if we have a patient_id
  let assistLevel = 3;
  if (patientId !== "default") {
    const db = getDb();
    const patient = db.prepare("SELECT assist_level FROM patients WHERE id = ?").get(patientId) as any;
    if (patient) assistLevel = patient.assist_level ?? 3;
  }

  // Release serial port so calibrate_patient.py can open it.
  // Close unconditionally — port may be mid-connect from previous session cleanup.
  serial.close();
  console.log("[SERIAL] Releasing port for calibration");

  calibBridge.start({
    port: process.env.EMG_PORT || process.env.SERIAL_PORT || "/dev/cu.usbmodem176627901",
    model: process.env.MODEL_PATH || path.join(__dirname, "..", "..", "..", "exohand_model.pkl"),
    patientId: String(patientId),
    mode,
    assistLevel,
  });

  res.json({ ok: true, mode, totalPhases: mode === "full" ? 6 : 3 });
});

/** Client calls this after showing pre_phase + countdown screens */
router.post("/phase-ready", (_req: Request, res: Response) => {
  calibBridge.phaseReady();
  res.json({ ok: true });
});

router.post("/stop", (_req: Request, res: Response) => {
  calibBridge.stop();
  res.json({ ok: true });
});

router.get("/status", (_req: Request, res: Response) => {
  res.json(calibBridge.getStatus());
});

export default router;
