import { Router, Request, Response } from "express";
import path from "path";
import { getDb } from "../db";
import { calibBridge, serial } from "../shared";

const router = Router();

/* ── Routes ── */

router.post("/start", (req: Request, res: Response) => {
  const rawMode = req.body.mode;
  const mode: "full" | "quick" | "paper22s" =
    rawMode === "full" ? "full" :
    rawMode === "paper22s" ? "paper22s" :
    "quick";
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

  // Default serial port: macOS-style device path. Windows users must set
  // EMG_PORT (e.g. "COM4") or SERIAL_PORT. If neither is set on non-macOS,
  // reject with a clear message rather than pass an invalid path to Python.
  const explicitPort = process.env.EMG_PORT || process.env.SERIAL_PORT;
  const port = explicitPort || (process.platform === "darwin"
    ? "/dev/cu.usbmodem176627901"
    : "");
  if (!port) {
    res.status(400).json({
      ok: false,
      error: `EMG_PORT (or SERIAL_PORT) env var must be set on ${process.platform}. Example: EMG_PORT=COM4 npm start`,
    });
    return;
  }

  calibBridge.start({
    port,
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
