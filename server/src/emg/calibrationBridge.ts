import { spawn, ChildProcess } from "child_process";
import { EventEmitter } from "events";
import path from "path";
import fs from "fs";

const GESTURE_FILE = "/tmp/exohand_gesture.txt";

export interface CalibrationProgress {
  phase: number;
  trial: number;
  total: number;
  gesture: string;
  remaining: number;
  percent: number;
}

/**
 * Status shape kept compatible with what Calibration.tsx polls.
 */
export interface CalibrationStatus {
  active: boolean;
  completed: boolean;
  mode: "full" | "quick";
  modelLoaded: boolean;
  phaseIndex: number;
  trialIndex: number;
  totalPhases: number;
  phaseName: string | null;
  phaseInstruction: string | null;
  phaseTargetAngle: number;
  phaseDurationSec: number;
  phaseTrials: number;
  phaseElapsedSec: number;
  phaseProgress: number;
  overallProgress: number;
  remainingSec: number;
  phaseWaiting: boolean;
  error: string | null;
  qualityReport: any | null;
}

const GESTURE_INSTRUCTIONS: Record<string, string> = {
  rest: "Relax your hand completely",
  close: "Close your hand",
  open: "Open your hand",
  loading: "Loading model...",
  connecting: "Connecting to sensor...",
  processing: "Processing calibration data...",
  complete: "Calibration complete!",
};

/** Uppercase display name for gestures */
const GESTURE_DISPLAY: Record<string, string> = {
  rest: "REST",
  close: "CLOSE",
  open: "OPEN",
  loading: "LOADING",
  connecting: "CONNECTING",
  processing: "PROCESSING",
  complete: "COMPLETE",
};

const GESTURE_ANGLES: Record<string, number> = {
  rest: 110,
  close: 180,
  open: 0,
};

// Estimated per-phase durations (seconds)
const PHASE_DURATIONS: Record<string, number> = {
  rest: 10,
  close: 30,
  open: 30,
  processing: 5,
};

export class CalibrationBridge extends EventEmitter {
  private process: ChildProcess | null = null;
  private buffer: string = "";
  private stderrBuffer: string = "";
  private status: CalibrationStatus = this.defaultStatus();
  private phaseStartedAt: number = 0;
  private currentGesture: string = "";
  private trialChangeCounter: number = 0;
  private gesturePhaseMap: Map<string, number> = new Map();
  private collectionStartAt: number = 0; // when phaseReady() was called — used for real-time remaining decrement

  private defaultStatus(): CalibrationStatus {
    return {
      active: false,
      completed: false,
      mode: "quick",
      modelLoaded: false,
      phaseIndex: 0,
      trialIndex: 0,
      totalPhases: 3,
      phaseName: null,
      phaseInstruction: null,
      phaseTargetAngle: 110,
      phaseDurationSec: 10,
      phaseTrials: 1,
      phaseElapsedSec: 0,
      phaseProgress: 0,
      overallProgress: 0,
      remainingSec: 0,
      phaseWaiting: false,
      error: null,
      qualityReport: null,
    };
  }

  start(options: {
    port: string;
    model: string;
    patientId: string;
    mode: "full" | "quick";
    assistLevel: number;
  }) {
    this.stop();

    const projectRoot = path.resolve(__dirname, "..", "..", "..");
    const script = path.join(projectRoot, "calibrate_patient.py");

    const args = [
      script,
      "--web-mode",
      "--port", options.port,
      "--model", options.model,
      "--patient-id", options.patientId,
      "--mode", options.mode,
      "--assist-level", String(options.assistLevel),
    ];

    const totalPhases = options.mode === "full" ? 6 : 3;

    this.status = {
      ...this.defaultStatus(),
      active: true,
      mode: options.mode,
      totalPhases,
      phaseName: "STARTING",
      phaseInstruction: "Loading model...",
      remainingSec: options.mode === "full" ? 360 : 90,
      modelLoaded: false,
      phaseWaiting: false, // Will become true once Python reports first real progress
    };

    // Signal simulator to stream rest data during model loading.
    // Delay slightly so the simulator processes the "session" reset from stop() first.
    setTimeout(() => this.writeGestureFile("rest"), 600);

    this.process = spawn("python3", args, {
      cwd: projectRoot,
      stdio: ["pipe", "pipe", "pipe"],
    });

    this.process.stdout?.on("data", (data: Buffer) => {
      this.buffer += data.toString();
      const lines = this.buffer.split("\n");
      this.buffer = lines.pop() || "";

      for (const line of lines) {
        this.parseLine(line.trim());
      }
    });

    this.process.stderr?.on("data", (data: Buffer) => {
      const text = data.toString();
      this.stderrBuffer += text;
      this.emit("log", text);
    });

    this.process.on("exit", (code) => {
      if (this.status.active && !this.status.completed) {
        this.status.active = false;
        const lastLines = this.stderrBuffer.trim().split("\n").slice(-3).join(" ");
        this.status.error = lastLines || `Calibration process exited with code ${code}`;
      }
      this.process = null;
      this.emit("exit", code);
    });
  }

  private parseLine(line: string) {
    if (!line.startsWith("{")) return;

    try {
      const data = JSON.parse(line);

      if (data.type === "progress") {
        const gesture = data.gesture || "rest";

        // Detect phase/gesture change → reset phase timer
        if (gesture !== this.currentGesture) {
          this.currentGesture = gesture;
          this.phaseStartedAt = Date.now();
          this.status.phaseDurationSec = PHASE_DURATIONS[gesture] ?? 10;
          this.status.phaseElapsedSec = 0;
        }

        this.status.phaseName = GESTURE_DISPLAY[gesture] || gesture.toUpperCase();
        this.status.phaseInstruction = GESTURE_INSTRUCTIONS[gesture] || gesture;
        this.status.phaseTargetAngle = GESTURE_ANGLES[gesture] ?? 110;
        this.status.overallProgress = (data.percent || 0) / 100;
        this.status.remainingSec = Math.round(data.remaining || 0);
        this.status.trialIndex = data.trial || 0;
        this.status.phaseProgress = data.total > 0 ? data.trial / data.total : 0;

        // Map Python phase numbers to frontend phaseIndex
        if (data.phase !== undefined) {
          this.status.phaseIndex = data.phase;
        }

        // First real trial progress means model is loaded and serial is connected.
        // "loading" and "connecting" are pre-trial setup stages.
        const isSetupGesture = gesture === "loading" || gesture === "connecting";
        if (!this.status.modelLoaded && !isSetupGesture) {
          this.status.modelLoaded = true;
          this.status.phaseWaiting = true; // Signal frontend to show pre_phase
        }

        this.emit("progress", data as CalibrationProgress);
      } else if (data.type === "trial_start") {
        // New trial beginning — pause for UI instruction + countdown
        const gesture = data.gesture || "rest";
        this.trialChangeCounter++;
        this.currentGesture = gesture;
        // Don't reset phaseStartedAt here — it resets in phaseReady()
        // after the frontend countdown completes

        this.status.phaseName = GESTURE_DISPLAY[gesture] || gesture.toUpperCase();
        this.status.phaseInstruction = GESTURE_INSTRUCTIONS[gesture] || gesture;
        this.status.phaseTargetAngle = GESTURE_ANGLES[gesture] ?? 110;
        this.status.phaseDurationSec = data.duration || 5;
        this.status.phaseElapsedSec = 0;
        this.status.phaseWaiting = true; // Triggers pre_phase → countdown in frontend

        // Track gesture-level phase index (rest=0, close=1, open=2, etc.)
        if (!this.gesturePhaseMap.has(gesture)) {
          this.gesturePhaseMap.set(gesture, this.gesturePhaseMap.size);
        }
        this.status.phaseIndex = this.trialChangeCounter; // unique per trial for frontend transition detection

        // Trial within current gesture
        if (data.total_trials > 0) {
          this.status.trialIndex = data.trial_idx || 0;
          this.status.phaseTrials = data.total_trials;
          this.status.phaseProgress = (data.trial_idx || 0) / data.total_trials;
        }

        // Track overall phase (gesture-level) for display
        this.status.totalPhases = this.status.mode === "full" ? 6 : 3;

        // Model is loaded if we're getting trial events
        if (!this.status.modelLoaded) {
          this.status.modelLoaded = true;
        }

        // Signal simulator to stream this gesture's EMG data
        this.writeGestureFile(gesture);

        this.emit("trial_start", data);

      } else if (data.type === "trial_rest") {
        // Rest period between trials — update display but no phaseWaiting
        this.currentGesture = "rest";
        this.writeGestureFile("rest");
        this.phaseStartedAt = Date.now();
        this.status.phaseName = "REST";
        this.status.phaseInstruction = "Relax your hand";
        this.status.phaseTargetAngle = GESTURE_ANGLES["rest"] ?? 110;
        this.status.phaseDurationSec = data.duration || 4;
        this.status.phaseElapsedSec = 0;
        // Don't set phaseWaiting — rest periods flow without instruction screen

      } else if (data.type === "emg") {
        this.emit("reading", {
          emg: data.emg || [0, 0, 0, 0],
          intent: data.gesture || "rest",
          confidence: 0,
          assistStrength: 0,
        });
      } else if (data.type === "complete") {
        this.status.active = false;
        this.status.completed = true;
        this.status.overallProgress = 1;
        this.status.remainingSec = 0;
        this.status.phaseName = "COMPLETE";
        this.status.phaseInstruction = "Calibration complete!";
        this.status.phaseWaiting = false;
        this.status.qualityReport = data.quality_report || null;
        this.writeGestureFile("session"); // Signal simulator to switch to session mode
        this.emit("complete", data);
      } else if (data.type === "error") {
        this.status.active = false;
        this.status.error = data.message || "Calibration failed";
        this.emit("error", data);
      }
    } catch {
      // Not valid JSON — ignore
    }
  }

  /**
   * Called by /calibration/phase-ready — signals the frontend countdown is done.
   * Resets the phase timer so the "running" screen starts counting from zero.
   */
  phaseReady() {
    this.status.phaseWaiting = false;
    this.phaseStartedAt = Date.now();
    this.collectionStartAt = Date.now();
    this.status.phaseElapsedSec = 0;
  }

  stop() {
    if (this.process) {
      this.process.kill("SIGTERM");
      this.process = null;
    }
    this.status = this.defaultStatus();
    this.buffer = "";
    this.stderrBuffer = "";
    this.gesturePhaseMap = new Map();
    this.trialChangeCounter = 0;
    this.writeGestureFile("session"); // Signal simulator to switch to session mode
  }

  /** Write current gesture to file so simulator streams correct EMG data */
  private writeGestureFile(gesture: string) {
    try {
      fs.writeFileSync(GESTURE_FILE, gesture);
    } catch {
      // Non-critical — only needed when using simulator
    }
  }

  getStatus(): CalibrationStatus {
    const status = { ...this.status };
    // Compute elapsed time dynamically
    if (status.active && this.phaseStartedAt > 0) {
      status.phaseElapsedSec = (Date.now() - this.phaseStartedAt) / 1000;
    }
    // Real-time remaining decrement — only during active data collection
    // (phaseWaiting=false), not during UI instruction/countdown pauses.
    // collectionStartAt resets each phaseReady() call; remaining resets each
    // Python progress callback. Together they give smooth real-time countdown
    // that pauses during UI transitions and resumes during collection.
    if (status.active && !status.phaseWaiting && this.collectionStartAt > 0) {
      const collectionElapsed = (Date.now() - this.collectionStartAt) / 1000;
      status.remainingSec = Math.max(0, Math.round(this.status.remainingSec - collectionElapsed));
    }
    return status;
  }

  isRunning(): boolean {
    return this.process !== null && !this.process.killed;
  }
}
