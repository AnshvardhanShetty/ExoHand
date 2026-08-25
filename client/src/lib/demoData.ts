/**
 * demoData.ts
 *
 * Replays the Adhi 2026-02-20 12-minute cued session as a live-looking
 * MotorFrame stream. Powers the mock WebSocket in demo mode.
 *
 * The data is fetched once from /demo/emg.json + /demo/cues.json, then a
 * lightweight state machine walks through it at 20 Hz. Between cues we
 * interpolate a smooth grip/targetAngle. Rep counts, classifier
 * confidence, and assist strength are derived from cue transitions.
 */

import type { MotorFrame } from "../hooks/useWebSocket";

interface RawEmgFrame {
  t: number;     // seconds since session start
  ch: number[]; // 4-channel P-P amplitudes
}
interface RawCue {
  t: number;      // seconds since session start
  label: string;   // "rest" | "close" | "open"
  block?: string;
}

interface DemoDataset {
  emg: RawEmgFrame[];
  cues: RawCue[];
  duration_sec: number;
  sample_rate_hz: number;
}

// Angle targets used by the deployed servo (from README §Hardware / Motor).
const ANGLE = { rest: 145, close: 180, open: 110 } as const;
// Grip: 0 = fully open, 1 = fully closed. Rest = intermediate.
const GRIP = { rest: 0.35, close: 1.0, open: 0.0 } as const;

let cache: DemoDataset | null = null;

async function loadDataset(): Promise<DemoDataset> {
  if (cache) return cache;
  const base = (import.meta.env.BASE_URL || "/").replace(/\/$/, "");
  const [emgRes, cuesRes, metaRes] = await Promise.all([
    fetch(`${base}/demo/emg.json`),
    fetch(`${base}/demo/cues.json`),
    fetch(`${base}/demo/meta.json`),
  ]);
  if (!emgRes.ok || !cuesRes.ok || !metaRes.ok) {
    throw new Error("Failed to load demo data files from /demo/*");
  }
  const emg = (await emgRes.json()) as RawEmgFrame[];
  const cues = (await cuesRes.json()) as RawCue[];
  const meta = await metaRes.json();
  cache = {
    emg,
    cues,
    duration_sec: meta.duration_sec,
    sample_rate_hz: meta.sample_rate_hz,
  };
  return cache;
}

function findCueAt(cues: RawCue[], t: number): RawCue {
  // Cues array is sorted by t. Find the most recent cue with cue.t <= t.
  let lo = 0;
  let hi = cues.length - 1;
  let best = cues[0];
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (cues[mid].t <= t) {
      best = cues[mid];
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return best;
}

interface ReplayCallbacks {
  onFrame: (frame: MotorFrame) => void;
  onConnected: (connected: boolean) => void;
  onEnd?: () => void;
  loop?: boolean; // default true
}

/**
 * Starts the timeline replay. Emits a MotorFrame every ~50 ms.
 * Returns a stop() function.
 */
export function startDemoReplay(cb: ReplayCallbacks): () => void {
  const loop = cb.loop !== false;
  let stopped = false;
  let intervalId: ReturnType<typeof setInterval> | null = null;

  let repCount = 0;
  let negativeRepCount = 0;
  let prevIntent = "rest";
  let gripCurrent = GRIP.rest;
  let angleCurrent = ANGLE.rest;

  loadDataset()
    .then((data) => {
      if (stopped) return;
      cb.onConnected(true);
      const startWallMs = performance.now();

      const tickMs = 50; // 20 Hz
      intervalId = setInterval(() => {
        const nowMs = performance.now() - startWallMs;
        let tSec = nowMs / 1000;
        if (tSec >= data.duration_sec) {
          if (loop) {
            // Restart the wall clock so we loop cleanly
            const wraps = Math.floor(tSec / data.duration_sec);
            tSec = tSec - wraps * data.duration_sec;
            // Reset rep counters when we wrap so the display doesn't grow forever
            repCount = 0;
            negativeRepCount = 0;
          } else {
            if (intervalId) clearInterval(intervalId);
            cb.onEnd?.();
            return;
          }
        }

        // Find the emg frame nearest tSec (data is dense at 20 Hz already).
        const idx = Math.min(
          data.emg.length - 1,
          Math.floor(tSec * data.sample_rate_hz)
        );
        const emgFrame = data.emg[idx];
        const cue = findCueAt(data.cues, tSec);
        const intent = cue.label; // "rest" | "close" | "open"

        // Rep counting on transitions AWAY from a movement into rest.
        if (prevIntent !== intent) {
          if (prevIntent === "close" && intent === "rest") repCount += 1;
          if (prevIntent === "open" && intent === "rest") negativeRepCount += 1;
          prevIntent = intent;
        }

        // Smooth grip / target-angle toward the cue target (mimics EMA smoothing
        // that the runtime does on predict_proba).
        const targetGrip =
          intent === "close" ? GRIP.close : intent === "open" ? GRIP.open : GRIP.rest;
        const targetAngle =
          intent === "close" ? ANGLE.close : intent === "open" ? ANGLE.open : ANGLE.rest;
        gripCurrent += (targetGrip - gripCurrent) * 0.25;
        angleCurrent += (targetAngle - angleCurrent) * 0.25;

        // Confidence: higher when we're mid-hold, lower right after a transition.
        const secSinceCue = tSec - cue.t;
        const confidence = Math.min(
          0.97,
          0.55 + Math.min(secSinceCue / 1.2, 0.42)
        );
        const assistStrength =
          intent === "rest" ? 0 : 0.55 + Math.min(secSinceCue / 3.0, 0.3);

        const frame: MotorFrame = {
          type: "frame",
          emg: emgFrame.ch,
          classifierConfidence: confidence,
          assistStrength,
          repCount,
          negativeRepCount,
          grip: gripCurrent,
          intent,
          stateCmd: intent.toUpperCase(),
          motionLocked: false,
          lockRemainingMs: 0,
          cooldownRemainingMs: 0,
          targetAngle: angleCurrent,
          stale: false,
        };
        cb.onFrame(frame);
      }, tickMs);
    })
    .catch((err) => {
      console.error("[demoData] failed to start replay:", err);
      cb.onConnected(false);
    });

  return () => {
    stopped = true;
    if (intervalId) clearInterval(intervalId);
    cb.onConnected(false);
  };
}

/** Prefetch the demo data on app boot so first play is instant. */
export function prefetchDemoData(): void {
  loadDataset().catch(() => { /* ignored — the replay will retry */ });
}
