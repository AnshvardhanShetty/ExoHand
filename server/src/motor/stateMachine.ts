import { EventEmitter } from "events";
import { getDb } from "../db";

/**
 * CANONICAL ANGLE DEFINITIONS:
 *   110° = FULL OPEN / EXTENSION
 *   145° = REST (neutral posture)
 *   180° = FULL CLOSE / FLEXION
 *
 * Opening = angle decreases (toward 110°)
 * Closing = angle increases (toward 180°)
 *
 * Serial protocol: "A###\n" where ### is 110..180 (3 digits, zero-padded)
 */

// States
export enum MotorState {
  IDLE_OPEN = "IDLE_OPEN",
  MOVING_TO_CLOSED = "MOVING_TO_CLOSED",
  IDLE_CLOSED = "IDLE_CLOSED",
  MOVING_TO_OPEN = "MOVING_TO_OPEN",
  IDLE_PARTIAL = "IDLE_PARTIAL",
  MOVING_TO_PARTIAL = "MOVING_TO_PARTIAL",
  IDLE_REST = "IDLE_REST",
  MOVING_TO_REST = "MOVING_TO_REST",
}

// Intents from EMG classifier
export type Intent = "open" | "close" | "partial" | "rest";

export interface MotorStateEvent {
  state: MotorState;
  stateCmd: string;
  motionLocked: boolean;
  lockRemainingMs: number;
  cooldownRemainingMs: number;
  targetAngle: number;
}

const REST_ANGLE = 145;
const OPEN_ANGLE = 110;
const MOTION_DURATION_MS = 700;
const COOLDOWN_MS = 800;
const STALE_TIMEOUT_MS = 1500;
const MAX_STATE_CHANGES_PER_SEC = 1;

/** Format angle as 3-digit zero-padded serial command */
function serialCmd(angle: number): string {
  const clamped = Math.max(OPEN_ANGLE, Math.min(180, Math.round(angle)));
  return `A${String(clamped).padStart(3, "0")}\n`;
}

export class LockedMotorStateMachine extends EventEmitter {
  private state: MotorState = MotorState.IDLE_REST;
  private motionStartedAt: number | null = null;
  private cooldownStartedAt: number | null = null;
  private lastEmgTimestamp: number = Date.now();
  private lastStateChangeAt: number = 0;
  private targetAngle: number = REST_ANGLE; // start at REST
  private sessionId: number | null = null;
  private staleCheckInterval: ReturnType<typeof setInterval> | null = null;
  private _running: boolean = false;

  constructor() {
    super();
  }

  setSessionId(id: number | null) {
    this.sessionId = id;
  }

  start() {
    this._running = true;
    this.state = MotorState.IDLE_REST;
    this.targetAngle = REST_ANGLE;
    this.lastEmgTimestamp = Date.now();
    this.staleCheckInterval = setInterval(() => this.checkStale(), 200);
  }

  stop() {
    if (this._running) {
      // Send motor to rest position before stopping
      this.emit("serial", serialCmd(REST_ANGLE));
    }
    this._running = false;
    if (this.staleCheckInterval) {
      clearInterval(this.staleCheckInterval);
      this.staleCheckInterval = null;
    }
    this.state = MotorState.IDLE_REST;
    this.targetAngle = REST_ANGLE;
  }

  isRunning(): boolean {
    return this._running;
  }

  getState(): MotorStateEvent {
    const now = Date.now();
    const motionLocked = this.isMotionLocked();
    let lockRemainingMs = 0;
    let cooldownRemainingMs = 0;

    if (this.motionStartedAt && this.isInMotion()) {
      lockRemainingMs = Math.max(
        0,
        MOTION_DURATION_MS - (now - this.motionStartedAt)
      );
    }
    if (this.cooldownStartedAt) {
      cooldownRemainingMs = Math.max(
        0,
        COOLDOWN_MS - (now - this.cooldownStartedAt)
      );
    }

    return {
      state: this.state,
      stateCmd: this.stateToCmd(),
      motionLocked,
      lockRemainingMs,
      cooldownRemainingMs,
      targetAngle: this.targetAngle,
    };
  }

  onIntent(intent: Intent, targetClosure: number = 50) {
    if (!this._running) return;
    this.lastEmgTimestamp = Date.now();
    this.updateMotionCompletion();

    if (this.isMotionLocked()) return;
    if (this.isInCooldown()) return;

    const now = Date.now();
    if (now - this.lastStateChangeAt < 1000 / MAX_STATE_CHANGES_PER_SEC) return;

    const cmd = this.processIntent(intent, targetClosure);
    if (cmd) {
      this.lastStateChangeAt = now;
      this.emit("serial", cmd);
      this.emit("stateChange", this.getState());
    }
  }

  private processIntent(intent: Intent, targetClosure: number): string | null {
    switch (this.state) {
      case MotorState.IDLE_REST:
      case MotorState.IDLE_OPEN:
        if (intent === "close") {
          this.state = MotorState.MOVING_TO_CLOSED;
          this.targetAngle = 180;
          this.motionStartedAt = Date.now();
          this.logSafety("motion_start", `${this.state} -> MOVING_TO_CLOSED (A180)`);
          return serialCmd(180);
        }
        if (intent === "partial") {
          // targetClosure 0-100 maps to REST(140)..CLOSE(180) range
          const angle = Math.round(REST_ANGLE + (targetClosure / 100) * (180 - REST_ANGLE));
          this.state = MotorState.MOVING_TO_PARTIAL;
          this.targetAngle = angle;
          this.motionStartedAt = Date.now();
          this.logSafety("motion_start", `${this.state} -> MOVING_TO_PARTIAL (A${angle})`);
          return serialCmd(angle);
        }
        if (intent === "open" && this.state !== MotorState.IDLE_OPEN) {
          this.state = MotorState.MOVING_TO_OPEN;
          this.targetAngle = OPEN_ANGLE;
          this.motionStartedAt = Date.now();
          this.logSafety("motion_start", `IDLE_REST -> MOVING_TO_OPEN (A${OPEN_ANGLE})`);
          return serialCmd(OPEN_ANGLE);
        }
        if (intent === "rest" && this.state === MotorState.IDLE_OPEN) {
          this.state = MotorState.MOVING_TO_REST;
          this.targetAngle = REST_ANGLE;
          this.motionStartedAt = Date.now();
          this.logSafety("motion_start", `IDLE_OPEN -> MOVING_TO_REST (A${REST_ANGLE})`);
          return serialCmd(REST_ANGLE);
        }
        return null;

      case MotorState.IDLE_CLOSED:
        if (intent === "open") {
          this.state = MotorState.MOVING_TO_OPEN;
          this.targetAngle = OPEN_ANGLE;
          this.motionStartedAt = Date.now();
          this.logSafety("motion_start", `IDLE_CLOSED -> MOVING_TO_OPEN (A${OPEN_ANGLE})`);
          return serialCmd(OPEN_ANGLE);
        }
        if (intent === "rest") {
          this.state = MotorState.MOVING_TO_REST;
          this.targetAngle = REST_ANGLE;
          this.motionStartedAt = Date.now();
          this.logSafety("motion_start", `IDLE_CLOSED -> MOVING_TO_REST (A${REST_ANGLE})`);
          return serialCmd(REST_ANGLE);
        }
        return null;

      case MotorState.IDLE_PARTIAL:
        if (intent === "open") {
          this.state = MotorState.MOVING_TO_OPEN;
          this.targetAngle = OPEN_ANGLE;
          this.motionStartedAt = Date.now();
          this.logSafety("motion_start", `IDLE_PARTIAL -> MOVING_TO_OPEN (A${OPEN_ANGLE})`);
          return serialCmd(OPEN_ANGLE);
        }
        if (intent === "rest") {
          this.state = MotorState.MOVING_TO_REST;
          this.targetAngle = REST_ANGLE;
          this.motionStartedAt = Date.now();
          this.logSafety("motion_start", `IDLE_PARTIAL -> MOVING_TO_REST (A${REST_ANGLE})`);
          return serialCmd(REST_ANGLE);
        }
        return null;

      default:
        return null;
    }
  }

  private updateMotionCompletion() {
    if (!this.motionStartedAt || !this.isInMotion()) return;

    const elapsed = Date.now() - this.motionStartedAt;
    if (elapsed >= MOTION_DURATION_MS) {
      const prevState = this.state;
      switch (this.state) {
        case MotorState.MOVING_TO_CLOSED:
          this.state = MotorState.IDLE_CLOSED;
          break;
        case MotorState.MOVING_TO_OPEN:
          this.state = MotorState.IDLE_OPEN;
          break;
        case MotorState.MOVING_TO_PARTIAL:
          this.state = MotorState.IDLE_PARTIAL;
          break;
        case MotorState.MOVING_TO_REST:
          this.state = MotorState.IDLE_REST;
          break;
      }
      this.motionStartedAt = null;
      this.cooldownStartedAt = Date.now();
      this.logSafety("motion_complete", `${prevState} -> ${this.state}`);
      this.emit("stateChange", this.getState());
    }
  }

  private checkStale() {
    const elapsed = Date.now() - this.lastEmgTimestamp;
    if (elapsed > STALE_TIMEOUT_MS) {
      this.failSafeRest("EMG stale");
    }
  }

  /** Fail-safe: return to REST, not full open */
  private failSafeRest(reason: string) {
    if (this.state === MotorState.IDLE_REST) return;

    this.logSafety("fail_safe_rest", reason);
    this.state = MotorState.MOVING_TO_REST;
    this.targetAngle = REST_ANGLE;
    this.motionStartedAt = Date.now();
    this.emit("serial", serialCmd(REST_ANGLE));
    this.emit("stateChange", this.getState());
    this.emit("safetyEvent", { type: "fail_safe_rest", reason });
  }

  onMotionTimeout() {
    this.failSafeRest("Motion timeout");
  }

  private isInMotion(): boolean {
    return (
      this.state === MotorState.MOVING_TO_CLOSED ||
      this.state === MotorState.MOVING_TO_OPEN ||
      this.state === MotorState.MOVING_TO_PARTIAL ||
      this.state === MotorState.MOVING_TO_REST
    );
  }

  private isInCooldown(): boolean {
    if (!this.cooldownStartedAt) return false;
    return Date.now() - this.cooldownStartedAt < COOLDOWN_MS;
  }

  private isMotionLocked(): boolean {
    return this.isInMotion() || this.isInCooldown();
  }

  private stateToCmd(): string {
    switch (this.state) {
      case MotorState.IDLE_OPEN:
        return "OPEN";
      case MotorState.MOVING_TO_CLOSED:
      case MotorState.MOVING_TO_OPEN:
      case MotorState.MOVING_TO_PARTIAL:
      case MotorState.MOVING_TO_REST:
        return "MOVING";
      case MotorState.IDLE_CLOSED:
        return "CLOSED";
      case MotorState.IDLE_PARTIAL:
        return "HOLDING";
      case MotorState.IDLE_REST:
        return "REST";
      default:
        return "REST";
    }
  }

  private logSafety(eventType: string, details: string) {
    if (!this.sessionId) return;
    try {
      const db = getDb();
      db.prepare(
        "INSERT INTO safety_events (session_id, event_type, details) VALUES (?, ?, ?)"
      ).run(this.sessionId, eventType, details);
    } catch {
      // Non-critical
    }
  }
}
