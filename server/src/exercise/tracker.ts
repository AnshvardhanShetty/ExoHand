import { EventEmitter } from "events";
import { getDb } from "../db";

export interface ExerciseDefinition {
  id: string;
  name: string;
  category: "close" | "open" | "combined";
  startAngle: number;
  targetAngle: number;
  holdSeconds: number;
  reps: number;
}

interface RepState {
  startedAt: number;
  reachedTargetAt: number | null;
  holdStartAt: number | null;
  confidenceSum: number;
  confidenceCount: number;
  peakConfidence: number;
  intent: string; // which intent triggered this rep
}

/**
 * Tracks exercise reps from EMG intent stream.
 *
 * Rep lifecycle (close exercise example):
 *   idle → intent becomes "close" → approaching
 *   approaching → sustained close intent → holding
 *   holding → intent returns to "rest" → rep complete
 *
 * Combined exercises:
 *   close → open → close → open (no rest required between phases)
 *
 * Negative reps: if the wrong gesture is detected (e.g., "open" during a
 * close exercise), it counts as a negative rep.
 */
export class ExerciseTracker extends EventEmitter {
  private exercise: ExerciseDefinition;
  private sessionId: number;
  private repCount: number = 0;
  private negativeRepCount: number = 0;
  private currentRep: RepState | null = null;
  private lastAcceptedIntent: string = "rest";

  // Debounce: require N consecutive same-intent frames before accepting
  // At ~4 Hz, 2 frames = 0.5s (smoother already provides 0.75s filtering)
  private pendingIntent: string = "rest";
  private pendingCount: number = 0;
  private readonly DEBOUNCE_FRAMES = 2;

  // Minimum time between completed reps (ms) — prevents rapid false reps
  private lastRepCompletedAt: number = 0;
  private readonly MIN_REP_INTERVAL_MS = 2000;
  // Combined exercises need faster transitions between phases
  private readonly MIN_COMBINED_INTERVAL_MS = 500;

  // Grace period: ignore first N seconds to let smoothing settle
  // Starts on first onFrame() call, not at construction
  private firstFrameAt: number = 0;
  private readonly GRACE_PERIOD_MS = 2000;
  private _failed: boolean = false;

  // For combined exercises — alternate close/open
  private combinedPhase: "close" | "open" = "close";

  constructor(exercise: ExerciseDefinition, sessionId: number) {
    super();
    this.exercise = exercise;
    this.sessionId = sessionId;
  }

  /**
   * Called on every EMG frame from PythonBridge.
   * Returns current rep count for inclusion in WS broadcast.
   */
  onFrame(intent: string, confidence: number): number {
    if (this.repCount >= this.exercise.reps) return this.repCount;
    if (this.negativeRepCount >= this.exercise.reps * 3) {
      if (!this._failed) {
        this._failed = true;
        this.emit("failed", { reason: "too_many_misses", missedReps: this.negativeRepCount });
      }
      return this.repCount;
    }

    // Start grace period on first frame (not at construction)
    if (this.firstFrameAt === 0) this.firstFrameAt = Date.now();

    // Grace period — let smoothing settle before counting anything
    if (Date.now() - this.firstFrameAt < this.GRACE_PERIOD_MS) {
      this.debounce(intent); // still feed debouncer to build up state
      return this.repCount;
    }

    // Debounce intent transitions
    const stableIntent = this.debounce(intent);
    const targetIntent = this.getTargetIntent();
    const now = Date.now();

    if (this.currentRep) {
      // Currently tracking a rep
      this.currentRep.confidenceSum += confidence;
      this.currentRep.confidenceCount += 1;
      this.currentRep.peakConfidence = Math.max(
        this.currentRep.peakConfidence,
        confidence
      );

      if (stableIntent === this.currentRep.intent) {
        // Still at target — mark reached if not already
        if (!this.currentRep.reachedTargetAt) {
          this.currentRep.reachedTargetAt = now;
          this.currentRep.holdStartAt = now;
        }
      } else if (this.currentRep.reachedTargetAt) {
        // Intent changed after reaching target
        if (this.exercise.category === "combined") {
          // Combined: require direct close↔open transition
          const nextPhase = this.combinedPhase === "close" ? "open" : "close";
          if (stableIntent === nextPhase) {
            // Direct transition to opposite movement — successful rep
            if (now - this.lastRepCompletedAt >= this.MIN_COMBINED_INTERVAL_MS) {
              this.completeRep(now, true);
            } else {
              this.currentRep = null;
            }
          } else if (stableIntent === "rest") {
            // Went through rest — negative rep, but still toggle phase
            if (now - this.lastRepCompletedAt >= this.MIN_COMBINED_INTERVAL_MS) {
              this.negativeRepCount++;
              this.lastRepCompletedAt = now;
              this.emit("negative_rep", {
                negativeRepCount: this.negativeRepCount,
                wrongIntent: "rest",
                expectedIntent: nextPhase,
              });
              this.combinedPhase = nextPhase;
              this.currentRep = null;
            } else {
              this.currentRep = null;
            }
          }
        } else {
          // Close/open: need to return to rest to complete
          if (stableIntent === "rest") {
            if (now - this.lastRepCompletedAt >= this.MIN_REP_INTERVAL_MS) {
              this.completeRep(now, true);
            } else {
              this.currentRep = null;
            }
          }
        }
      } else if (
        now - this.currentRep.startedAt > 8000
      ) {
        // Timeout — abandon incomplete rep
        this.currentRep = null;
      }
    } else {
      // Idle — looking for rep start or wrong gesture
      if (
        stableIntent === targetIntent &&
        this.lastAcceptedIntent !== targetIntent
      ) {
        // Correct gesture — start tracking rep
        this.currentRep = {
          startedAt: now,
          reachedTargetAt: null,
          holdStartAt: null,
          confidenceSum: confidence,
          confidenceCount: 1,
          peakConfidence: confidence,
          intent: targetIntent,
        };
      }
    }

    this.lastAcceptedIntent = stableIntent;
    return this.repCount;
  }

  private debounce(raw: string): string {
    if (raw === this.pendingIntent) {
      this.pendingCount++;
    } else {
      this.pendingIntent = raw;
      this.pendingCount = 1;
    }

    if (this.pendingCount >= this.DEBOUNCE_FRAMES) {
      return this.pendingIntent;
    }
    // Not yet stable — return last accepted
    return this.lastAcceptedIntent;
  }

  private getTargetIntent(): string {
    if (this.exercise.category === "close") return "close";
    if (this.exercise.category === "open") return "open";
    // Combined: alternate
    return this.combinedPhase;
  }

  private completeRep(now: number, success: boolean) {
    if (!this.currentRep) return;

    const timeToTarget = this.currentRep.reachedTargetAt
      ? (this.currentRep.reachedTargetAt - this.currentRep.startedAt) / 1000
      : (now - this.currentRep.startedAt) / 1000;

    const holdDuration = this.currentRep.holdStartAt
      ? (now - this.currentRep.holdStartAt) / 1000
      : 0;

    const avgConfidence =
      this.currentRep.confidenceCount > 0
        ? this.currentRep.confidenceSum / this.currentRep.confidenceCount
        : 0;

    // Accuracy: based on average classifier confidence during the rep
    const accuracy = Math.min(100, avgConfidence * 100);

    // Stability: based on peak confidence (higher = more consistent signal)
    const stability = Math.min(100, this.currentRep.peakConfidence * 100);

    // Success: reached target AND held long enough (if hold required)
    const heldEnough =
      this.exercise.holdSeconds === 0 ||
      holdDuration >= this.exercise.holdSeconds * 0.8;
    const repSuccess = this.currentRep.reachedTargetAt !== null && heldEnough && success;

    // Hold exercises: if not held long enough, count as missed rep instead
    if (this.exercise.holdSeconds > 0 && !heldEnough) {
      this.negativeRepCount++;
      this.lastRepCompletedAt = now;
      this.emit("negative_rep", {
        negativeRepCount: this.negativeRepCount,
        wrongIntent: "insufficient_hold",
        expectedIntent: this.getTargetIntent(),
        holdDuration,
        requiredHold: this.exercise.holdSeconds,
      });
      this.currentRep = null;
      if (this.exercise.category === "combined") {
        this.combinedPhase = this.combinedPhase === "close" ? "open" : "close";
      }
      return;
    }

    this.repCount++;
    this.lastRepCompletedAt = now;

    // Save to DB
    try {
      const db = getDb();
      db.prepare(
        `INSERT INTO reps (session_id, rep_number, accuracy, stability, time_to_target, success)
         VALUES (?, ?, ?, ?, ?, ?)`
      ).run(
        this.sessionId,
        this.repCount,
        Math.round(accuracy * 10) / 10,
        Math.round(stability * 10) / 10,
        Math.round(timeToTarget * 100) / 100,
        repSuccess ? 1 : 0
      );
    } catch {
      /* non-critical */
    }

    this.emit("rep", {
      repNumber: this.repCount,
      totalReps: this.exercise.reps,
      accuracy,
      stability,
      timeToTarget,
      holdDuration,
      success: repSuccess,
    });

    // Combined: toggle phase after each rep
    if (this.exercise.category === "combined") {
      this.combinedPhase =
        this.combinedPhase === "close" ? "open" : "close";
    }

    this.currentRep = null;

    if (this.repCount >= this.exercise.reps) {
      this.emit("complete");
    }
  }

  /**
   * Clamp impossible intents for the current exercise.
   * Close exercise: "open" → "rest" (can't happen in real usage)
   * Open exercise: "close" → "rest"
   * Combined: no clamping (both are valid)
   */
  filterIntent(intent: string): string {
    if (this.exercise.category === "close" && intent === "open") return "rest";
    if (this.exercise.category === "open" && intent === "close") return "rest";
    return intent;
  }

  getRepCount(): number {
    return this.repCount;
  }

  getNegativeRepCount(): number {
    return this.negativeRepCount;
  }

  getTotalReps(): number {
    return this.exercise.reps;
  }

  isComplete(): boolean {
    return this.repCount >= this.exercise.reps;
  }

  isFailed(): boolean {
    return this._failed;
  }

  getScore(): { successRate: number; negativeRate: number; score: number } {
    const totalAttempts = this.repCount + this.negativeRepCount;
    const successRate = totalAttempts > 0 ? this.repCount / totalAttempts : 0;
    const negativeRate = totalAttempts > 0 ? this.negativeRepCount / totalAttempts : 0;
    // Score: percentage of target reps achieved minus penalty for negative reps
    const score = Math.max(0, Math.min(100,
      (this.repCount / this.exercise.reps) * 100 - this.negativeRepCount * 5
    ));
    return { successRate, negativeRate, score };
  }

  reset() {
    this.repCount = 0;
    this.negativeRepCount = 0;
    this.currentRep = null;
    this.lastAcceptedIntent = "rest";
    this.pendingIntent = "rest";
    this.pendingCount = 0;
    this.combinedPhase = "close";
    this.lastRepCompletedAt = 0;
    this.firstFrameAt = 0;
    this._failed = false;
  }
}
