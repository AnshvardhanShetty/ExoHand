// Per-rep scoring
export interface RepScore {
  accuracy: number; // 0-100: distance from target
  stability: number; // 0-100: hold variance
  timeToTarget: number; // seconds
  success: boolean;
}

// Session scoring
export interface SessionScore {
  overallScore: number;
  completionRate: number;
  avgStability: number;
  avgAccuracy: number;
}

// Score a single rep
export function scoreRep(
  actualAngle: number,
  targetAngle: number,
  holdVariance: number,
  timeToTarget: number,
  holdDuration: number,
  requiredHold: number,
  tolerance: number = 10
): RepScore {
  // Accuracy: how close to target (100 = perfect)
  const angleDiff = Math.abs(actualAngle - targetAngle);
  const accuracy = Math.max(0, 100 - (angleDiff / 90) * 100);

  // Stability: inverse of hold variance (100 = rock steady)
  const stability = Math.max(0, 100 - holdVariance * 10);

  // Success: reached target within tolerance AND held stable AND no jitter
  const reachedTarget = angleDiff <= tolerance;
  const heldLongEnough = holdDuration >= requiredHold * 0.8;
  const wasStable = stability >= 50;
  const success = reachedTarget && heldLongEnough && wasStable;

  return { accuracy, stability, timeToTarget, success };
}

// Compute session-level scores from rep data
// targetReps: the exercise's target rep count (default 10)
export function computeSessionScore(
  reps: Array<{
    accuracy: number;
    stability: number;
    time_to_target: number;
    success: number;
  }>,
  targetReps: number = 10
): SessionScore {
  if (reps.length === 0) {
    return {
      overallScore: 0,
      completionRate: 0,
      avgStability: 0,
      avgAccuracy: 0,
    };
  }

  // R2: completion = successful reps / target reps (not / recorded reps)
  const successCount = reps.filter((r) => r.success).length;
  const completionRate = Math.min(100, (successCount / targetReps) * 100);

  const avgStability =
    reps.reduce((sum, r) => sum + r.stability, 0) / reps.length;
  const avgAccuracy =
    reps.reduce((sum, r) => sum + r.accuracy, 0) / reps.length;

  // Overall score: weighted combination
  // 40% accuracy + 30% stability + 30% completion
  const overallScore =
    avgAccuracy * 0.4 + avgStability * 0.3 + completionRate * 0.3;

  return {
    overallScore: Math.round(overallScore * 10) / 10,
    completionRate: Math.round(completionRate * 10) / 10,
    avgStability: Math.round(avgStability * 10) / 10,
    avgAccuracy: Math.round(avgAccuracy * 10) / 10,
  };
}
