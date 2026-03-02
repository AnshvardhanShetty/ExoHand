/**
 * Shared singleton instances — avoids circular imports between
 * index.ts and route files that need access to bridge/wsManager.
 */
import { PythonBridge } from "./emg/bridge";
import { CalibrationBridge } from "./emg/calibrationBridge";
import { LockedMotorStateMachine } from "./motor/stateMachine";
import { WebSocketManager } from "./ws";
import { ExerciseTracker } from "./exercise/tracker";
import { SerialController } from "./motor/serial";

export const stateMachine = new LockedMotorStateMachine();
export const wsManager = new WebSocketManager(stateMachine);
export const bridge = new PythonBridge();
export const calibBridge = new CalibrationBridge();
export const serial = new SerialController();

// Mutable — set when session starts, cleared when it ends
let _exerciseTracker: ExerciseTracker | null = null;
export function getExerciseTracker(): ExerciseTracker | null {
  return _exerciseTracker;
}
export function setExerciseTracker(t: ExerciseTracker | null) {
  _exerciseTracker = t;
}
