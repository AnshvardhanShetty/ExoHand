import { WebSocketServer, WebSocket } from "ws";
import { Server } from "http";
import { LockedMotorStateMachine } from "../motor/stateMachine";

export interface EmgFrame {
  emg: number[];
  classifierConfidence: number;
  assistStrength: number;
  repCount: number;
  negativeRepCount: number;
  sessionFailed: boolean;
  grip: number;
  intent: string;
}

// Grip pipeline parameters
const EMA_ALPHA = 0.25;
const STALE_TIMEOUT_MS = 500;
const DECAY_RATE = 0.95;

export class WebSocketManager {
  private wss: WebSocketServer | null = null;
  private clients: Set<WebSocket> = new Set();
  private gripSmoothed: number = 0;
  private lastFrameTime: number = Date.now();
  private staleCheckInterval: ReturnType<typeof setInterval> | null = null;
  private isStale: boolean = false;
  private stateMachine: LockedMotorStateMachine;
  private _onAllDisconnected: (() => void) | null = null;
  private _onSessionStart: (() => void) | null = null;
  private disconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(stateMachine: LockedMotorStateMachine) {
    this.stateMachine = stateMachine;

    // Forward state machine events to clients
    stateMachine.on("stateChange", (state) => {
      this.broadcast({
        type: "motorState",
        ...state,
      });
    });

    stateMachine.on("safetyEvent", (event) => {
      this.broadcast({
        type: "safetyEvent",
        ...event,
      });
    });
  }

  attach(server: Server) {
    this.wss = new WebSocketServer({ server });

    this.wss.on("connection", (ws: WebSocket) => {
      this.clients.add(ws);
      // Cancel pending disconnect timer on new connection
      if (this.disconnectTimer) {
        clearTimeout(this.disconnectTimer);
        this.disconnectTimer = null;
      }
      // Send current state on connect
      ws.send(
        JSON.stringify({
          type: "motorState",
          ...this.stateMachine.getState(),
        })
      );

      ws.on("close", () => {
        this.clients.delete(ws);
        if (this.clients.size === 0 && this._onAllDisconnected) {
          // Debounce: wait 5s before stopping — page reloads / HMR cause brief disconnects
          if (this.disconnectTimer) clearTimeout(this.disconnectTimer);
          this.disconnectTimer = setTimeout(() => {
            if (this.clients.size === 0 && this._onAllDisconnected) {
              this._onAllDisconnected();
            }
          }, 5000);
        }
      });

      ws.on("message", (data: Buffer) => {
        try {
          const msg = JSON.parse(data.toString());
          this.handleMessage(ws, msg);
        } catch {
          // ignore malformed messages
        }
      });
    });

    // Stale detection
    this.staleCheckInterval = setInterval(() => {
      const elapsed = Date.now() - this.lastFrameTime;
      if (elapsed > STALE_TIMEOUT_MS && !this.isStale) {
        this.isStale = true;
        // Decay grip
        this.gripSmoothed *= DECAY_RATE;
        this.broadcast({
          type: "stale",
          grip: this.gripSmoothed,
          stale: true,
        });
      }
    }, 100);
  }

  // Called by EMG bridge on each prediction
  onEmgFrame(frame: EmgFrame) {
    this.lastFrameTime = Date.now();
    this.isStale = false;

    // EMA smoothing on grip (for smooth visual display)
    this.gripSmoothed =
      EMA_ALPHA * frame.grip + (1 - EMA_ALPHA) * this.gripSmoothed;

    // Use the classifier's intent directly for state machine
    // (the grip hysteresis was causing the hand to stay stuck)
    const intent = frame.intent === "close" ? "close"
      : frame.intent === "open" ? "open"
      : frame.intent === "rest" ? "rest"
      : "rest";
    this.stateMachine.onIntent(intent as any);

    const motorState = this.stateMachine.getState();

    // rawIntent: unfiltered classifier output for immediate UI state display (S1)
    // stateCmd: state machine output for motor control (may lag due to motion locks)
    this.broadcast({
      type: "frame",
      emg: frame.emg,
      classifierConfidence: frame.classifierConfidence,
      assistStrength: frame.assistStrength,
      repCount: frame.repCount,
      negativeRepCount: frame.negativeRepCount,
      sessionFailed: frame.sessionFailed,
      grip: this.gripSmoothed,
      intent: frame.intent,
      rawIntent: frame.intent,
      stateCmd: motorState.stateCmd,
      motionLocked: motorState.motionLocked,
      lockRemainingMs: motorState.lockRemainingMs,
      cooldownRemainingMs: motorState.cooldownRemainingMs,
      targetAngle: motorState.targetAngle,
      stale: false,
    });
  }

  private handleMessage(_ws: WebSocket, msg: any) {
    if (msg.type === "startSession") {
      this.stateMachine.start();
      this._onSessionStart?.();
    } else if (msg.type === "endSession") {
      this.stateMachine.stop();
    }
  }

  broadcast(data: any) {
    // Keep stale detection satisfied when external broadcasts send frame data
    if (data.type === "frame") {
      this.lastFrameTime = Date.now();
      this.isStale = false;
    }
    const json = JSON.stringify(data);
    for (const client of this.clients) {
      if (client.readyState === WebSocket.OPEN) {
        client.send(json);
      }
    }
  }

  onAllDisconnected(callback: () => void) {
    this._onAllDisconnected = callback;
  }

  onSessionStart(callback: () => void) {
    this._onSessionStart = callback;
  }

  close() {
    if (this.staleCheckInterval) {
      clearInterval(this.staleCheckInterval);
    }
    if (this.wss) {
      this.wss.close();
    }
  }
}
