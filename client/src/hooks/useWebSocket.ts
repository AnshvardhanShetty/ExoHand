import { useEffect, useRef, useState, useCallback } from "react";

export interface MotorFrame {
  type: string;
  emg: number[];
  classifierConfidence: number;
  assistStrength: number;
  repCount: number;
  negativeRepCount: number;
  grip: number;
  intent: string;
  stateCmd: string;
  motionLocked: boolean;
  lockRemainingMs: number;
  cooldownRemainingMs: number;
  targetAngle: number;
  stale: boolean;
}

const DEFAULT_MOTOR_STATE: Partial<MotorFrame> = {
  stateCmd: "REST",
  motionLocked: false,
  lockRemainingMs: 0,
  cooldownRemainingMs: 0,
  targetAngle: 110,
  stale: false,
};

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [connected, setConnected] = useState(false);
  const [frame, setFrame] = useState<MotorFrame | null>(null);
  const [stale, setStale] = useState(false);

  useEffect(() => {
    let unmounted = false;

    function connect() {
      if (unmounted) return;
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${protocol}//${window.location.hostname}:3001`);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!unmounted) setConnected(true);
      };

      ws.onclose = () => {
        if (unmounted) return;
        setConnected(false);
        wsRef.current = null;
        // Reconnect after 2s
        reconnectRef.current = setTimeout(connect, 2000);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "frame") {
            setFrame(data);
            setStale(data.stale || false);
          } else if (data.type === "stale") {
            setStale(true);
          } else if (data.type === "motorState") {
            // Merge motor state — handle case where no frame has arrived yet
            setFrame((prev) => ({
              ...(prev || DEFAULT_MOTOR_STATE as MotorFrame),
              stateCmd: data.stateCmd,
              motionLocked: data.motionLocked,
              lockRemainingMs: data.lockRemainingMs,
              cooldownRemainingMs: data.cooldownRemainingMs,
              targetAngle: data.targetAngle,
            }));
          }
        } catch {
          // ignore
        }
      };
    }

    connect();

    return () => {
      unmounted = true;
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const send = useCallback(
    (msg: any) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(msg));
      }
    },
    []
  );

  return { connected, frame, stale, send };
}
