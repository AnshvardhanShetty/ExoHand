import { spawn, ChildProcess } from "child_process";
import { EventEmitter } from "events";
import path from "path";

export interface EmgReading {
  emg: number[];
  intent: string;
  confidence: number;
  assistStrength: number;
}

export class PythonBridge extends EventEmitter {
  private process: ChildProcess | null = null;
  private buffer: string = "";
  private _ready: boolean = false;
  private _error: string | null = null;
  private stderrBuffer: string = "";

  start(options: {
    port: string;
    model: string;
    assistLevel: number;
    patientId?: string;
  }) {
    // Stop any existing process
    this.stop();

    const projectRoot = path.resolve(__dirname, "..", "..", "..");
    const script = path.join(projectRoot, "run_exohand.py");

    const args = [
      script,
      "--port",
      options.port,
      "--model",
      options.model,
      "--assist-level",
      String(options.assistLevel),
      "--node",
      "--skip-rest-cal",
    ];

    if (options.patientId) {
      args.push("--patient-id", options.patientId);
    }

    this._ready = false;
    this._error = null;
    this.stderrBuffer = "";

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
      if (!this._ready && code !== 0) {
        // Python crashed before becoming ready — extract useful error
        const lastLines = this.stderrBuffer.trim().split("\n").slice(-3).join(" ");
        this._error = lastLines || `Process exited with code ${code}`;
      }
      this._ready = false;
      this.process = null;
      this.emit("exit", code);
    });
  }

  private parseLine(line: string) {
    if (!line.startsWith("{")) return;

    try {
      const data = JSON.parse(line);

      // Ready signal from Python after model load
      if (data.type === "ready") {
        this._ready = true;
        this.emit("ready");
        return;
      }

      this.emit("reading", {
        emg: data.emg || [0, 0, 0, 0],
        intent: data.intent || "rest",
        confidence: data.confidence || 0,
        assistStrength: data.assist_strength || 0,
      } as EmgReading);
    } catch {
      // Not JSON — ignore
    }
  }

  stop() {
    if (this.process) {
      this.process.kill("SIGTERM");
      this.process = null;
    }
    this._ready = false;
    this._error = null;
    this.buffer = "";
    this.stderrBuffer = "";
  }

  sendCommand(cmd: string) {
    if (this.process?.stdin?.writable) {
      this.process.stdin.write(cmd);
    }
  }

  isRunning(): boolean {
    return this.process !== null && !this.process.killed;
  }

  isReady(): boolean {
    return this._ready;
  }

  getError(): string | null {
    return this._error;
  }
}
