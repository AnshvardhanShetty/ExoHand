import { EventEmitter } from "events";

// Serial protocol:
//   "A###\n" — move to angle ### (110=fully open, 145=rest, 180=fully closed)
//   3-digit zero-padded. Examples: A110\n A145\n A180\n
//   Only one command per transition (locked motion state machine enforces this).

export class SerialController extends EventEmitter {
  private port: any = null;
  private connected: boolean = false;

  async connect(portPath: string, baudRate: number = 115200) {
    try {
      const { SerialPort } = await import("serialport");
      this.port = new SerialPort({ path: portPath, baudRate });

      this.port.on("open", () => {
        this.connected = true;
        this.emit("connected");
      });

      this.port.on("error", (err: Error) => {
        this.emit("error", err);
      });

      this.port.on("close", () => {
        this.connected = false;
        this.emit("disconnected");
      });

      // Read data from Teensy (EMG lines)
      this.port.on("data", (data: Buffer) => {
        const lines = data.toString().split("\n").filter(Boolean);
        for (const line of lines) {
          this.emit("data", line.trim());
        }
      });
    } catch {
      this.emit("mock", "Serial port not available, running in mock mode");
    }
  }

  send(cmd: string) {
    if (this.port && this.connected) {
      this.port.write(cmd);
    }
    this.emit("sent", cmd.trim());
  }

  sendMotorCommand(cmd: string) {
    // cmd is already formatted as "A###\n"
    this.send(cmd);
  }

  isConnected(): boolean {
    return this.connected;
  }

  close() {
    if (this.port) {
      const oldPort = this.port;
      this.port = null;
      this.connected = false;
      // Remove listeners so a stale port can't corrupt our state,
      // but keep a no-op error handler to prevent unhandled "error"
      // events from crashing the process.
      oldPort.removeAllListeners();
      oldPort.on("error", () => {});
      try {
        oldPort.close();
      } catch {
        // Port may not be fully open yet — safe to ignore
      }
    }
  }
}
