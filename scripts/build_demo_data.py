"""
Build the static demo assets for the client from the Adhi 2026-02-20 recording.

Outputs (into client/public/demo/):
  emg.json   — array of {t, ch: [4]} at ~20 Hz for the whole 733-second session
  cues.json  — array of {t, label} (rest / close / open) derived from cues.csv
  meta.json  — high-level session metadata (duration, sample rate, block layout)

The client's mock WebSocket walks the emg + cues arrays in real time,
emits MotorFrames matching the runtime's format, and loops when it hits
the end.
"""

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSION_DIR = PROJECT_ROOT / "sessions" / "2026-02-20_18-51"
OUT_DIR = PROJECT_ROOT / "client" / "public" / "demo"


def _read_emg(csv_path: Path):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "t": round(float(r["timestamp_sec"]), 3),
                "ch": [
                    int(round(float(r["ch0"]))),
                    int(round(float(r["ch1"]))),
                    int(round(float(r["ch2"]))),
                    int(round(float(r["ch3"]))),
                ],
            })
    return rows


def _read_cues(csv_path: Path):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "t": round(float(r["timestamp_sec"]), 3),
                "label": r["cue_label"].strip().lower(),
                "block": r.get("block_description", ""),
            })
    return rows


def _meta(emg, cues, session_info):
    def _count(label):
        return sum(1 for c in cues if c["label"] == label)
    return {
        "duration_sec": float(session_info["duration_sec"]),
        "sample_rate_hz": float(session_info.get("approx_sample_rate_hz", 20.0)),
        "num_frames": len(emg),
        "num_cues": len(cues),
        "cue_counts": {
            "rest":  _count("rest"),
            "close": _count("close"),
            "open":  _count("open"),
        },
        "session_id": SESSION_DIR.name,
        "recorded_at": session_info["date"],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session_info = json.loads((SESSION_DIR / "session_info.json").read_text())
    emg = _read_emg(SESSION_DIR / "raw_emg.csv")
    cues = _read_cues(SESSION_DIR / "cues.csv")
    meta = _meta(emg, cues, session_info)

    (OUT_DIR / "emg.json").write_text(
        json.dumps(emg, separators=(",", ":"))
    )
    (OUT_DIR / "cues.json").write_text(
        json.dumps(cues, separators=(",", ":"), ensure_ascii=False)
    )
    (OUT_DIR / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )

    print(f"Wrote demo assets to {OUT_DIR}")
    print(f"  emg.json:  {(OUT_DIR / 'emg.json').stat().st_size / 1024:.0f} KB "
          f"({meta['num_frames']} frames)")
    print(f"  cues.json: {(OUT_DIR / 'cues.json').stat().st_size / 1024:.0f} KB "
          f"({meta['num_cues']} cues)")
    print(f"  meta:      duration={meta['duration_sec']:.1f}s  "
          f"rate={meta['sample_rate_hz']:.1f}Hz  "
          f"cues={meta['cue_counts']}")


if __name__ == "__main__":
    main()
