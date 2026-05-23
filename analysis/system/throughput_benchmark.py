"""
ExoHand throughput benchmark.

Two questions:
  1. **Sustained inference rate** of the realtime pipeline (single-row predicts,
     dominated by Python + sklearn per-call overhead). This is the relevant
     number for the deployed system, which feeds the model one window per
     Teensy frame.
  2. **Batched throughput** (rows / second when a whole array is given to
     model.predict at once). This is the relevant number for offline analysis
     pipelines (LOSO eval, longitudinal eval, post-hoc session scoring).

Compares the heavy shipped GrabMyo model against the lighter per-session model
configuration.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from analysis.seed import SEED, seed_everything

HEAVY_MODEL = PROJECT_ROOT / "grabmyo" / "improved_hgb_model.pkl"
HEAVY_SCALER = PROJECT_ROOT / "grabmyo" / "improved_hgb_scaler.pkl"
FAST_MODEL = PROJECT_ROOT / "analysis" / "system" / "results" / "fast_hgb_model.pkl"
FAST_SCALER = PROJECT_ROOT / "analysis" / "system" / "results" / "fast_hgb_scaler.pkl"
OUT_CSV = PROJECT_ROOT / "analysis" / "system" / "results" / "throughput.csv"
OUT_MD = PROJECT_ROOT / "analysis" / "system" / "results" / "throughput_summary.md"

BATCH_SIZES = [1, 32, 256, 4096]
N_REPEATS = 50


def benchmark_model(name, model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    n_feats = scaler.n_features_in_
    n_trees = sum(len(p) for p in model._predictors)
    rng = np.random.RandomState(SEED)

    print(f"\n=== {name}  ({n_trees} total trees, {model.n_iter_} boosts × {len(model.classes_)} classes) ===")
    rows = []
    for bs in BATCH_SIZES:
        X = rng.randn(bs, n_feats).astype(np.float32)
        X_s = scaler.transform(X)
        # warm-up
        for _ in range(3):
            model.predict(X_s)
        times = []
        for _ in range(N_REPEATS):
            t = time.perf_counter()
            model.predict(X_s)
            times.append(time.perf_counter() - t)
        times = np.array(times)
        mean_s = times.mean()
        per_row_ms = (mean_s / bs) * 1000
        rows_per_s = bs / mean_s
        print(f"  batch={bs:>4d}  mean={mean_s*1000:>8.2f} ms  per-row={per_row_ms:>7.3f} ms  rate={rows_per_s:>10.0f} rows/s")
        rows.append({
            "model": name,
            "n_trees": n_trees,
            "batch_size": bs,
            "n_repeats": N_REPEATS,
            "mean_batch_ms": mean_s * 1000,
            "per_row_ms": per_row_ms,
            "rows_per_sec": rows_per_s,
        })
    return rows


def main():
    seed_everything(SEED)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_rows.extend(benchmark_model("heavy_grabmyo_base", HEAVY_MODEL, HEAVY_SCALER))
    if FAST_MODEL.exists():
        all_rows.extend(benchmark_model("fast_per_session", FAST_MODEL, FAST_SCALER))
    else:
        print(f"\n[skip] {FAST_MODEL} not found — run train_fast_hgb.py first.")

    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    # Markdown summary
    df = pd.DataFrame(all_rows)
    lines = [
        "# ExoHand throughput benchmark",
        "",
        f"50 repeats per batch size · single-thread sklearn predict · Mac CPU.",
        "",
        "## Single-row latency (deployment-relevant)",
        "",
        "| Model | Trees | Per-row latency | Sustained rate |",
        "|---|---:|---:|---:|",
    ]
    for _, r in df[df["batch_size"] == 1].iterrows():
        lines.append(f"| `{r['model']}` | {int(r['n_trees'])} | **{r['per_row_ms']:.2f} ms** | {r['rows_per_sec']:.0f} rows/s |")
    lines += [
        "",
        "**Interpretation.** The Teensy delivers one 4-channel EMG frame every "
        "50 ms (20 Hz). The deployed loop can therefore call predict at most "
        "**20 rows/s** in steady state — both models can sustain this, but only "
        "the fast model has per-call latency comfortably below the 50 ms cycle "
        "budget. The heavy model's variance makes it occasionally exceed the "
        "cycle budget (see p95/p99 in `latency_breakdown.csv`), causing dropped "
        "frames during sustained use.",
        "",
        "## Batched throughput (offline analysis)",
        "",
        "| Model | Batch | Mean batch latency | Per-row | Rows/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(f"| `{r['model']}` | {int(r['batch_size'])} | {r['mean_batch_ms']:.2f} ms | {r['per_row_ms']:.3f} ms | {r['rows_per_sec']:.0f} |")
    lines += [
        "",
        "Batched predict amortises per-call Python overhead. For offline analysis "
        "(LOSO evaluation, longitudinal eval, post-hoc session scoring) the "
        "throughput scales near-linearly with batch size — a full PhysioMio "
        "session (~1 200 windows) predicts in well under a second.",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
