"""
Lucchetti full deployed pipeline — applies the same 6-layer post-processing
as analysis/physiomio/full_deployed_pipeline.py to Lucchetti per-window
probabilities. Sweeps all 5 assist profiles + raw baseline.

Outputs:
  analysis/lucchetti/results/full_deployed_pipeline_strict.{md,csv,json}
  analysis/lucchetti/results/full_deployed_pipeline_relaxed.{md,csv,json}
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from analysis.seed import SEED, seed_everything
from analysis.physiomio.transition_accuracy import (
    compute_session_transitions, bootstrap_mean_ci,
    REACTION_BUFFER_WIN, MIN_HOLD_WINDOWS, DEFAULT_MAINT_CAP,
)
from analysis.physiomio.full_deployed_pipeline import apply_full_pipeline, WINDOW_MS
from runtime.assist_profile import get_profile

PROBAS = PROJECT_ROOT / "analysis" / "lucchetti" / "results" / "per_window_probas.parquet"
OUT_DIR = PROJECT_ROOT / "analysis" / "lucchetti" / "results"


def evaluate_profile(probas_all, alpha, N, enter_t, exit_t, floor_t, cooldown_ms,
                     reaction_buf, max_err, min_hold, maint_cap):
    sess = []
    for (subj, session), g in probas_all.groupby(["participant", "session"]):
        gg = g.sort_values(["trial", "t_rel_s"]).reset_index(drop=True)
        proba = gg[["proba_rest", "proba_close", "proba_open"]].values.astype(np.float32)
        effective = apply_full_pipeline(proba, alpha, N, enter_t, exit_t, floor_t, cooldown_ms)
        smd = pd.DataFrame({
            "trial": gg["trial"].values, "t_rel_s": gg["t_rel_s"].values,
            "gt_intent": gg["gt_intent"].values, "pred_intent": effective,
        })
        _, summary = compute_session_transitions(smd, reaction_buf, max_err, min_hold, maint_cap)
        summary["participant"] = subj
        summary["arm"] = g["arm"].iloc[0]
        sess.append(summary)
    ok = pd.DataFrame(sess).dropna(subset=["transition_acc"])
    sm, slo, shi = bootstrap_mean_ci(ok["transition_acc"].values)
    pm = ok.groupby("participant")["transition_acc"].mean().values
    pmm, plo, phi = bootstrap_mean_ci(pm)
    raw_m = float(ok["raw_acc"].mean())
    return {
        "session_mean": sm, "session_ci": [slo, shi],
        "patient_mean": pmm, "patient_ci": [plo, phi],
        "raw_acc": raw_m, "n_sessions": int(len(ok)),
    }


def main():
    seed_everything(SEED)
    probas = pd.read_parquet(PROBAS)
    print(f"Loaded {len(probas):,} windows × {probas['participant'].nunique()} subjects × "
          f"{probas.groupby('participant')['session'].nunique().sum()} sessions")

    for max_err, label in [(0.0, "strict"), (0.10, "relaxed")]:
        rows = []
        # Raw classifier baseline
        raw_results = []
        for (subj, session), g in probas.groupby(["participant", "session"]):
            gg = g.sort_values(["trial", "t_rel_s"]).reset_index(drop=True)
            proba = gg[["proba_rest", "proba_close", "proba_open"]].values.astype(np.float32)
            raw_pred = np.argmax(proba, axis=1).astype(np.int8)
            smd = pd.DataFrame({
                "trial": gg["trial"].values, "t_rel_s": gg["t_rel_s"].values,
                "gt_intent": gg["gt_intent"].values, "pred_intent": raw_pred,
            })
            _, s = compute_session_transitions(smd, REACTION_BUFFER_WIN, max_err,
                                               MIN_HOLD_WINDOWS, DEFAULT_MAINT_CAP)
            s["participant"] = subj
            raw_results.append(s)
        raw_ok = pd.DataFrame(raw_results).dropna(subset=["transition_acc"])
        raw_pm = raw_ok.groupby("participant")["transition_acc"].mean().values
        raw_p_mean, raw_p_lo, raw_p_hi = bootstrap_mean_ci(raw_pm)
        rows.append({
            "level": 0, "label": "Raw HGB (no pipeline)", "N": 1, "alpha": 0.0,
            "enter": 0.0, "exit": 0.0, "floor": 0.0, "cooldown_ms": 0,
            "patient_transition_acc": raw_p_mean, "patient_ci_lo": raw_p_lo, "patient_ci_hi": raw_p_hi,
            "raw_full_stream_acc": float(raw_ok["raw_acc"].mean()),
            "added_latency_ms": 0,
        })

        print(f"\n[{label}] Raw HGB: patient_mean = {raw_p_mean:.4f} [{raw_p_lo:.4f}, {raw_p_hi:.4f}]")
        print(f"{'lvl':>4}  {'label':<24}  {'N':>2}  {'α':>4}  patient_acc                latency")
        for lvl in [1, 2, 3, 4, 5]:
            p = get_profile(lvl)
            r = evaluate_profile(
                probas, p.proba_ema_alpha, p.stability_required,
                p.hysteresis_enter, p.hysteresis_exit, p.confidence_floor, p.cooldown_ms,
                REACTION_BUFFER_WIN, max_err, MIN_HOLD_WINDOWS, DEFAULT_MAINT_CAP,
            )
            added_lat = (p.stability_required - 1) * WINDOW_MS
            marker = " ← deployed default" if lvl == 3 else ""
            rows.append({
                "level": lvl, "label": p.label,
                "N": p.stability_required, "alpha": p.proba_ema_alpha,
                "enter": p.hysteresis_enter, "exit": p.hysteresis_exit,
                "floor": p.confidence_floor, "cooldown_ms": p.cooldown_ms,
                "patient_transition_acc": r["patient_mean"],
                "patient_ci_lo": r["patient_ci"][0], "patient_ci_hi": r["patient_ci"][1],
                "raw_full_stream_acc": r["raw_acc"],
                "added_latency_ms": added_lat,
            })
            print(f"{lvl:>4}  {p.label:<24}  {p.stability_required:>2}  {p.proba_ema_alpha:>4.2f}  "
                  f"{r['patient_mean']:.4f} [{r['patient_ci'][0]:.4f}, {r['patient_ci'][1]:.4f}]  +{added_lat} ms{marker}")

        df = pd.DataFrame(rows)
        df.to_csv(OUT_DIR / f"full_deployed_pipeline_{label}.csv", index=False)
        (OUT_DIR / f"full_deployed_pipeline_{label}.json").write_text(
            json.dumps({"label": label, "rows": df.to_dict("records")}, indent=2)
        )

        md = [
            f"# Lucchetti full deployed pipeline — {label}",
            "",
            f"Strict = 0% maint error tolerance; relaxed = 10% maint error tolerance. "
            f"Reaction buffer {REACTION_BUFFER_WIN*50} ms, maint cap {DEFAULT_MAINT_CAP*50} ms.",
            "",
            "| Config | Level | Label | N | α | enter | exit | floor | cd(ms) | Patient transition acc | Raw full-stream | +Latency |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
        for r in rows:
            ci = f"[{r['patient_ci_lo']:.3f}, {r['patient_ci_hi']:.3f}]"
            config = "raw" if r["level"] == 0 else f"L{r['level']}"
            marker = " ★" if r["level"] == 3 else ""
            md.append(
                f"| {config}{marker} | {r['level']} | {r['label']} | "
                f"{int(r['N'])} | {r['alpha']:.2f} | {r['enter']:.2f} | {r['exit']:.2f} | "
                f"{r['floor']:.2f} | {int(r['cooldown_ms'])} | "
                f"**{r['patient_transition_acc']:.3f}** {ci} | "
                f"{r['raw_full_stream_acc']:.3f} | +{int(r['added_latency_ms'])} ms |"
            )
        md.append("\n★ = deployed default.")
        (OUT_DIR / f"full_deployed_pipeline_{label}.md").write_text("\n".join(md))
        print(f"  Wrote full_deployed_pipeline_{label}.{{csv,json,md}}")


if __name__ == "__main__":
    main()
