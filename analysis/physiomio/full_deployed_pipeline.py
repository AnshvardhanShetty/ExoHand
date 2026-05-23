"""
Full deployed-runtime pipeline applied to PhysioMio per-window probabilities.

Replicates the post-classifier processing in `runtime/run_exohand.py`:

    1. EMA on predict_proba          (_smooth_proba, alpha = proba_ema_alpha)
    2. argmax of smoothed proba       → pred
    3. confidence = max(smoothed)
    4. Stability filter: N consecutive same predictions before output changes
    5. Cooldown:        no transitions within cooldown_ms of the last one
    6. Hysteresis:      enter / exit thresholds on confidence
    7. Confidence floor: if effective != rest and confidence < floor → rest

All 5 assist profiles (`runtime/assist_profile.py`) are evaluated and a
"raw" baseline (no smoothing) is included for reference. The deployed
default is **Level 3 / Moderate Assist** (per `run_exohand.py` arg default).

Adaptive gain (which scales hysteresis thresholds by current EMG gain)
is NOT applied in this offline replay because we don't have the gain
history. The deployed runtime would apply additional threshold reduction
(by factor 0.4-1.0) under weak-signal conditions, which would make the
deployment slightly more lenient than these numbers — so the values here
are a conservative lower bound on true deployed transition accuracy.

Inputs:
  analysis/physiomio/results/per_window_probas.parquet
  runtime/assist_profile.py  (parameter definitions)

Outputs:
  analysis/physiomio/results/full_deployed_pipeline.{csv,md,json}
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from analysis.seed import SEED, seed_everything
from analysis.physiomio.transition_accuracy import (
    compute_session_transitions, bootstrap_mean_ci, REACTION_BUFFER_WIN,
    MIN_HOLD_WINDOWS, DEFAULT_MAINT_CAP,
)
from runtime.assist_profile import ASSIST_PROFILES, get_profile


PROBAS_PARQUET = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "per_window_probas.parquet"
OUT_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "full_deployed_pipeline.csv"
OUT_JSON = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "full_deployed_pipeline.json"
OUT_MD = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "full_deployed_pipeline.md"

WINDOW_MS = 50

# In PhysioMio data: rest=0, close=1, open=2
REST_IDX = 0


def apply_full_pipeline(proba_stream: np.ndarray, alpha: float, N: int,
                        enter_thresh: float, exit_thresh: float, floor_thresh: float,
                        cooldown_ms: float) -> np.ndarray:
    """Replicate runtime/_smooth_proba + _apply_stability faithfully.

    Args:
        proba_stream: (T, 3) array with class order [rest, close, open]
        alpha, N, enter_thresh, exit_thresh, floor_thresh, cooldown_ms: profile params

    Returns:
        effective_intent: (T,) int array of motor-command intents per window
    """
    T = len(proba_stream)
    out = np.zeros(T, dtype=np.int8)
    if T == 0:
        return out

    cooldown_windows = int(cooldown_ms / WINDOW_MS)

    # State
    smoothed = np.array([1.0, 0.0, 0.0], dtype=np.float64)   # start as rest (rest=0)
    current_intent = REST_IDX
    recent_preds = []
    last_transition_window = -cooldown_windows - 1   # so no cooldown at t=0

    for t in range(T):
        # 1. EMA smoothing
        smoothed = alpha * smoothed + (1.0 - alpha) * proba_stream[t].astype(np.float64)
        s = smoothed.sum()
        if s > 0:
            smoothed = smoothed / s

        # 2. argmax + confidence
        pred = int(np.argmax(smoothed))
        confidence = float(smoothed[pred])

        # 3. Stability filter (N consecutive same)
        recent_preds.append(pred)
        if len(recent_preds) > N:
            recent_preds.pop(0)
        stable = (len(recent_preds) == N) and all(p == pred for p in recent_preds)

        # 4. Cooldown + hysteresis transitions
        if stable and pred != current_intent and (t - last_transition_window) >= cooldown_windows:
            if current_intent == REST_IDX:
                # Leaving rest → movement: enter threshold
                if confidence >= enter_thresh:
                    current_intent = pred
                    last_transition_window = t
            else:
                if pred == REST_IDX:
                    # Going to rest: exit threshold
                    if confidence >= exit_thresh:
                        current_intent = pred
                        last_transition_window = t
                else:
                    # Switching close↔open: enter threshold
                    if confidence >= enter_thresh:
                        current_intent = pred
                        last_transition_window = t

        # 5. Confidence floor: drop to rest if confidence too low
        effective = current_intent
        if effective != REST_IDX and confidence < floor_thresh:
            effective = REST_IDX

        out[t] = effective

    return out


def evaluate_profile(probas_all: pd.DataFrame,
                     alpha: float, N: int,
                     enter_thresh: float, exit_thresh: float, floor_thresh: float,
                     cooldown_ms: float,
                     reaction_buffer: int, maint_error_frac: float,
                     min_hold: int, maint_cap: int) -> dict:
    """Apply pipeline + compute transition accuracy across all sessions."""
    session_summaries = []

    for (patient, session), group in probas_all.groupby(["participant", "session"]):
        g = group.sort_values(["trial", "t_rel_s"]).reset_index(drop=True)
        proba_arr = g[["proba_rest", "proba_close", "proba_open"]].values.astype(np.float32)
        effective = apply_full_pipeline(
            proba_arr, alpha, N, enter_thresh, exit_thresh, floor_thresh, cooldown_ms,
        )
        # Feed through the standard transition-accuracy logic
        sm_df = pd.DataFrame({
            "trial": g["trial"].values,
            "t_rel_s": g["t_rel_s"].values,
            "gt_intent": g["gt_intent"].values,
            "pred_intent": effective,
        })
        _, summary = compute_session_transitions(
            sm_df, reaction_buffer, maint_error_frac, min_hold, maint_cap,
        )
        summary["participant"] = patient
        summary["session"] = session
        summary["arm"] = group["arm"].iloc[0]
        session_summaries.append(summary)

    ok = pd.DataFrame(session_summaries).dropna(subset=["transition_acc"])
    session_mean, session_lo, session_hi = bootstrap_mean_ci(ok["transition_acc"].values)
    pat_means = ok.groupby("participant")["transition_acc"].mean().values
    pat_mean, pat_lo, pat_hi = bootstrap_mean_ci(pat_means)
    raw_acc = float(ok["raw_acc"].mean())
    return {
        "n_sessions": int(len(ok)),
        "n_patients": int(len(pat_means)),
        "session_transition_acc": session_mean,
        "session_ci_lo": session_lo, "session_ci_hi": session_hi,
        "patient_transition_acc": pat_mean,
        "patient_ci_lo": pat_lo, "patient_ci_hi": pat_hi,
        "raw_full_stream_acc": raw_acc,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-maint-error", type=float, default=0.0,
                        help="Maintenance error tolerance (0.0 = strict / ReactEMG-faithful)")
    parser.add_argument("--reaction-buffer", type=int, default=REACTION_BUFFER_WIN)
    parser.add_argument("--maint-cap", type=int, default=DEFAULT_MAINT_CAP)
    parser.add_argument("--min-hold", type=int, default=MIN_HOLD_WINDOWS)
    args = parser.parse_args()

    seed_everything(SEED)

    if not PROBAS_PARQUET.exists():
        print(f"ERROR: {PROBAS_PARQUET} not found. Run save_predict_probas.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {PROBAS_PARQUET}...")
    probas_all = pd.read_parquet(PROBAS_PARQUET)
    print(f"  {len(probas_all):,} windows × {probas_all['participant'].nunique()} patients × "
          f"{probas_all.groupby('participant')['session'].nunique().sum()} sessions")
    print(f"params: reaction_buffer={args.reaction_buffer}, maint_cap={args.maint_cap}, "
          f"max_maint_error={args.max_maint_error}")

    rows = []
    # First: raw classifier (no pipeline) for reference — argmax of un-smoothed proba per window
    print(f"\n--- Raw classifier (no smoothing, no hysteresis, no cooldown) ---")
    raw_summaries = []
    for (patient, session), group in probas_all.groupby(["participant", "session"]):
        g = group.sort_values(["trial", "t_rel_s"]).reset_index(drop=True)
        proba_arr = g[["proba_rest", "proba_close", "proba_open"]].values.astype(np.float32)
        raw_pred = np.argmax(proba_arr, axis=1).astype(np.int8)
        sm_df = pd.DataFrame({"trial": g["trial"].values, "t_rel_s": g["t_rel_s"].values,
                              "gt_intent": g["gt_intent"].values, "pred_intent": raw_pred})
        _, summary = compute_session_transitions(sm_df, args.reaction_buffer, args.max_maint_error,
                                                 args.min_hold, args.maint_cap)
        summary["participant"] = patient
        raw_summaries.append(summary)
    raw_ok = pd.DataFrame(raw_summaries).dropna(subset=["transition_acc"])
    raw_session, raw_lo, raw_hi = bootstrap_mean_ci(raw_ok["transition_acc"].values)
    raw_pat = raw_ok.groupby("participant")["transition_acc"].mean().values
    raw_pat_mean, raw_pat_lo, raw_pat_hi = bootstrap_mean_ci(raw_pat)
    raw_acc = float(raw_ok["raw_acc"].mean())
    print(f"  patient_acc={raw_pat_mean:.4f} [{raw_pat_lo:.4f}, {raw_pat_hi:.4f}]  raw_acc={raw_acc:.4f}")
    rows.append({
        "config": "raw_classifier",
        "level": 0, "label": "Raw HGB (no pipeline)",
        "N": 1, "alpha": 0.0, "enter": 0.0, "exit": 0.0, "floor": 0.0, "cooldown_ms": 0,
        "patient_transition_acc": raw_pat_mean, "patient_ci_lo": raw_pat_lo, "patient_ci_hi": raw_pat_hi,
        "session_transition_acc": raw_session, "session_ci_lo": raw_lo, "session_ci_hi": raw_hi,
        "raw_full_stream_acc": raw_acc,
        "added_latency_ms": 0,
    })

    # Sweep all 5 deployed profiles
    print(f"\n{'level':>5}  {'label':<24}  {'N':>2}  {'α':>4}  {'en':>4}  {'ex':>4}  {'fl':>4}  {'cd_ms':>5}  {'patient_acc':<22}  {'raw_acc':>8}  {'+lat':>5}")
    for lvl in [1, 2, 3, 4, 5]:
        p = get_profile(lvl)
        result = evaluate_profile(
            probas_all,
            alpha=p.proba_ema_alpha, N=p.stability_required,
            enter_thresh=p.hysteresis_enter, exit_thresh=p.hysteresis_exit,
            floor_thresh=p.confidence_floor, cooldown_ms=p.cooldown_ms,
            reaction_buffer=args.reaction_buffer, maint_error_frac=args.max_maint_error,
            min_hold=args.min_hold, maint_cap=args.maint_cap,
        )
        added_latency = (p.stability_required - 1) * WINDOW_MS
        deployed_marker = " ← deployed default" if lvl == 3 else ""
        rows.append({
            "config": f"level_{lvl}",
            "level": lvl, "label": p.label,
            "N": p.stability_required, "alpha": p.proba_ema_alpha,
            "enter": p.hysteresis_enter, "exit": p.hysteresis_exit,
            "floor": p.confidence_floor, "cooldown_ms": p.cooldown_ms,
            "patient_transition_acc": result["patient_transition_acc"],
            "patient_ci_lo": result["patient_ci_lo"], "patient_ci_hi": result["patient_ci_hi"],
            "session_transition_acc": result["session_transition_acc"],
            "session_ci_lo": result["session_ci_lo"], "session_ci_hi": result["session_ci_hi"],
            "raw_full_stream_acc": result["raw_full_stream_acc"],
            "added_latency_ms": added_latency,
        })
        ci = f"{result['patient_transition_acc']:.4f} [{result['patient_ci_lo']:.4f}, {result['patient_ci_hi']:.4f}]"
        print(f"{lvl:>5}  {p.label:<24}  {p.stability_required:>2}  {p.proba_ema_alpha:>4.2f}  "
              f"{p.hysteresis_enter:>4.2f}  {p.hysteresis_exit:>4.2f}  {p.confidence_floor:>4.2f}  "
              f"{int(p.cooldown_ms):>5}  {ci:<22}  {result['raw_full_stream_acc']:>8.4f}  "
              f"{added_latency:>3} ms{deployed_marker}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps({"params": vars(args), "rows": df.to_dict("records")}, indent=2))

    label = ("strict (ReactEMG-faithful, 0% maint error)" if args.max_maint_error == 0.0
             else f"relaxed ({args.max_maint_error:.0%} maint error tolerance)")
    md = [
        f"# Full deployed pipeline — transition accuracy ({label})",
        "",
        f"Per-window probabilities (`per_window_probas.parquet`) passed through the full deployed runtime pipeline (EMA → argmax → stability → cooldown → hysteresis → confidence floor), exactly as in `runtime/run_exohand.py:_apply_stability + _smooth_proba`, with parameters from `runtime/assist_profile.py` for each of the five assist levels.",
        "",
        f"Common: reaction buffer = {args.reaction_buffer} win ({args.reaction_buffer*50} ms), maint cap = {args.maint_cap} win ({args.maint_cap*50} ms), maint error tolerance = {args.max_maint_error:.0%}.",
        "",
        "## Sweep across deployed profiles",
        "",
        "| Config | Level | Label | N | α | enter | exit | floor | cd(ms) | Patient transition acc (95 % CI) | Full-stream raw acc | +Latency |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        ci = f"[{r['patient_ci_lo']:.3f}, {r['patient_ci_hi']:.3f}]"
        config_label = "raw" if r["level"] == 0 else f"L{r['level']}"
        deployed = " ★" if r["level"] == 3 else ""
        md.append(
            f"| {config_label}{deployed} | {r['level']} | {r['label']} | "
            f"{int(r['N'])} | {r['alpha']:.2f} | {r['enter']:.2f} | {r['exit']:.2f} | "
            f"{r['floor']:.2f} | {int(r['cooldown_ms'])} | "
            f"**{r['patient_transition_acc']:.3f}** {ci} | "
            f"{r['raw_full_stream_acc']:.3f} | +{int(r['added_latency_ms'])} ms |"
        )
    md += [
        "",
        "★ = deployed default (`runtime/run_exohand.py --assist-level 3`).",
        "",
        "## Caveats",
        "",
        "- Adaptive gain (per-channel signal-strength threshold scaling, `_apply_adaptive_gain`) is not replayed offline because we don't have the gain history. In live deployment, weak-signal patients (e.g. severely impaired) get hysteresis thresholds scaled by 0.4-0.7×, which would make transitions *easier* to accept and likely raise these numbers. The values here are therefore a slight under-estimate of true deployed transition accuracy on the most-impaired sub-cohort.",
        "- Adaptive confidence floor reductions (same source) are similarly not replayed.",
        "- All 5 profiles are evaluated; the deployed default in `run_exohand.py` is **Level 3** (Moderate Assist).",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
