"""
Transition accuracy under the deployed runtime's stability filter.

The deployed system (`runtime/run_exohand.py:_apply_stability`,
`runtime/assist_profile.py`) applies an N-window consistency requirement
to per-window classifier output before changing motor commands:

    recent_preds = deque(maxlen=N)
    output = initial_state
    for raw_pred in stream:
        recent_preds.append(raw_pred)
        if len(recent_preds) == N and all(p == raw_pred for p in recent_preds):
            output = raw_pred       # accept the new intent
        emit(output)                # else hold previous intent

This script applies the same logic to PhysioMio per-window predictions
(`results/per_window_predictions.parquet`) and recomputes transition accuracy
for the SMOOTHED output stream. We sweep N ∈ {1, 2, 3, 5, 10} to map the
tradeoff between flicker suppression and added decision latency.

N → assist profile (assist_profile.py):
    N = 1: levels 1-2 (Max / High Assist)
    N = 2: level 3 (Moderate Assist)
    N = 3: levels 4-5 (Light / Minimal Assist)    ← deployed default
    N = 5, 10: not in any profile, sensitivity sweep only

Added latency: changing intent now requires (N-1) consecutive in-class
predictions on top of the raw classification, so each transition incurs
up to (N-1)×50 ms of additional decision latency before the new motor
command is issued.

Note: this script implements only the STABILITY layer. The deployed system
also applies (a) EMA on predict_proba, (b) hysteresis on confidence, (c)
cooldown, (d) confidence floor. Those layers act on `predict_proba` not on
`predict`, so reproducing them requires re-running model inference with
`predict_proba` on the cached per-session models. The stability layer
alone captures the dominant flicker-suppression effect; we document the
remaining layers in §6.

Outputs:
    analysis/physiomio/results/deployed_transition_accuracy.csv
    analysis/physiomio/results/deployed_transition_accuracy.md
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
    MIN_HOLD_WINDOWS, DEFAULT_MAINT_CAP, CLASSES, CLASS_NAMES,
)


PREDS_PARQUET = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "per_window_predictions.parquet"
OUT_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "deployed_transition_accuracy.csv"
OUT_MD = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "deployed_transition_accuracy.md"
OUT_JSON = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "deployed_transition_accuracy.json"

# N values to sweep — including the actual deployed levels (1, 2, 3) plus
# heavier smoothing (5, 10) for sensitivity.
N_SWEEP = [1, 2, 3, 5, 10]
PROFILE_FOR_N = {
    1: "Max / High Assist (Levels 1-2)",
    2: "Moderate Assist (Level 3)",
    3: "Light / Minimal Assist (Levels 4-5) — deployed default",
    5: "(sensitivity only — not in any profile)",
    10: "(sensitivity only — not in any profile)",
}


def apply_stability_filter(raw_preds: np.ndarray, N: int) -> np.ndarray:
    """Replicate runtime/run_exohand.py:_apply_stability — N-window consistency.

    Output is initialised to the first prediction (warm assumption — in
    deployment the system warms up over rest baseline before patient effort).
    """
    if N <= 1:
        return raw_preds.copy()
    out = np.empty_like(raw_preds)
    current = int(raw_preds[0])
    out[0] = current
    # Track last N raw predictions
    window = [current]
    for i in range(1, len(raw_preds)):
        p = int(raw_preds[i])
        window.append(p)
        if len(window) > N:
            window.pop(0)
        if len(window) == N and all(w == p for w in window) and p != current:
            current = p
        out[i] = current
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-maint-error", type=float, default=0.0,
                        help="Maintenance error tolerance (0.0 = strict / ReactEMG-faithful)")
    parser.add_argument("--reaction-buffer", type=int, default=REACTION_BUFFER_WIN)
    parser.add_argument("--maint-cap", type=int, default=DEFAULT_MAINT_CAP)
    parser.add_argument("--min-hold", type=int, default=MIN_HOLD_WINDOWS)
    args = parser.parse_args()

    seed_everything(SEED)
    preds_all = pd.read_parquet(PREDS_PARQUET)
    print(f"Loading {PREDS_PARQUET}: {len(preds_all):,} windows × "
          f"{preds_all['participant'].nunique()} patients × "
          f"{preds_all.groupby('participant')['session'].nunique().sum()} sessions")
    print(f"params: reaction_buffer={args.reaction_buffer} win, maint_cap={args.maint_cap} win, "
          f"max_maint_error={args.max_maint_error}")

    # Sweep N values
    rows = []
    print(f"\n{'N':>3} {'profile':<55} {'patient_acc':>12} {'session_acc':>12} {'raw_acc':>10} {'+latency_ms':>12}")
    for N in N_SWEEP:
        # Apply filter per session, then run the same transition logic
        smoothed_dfs = []
        for (patient, session), group in preds_all.groupby(["participant", "session"]):
            g = group.sort_values(["trial", "t_rel_s"]).reset_index(drop=True)
            smoothed = apply_stability_filter(g["pred_intent"].values.astype(np.int8), N)
            smoothed_df = g.copy()
            smoothed_df["pred_intent"] = smoothed
            smoothed_dfs.append(smoothed_df)
        smoothed_all = pd.concat(smoothed_dfs, ignore_index=True)

        # Compute transition accuracy (strict + relaxed) on smoothed stream
        session_rows = []
        for (patient, session), group in smoothed_all.groupby(["participant", "session"]):
            _, summary = compute_session_transitions(
                group, args.reaction_buffer, args.max_maint_error,
                args.min_hold, args.maint_cap,
            )
            summary["participant"] = patient
            summary["session"] = session
            summary["arm"] = group["arm"].iloc[0]
            session_rows.append(summary)
        ok = pd.DataFrame(session_rows).dropna(subset=["transition_acc"])
        session_mean, session_lo, session_hi = bootstrap_mean_ci(ok["transition_acc"].values)
        pat_means = ok.groupby("participant")["transition_acc"].mean().values
        pat_mean, pat_lo, pat_hi = bootstrap_mean_ci(pat_means)
        raw_mean = float((smoothed_all["pred_intent"] == smoothed_all["gt_intent"]).mean())
        added_latency_ms = (N - 1) * 50

        rows.append({
            "N": N,
            "profile": PROFILE_FOR_N.get(N, ""),
            "patient_transition_acc": pat_mean,
            "patient_ci_lo": pat_lo,
            "patient_ci_hi": pat_hi,
            "session_transition_acc": session_mean,
            "session_ci_lo": session_lo,
            "session_ci_hi": session_hi,
            "raw_full_stream_acc": raw_mean,
            "added_latency_ms": added_latency_ms,
        })
        print(f"{N:>3} {PROFILE_FOR_N.get(N, ''):<55} "
              f"{pat_mean:.4f} [{pat_lo:.4f},{pat_hi:.4f}]  "
              f"{session_mean:.4f}        {raw_mean:.4f}        {added_latency_ms:>3} ms")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    # Markdown summary
    suffix = "strict" if args.max_maint_error == 0.0 else f"relaxed_{args.max_maint_error:.0%}"
    label = ("strict (ReactEMG-faithful, 0% maint error)" if args.max_maint_error == 0.0
             else f"relaxed ({args.max_maint_error:.0%} maint error tolerance)")
    md = [
        f"# Deployed-configuration transition accuracy ({label})",
        "",
        f"Per-window predictions in `per_window_predictions.parquet` were "
        f"passed through the runtime's N-window consistency filter "
        f"(`runtime/run_exohand.py:_apply_stability`, `stability_required` "
        f"parameter from `runtime/assist_profile.py`). The deployed assist "
        f"profile uses **N=3** (Light / Minimal Assist), corresponding to "
        f"100 ms of added decision latency before a new motor command is "
        f"issued.",
        "",
        f"Common parameters across all N: reaction buffer = {args.reaction_buffer} windows "
        f"({args.reaction_buffer*50} ms), maintenance cap = {args.maint_cap} windows "
        f"({args.maint_cap*50} ms), maintenance error tolerance = "
        f"{args.max_maint_error:.0%}.",
        "",
        "## Sweep results",
        "",
        "| N | Profile | Patient-level transition acc (95 % CI) | Raw acc (full-stream, post-filter) | Added latency |",
        "|---:|---|---|---:|---:|",
    ]
    for _, r in df.iterrows():
        ci = f"[{r['patient_ci_lo']:.3f}, {r['patient_ci_hi']:.3f}]"
        md.append(f"| **{int(r['N'])}** | {r['profile']} | **{r['patient_transition_acc']:.3f}** {ci} | {r['raw_full_stream_acc']:.3f} | +{int(r['added_latency_ms'])} ms |")
    md += [
        "",
        "## Comparison to raw classifier output (no stability filter)",
        "",
        f"Without the stability filter (N = 1, equivalent to "
        f"`transition_accuracy.py` baseline): "
        f"patient mean **{df[df['N']==1]['patient_transition_acc'].iloc[0]:.3f}**.",
        "",
        f"With the deployed N = 3 stability filter: patient mean "
        f"**{df[df['N']==3]['patient_transition_acc'].iloc[0]:.3f}**, "
        f"a **+{(df[df['N']==3]['patient_transition_acc'].iloc[0] - df[df['N']==1]['patient_transition_acc'].iloc[0])*100:.1f} pp** "
        f"absolute improvement at the cost of +100 ms decision latency.",
        "",
        "ReactEMG Stroke's reported transition accuracy (their best: 0.61 with "
        "LoRA full fine-tuning) is measured on their raw classifier output. "
        "The deployed configuration of our system applies additional "
        "stability filtering before motor command — a documented difference "
        "in evaluation framing.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")

    OUT_JSON.write_text(json.dumps({"label": label, "params": vars(args),
                                    "rows": df.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
