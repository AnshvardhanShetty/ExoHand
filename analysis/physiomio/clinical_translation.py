"""
Clinical-outcome translation of the deployed-pipeline transition accuracy.

Raw accuracy numbers don't speak to clinical relevance. This script translates
the deployed pipeline's behaviour into three clinically-meaningful metrics
that a rehab clinician would care about:

  1. **Per-rep success rate** — of N intended grasps in a session, what fraction
     does the system correctly recognise and *sustain* the motor command for?
  2. **False-activation rate during rest** — during patient rest, how often
     does the system incorrectly issue a close/open command? Reported as
     events per minute of rest.
  3. **Time-to-correct-command** — from cue/transition onset, how long until
     the first correct sustained prediction? (Latency-to-action.)

All metrics computed on the existing per-window predictions parquet, after
applying the deployed N=3 stability filter (Stage 2, Level 3 default profile).

Output:
  analysis/physiomio/results/clinical_translation.{csv,json,md}
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from analysis.seed import SEED, seed_everything

PREDS = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "per_window_predictions.parquet"
OUT_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "clinical_translation.csv"
OUT_JSON = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "clinical_translation.json"
OUT_MD = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "clinical_translation.md"

WINDOW_MS = 50
N_STABILITY = 3   # deployed default (L3 Moderate Assist)
SUSTAIN_MS = 250  # how long the correct prediction must be held to count as "success" (per-rep success)
N_BOOT = 2000


def apply_stage2(predictions: np.ndarray, N: int) -> np.ndarray:
    """N-window consistency filter — same as runtime/run_exohand.py:_apply_stability."""
    if N <= 1:
        return predictions.copy()
    out = np.empty_like(predictions)
    current = int(predictions[0])
    out[0] = current
    window = [current]
    for i in range(1, len(predictions)):
        p = int(predictions[i])
        window.append(p)
        if len(window) > N:
            window.pop(0)
        if len(window) == N and all(w == p for w in window) and p != current:
            current = p
        out[i] = current
    return out


def boot_mean_ci(x, n=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (float("nan"),) * 3
    idx = rng.randint(0, len(x), size=(n, len(x)))
    s = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def session_metrics(session_df: pd.DataFrame, N: int, sustain_windows: int) -> dict:
    """Per-session clinical metrics on the deployed (post-Stage 2) output."""
    g = session_df.sort_values(["trial", "t_rel_s"]).reset_index(drop=True)
    gt = g["gt_intent"].values
    raw_pred = g["pred_intent"].values
    pred = apply_stage2(raw_pred, N)

    # Identify rest-period windows and movement-period windows
    rest_mask = gt == 0
    n_rest_windows = int(rest_mask.sum())
    n_total_windows = len(gt)

    # 1. False-activation rate during rest — number of windows where GT=rest but pred ≠ rest
    n_false_act_windows = int(np.logical_and(rest_mask, pred != 0).sum())
    rest_duration_s = n_rest_windows * WINDOW_MS / 1000.0
    # Discrete false-activation EVENTS (transitions of pred from rest to non-rest during GT-rest)
    if n_rest_windows > 0:
        in_rest = rest_mask
        prev_was_rest = np.concatenate([[True], pred[:-1] == 0])
        false_act_events = int(np.logical_and.reduce([
            in_rest, pred != 0, prev_was_rest,
        ]).sum())
        false_act_per_min_rest = false_act_events / (rest_duration_s / 60.0) if rest_duration_s > 0 else float("nan")
    else:
        false_act_events = 0
        false_act_per_min_rest = float("nan")

    # 2. Per-rep success rate — for each ground-truth transition into close/open,
    #    does the system produce a *sustained* (≥ sustain_windows windows) correct prediction
    #    within the new segment?
    change_idx = np.where(np.diff(gt) != 0)[0] + 1
    # Boundaries: [0, change_idx..., end]
    boundaries = np.concatenate([[0], change_idx, [len(g)]])
    segments = list(zip(boundaries[:-1], boundaries[1:]))

    n_grasps = 0   # ground-truth transitions into close or open
    n_grasps_succeeded = 0
    time_to_correct_ms = []
    for i in range(1, len(segments)):
        s, e = segments[i]
        new_class = int(gt[s])
        if new_class == 0:
            continue  # not a "grasp" (transition into rest)
        n_grasps += 1
        seg_pred = pred[s:e]
        # Did we hit `new_class` for at least sustain_windows consecutive windows somewhere in this segment?
        if len(seg_pred) < sustain_windows:
            continue
        match = (seg_pred == new_class).astype(int)
        # Convolve to find any run of sustain_windows consecutive matches
        if len(match) >= sustain_windows:
            run = np.convolve(match, np.ones(sustain_windows, dtype=int), mode="valid")
            if (run >= sustain_windows).any():
                n_grasps_succeeded += 1
                # First sustained correct prediction position → time-to-correct from segment start
                first_run_start = int(np.where(run >= sustain_windows)[0][0])
                time_to_correct_ms.append(first_run_start * WINDOW_MS)

    per_rep_success = n_grasps_succeeded / n_grasps if n_grasps > 0 else float("nan")
    mean_ttc = float(np.mean(time_to_correct_ms)) if time_to_correct_ms else float("nan")
    median_ttc = float(np.median(time_to_correct_ms)) if time_to_correct_ms else float("nan")

    return {
        "patient": g["participant"].iloc[0],
        "session": g["session"].iloc[0],
        "arm": g["arm"].iloc[0],
        "n_rest_windows": n_rest_windows,
        "rest_duration_s": rest_duration_s,
        "n_false_activation_windows": n_false_act_windows,
        "n_false_activation_events": false_act_events,
        "false_act_per_min_rest": false_act_per_min_rest,
        "n_grasps": n_grasps,
        "n_grasps_succeeded": n_grasps_succeeded,
        "per_rep_success_rate": per_rep_success,
        "mean_time_to_correct_ms": mean_ttc,
        "median_time_to_correct_ms": median_ttc,
    }


def main():
    seed_everything(SEED)
    df = pd.read_parquet(PREDS)
    print(f"Loaded {len(df):,} windows × {df['participant'].nunique()} patients × "
          f"{df.groupby('participant')['session'].nunique().sum()} sessions")
    print(f"Applying Stage 2 N={N_STABILITY} (deployed default) + computing clinical metrics...")

    sustain_windows = int(SUSTAIN_MS / WINDOW_MS)
    rows = []
    for (subj, session), g in df.groupby(["participant", "session"]):
        rows.append(session_metrics(g, N_STABILITY, sustain_windows))
    sess_df = pd.DataFrame(rows)
    sess_df.to_csv(OUT_CSV, index=False)

    # ─── Aggregate ──
    valid_rep = sess_df["per_rep_success_rate"].dropna().values
    valid_fa = sess_df["false_act_per_min_rest"].dropna().values
    valid_ttc = sess_df["mean_time_to_correct_ms"].dropna().values

    rep_mean, rep_lo, rep_hi = boot_mean_ci(valid_rep)
    fa_mean, fa_lo, fa_hi = boot_mean_ci(valid_fa)
    ttc_mean, ttc_lo, ttc_hi = boot_mean_ci(valid_ttc)

    # Per-arm
    by_arm = {}
    for arm in ["healthy", "impaired"]:
        sub = sess_df[sess_df["arm"] == arm]
        if len(sub) == 0: continue
        rep_m, rep_l, rep_h = boot_mean_ci(sub["per_rep_success_rate"].dropna().values)
        fa_m, fa_l, fa_h = boot_mean_ci(sub["false_act_per_min_rest"].dropna().values)
        ttc_m, ttc_l, ttc_h = boot_mean_ci(sub["mean_time_to_correct_ms"].dropna().values)
        by_arm[arm] = {
            "n_sessions": int(len(sub)),
            "per_rep_success": {"mean": rep_m, "ci": [rep_l, rep_h]},
            "false_act_per_min_rest": {"mean": fa_m, "ci": [fa_l, fa_h]},
            "time_to_correct_ms": {"mean": ttc_m, "ci": [ttc_l, ttc_h]},
        }

    # Also report on a typical session — assume 10 grasp reps/session for narrative
    EXPECTED_REPS_PER_SESSION = 10
    expected_successful_reps = rep_mean * EXPECTED_REPS_PER_SESSION
    # And false activations per a hypothetical 5-minute therapy session at 70% rest
    hyp_rest_min = 5 * 0.7   # 5 min session, 70% rest
    false_act_per_session = fa_mean * hyp_rest_min if not np.isnan(fa_mean) else float("nan")

    summary = {
        "config": {
            "stability_N": N_STABILITY,
            "sustain_window_ms": SUSTAIN_MS,
            "window_ms": WINDOW_MS,
        },
        "n_sessions": int(len(sess_df)),
        "n_patients": int(sess_df["patient"].nunique()),
        "per_rep_success_rate": {
            "mean": rep_mean, "ci95": [rep_lo, rep_hi],
            "interpretation": (f"Of every {EXPECTED_REPS_PER_SESSION} cued grasps in a session, "
                              f"the deployed pipeline produces a sustained (≥{SUSTAIN_MS} ms) "
                              f"correct motor command for {expected_successful_reps:.1f}."),
        },
        "false_activation_per_min_rest": {
            "mean": fa_mean, "ci95": [fa_lo, fa_hi],
            "interpretation": (f"During patient rest periods, the deployed pipeline incorrectly "
                              f"issues a non-rest motor command {fa_mean:.2f} times per minute "
                              f"of rest on average; "
                              f"in a 5-min therapy session with ~70% rest, this is "
                              f"~{false_act_per_session:.1f} spurious activations."),
        },
        "time_to_correct_command_ms": {
            "mean": ttc_mean, "ci95": [ttc_lo, ttc_hi],
            "interpretation": (f"From the ground-truth cue/transition onset, the system takes "
                              f"~{ttc_mean:.0f} ms (mean) to issue its first sustained correct "
                              f"motor command. This is the deployment latency from "
                              f"intent → action that a patient experiences."),
        },
        "by_arm": by_arm,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))

    md = [
        "# Clinical-outcome translation",
        "",
        "Raw accuracy doesn't speak to clinical relevance. Here we translate the "
        f"deployed pipeline (Stage 2, N={N_STABILITY} stability filter — the "
        "`run_exohand.py --assist-level 3` default) into three metrics a rehab "
        "clinician would actually use.",
        "",
        f"All metrics on n = {int(sess_df['patient'].nunique())} patients, "
        f"{len(sess_df)} sessions, {len(df):,} per-window predictions.",
        "",
        "## Headline metrics (overall, both arms)",
        "",
        "| Clinical metric | Mean (95 % CI) | Plain-language reading |",
        "|---|---:|---|",
        f"| **Per-rep success rate** | **{rep_mean:.3f}** [{rep_lo:.3f}, {rep_hi:.3f}] | Of every 10 cued grasps in a session, ~{expected_successful_reps:.1f} produce a sustained ≥ {SUSTAIN_MS} ms correct motor command. |",
        f"| **False activations / min of rest** | **{fa_mean:.2f}** [{fa_lo:.2f}, {fa_hi:.2f}] | During rest, the system spuriously issues a non-rest command {fa_mean:.1f}× per minute on average. |",
        f"| **Time-to-correct-command** | **{ttc_mean:.0f} ms** [{ttc_lo:.0f}, {ttc_hi:.0f}] | After an intent change, the system reaches a sustained correct command in ~{ttc_mean:.0f} ms (mean). |",
        "",
        "## By arm",
        "",
        "| Arm | n sessions | Per-rep success (95 % CI) | False-act/min rest | Time-to-correct (ms) |",
        "|---|---:|---|---:|---:|",
    ]
    for arm, v in by_arm.items():
        rs = v["per_rep_success"]; fa = v["false_act_per_min_rest"]; ttc = v["time_to_correct_ms"]
        md.append(f"| {arm} | {v['n_sessions']} | "
                  f"{rs['mean']:.3f} [{rs['ci'][0]:.3f}, {rs['ci'][1]:.3f}] | "
                  f"{fa['mean']:.2f} | {ttc['mean']:.0f} |")

    md += [
        "",
        "## Definitions",
        "",
        "- **Per-rep success.** For every ground-truth transition into close or open, "
        f"counts as successful iff the post-Stage-2 prediction matches the new "
        f"class for at least {SUSTAIN_MS // WINDOW_MS} consecutive windows ({SUSTAIN_MS} ms) "
        "anywhere in the segment. This is the clinically-meaningful question: not just "
        "*did the model see the intent*, but *did it issue a stable command long enough "
        "to drive the actuator*.",
        "- **False activations / min of rest.** Per session, count discrete transitions "
        "of post-Stage-2 output from rest → non-rest while ground truth is in a rest "
        "period; divide by total rest duration in minutes. This is the spurious-command "
        "rate a patient experiences when not trying to move.",
        "- **Time-to-correct-command.** For each successful rep, the latency (ms) from "
        "ground-truth segment start to the first window of the sustained correct run. "
        "Adds to the system's other latencies (~225 ms hardware/software pipeline, §3) "
        "to give the full intent → action delay.",
        "",
        "## How this enters the paper",
        "",
        "One paragraph in §4 or §6 (clinical relevance), positioned between the "
        "transition-accuracy and limitations sections:",
        "",
        "> *Translating the deployed pipeline (Stage 2, N = 3) into clinical metrics: "
        f"the system completes **{rep_mean*100:.1f}% of cued grasps** with a sustained "
        f"motor command (≥ {SUSTAIN_MS} ms), with **{fa_mean:.1f} false activations per "
        "minute of rest** and a **{ttc_mean:.0f}-ms median time-to-correct-command**. "
        "For a typical 5-minute therapy session with 70 % rest, a patient would "
        "experience ~{false_act_per_session:.0f} spurious activations and successfully "
        "trigger ~{expected_successful_reps:.0f} of every 10 attempted grasps. The "
        "spurious-activation rate is the dominant remaining limitation for fully "
        "autonomous use; the deployed runtime's cooldown + hysteresis layers (§3.4) "
        "reduce it further in practice but were not replayed offline here.*",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nPer-rep success rate: {rep_mean:.4f} [{rep_lo:.4f}, {rep_hi:.4f}]")
    print(f"False-act per min rest: {fa_mean:.2f} [{fa_lo:.2f}, {fa_hi:.2f}]")
    print(f"Time-to-correct (ms):  {ttc_mean:.0f} [{ttc_lo:.0f}, {ttc_hi:.0f}]")
    print(f"\nWrote {OUT_CSV}, {OUT_JSON}, {OUT_MD}")


if __name__ == "__main__":
    main()
