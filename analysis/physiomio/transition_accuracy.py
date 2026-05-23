"""
Transition accuracy on PhysioMio per-session predictions.

Implements the metric defined by Wang et al. (ReactEMG, arXiv 2506.19815;
ReactEMG-Stroke, arXiv 2601.22090). Transition accuracy is the fraction of
ground-truth intent transitions for which the model:

    (1) emits ≥1 correct prediction of the new class within a short
        REACTION BUFFER after the transition, AND
    (2) maintains the new class with zero (or near-zero) errors throughout
        the MAINTENANCE PERIOD until the next transition.

Why this metric: raw window-level accuracy is dominated by long uniform
holds (rest, sustained grasp) where the model's per-window correctness
matters less for control. Transition accuracy reflects two real failure
modes of real-time intent-controlled orthoses — delayed onset and
mid-hold flicker.

Parameters (default values are documented choices; values aren't pinned by
the paper because their reaction-buffer/maintenance constants depend on
the cue protocol):

    REACTION_BUFFER_WIN = 10         # 10 windows × 50 ms stride = 500 ms reaction budget
    MAINT_ERROR_FRAC    = 0.0        # strict — zero tolerance during maintenance
                                     # (a relaxed --max-maint-error variant is exposed below)
    MIN_HOLD_WINDOWS    = 5          # ignore micro-segments shorter than 250 ms

Input: analysis/physiomio/results/per_window_predictions.parquet
       (one row per (patient, session, t_rel_s) — produced by
       per_session_eval.py with --use-cached after a one-time cache build.)

Output:
    analysis/physiomio/results/transition_accuracy_per_session.csv
    analysis/physiomio/results/transition_accuracy_summary.{json,md}
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


PREDS_PARQUET = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "per_window_predictions.parquet"
OUT_PER_SESSION = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "transition_accuracy_per_session.csv"
OUT_SUMMARY_JSON = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "transition_accuracy_summary.json"
OUT_SUMMARY_MD = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "transition_accuracy_summary.md"

REACTION_BUFFER_WIN = 10       # 500 ms
MIN_HOLD_WINDOWS = 5           # 250 ms
DEFAULT_MAINT_CAP = 110        # 5.5 s of post-buffer maintenance — matches ReactEMG's 6 s O/C
                               # cue segments minus the reaction buffer.
N_BOOT = 2000
CLASSES = [0, 1, 2]
CLASS_NAMES = {0: "rest", 1: "close", 2: "open"}


def compute_session_transitions(session_df: pd.DataFrame,
                                reaction_buffer: int,
                                maint_error_frac: float,
                                min_hold: int,
                                maint_cap: int) -> tuple:
    """Score every ground-truth transition within one session's temporal sequence.

    Returns (rows, summary):
      rows: list of per-transition dicts (gt_old, gt_new, buffer_correct, maint_correct, ...)
      summary: per-session aggregate stats
    """
    # PhysioMio's t_rel_s is per-trial (0..4 s within each gesture), not per-session.
    # The session-level temporal order is (trial, t_rel_s) — trials are gesture
    # indices in presentation order. Sorting only by t_rel_s would interleave
    # gestures and produce ~150 spurious "transitions" per session; using
    # (trial, t_rel_s) yields the natural ~2-3 between-gesture transitions
    # (e.g. rest→close, then close→close repeats, then close→open).
    df = session_df.sort_values(["trial", "t_rel_s"]).reset_index(drop=True)
    gt = df["gt_intent"].values
    pred = df["pred_intent"].values

    # Find indices where gt changes from previous window
    change_idx = np.where(np.diff(gt) != 0)[0] + 1
    if len(change_idx) == 0:
        return [], {
            "n_transitions": 0, "n_correct": 0, "transition_acc": np.nan,
            "n_segments": 1, "raw_acc": float((pred == gt).mean()),
        }

    # Segment boundaries: [0, change_idx[0], change_idx[1], ..., len(df)]
    boundaries = np.concatenate([[0], change_idx, [len(df)]])
    segments = list(zip(boundaries[:-1], boundaries[1:]))
    # Filter micro-segments
    valid_segments = [(s, e) for s, e in segments if (e - s) >= min_hold]

    rows = []
    for i in range(1, len(valid_segments)):
        seg_start, seg_end = valid_segments[i]
        new_class = int(gt[seg_start])
        prev_class = int(gt[valid_segments[i - 1][1] - 1])

        # Reaction buffer: first `reaction_buffer` windows of this segment
        buf_end = min(seg_start + reaction_buffer, seg_end)
        in_buffer = pred[seg_start:buf_end]
        buffer_correct = bool((in_buffer == new_class).any())

        # Maintenance period: from end of buffer to min(end of segment, buf_end + maint_cap)
        # The cap matches ReactEMG's natural cue length (~6 s = ~120 windows); without it
        # PhysioMio's 40 s stacked-close segments would be unfairly penalised.
        maint_end = min(seg_end, buf_end + maint_cap)
        maint = pred[buf_end:maint_end]
        if len(maint) == 0:
            maint_correct = True
            maint_error_rate = 0.0
        else:
            maint_error_rate = float((maint != new_class).mean())
            maint_correct = bool(maint_error_rate <= maint_error_frac)

        transition_correct = bool(buffer_correct and maint_correct)
        rows.append({
            "seg_idx": i,
            "from_class": prev_class,
            "to_class": new_class,
            "seg_len": int(seg_end - seg_start),
            "buffer_correct": buffer_correct,
            "maint_error_rate": maint_error_rate,
            "maint_correct": maint_correct,
            "transition_correct": transition_correct,
        })

    if rows:
        n_correct = sum(r["transition_correct"] for r in rows)
        ta = n_correct / len(rows)
    else:
        n_correct = 0
        ta = np.nan
    summary = {
        "n_transitions": len(rows),
        "n_correct": n_correct,
        "transition_acc": ta,
        "n_segments": len(valid_segments),
        "raw_acc": float((pred == gt).mean()),
    }
    return rows, summary


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple:
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.randint(0, n, size=(n_boot, n))
    samples = values[idx].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction-buffer", type=int, default=REACTION_BUFFER_WIN,
                        help=f"Reaction buffer length in windows (default {REACTION_BUFFER_WIN} = 500 ms)")
    parser.add_argument("--max-maint-error", type=float, default=0.0,
                        help="Max fraction of incorrect predictions during maintenance (default 0.0 = strict per paper)")
    parser.add_argument("--min-hold", type=int, default=MIN_HOLD_WINDOWS,
                        help=f"Minimum segment length to count (default {MIN_HOLD_WINDOWS} = 250 ms)")
    parser.add_argument("--maint-cap", type=int, default=DEFAULT_MAINT_CAP,
                        help=f"Maintenance scan window cap (default {DEFAULT_MAINT_CAP} = 5.5 s, "
                             f"matches ReactEMG's 6 s O/C segment minus reaction buffer)")
    args = parser.parse_args()

    seed_everything(SEED)

    if not PREDS_PARQUET.exists():
        print(f"ERROR: {PREDS_PARQUET} not found.", file=sys.stderr)
        print("Run per_session_eval.py first to populate per-window predictions.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {PREDS_PARQUET}...")
    preds_all = pd.read_parquet(PREDS_PARQUET)
    print(f"  {len(preds_all):,} windows × {preds_all['participant'].nunique()} patients × "
          f"{preds_all.groupby('participant')['session'].nunique().sum()} sessions")
    print(f"  reaction_buffer={args.reaction_buffer} windows ({args.reaction_buffer*50} ms), "
          f"max_maint_error={args.max_maint_error:.2f}, min_hold={args.min_hold} windows")

    # Per-session
    session_rows = []
    all_trans = []
    for (patient, session), group in preds_all.groupby(["participant", "session"]):
        trans_rows, summary = compute_session_transitions(
            group, args.reaction_buffer, args.max_maint_error, args.min_hold, args.maint_cap,
        )
        for r in trans_rows:
            r["participant"] = patient
            r["session"] = session
            r["arm"] = group["arm"].iloc[0]
            all_trans.append(r)
        session_rows.append({
            "participant": patient,
            "session": session,
            "arm": group["arm"].iloc[0],
            **summary,
        })

    sess_df = pd.DataFrame(session_rows)
    trans_df = pd.DataFrame(all_trans)
    sess_df.to_csv(OUT_PER_SESSION, index=False)
    print(f"\nWrote {OUT_PER_SESSION}  ({len(sess_df)} sessions, {len(trans_df)} transitions)")

    # Aggregate
    ok = sess_df.dropna(subset=["transition_acc"])
    ta_session_mean, ta_lo, ta_hi = bootstrap_mean_ci(ok["transition_acc"].values)
    # Patient-level mean of session means
    pat_means = ok.groupby("participant")["transition_acc"].mean().values
    ta_pat_mean, ta_pat_lo, ta_pat_hi = bootstrap_mean_ci(pat_means)
    # Raw accuracy for comparison
    raw_session_mean, raw_lo, raw_hi = bootstrap_mean_ci(ok["raw_acc"].values)

    # Per-transition-type breakdown
    if len(trans_df) > 0:
        per_type = trans_df.groupby(["from_class", "to_class"]).agg(
            n=("transition_correct", "size"),
            acc=("transition_correct", "mean"),
            buffer_acc=("buffer_correct", "mean"),
            maint_acc=("maint_correct", "mean"),
        ).reset_index()
        per_type["from"] = per_type["from_class"].map(CLASS_NAMES)
        per_type["to"] = per_type["to_class"].map(CLASS_NAMES)
    else:
        per_type = pd.DataFrame()

    # By arm
    by_arm = {}
    for arm in ["healthy", "impaired"]:
        sub = ok[ok["arm"] == arm]
        if len(sub) == 0:
            continue
        m, lo, hi = bootstrap_mean_ci(sub["transition_acc"].values)
        by_arm[arm] = {"n_sessions": int(len(sub)), "transition_acc_mean": m,
                       "transition_acc_lo": lo, "transition_acc_hi": hi}

    summary = {
        "params": {
            "reaction_buffer_windows": args.reaction_buffer,
            "reaction_buffer_ms": args.reaction_buffer * 50,
            "max_maint_error_frac": args.max_maint_error,
            "min_hold_windows": args.min_hold,
            "maint_cap_windows": args.maint_cap,
            "maint_cap_ms": args.maint_cap * 50,
        },
        "session_level": {
            "n_sessions": int(len(ok)),
            "transition_acc_mean": ta_session_mean,
            "transition_acc_ci95": [ta_lo, ta_hi],
            "raw_acc_mean": raw_session_mean,
            "raw_acc_ci95": [raw_lo, raw_hi],
        },
        "patient_level": {
            "n_patients": int(len(pat_means)),
            "transition_acc_mean": ta_pat_mean,
            "transition_acc_ci95": [ta_pat_lo, ta_pat_hi],
        },
        "by_arm": by_arm,
        "per_transition_type": per_type.to_dict("records") if len(per_type) else [],
        "n_total_transitions": int(len(trans_df)),
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str))

    # Markdown summary
    md = [
        "# PhysioMio transition accuracy",
        "",
        f"Metric per Wang et al. (ReactEMG Stroke, arXiv 2601.22090). "
        f"Reaction buffer = {args.reaction_buffer} windows ({args.reaction_buffer*50} ms); "
        f"max maintenance error = {args.max_maint_error:.2f} "
        f"({'strict / zero tolerance' if args.max_maint_error == 0.0 else 'relaxed'}); "
        f"min segment length = {args.min_hold} windows.",
        "",
        "## Headline numbers",
        "",
        "| Aggregation | Mean | 95% bootstrap CI |",
        "|---|---:|---|",
        f"| Session-level transition accuracy | **{ta_session_mean:.4f}** | [{ta_lo:.4f}, {ta_hi:.4f}] |",
        f"| Patient-level transition accuracy | **{ta_pat_mean:.4f}** | [{ta_pat_lo:.4f}, {ta_pat_hi:.4f}] |",
        f"| Session-level raw accuracy (full-stream, for ref) | {raw_session_mean:.4f} | [{raw_lo:.4f}, {raw_hi:.4f}] |",
        "",
        f"Based on **{len(ok)} sessions** ({len(pat_means)} patients) and **{len(trans_df)} total transitions**.",
        "",
    ]
    if by_arm:
        md += [
            "## By arm",
            "",
            "| Arm | n sessions | Transition acc (95% CI) |",
            "|---|---:|---:|",
        ]
        for arm, v in by_arm.items():
            md.append(f"| {arm} | {v['n_sessions']} | {v['transition_acc_mean']:.4f} [{v['transition_acc_lo']:.4f}, {v['transition_acc_hi']:.4f}] |")
        md.append("")
    if len(per_type) > 0:
        md += [
            "## Per transition type",
            "",
            "| from → to | n | Buffer acc | Maint acc | Combined transition acc |",
            "|---|---:|---:|---:|---:|",
        ]
        for _, r in per_type.iterrows():
            md.append(f"| {r['from']} → {r['to']} | {int(r['n'])} | {r['buffer_acc']:.3f} | {r['maint_acc']:.3f} | **{r['acc']:.3f}** |")
        md.append("")
    md += [
        "## Comparison to ReactEMG Stroke (Wang et al. 2026)",
        "",
        "Wang et al. report transition accuracy averaged across 3 chronic-stroke participants "
        "(FMA-UE 26-35) on 5 held-out distribution-shifted test sets:",
        "",
        "| Method | Raw acc | Transition acc |",
        "|---|---:|---:|",
        "| Zero-shot (healthy-pretrained, frozen) | 0.60 | 0.13 |",
        "| Stroke-only training from scratch | 0.69 | 0.42 |",
        "| Head-only fine-tune | 0.75 | 0.53 |",
        "| LoRA fine-tune | 0.78 | 0.61 |",
        "| Full fine-tune | 0.78 | 0.61 |",
        "| **This work (PhysioMio, per-session cal, n=48)** | "
        f"**{raw_session_mean:.2f}** | **{ta_session_mean:.2f}** |",
        "",
        "Notes for fair comparison:",
        "",
        "- Cohort: 48 patients (PhysioMio) vs 3 (ReactEMG Stroke). Larger cohort → tighter CIs but different severity mix.",
        "- Method: classical ML (HistGradientBoosting + weighted refit on a 43-subject GrabMyo base) vs transformer + LoRA pretrained on 650+ subjects.",
        "- Test protocol: PhysioMio sessions average ~48 s of gesture data (16 gestures × 4 s, then 3-class-mapped to 12 gestures), with 2-3 ground-truth transitions per session. ReactEMG runs ~18-minute sessions with explicitly interleaved RCRCRCR cue sequences (5-6 transitions/set × multiple sets). The PhysioMio transition counts per session are therefore an order of magnitude smaller; aggregate transition acc estimate is more variance-stable thanks to the 48-patient cohort but per-session is noisier.",
        "- Distribution shift: ReactEMG evaluates 5 held-out perturbation conditions (within-session drift, unseen posture, sensor placement, device-driven motion); the PhysioMio number here is on the per-session balanced split without explicit perturbations. Adding perturbation evaluation is tracked separately.",
        "- Deployment: this metric is on the per-session calibrated model that fits inside the 50 ms Teensy cycle (~17 ms/cycle p50, CPU). ReactEMG runs a transformer that needs GPU at inference time.",
    ]
    OUT_SUMMARY_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_SUMMARY_MD}")
    print(f"Wrote {OUT_SUMMARY_JSON}")
    print()
    print(f"Headline: session-level transition_acc = {ta_session_mean:.4f} [{ta_lo:.4f}, {ta_hi:.4f}]")
    print(f"          patient-level transition_acc = {ta_pat_mean:.4f} [{ta_pat_lo:.4f}, {ta_pat_hi:.4f}]")


if __name__ == "__main__":
    main()
