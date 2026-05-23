"""
Lucchetti transition accuracy — wraps the PhysioMio transition_accuracy logic
against analysis/lucchetti/results/per_window_predictions.parquet.

Computes both:
  - Stage 1 raw (per-window argmax)
  - Stage 2 deployed pipeline (full 6-layer post-processing, all 5 assist profiles)

Outputs:
  analysis/lucchetti/results/transition_accuracy_strict.{md,csv,json}
  analysis/lucchetti/results/transition_accuracy_relaxed.{md,csv,json}
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
    REACTION_BUFFER_WIN, MIN_HOLD_WINDOWS, DEFAULT_MAINT_CAP, CLASS_NAMES,
)

PREDS_PARQUET = PROJECT_ROOT / "analysis" / "lucchetti" / "results" / "per_window_predictions.parquet"
OUT_DIR = PROJECT_ROOT / "analysis" / "lucchetti" / "results"


def run_strict_relaxed_pair():
    seed_everything(SEED)
    if not PREDS_PARQUET.exists():
        print(f"ERROR: {PREDS_PARQUET} not found. Run per_session_eval.py first.", file=sys.stderr)
        sys.exit(1)
    preds = pd.read_parquet(PREDS_PARQUET)
    print(f"Loaded {len(preds):,} windows × {preds['participant'].nunique()} subjects × "
          f"{preds.groupby('participant')['session'].nunique().sum()} sessions")

    for max_err, label in [(0.0, "strict"), (0.10, "relaxed")]:
        rows = []
        all_trans = []
        for (subj, session), group in preds.groupby(["participant", "session"]):
            trans, summary = compute_session_transitions(
                group, REACTION_BUFFER_WIN, max_err, MIN_HOLD_WINDOWS, DEFAULT_MAINT_CAP,
            )
            for r in trans:
                r["participant"] = subj
                r["session"] = session
                r["arm"] = group["arm"].iloc[0]
                all_trans.append(r)
            summary["participant"] = subj
            summary["session"] = session
            summary["arm"] = group["arm"].iloc[0]
            rows.append(summary)

        sess = pd.DataFrame(rows).dropna(subset=["transition_acc"])
        trans_df = pd.DataFrame(all_trans)
        s_mean, s_lo, s_hi = bootstrap_mean_ci(sess["transition_acc"].values)
        pat_m = sess.groupby("participant")["transition_acc"].mean().values
        p_mean, p_lo, p_hi = bootstrap_mean_ci(pat_m)
        raw_m, raw_lo, raw_hi = bootstrap_mean_ci(sess["raw_acc"].values)

        per_type = (trans_df.groupby(["from_class", "to_class"]).agg(
            n=("transition_correct", "size"),
            acc=("transition_correct", "mean"),
            buffer_acc=("buffer_correct", "mean"),
            maint_acc=("maint_correct", "mean"),
        ).reset_index() if len(trans_df) > 0 else pd.DataFrame())

        by_arm = {}
        for arm in ["healthy", "impaired"]:
            sub = sess[sess["arm"] == arm]
            if len(sub) == 0: continue
            m, lo, hi = bootstrap_mean_ci(sub["transition_acc"].values)
            by_arm[arm] = {"n_sessions": int(len(sub)), "mean": m, "ci": [lo, hi]}

        out = {
            "label": label,
            "n_sessions": int(len(sess)),
            "n_patients": int(sess["participant"].nunique()),
            "n_transitions": int(len(trans_df)),
            "session_transition_acc": {"mean": s_mean, "ci": [s_lo, s_hi]},
            "patient_transition_acc": {"mean": p_mean, "ci": [p_lo, p_hi]},
            "raw_full_stream_acc": {"mean": raw_m, "ci": [raw_lo, raw_hi]},
            "by_arm": by_arm,
            "per_transition_type": per_type.to_dict("records") if len(per_type) > 0 else [],
        }
        (OUT_DIR / f"transition_accuracy_{label}.json").write_text(json.dumps(out, indent=2, default=str))

        md = [
            f"# Lucchetti transition accuracy ({label})",
            "",
            f"n = {out['n_sessions']} sessions, {out['n_patients']} subjects, {out['n_transitions']} transitions.",
            f"Reaction buffer = {REACTION_BUFFER_WIN*50} ms, maint cap = {DEFAULT_MAINT_CAP*50} ms, maint error tolerance = {max_err*100:.0f}%.",
            "",
            "## Headline",
            "",
            "| | Mean | 95 % CI |",
            "|---|---:|---|",
            f"| Patient-level transition acc | **{p_mean:.4f}** | [{p_lo:.4f}, {p_hi:.4f}] |",
            f"| Session-level transition acc | {s_mean:.4f} | [{s_lo:.4f}, {s_hi:.4f}] |",
            f"| Session-level raw acc (full stream) | {raw_m:.4f} | [{raw_lo:.4f}, {raw_hi:.4f}] |",
            "",
        ]
        if by_arm:
            md += ["## By arm", "", "| Arm | n | Transition acc (95 % CI) |", "|---|---:|---|"]
            for arm, v in by_arm.items():
                md.append(f"| {arm} | {v['n_sessions']} | {v['mean']:.4f} [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}] |")
            md.append("")
        if len(per_type) > 0:
            md += ["## Per transition type", "",
                   "| from → to | n | Buffer acc | Maint acc | Combined |",
                   "|---|---:|---:|---:|---:|"]
            for _, r in per_type.iterrows():
                from_n = CLASS_NAMES.get(int(r["from_class"]), str(r["from_class"]))
                to_n = CLASS_NAMES.get(int(r["to_class"]), str(r["to_class"]))
                md.append(f"| {from_n} → {to_n} | {int(r['n'])} | {r['buffer_acc']:.3f} | {r['maint_acc']:.3f} | **{r['acc']:.3f}** |")
        (OUT_DIR / f"transition_accuracy_{label}.md").write_text("\n".join(md))
        sess.to_csv(OUT_DIR / f"transition_accuracy_{label}_per_session.csv", index=False)
        print(f"  [{label}] patient-mean = {p_mean:.4f} [{p_lo:.4f}, {p_hi:.4f}]")


if __name__ == "__main__":
    run_strict_relaxed_pair()
