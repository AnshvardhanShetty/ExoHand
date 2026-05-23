"""
PhysioMio error analysis — confusion matrices + error-vs-time-from-transition.

Uses analysis/physiomio/results/per_window_predictions.parquet (165k windows
across 329 sessions) to answer two questions:

1. **Where do errors concentrate?**
   - Overall 3-class confusion matrix
   - Per-arm (healthy / impaired)
   - Per-severity (FMA tertile, from severity_analysis.py source data)

2. **When do errors happen relative to GT transitions?**
   - For each ground-truth class transition in the temporal stream, compute
     the per-window error rate at offsets {-5..+50} windows from the transition.
   - Expected: errors spike around transition, decay to floor during maintenance.

Output:
  analysis/physiomio/results/confusion_overall.csv
  analysis/physiomio/results/confusion_by_arm.csv
  analysis/physiomio/results/confusion_by_severity.csv
  analysis/physiomio/results/error_vs_transition.csv
  analysis/physiomio/results/error_analysis.{png,pdf}
  analysis/physiomio/results/error_analysis_summary.md
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from analysis.seed import SEED, seed_everything

PREDS_PARQUET = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "per_window_predictions.parquet"
SEVERITY_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "severity_per_patient.csv"
OUT_DIR = PROJECT_ROOT / "analysis" / "physiomio" / "results"

CLASSES = [0, 1, 2]
CLASS_NAMES = {0: "rest", 1: "close", 2: "open"}

TRANSITION_OFFSETS = list(range(-5, 51))   # -5 to +50 windows around transition


def confusion(df):
    """Return (3, 3) confusion matrix in canonical class order [rest, close, open]."""
    cm = np.zeros((3, 3), dtype=int)
    for gt, pr in zip(df["gt_intent"].values, df["pred_intent"].values):
        cm[int(gt), int(pr)] += 1
    return cm


def cm_to_records(cm, label):
    rows = []
    for gt in CLASSES:
        for pr in CLASSES:
            rows.append({
                "context": label,
                "gt": CLASS_NAMES[gt], "pred": CLASS_NAMES[pr],
                "count": int(cm[gt, pr]),
                "row_pct": float(cm[gt, pr] / cm[gt].sum() * 100) if cm[gt].sum() else 0.0,
            })
    return rows


def main():
    seed_everything(SEED)
    print(f"Loading {PREDS_PARQUET}...")
    df = pd.read_parquet(PREDS_PARQUET)
    print(f"  {len(df):,} windows × {df['participant'].nunique()} patients × "
          f"{df.groupby('participant')['session'].nunique().sum()} sessions")

    # ── 1. Confusion matrices ──
    print("\n[1] Confusion matrices...")
    cm_overall = confusion(df)
    rows = cm_to_records(cm_overall, "overall")
    print("\nOVERALL confusion matrix (row = GT, col = pred; row %):")
    print(f"        rest   close  open")
    for gt in CLASSES:
        rp = cm_overall[gt] / cm_overall[gt].sum() * 100 if cm_overall[gt].sum() else cm_overall[gt]
        print(f"  {CLASS_NAMES[gt]:>5}: {rp[0]:>6.1f} {rp[1]:>6.1f} {rp[2]:>6.1f}")

    # Per arm
    for arm in ["healthy", "impaired"]:
        cm = confusion(df[df["arm"] == arm])
        rows.extend(cm_to_records(cm, f"arm:{arm}"))
        print(f"\n{arm.upper()} arm:")
        for gt in CLASSES:
            rp = cm[gt] / cm[gt].sum() * 100 if cm[gt].sum() else cm[gt]
            print(f"  {CLASS_NAMES[gt]:>5}: {rp[0]:>6.1f} {rp[1]:>6.1f} {rp[2]:>6.1f}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "confusion_overall.csv", index=False)

    # Per severity (FMA tertile from severity_per_patient.csv)
    if SEVERITY_CSV.exists():
        sev = pd.read_csv(SEVERITY_CSV)
        # FMA-equivalent severity column varies — use the "fma_per_gesture" or first numeric col
        sev_col = "fma_per_gesture" if "fma_per_gesture" in sev.columns else None
        if sev_col is None:
            # Find a plausible numeric column
            for c in sev.columns:
                if c not in ("participant",) and pd.api.types.is_numeric_dtype(sev[c]):
                    sev_col = c; break
        if sev_col:
            try:
                sev["tertile"] = pd.qcut(sev[sev_col], 3, labels=["severe", "moderate", "mild"], duplicates="drop")
            except ValueError:
                # Score too discrete for tertiles — use binary split on median
                median = sev[sev_col].median()
                sev["tertile"] = np.where(sev[sev_col] <= median, "more_impaired", "less_impaired")
            df_t = df.merge(sev[["participant", "tertile"]], on="participant", how="left")
            sev_rows = []
            for t in sev["tertile"].dropna().unique():
                cm = confusion(df_t[df_t["tertile"] == t])
                if cm.sum() > 0:
                    sev_rows.extend(cm_to_records(cm, f"severity:{t}"))
            pd.DataFrame(sev_rows).to_csv(OUT_DIR / "confusion_by_severity.csv", index=False)
            print(f"\nSeverity tertiles (col '{sev_col}'): wrote confusion_by_severity.csv")

    # ── 2. Error vs time from transition ──
    print("\n[2] Error vs time-from-nearest-transition...")
    offset_errs = {off: {"n": 0, "errs": 0} for off in TRANSITION_OFFSETS}
    for (subj, session), g in df.groupby(["participant", "session"]):
        gg = g.sort_values(["trial", "t_rel_s"]).reset_index(drop=True)
        gt = gg["gt_intent"].values
        pred = gg["pred_intent"].values
        # Transition indices = where gt changes from previous
        change_idx = np.where(np.diff(gt) != 0)[0] + 1
        if len(change_idx) == 0:
            continue
        for offset in TRANSITION_OFFSETS:
            # For each transition, look at the window at offset from it
            target_idx = change_idx + offset
            valid = (target_idx >= 0) & (target_idx < len(gg))
            target_idx = target_idx[valid]
            if len(target_idx) == 0:
                continue
            errs = (pred[target_idx] != gt[target_idx]).sum()
            offset_errs[offset]["n"] += len(target_idx)
            offset_errs[offset]["errs"] += int(errs)

    err_rows = []
    for off, v in offset_errs.items():
        rate = v["errs"] / v["n"] if v["n"] > 0 else float("nan")
        err_rows.append({"offset_windows": off, "offset_ms": off * 50,
                         "n": v["n"], "errors": v["errs"], "error_rate": rate})
    err_df = pd.DataFrame(err_rows)
    err_df.to_csv(OUT_DIR / "error_vs_transition.csv", index=False)
    print(f"  Wrote error_vs_transition.csv")

    # ── 3. Plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                          "axes.spines.top": False, "axes.spines.right": False})

    fig = plt.figure(figsize=(12, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.4], wspace=0.4)

    # (a) Overall confusion matrix
    ax0 = fig.add_subplot(gs[0, 0])
    cm = cm_overall / cm_overall.sum(axis=1, keepdims=True) * 100
    im = ax0.imshow(cm, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(3):
        for j in range(3):
            ax0.text(j, i, f"{cm[i, j]:.0f}%", ha="center", va="center",
                     color="white" if cm[i, j] > 50 else "black", fontsize=10)
    ax0.set_xticks([0, 1, 2]); ax0.set_xticklabels(["rest", "close", "open"])
    ax0.set_yticks([0, 1, 2]); ax0.set_yticklabels(["rest", "close", "open"])
    ax0.set_xlabel("Predicted"); ax0.set_ylabel("Ground truth")
    ax0.set_title("(a)  Overall confusion (row %)", loc="left", fontsize=10, pad=8)

    # (b) Per-arm side-by-side
    ax1 = fig.add_subplot(gs[0, 1])
    cm_h = confusion(df[df["arm"] == "healthy"])
    cm_i = confusion(df[df["arm"] == "impaired"])
    h_norm = (cm_h / cm_h.sum(axis=1, keepdims=True) * 100).diagonal()
    i_norm = (cm_i / cm_i.sum(axis=1, keepdims=True) * 100).diagonal()
    x = np.arange(3)
    ax1.bar(x - 0.2, h_norm, width=0.38, label="healthy arm", color="#3498db")
    ax1.bar(x + 0.2, i_norm, width=0.38, label="impaired arm", color="#c0392b")
    ax1.set_xticks(x); ax1.set_xticklabels(["rest", "close", "open"])
    ax1.set_ylabel("Per-class recall (%)")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="lower right", frameon=False, fontsize=8)
    ax1.set_title("(b)  Per-class recall by arm", loc="left", fontsize=10, pad=8)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    # (c) Error rate vs time-from-transition
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(err_df["offset_ms"], err_df["error_rate"] * 100, "-o", color="#2c3e50", markersize=3)
    ax2.axvline(0, color="red", linestyle="--", alpha=0.6, linewidth=1, label="transition")
    ax2.set_xlabel("Time from ground-truth transition (ms)")
    ax2.set_ylabel("Per-window error rate (%)")
    ax2.set_title("(c)  Error rate around transitions", loc="left", fontsize=10, pad=8)
    ax2.grid(linestyle=":", alpha=0.4)
    ax2.legend(loc="upper right", frameon=False, fontsize=8)
    ax2.set_xlim(err_df["offset_ms"].min(), err_df["offset_ms"].max())

    plt.savefig(OUT_DIR / "error_analysis.png", dpi=180, bbox_inches="tight")
    plt.savefig(OUT_DIR / "error_analysis.pdf", bbox_inches="tight")
    print(f"  Wrote error_analysis.{{png,pdf}}")

    # ── 4. Markdown summary ──
    md = [
        "# PhysioMio error analysis",
        "",
        f"n = {df['participant'].nunique()} patients, "
        f"{df.groupby('participant')['session'].nunique().sum()} sessions, "
        f"{len(df):,} per-window predictions (Stage 1 classifier output).",
        "",
        "## Overall confusion matrix (row % normalised)",
        "",
        "| GT \\ Pred | rest | close | open |",
        "|---|---:|---:|---:|",
    ]
    for gt in CLASSES:
        row = cm_overall[gt] / cm_overall[gt].sum() * 100
        md.append(f"| {CLASS_NAMES[gt]} | {row[0]:.1f}% | {row[1]:.1f}% | {row[2]:.1f}% |")
    md += [
        "",
        "Diagonal = per-class recall. Off-diagonal patterns:",
    ]
    # Identify top off-diag entries
    flat = []
    for gt in CLASSES:
        for pr in CLASSES:
            if gt != pr:
                pct = cm_overall[gt, pr] / cm_overall[gt].sum() * 100
                flat.append((gt, pr, pct))
    flat.sort(key=lambda x: -x[2])
    for gt, pr, pct in flat[:3]:
        md.append(f"- **{CLASS_NAMES[gt]} → {CLASS_NAMES[pr]}** confusion: {pct:.1f}%")

    md += [
        "",
        "## Per-arm recall",
        "",
        "| Class | Healthy arm | Impaired arm | Δ (impaired − healthy) |",
        "|---|---:|---:|---:|",
    ]
    h_recall = (cm_h / cm_h.sum(axis=1, keepdims=True) * 100).diagonal()
    i_recall = (cm_i / cm_i.sum(axis=1, keepdims=True) * 100).diagonal()
    for gt, name in CLASS_NAMES.items():
        md.append(f"| {name} | {h_recall[gt]:.1f}% | {i_recall[gt]:.1f}% | {i_recall[gt] - h_recall[gt]:+.1f}% |")

    md += [
        "",
        "## Error rate vs time from ground-truth transition",
        "",
        "Per-window error rate at offsets {-250 ms .. +2500 ms} from each GT class change:",
        "",
        "| Offset (ms) | Error rate | Description |",
        "|---:|---:|---|",
        f"| -250 (pre-transition, old class) | {err_df.loc[err_df['offset_ms']==-250, 'error_rate'].values[0]*100:.1f}% | Last window of old class |",
        f"| +0 (transition) | {err_df.loc[err_df['offset_ms']==0, 'error_rate'].values[0]*100:.1f}% | First window of new class |",
        f"| +500 (reaction-buffer end) | {err_df.loc[err_df['offset_ms']==500, 'error_rate'].values[0]*100:.1f}% | End of typical reaction budget |",
        f"| +2500 (deep maintenance) | {err_df.loc[err_df['offset_ms']==2500, 'error_rate'].values[0]*100:.1f}% | 2.5 s into new class |",
        "",
        "If error rate is highest at +0 and decays through maintenance, the model is *latency-limited* (detects transitions slightly late but holds correctly). If error rate is flat across the window, the model is *class-confusion-limited* (it just gets the wrong class persistently).",
        "",
        "## Where the errors are",
        "",
        f"- **Most-confused pair**: {CLASS_NAMES[flat[0][0]]} → {CLASS_NAMES[flat[0][1]]} ({flat[0][2]:.1f}%)",
        f"- **Impaired-arm penalty on rest recall**: {i_recall[0] - h_recall[0]:+.1f}%",
        f"- **Impaired-arm penalty on close recall**: {i_recall[1] - h_recall[1]:+.1f}%",
        f"- **Impaired-arm penalty on open recall**: {i_recall[2] - h_recall[2]:+.1f}%",
    ]
    (OUT_DIR / "error_analysis_summary.md").write_text("\n".join(md))
    print(f"  Wrote error_analysis_summary.md")

    print("\nDone.")


if __name__ == "__main__":
    main()
