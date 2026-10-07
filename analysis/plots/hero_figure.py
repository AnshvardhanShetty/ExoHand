"""
Single composite hero figure for the paper.

Three panels:
  (a) Per-patient calibration recovery across both stroke datasets
      (58 patients: PhysioMio n=48 + Lucchetti n=10 stroke). Strip plot.
  (b) Within-subject longitudinal degradation curve + cross-arm transfer.
      Conveys *why* per-session recalibration is necessary.
  (c) Deployment characterisation summary: BOM, latency, real-device metrics.

This figure is meant to land the paper's whole story in one glance.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from analysis.plots.style import apply_style, PALETTE, save_pair
from analysis.seed import SEED

OUT = PROJECT_ROOT / "analysis" / "plots" / "figures" / "hero"


def boot_ci(x, n=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.randint(0, len(x), size=(n, len(x)))
    s = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    apply_style()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # ─── Load data ──
    physiomio_zs = pd.read_csv(PROJECT_ROOT / "analysis/physiomio/results/zero_shot_per_session.csv")
    physiomio_cal = pd.read_csv(PROJECT_ROOT / "analysis/physiomio/results/per_session_results.csv")
    physiomio_cal = physiomio_cal[physiomio_cal["status"] == "ok"]

    lucchetti_zs = pd.read_csv(PROJECT_ROOT / "analysis/lucchetti/results/zero_shot_per_session.csv")
    lucchetti_cal = pd.read_csv(PROJECT_ROOT / "analysis/lucchetti/results/per_session_results.csv")
    lucchetti_cal = lucchetti_cal[lucchetti_cal["status"] == "ok"]
    lucchetti_binary = pd.read_csv(PROJECT_ROOT / "analysis/lucchetti/results/binary_per_session.csv")

    # Per-patient impaired-arm mean acc for each cohort, zero-shot and calibrated
    def per_patient_impaired(df, col):
        return df[df["arm"] == "impaired"].groupby(
            df.columns[0] if "patient" not in df.columns else "patient"
        )[col].mean()
    # PhysioMio uses 'participant'; Lucchetti uses 'participant' too
    def agg(df, col):
        return df[df["arm"] == "impaired"].groupby("participant")[col].mean()

    pm_zs = agg(physiomio_zs, "acc")
    pm_cal = agg(physiomio_cal, "acc")
    luc_zs = agg(lucchetti_zs, "acc")
    luc_cal = agg(lucchetti_cal, "acc")
    luc_bin = agg(lucchetti_binary, "acc_binary")

    # Longitudinal
    long_df = pd.read_csv(PROJECT_ROOT / "analysis/physiomio/results/longitudinal_per_session.csv")
    imp = long_df[long_df["arm"] == "impaired"].copy()
    imp["impaired_session_distance"] = imp["impaired_session_distance"].astype(int)
    distances = sorted(imp["impaired_session_distance"].unique())
    long_means = []
    long_los = []
    long_his = []
    for d in distances:
        sub = imp[imp["impaired_session_distance"] == d]["acc"].values
        m, lo, hi = boot_ci(sub)
        long_means.append(m); long_los.append(lo); long_his.append(hi)

    # Clinical translation summary
    clin = json.load(open(PROJECT_ROOT / "analysis/physiomio/results/clinical_translation.json"))
    live = json.load(open(PROJECT_ROOT / "analysis/system/results/live_deployment_eval.json"))

    # ─── Build figure ──
    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.2, 1.1], wspace=0.30)

    # ── Panel (a) Per-patient calibration recovery, PhysioMio + Lucchetti
    ax = fig.add_subplot(gs[0, 0])
    # Combine 48 PhysioMio + 10 Lucchetti, sort by zero-shot acc ascending
    pm = pd.DataFrame({"zs": pm_zs, "cal": pm_cal, "cohort": "PhysioMio"}).dropna()
    luc = pd.DataFrame({"zs": luc_zs, "cal": luc_cal, "cohort": "Lucchetti"}).dropna()
    combined = pd.concat([pm, luc]).sort_values("zs").reset_index(drop=True)
    x = np.arange(len(combined))

    for i, row in combined.iterrows():
        c = PALETTE["impaired_arm"] if row["cohort"] == "PhysioMio" else "#8e44ad"
        ax.plot([x[i], x[i]], [row["zs"], row["cal"]], color=c, alpha=0.35, linewidth=0.8, zorder=1)
    # Cohort-marker shapes
    pm_mask = combined["cohort"] == "PhysioMio"
    luc_mask = ~pm_mask
    ax.scatter(x[pm_mask], combined.loc[pm_mask, "zs"], s=14, color=PALETTE["zero_shot"], edgecolor="none",
               zorder=2, label="zero-shot")
    ax.scatter(x[luc_mask], combined.loc[luc_mask, "zs"], s=14, color=PALETTE["zero_shot"],
               edgecolor="none", zorder=2)
    ax.scatter(x[pm_mask], combined.loc[pm_mask, "cal"], s=20, color=PALETTE["calibrated"],
               edgecolor="white", linewidth=0.4, zorder=3, label="+ per-session cal (PhysioMio)")
    ax.scatter(x[luc_mask], combined.loc[luc_mask, "cal"], s=24, color="#8e44ad", marker="D",
               edgecolor="white", linewidth=0.4, zorder=3, label="+ per-session cal (Lucchetti)")
    ax.set_xticks([])
    ax.set_ylabel("Impaired-arm test accuracy")
    ax.set_xlabel("Patients (n = 58, sorted by zero-shot accuracy)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.5, len(combined) - 0.5)
    ax.set_title(
        "(a)  Per-session calibration closes the cross-population gap\n"
        "for every patient, on two independent stroke datasets",
        loc="left", fontsize=10, pad=8,
    )
    ax.legend(loc="lower right", fontsize=7, handletextpad=0.4, borderaxespad=0.4, ncol=1)
    # Annotate means
    zs_mean_all = combined["zs"].mean()
    cal_mean_all = combined["cal"].mean()
    ax.axhline(zs_mean_all, color=PALETTE["zero_shot"], linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(cal_mean_all, color=PALETTE["calibrated"], linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(len(combined) - 0.5, zs_mean_all - 0.04, f"mean {zs_mean_all:.2f}",
            ha="right", va="top", fontsize=7, color=PALETTE["muted_text"])
    ax.text(len(combined) - 0.5, cal_mean_all + 0.025, f"mean {cal_mean_all:.2f}",
            ha="right", va="bottom", fontsize=7, color=PALETTE["calibrated"])

    # ── Panel (b) Longitudinal degradation + cross-arm transfer
    ax2 = fig.add_subplot(gs[0, 1])
    distances_arr = np.array(distances)
    long_means_arr = np.array(long_means)
    long_los_arr = np.array(long_los)
    long_his_arr = np.array(long_his)
    ax2.plot(distances_arr, long_means_arr, "-o", color=PALETTE["impaired_arm"],
             markersize=4, linewidth=1.4, label="one-time cal", zorder=3)
    ax2.fill_between(distances_arr, long_los_arr, long_his_arr,
                     color=PALETTE["impaired_arm"], alpha=0.18, zorder=1)
    ax2.axhline(0.875, color=PALETTE["calibrated"], linestyle="--", linewidth=1.0, alpha=0.85,
                label="per-session recal (0.88)")
    # Annotate cross-arm transfer failure
    cross_arm_acc = long_df[long_df["arm"] == "healthy"]["acc"].mean()
    ax2.axhline(cross_arm_acc, color="#7f8c8d", linestyle=":", linewidth=1.0, alpha=0.8,
                label=f"cross-arm transfer ({cross_arm_acc:.2f})")
    ax2.set_xlabel("Sessions since calibration")
    ax2.set_ylabel("Impaired-arm test accuracy")
    ax2.set_ylim(0.3, 1.0)
    ax2.set_xticks(distances_arr[::2])
    ax2.legend(loc="lower left", fontsize=7, handletextpad=0.4, borderaxespad=0.4)
    ax2.set_title("(b)  One-time cal degrades across sessions\nper-session recalibration is necessary",
                  loc="left", fontsize=10, pad=8)

    # ── Panel (c) Deployment numbers callout
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.axis("off")
    ax3.set_title("(c)  Real-device deployment characterisation",
                  loc="left", fontsize=10, pad=8)

    # Big numbers in a clean grid
    entries = [
        ("£180", "BOM, full hardware",                       "#27ae60"),
        ("< 50 ms", "single-cycle ML on CPU",                "#27ae60"),
        ("~275 ms", "end-to-end intent\nto motor command",   "#2980b9"),
        ("99.2 %", "live device, within-session\nheld-out (12-min cued)",  "#2980b9"),
        ("97.7 %", "cued grasps successfully\ncompleted (Stage 2 deployed)", "#c0392b"),
        (f"{clin['time_to_correct_command_ms']['mean']:.0f} ms",
         "time-to-correct command\nafter intent change",     "#c0392b"),
    ]
    cols = 2
    rows = 3
    for i, (val, label, col) in enumerate(entries):
        r = i // cols
        c = i % cols
        x0 = 0.04 + c * 0.50
        y0 = 0.86 - r * 0.32
        # Numeric value
        ax3.text(x0, y0, val, fontsize=14, fontweight="bold", color=col, ha="left", va="top")
        # Label below
        ax3.text(x0, y0 - 0.10, label, fontsize=7.5, color=PALETTE["muted_text"],
                 ha="left", va="top", linespacing=1.2)

    # No suptitle — panel titles tell the story; suptitle overlaps panel (a)'s subtitle.
    plt.tight_layout()
    save_pair(fig, OUT)
    print(f"Wrote {OUT}.{{pdf,png}}")


if __name__ == "__main__":
    main()
