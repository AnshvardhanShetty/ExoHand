"""
Figure 4 — Per-class F1 distribution across patients.

For each of the 3 classes (rest, close, open), show the distribution of
per-patient mean F1 as a violin / box plot. Annotate the rest > close > open
ordering that drives the headline.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.plots.style import apply_style, PALETTE, save_pair

CAL_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "per_session_results.csv"
PAT_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "per_patient_results.csv"
ZS_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "zero_shot_per_session.csv"
OUT = PROJECT_ROOT / "analysis" / "plots" / "figures" / "fig4_per_class_f1"


def main():
    apply_style()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cal = pd.read_csv(CAL_CSV)
    cal = cal[cal["status"] == "ok"]

    # Per-patient mean F1 per class (calibrated) — zero-shot per-class F1 wasn't saved
    # in zero_shot_per_session.csv, so we derive zero-shot from per_window_predictions
    # is not possible (no zero-shot predictions parquet); instead we just show the
    # calibrated distribution which is the paper-relevant story.
    cal_per_patient = cal.groupby("participant").agg(
        f1_rest=("f1_rest", "mean"),
        f1_close=("f1_close", "mean"),
        f1_open=("f1_open", "mean"),
    ).reset_index()

    classes = ["rest", "close", "open"]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    positions = np.arange(len(classes))
    data_cal = [cal_per_patient[f"f1_{c}"].values for c in classes]
    class_colors = [PALETTE["rest"], PALETTE["close"], PALETTE["open"]]

    bp = ax.boxplot(data_cal, positions=positions, widths=0.55,
                    patch_artist=True, showfliers=False,
                    medianprops={"color": "white", "linewidth": 1.4},
                    whiskerprops={"color": "black", "linewidth": 0.6},
                    capprops={"color": "black", "linewidth": 0.6})
    for patch, color in zip(bp["boxes"], class_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)
        patch.set_alpha(0.85)

    rng = np.random.RandomState(0)
    for i, d in enumerate(data_cal):
        jit = rng.uniform(-0.15, 0.15, len(d))
        ax.scatter(np.full(len(d), positions[i]) + jit, d,
                   s=12, color="white", edgecolor="black", linewidth=0.4,
                   alpha=0.95, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(["rest", "close", "open"])
    ax.set_ylabel("Per-patient mean F1 (calibrated)")
    ax.set_ylim(-0.02, 1.05)

    # Median annotations
    for i, c in enumerate(classes):
        med = np.median(data_cal[i])
        ax.text(positions[i], med + 0.04, f"{med:.2f}", ha="center", va="bottom",
                fontsize=8, color="black", fontweight="bold")

    ax.set_title("Per-class F1 distribution after calibration (n = 48 patients)\n"
                 "Rest recovers near-perfectly. Open is the hardest class — extensor weakness in stroke",
                 loc="left", fontsize=10, pad=10)

    save_pair(fig, OUT)
    print(f"Wrote {OUT}.{{pdf,png}}")


if __name__ == "__main__":
    main()
