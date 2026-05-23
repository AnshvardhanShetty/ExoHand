"""
Figure 1 — Per-patient calibration improvement strip plot.

For each of 48 PhysioMio patients (sorted by zero-shot accuracy), show two
points connected by a line: zero-shot accuracy → calibrated accuracy. Split
into healthy-arm and impaired-arm panels. Every patient should improve.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.plots.style import apply_style, PALETTE, save_pair

ZS_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "zero_shot_per_session.csv"
CAL_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "per_session_results.csv"
OUT = PROJECT_ROOT / "analysis" / "plots" / "figures" / "fig1_strip"


def main():
    apply_style()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    zs = pd.read_csv(ZS_CSV)
    cal = pd.read_csv(CAL_CSV)
    cal = cal[cal["status"] == "ok"]

    # Aggregate per-patient per-arm mean accuracy
    def agg(df, name):
        return df.groupby(["participant", "arm"])["acc"].mean().reset_index().rename(columns={"acc": name})

    zs_pat = agg(zs, "zs")
    cal_pat = agg(cal, "cal")
    merged = zs_pat.merge(cal_pat, on=["participant", "arm"])

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), sharey=True,
                              gridspec_kw={"wspace": 0.15})

    for ax, arm, ax_title in zip(axes, ["impaired", "healthy"], ["(a)  Impaired arm", "(b)  Healthy arm"]):
        sub = merged[merged["arm"] == arm].copy()
        # Sort by zero-shot acc ascending → patient1 (worst zs) at left
        sub = sub.sort_values("zs", ascending=True).reset_index(drop=True)
        x = np.arange(len(sub))
        # Connecting lines
        for i in range(len(sub)):
            color = PALETTE["impaired_arm"] if arm == "impaired" else PALETTE["healthy_arm"]
            ax.plot([x[i], x[i]], [sub["zs"].iloc[i], sub["cal"].iloc[i]],
                    color=color, alpha=0.4, linewidth=0.9, zorder=1)
        # Zero-shot points
        ax.scatter(x, sub["zs"], s=18, color=PALETTE["zero_shot"],
                   edgecolor="none", label="zero-shot", zorder=2)
        # Calibrated points
        ax.scatter(x, sub["cal"], s=22, color=PALETTE["calibrated"],
                   edgecolor="white", linewidth=0.5, label="+ calibration", zorder=3)

        zs_mean = sub["zs"].mean()
        cal_mean = sub["cal"].mean()
        ax.axhline(zs_mean, color=PALETTE["zero_shot"], linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(cal_mean, color=PALETTE["calibrated"], linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(len(sub) - 1, zs_mean - 0.025, f"mean {zs_mean:.2f}", fontsize=7,
                color=PALETTE["muted_text"], ha="right", va="top")
        ax.text(len(sub) - 1, cal_mean + 0.015, f"mean {cal_mean:.2f}", fontsize=7,
                color=PALETTE["calibrated"], ha="right", va="bottom")

        ax.set_title(ax_title, loc="left")
        ax.set_xlabel(f"Patients (n = {len(sub)}, sorted by zero-shot accuracy)")
        ax.set_xticks([])
        ax.set_ylim(0, 1.02)
        ax.set_xlim(-0.5, len(sub) - 0.5)
        if arm == "impaired":
            ax.set_ylabel("Session-mean accuracy")
            ax.legend(loc="lower right", ncol=1, handletextpad=0.4, borderaxespad=0.4)

    fig.suptitle("Per-patient cross-population calibration recovery  ·  PhysioMio (n = 48 stroke patients)",
                 fontsize=10, y=1.02)
    save_pair(fig, OUT)
    print(f"Wrote {OUT}.{{pdf,png}}")


if __name__ == "__main__":
    main()
