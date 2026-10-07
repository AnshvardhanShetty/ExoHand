"""
Figure 2 (paper) — Combined cross-cohort per-patient calibration recovery.

Single-panel strip plot showing all 58 stroke patients across both cohorts
(48 PhysioMio + 10 Lucchetti, impaired arm), sorted by zero-shot accuracy.
For each patient: zero-shot accuracy connected to calibrated accuracy.
Cohort distinguished by marker shape (PhysioMio = circles, Lucchetti =
diamonds); zero-shot vs calibrated by colour.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analysis.plots.style import apply_style, PALETTE, save_pair

PM_ZS = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "zero_shot_per_session.csv"
PM_CAL = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "per_session_results.csv"
LUC_ZS = PROJECT_ROOT / "analysis" / "lucchetti" / "results" / "zero_shot_per_session.csv"
LUC_CAL = PROJECT_ROOT / "analysis" / "lucchetti" / "results" / "per_session_results.csv"

OUT = PROJECT_ROOT / "analysis" / "plots" / "figures" / "fig1_strip_combined"


def _agg_impaired(zs_path, cal_path, cohort):
    zs = pd.read_csv(zs_path)
    cal = pd.read_csv(cal_path)
    if "status" in cal.columns:
        cal = cal[cal["status"] == "ok"]
    zs = zs[zs["arm"] == "impaired"]
    cal = cal[cal["arm"] == "impaired"]
    zs_pat = zs.groupby("participant")["acc"].mean().rename("zs")
    cal_pat = cal.groupby("participant")["acc"].mean().rename("cal")
    df = pd.concat([zs_pat, cal_pat], axis=1).dropna().reset_index()
    df["cohort"] = cohort
    return df


def main():
    apply_style()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    pm = _agg_impaired(PM_ZS, PM_CAL, "PhysioMio")
    luc = _agg_impaired(LUC_ZS, LUC_CAL, "Lucchetti")
    combined = pd.concat([pm, luc], ignore_index=True)
    combined = combined.sort_values("zs", ascending=True).reset_index(drop=True)
    n_total = len(combined)
    n_pm = len(pm)
    n_luc = len(luc)
    x = np.arange(n_total)

    fig, ax = plt.subplots(figsize=(9.5, 4.4))

    # Connecting lines (per patient)
    pm_mask = (combined["cohort"] == "PhysioMio").values
    luc_mask = ~pm_mask
    for i in range(n_total):
        col = PALETTE["impaired_arm"] if pm_mask[i] else "#8e44ad"   # purple for Lucchetti
        ax.plot([x[i], x[i]],
                [combined["zs"].iloc[i], combined["cal"].iloc[i]],
                color=col, alpha=0.45, linewidth=0.9, zorder=1)

    # Zero-shot markers (cohort-distinguished by shape)
    ax.scatter(x[pm_mask], combined.loc[pm_mask, "zs"],
               s=22, color=PALETTE["zero_shot"], edgecolor="none",
               marker="o", zorder=2)
    ax.scatter(x[luc_mask], combined.loc[luc_mask, "zs"],
               s=28, color=PALETTE["zero_shot"], edgecolor="#444444",
               linewidth=0.5, marker="D", zorder=2)

    # Calibrated markers
    ax.scatter(x[pm_mask], combined.loc[pm_mask, "cal"],
               s=26, color=PALETTE["calibrated"], edgecolor="white",
               linewidth=0.5, marker="o", zorder=3)
    ax.scatter(x[luc_mask], combined.loc[luc_mask, "cal"],
               s=32, color="#8e44ad", edgecolor="white",
               linewidth=0.5, marker="D", zorder=3)

    # Pooled mean lines (across all 58 patients) — single dashed line each,
    # with cohort-specific means inline in the annotation.
    zs_mean = combined["zs"].mean()
    cal_mean = combined["cal"].mean()
    pm_zs_mean = pm["zs"].mean()
    pm_cal_mean = pm["cal"].mean()
    luc_zs_mean = luc["zs"].mean()
    luc_cal_mean = luc["cal"].mean()

    ax.axhline(zs_mean, color=PALETTE["zero_shot"], linestyle="--",
               linewidth=0.8, alpha=0.7)
    ax.axhline(cal_mean, color=PALETTE["calibrated"], linestyle="--",
               linewidth=0.8, alpha=0.7)

    # Mean labels on the LEFT (clear area, lowest-accuracy patients there)
    bbox_kw = dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.92)
    ax.text(0, zs_mean - 0.028,
            f"pooled {zs_mean:.2f}  (PhysioMio {pm_zs_mean:.2f}, Lucchetti {luc_zs_mean:.2f})",
            fontsize=8, color=PALETTE["muted_text"],
            ha="left", va="top", bbox=bbox_kw)
    ax.text(0, cal_mean + 0.018,
            f"pooled {cal_mean:.2f}  (PhysioMio {pm_cal_mean:.2f}, Lucchetti {luc_cal_mean:.2f})",
            fontsize=8, color=PALETTE["calibrated"],
            ha="left", va="bottom", bbox=bbox_kw)

    # Axis cosmetics
    ax.set_xlabel(f"Patients (n = {n_total}, sorted by zero-shot accuracy)")
    ax.set_ylabel("Session-mean accuracy")
    ax.set_xticks([])
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.5, n_total - 0.5)

    # Custom legend — cohort by shape, measurement by colour
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["zero_shot"],
               markersize=7, label=f"PhysioMio zero-shot (n={n_pm})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["calibrated"],
               markeredgecolor="white", markeredgewidth=0.5, markersize=8,
               label="PhysioMio + calibration"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["zero_shot"],
               markeredgecolor="#444444", markeredgewidth=0.5, markersize=7,
               label=f"Lucchetti zero-shot (n={n_luc})"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#8e44ad",
               markeredgecolor="white", markeredgewidth=0.5, markersize=8,
               label="Lucchetti + calibration"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=8, ncol=2, columnspacing=1.0,
              handletextpad=0.3, borderaxespad=0.4, frameon=True)

    ax.set_title(
        f"Per-patient calibration recovery across two stroke cohorts "
        f"(n = {n_total}: PhysioMio + Lucchetti, impaired arm)",
        loc="left", fontsize=10, pad=8)

    plt.tight_layout()
    save_pair(fig, OUT)
    print(f"Wrote {OUT}.{{pdf,png}}")
    print(f"Pooled: zs {zs_mean:.4f} → cal {cal_mean:.4f}  (n_pm={n_pm}, n_luc={n_luc})")


if __name__ == "__main__":
    main()
