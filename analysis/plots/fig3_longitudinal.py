"""
Figure 3 — Longitudinal degradation curve.

For each session distance from impaired_01 calibration, mean accuracy +
bootstrap 95% CI. Horizontal line for per-session cal baseline (0.875).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.plots.style import apply_style, PALETTE, save_pair
from analysis.seed import SEED

LONG_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "longitudinal_per_session.csv"
OUT = PROJECT_ROOT / "analysis" / "plots" / "figures" / "fig3_longitudinal"

PER_SESSION_BASELINE = 0.875   # PhysioMio per-session cal headline


def boot_ci(x, n=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    x = np.asarray(x)
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    idx = rng.randint(0, len(x), size=(n, len(x)))
    samples = x[idx].mean(axis=1)
    return x.mean(), np.percentile(samples, 2.5), np.percentile(samples, 97.5)


def main():
    apply_style()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LONG_CSV)

    # Impaired-arm by distance
    imp = df[df["arm"] == "impaired"].copy()
    imp["impaired_session_distance"] = imp["impaired_session_distance"].astype(int)

    distances = sorted(imp["impaired_session_distance"].unique())
    means, los, his, ns = [], [], [], []
    for d in distances:
        sub = imp[imp["impaired_session_distance"] == d]["acc"].values
        m, lo, hi = boot_ci(sub)
        means.append(m); los.append(lo); his.append(hi); ns.append(len(sub))

    # Healthy arm (cross-arm transfer)
    h = df[df["arm"] == "healthy"]
    h_summary = h.groupby("test_session")["acc"].agg(["mean", "std", "count"]).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})

    # (a) impaired-arm degradation curve
    ax = axes[0]
    distances_arr = np.array(distances)
    means_arr = np.array(means)
    los_arr = np.array(los)
    his_arr = np.array(his)
    ax.plot(distances_arr, means_arr, "-o", color=PALETTE["impaired_arm"],
            markersize=5, linewidth=1.4, label="cal trained on impaired_01")
    ax.fill_between(distances_arr, los_arr, his_arr, color=PALETTE["impaired_arm"], alpha=0.18)
    ax.axhline(PER_SESSION_BASELINE, color=PALETTE["calibrated"], linestyle="--",
               linewidth=1.0, alpha=0.85,
               label=f"per-session recal baseline ({PER_SESSION_BASELINE:.2f})")
    for d, m, n in zip(distances_arr, means_arr, ns):
        ax.text(d, m - 0.04, f"n={n}", ha="center", va="top", fontsize=6,
                color=PALETTE["muted_text"])
    ax.set_xlabel("Sessions since calibration (impaired_01 = 0)")
    ax.set_ylabel("Impaired-arm test accuracy")
    ax.set_ylim(0.3, 1.0)
    ax.set_xticks(distances_arr)
    ax.legend(loc="lower left", borderaxespad=0.4)
    ax.set_title("(a)  Within-arm longitudinal degradation", loc="left")

    # (b) cross-arm transfer (healthy arm performance with impaired-arm cal)
    ax = axes[1]
    if len(h_summary) > 0:
        x = np.arange(len(h_summary))
        ax.bar(x, h_summary["mean"], yerr=h_summary["std"], capsize=3,
               color=PALETTE["healthy_arm"], alpha=0.85,
               error_kw={"linewidth": 0.8, "ecolor": PALETTE["muted_text"]})
        for i, (m, n) in enumerate(zip(h_summary["mean"], h_summary["count"])):
            ax.text(i, m + 0.025, f"n={int(n)}", ha="center", va="bottom",
                    fontsize=7, color=PALETTE["muted_text"])
        ax.set_xticks(x)
        ax.set_xticklabels(h_summary["test_session"], rotation=0)
        ax.set_ylabel("Healthy-arm test accuracy")
        ax.set_ylim(0.3, 1.0)
    ax.axhline(PER_SESSION_BASELINE, color=PALETTE["calibrated"], linestyle="--",
               linewidth=1.0, alpha=0.85)
    ax.set_title("(b)  Cross-arm transfer", loc="left")

    fig.suptitle("One-time impaired-arm cal degrades over sessions and fails to cross-transfer",
                 fontsize=10, y=1.02)
    save_pair(fig, OUT)
    print(f"Wrote {OUT}.{{pdf,png}}")


if __name__ == "__main__":
    main()
