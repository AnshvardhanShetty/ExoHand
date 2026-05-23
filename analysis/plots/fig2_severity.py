"""
Figure 2 — Severity tertile bar chart showing flat profile across FMA tertiles.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.plots.style import apply_style, PALETTE, save_pair

SEVERITY_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "severity_per_patient.csv"
SUM_JSON = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "severity_summary.json"
OUT = PROJECT_ROOT / "analysis" / "plots" / "figures" / "fig2_severity"


def main():
    apply_style()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sev = pd.read_csv(SEVERITY_CSV)

    # The impaired-arm FMA is what we stratify by — it has the meaningful spread
    # (range 0-2, 42 unique values across 48 patients). Healthy-arm FMA is
    # nearly constant (most patients at 2.0).
    sev_col = "impaired_fma_mean"
    sev["tertile"] = pd.qcut(sev[sev_col], 3,
                              labels=["severe", "moderate", "mild"], duplicates="drop")

    # We need both zero-shot and calibrated per-patient. Load both.
    zs = pd.read_csv(PROJECT_ROOT / "analysis" / "physiomio" / "results" / "zero_shot_per_session.csv")
    cal = pd.read_csv(PROJECT_ROOT / "analysis" / "physiomio" / "results" / "per_session_results.csv")
    cal = cal[cal["status"] == "ok"]

    # Per-patient mean (impaired arm only — that's what stratifies by severity)
    zs_pat = zs[zs["arm"] == "impaired"].groupby("participant")["acc"].mean().rename("zs")
    cal_pat = cal[cal["arm"] == "impaired"].groupby("participant")["acc"].mean().rename("cal")
    merged = sev.merge(zs_pat, on="participant").merge(cal_pat, on="participant")

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    tertiles = list(merged["tertile"].dropna().unique())
    # Order
    order = ["severe", "moderate", "mild", "more_impaired", "less_impaired"]
    tertiles = [t for t in order if t in tertiles]

    x = np.arange(len(tertiles))
    w = 0.35
    zs_means = [merged[merged["tertile"] == t]["zs"].mean() for t in tertiles]
    zs_stds = [merged[merged["tertile"] == t]["zs"].std() for t in tertiles]
    cal_means = [merged[merged["tertile"] == t]["cal"].mean() for t in tertiles]
    cal_stds = [merged[merged["tertile"] == t]["cal"].std() for t in tertiles]
    n_per = [int(merged[merged["tertile"] == t].shape[0]) for t in tertiles]

    bars_zs = ax.bar(x - w/2, zs_means, w, yerr=zs_stds, capsize=3,
                     color=PALETTE["zero_shot"], label="zero-shot",
                     error_kw={"linewidth": 0.8, "ecolor": PALETTE["muted_text"]})
    bars_cal = ax.bar(x + w/2, cal_means, w, yerr=cal_stds, capsize=3,
                      color=PALETTE["calibrated"], label="+ calibration",
                      error_kw={"linewidth": 0.8, "ecolor": PALETTE["muted_text"]})

    for i, (zm, cm) in enumerate(zip(zs_means, cal_means)):
        ax.text(x[i] - w/2, zm + 0.025, f"{zm:.2f}", ha="center", va="bottom",
                fontsize=7, color=PALETTE["muted_text"])
        ax.text(x[i] + w/2, cm + 0.025, f"{cm:.2f}", ha="center", va="bottom",
                fontsize=7, color=PALETTE["calibrated"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n = {n})" for t, n in zip(tertiles, n_per)])
    ax.set_ylabel("Impaired-arm mean accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", borderaxespad=0.4)
    # Spearman from JSON
    try:
        summ = json.load(open(SUM_JSON))
        rho = summ.get("spearman_fma_vs_cal_benefit", summ.get("spearman_L_CA_vs_cal_benefit", {})).get("rho", None)
        p = summ.get("spearman_fma_vs_cal_benefit", summ.get("spearman_L_CA_vs_cal_benefit", {})).get("p", None)
        if rho is not None:
            ax.set_title(f"Calibration benefit is severity-independent\nSpearman ρ(severity, Δ) = {rho:+.3f}, p = {p:.3f}",
                         loc="left", fontsize=10, pad=10)
        else:
            ax.set_title("Calibration benefit by severity tertile", loc="left", fontsize=10, pad=10)
    except Exception:
        ax.set_title("Calibration benefit by severity tertile", loc="left", fontsize=10, pad=10)

    save_pair(fig, OUT)
    print(f"Wrote {OUT}.{{pdf,png}}")


if __name__ == "__main__":
    main()
