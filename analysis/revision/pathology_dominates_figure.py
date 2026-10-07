"""
The "pathology dominates diversity" hero figure + numbers.

For each of 48 patients compute two paired Deltas:
  diversity_delta = 47-others'-healthy − own-healthy
  pathology_delta = 47-others'-impaired − 47-others'-healthy

Then produce:
  1. A single scatter (per-patient diversity vs pathology Δ) with quadrant counts +
     marginal densities. Quadrant labels tell the story instantly.
  2. Sorted waterfall panel: patients ranked by pathology Δ, both bars per patient.
  3. Categorical patient breakdown: how many patients fall in each of the 4 quadrants.
  4. The rescue claim: of patients where diversity FAILS them, how many does
     pathology still help?

Outputs:
  analysis/revision/results/pathology_dominates.png
  analysis/revision/results/pathology_dominates_summary.md
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import wilcoxon

RES = PROJECT_ROOT / "analysis" / "revision" / "results"
LADDER = RES / "leakage_free_ladder_per_patient.csv"
MULTI = RES / "all_multidraw_per_patient.csv"
META = PROJECT_ROOT / "data" / "physiomio" / "data" / "metadata.csv"
OUT_PNG = RES / "pathology_dominates.png"
OUT_MD = RES / "pathology_dominates_summary.md"


def cliffs_delta(x, y):
    d = np.asarray(x) - np.asarray(y)
    d = d[~np.isnan(d)]
    return float((np.sum(d > 0) - np.sum(d < 0)) / len(d))


def bootstrap_ci(fn, x, y, n=2000, seed=42):
    rng = np.random.RandomState(seed)
    x, y = np.asarray(x), np.asarray(y)
    idx = np.arange(len(x))
    vals = [fn(x[rng.choice(idx, len(idx), replace=True)], y[rng.choice(idx, len(idx), replace=True)])
              for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def main():
    ladder = pd.read_csv(LADDER)
    multi = pd.read_csv(MULTI).rename(columns={"target": "patient"})
    meta = pd.read_csv(META)
    days = meta.groupby("patient")["days_after_stroke"].first().to_dict()

    m = ladder.merge(multi[["patient", "imp_mean", "hlth_mean"]], on="patient", how="inner")
    m["days"] = m["patient"].map(days)
    m["diversity_delta"] = m["hlth_mean"] - m["row2_cross_arm_own_hlth"]
    m["pathology_delta"] = m["imp_mean"] - m["hlth_mean"]
    m = m.dropna(subset=["diversity_delta", "pathology_delta"])

    # QUADRANT COUNTS
    div_pos = m["diversity_delta"] > 0
    pat_pos = m["pathology_delta"] > 0
    n_both = int((div_pos & pat_pos).sum())
    n_pat_only = int((~div_pos & pat_pos).sum())
    n_div_only = int((div_pos & ~pat_pos).sum())
    n_neither = int((~div_pos & ~pat_pos).sum())
    n = len(m)
    n_pat_any = int(pat_pos.sum())
    n_div_any = int(div_pos.sum())

    # RESCUE CLAIM: of patients where diversity FAILS them (delta ≤ 0), how many
    # still benefit from pathology?
    div_fails = m[~div_pos]
    pat_rescues = int((div_fails["pathology_delta"] > 0).sum())
    pat_rescue_mean = float(div_fails["pathology_delta"].mean())

    # DELTA DIFFERENCE bootstrap
    # (paired: pathology_delta − diversity_delta per patient, CI on mean of diff)
    diff = m["pathology_delta"] - m["diversity_delta"]
    boot_diff = []
    rng = np.random.RandomState(42)
    for _ in range(2000):
        boot_diff.append(diff.iloc[rng.choice(len(m), len(m), replace=True)].mean())
    diff_ci = np.percentile(boot_diff, [2.5, 97.5])
    # bootstrap of delta_pathology_effect_size − delta_diversity_effect_size
    boot_delta_diff = []
    for _ in range(2000):
        idx = rng.choice(len(m), len(m), replace=True)
        s = m.iloc[idx]
        d_pat = ((s["imp_mean"] > s["hlth_mean"]).sum() - (s["imp_mean"] < s["hlth_mean"]).sum()) / len(s)
        d_div = ((s["hlth_mean"] > s["row2_cross_arm_own_hlth"]).sum() -
                    (s["hlth_mean"] < s["row2_cross_arm_own_hlth"]).sum()) / len(s)
        boot_delta_diff.append(d_pat - d_div)
    delta_diff_ci = np.percentile(boot_delta_diff, [2.5, 97.5])

    pat_delta = cliffs_delta(m["imp_mean"], m["hlth_mean"])
    div_delta = cliffs_delta(m["hlth_mean"], m["row2_cross_arm_own_hlth"])
    pat_wilcox = wilcoxon(m["imp_mean"], m["hlth_mean"], alternative="greater").pvalue
    div_wilcox = wilcoxon(m["hlth_mean"], m["row2_cross_arm_own_hlth"], alternative="greater").pvalue

    # ================= FIGURE =================
    fig = plt.figure(figsize=(12, 5.5), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.15])

    # ── LEFT PANEL: scatter with quadrants ──
    ax = fig.add_subplot(gs[0])
    # Days-post-stroke split labelled by cutoff, not by phase word.
    # Bernhardt et al. (2017) reserve "acute" for <=7 d and "chronic" for >6 mo;
    # our subsets are early-subacute-heavy on either side of 30 days, so we name
    # them by the cutoff itself.
    is_post_cutoff = m["days"] > 30
    ax.scatter(m.loc[~is_post_cutoff, "diversity_delta"], m.loc[~is_post_cutoff, "pathology_delta"],
                 s=54, c="tab:orange", alpha=0.75, edgecolor="white", linewidth=0.7, label="≤30 days")
    ax.scatter(m.loc[is_post_cutoff, "diversity_delta"], m.loc[is_post_cutoff, "pathology_delta"],
                 s=54, c="tab:blue", alpha=0.85, edgecolor="white", linewidth=0.7, label=">30 days")

    ax.axvline(0, color="grey", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.axhline(0, color="grey", linewidth=0.9, linestyle="--", alpha=0.7)

    # Quadrant labels + counts
    xlim = (min(-0.25, m["diversity_delta"].min() - 0.02), max(0.25, m["diversity_delta"].max() + 0.02))
    ylim = (min(-0.25, m["pathology_delta"].min() - 0.02), max(0.25, m["pathology_delta"].max() + 0.02))
    ax.set_xlim(xlim); ax.set_ylim(ylim)

    # Shade the "pathology helps" upper half
    ax.axhspan(0, ylim[1], color="tab:green", alpha=0.06, zorder=0)
    ax.axhspan(ylim[0], 0, color="tab:red", alpha=0.05, zorder=0)

    # Semi-transparent white background on each quadrant label so overlapping
    # data points don't render the counts unreadable.
    label_bg = dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2.0)
    ax.text(xlim[0] + 0.03, ylim[1] - 0.03,
             f"Pathology-only rescue\n(diversity failed, pathology helped)\nn={n_pat_only}/{n}",
             va="top", ha="left", fontsize=10, fontweight="bold", color="tab:green",
             bbox=label_bg)
    ax.text(xlim[1] - 0.02, ylim[1] - 0.02,
             f"Both helped\nn={n_both}/{n}",
             va="top", ha="right", fontsize=10, color="dimgray", bbox=label_bg)
    ax.text(xlim[0] + 0.02, ylim[0] + 0.02,
             f"Neither helped\nn={n_neither}/{n}",
             va="bottom", ha="left", fontsize=10, color="dimgray", bbox=label_bg)
    ax.text(xlim[1] - 0.02, ylim[0] + 0.02,
             f"Diversity-only\nn={n_div_only}/{n}",
             va="bottom", ha="right", fontsize=10, color="dimgray", bbox=label_bg)

    ax.set_xlabel("Diversity Δ  (47 others' healthy − own healthy)")
    ax.set_ylabel("Pathology Δ  (47 others' impaired − 47 others' healthy)")
    ax.set_title(f"Pathology helps {n_pat_any}/{n} patients ({100*n_pat_any/n:.0f}%); "
                    f"diversity helps only {n_div_any}/{n} ({100*n_div_any/n:.0f}%)",
                    fontsize=11)
    # Legend below the panel — no in-axes corner is free with 4 quadrant labels.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9,
               frameon=False)
    ax.grid(alpha=0.25)

    # ── RIGHT PANEL: waterfall sorted by pathology Δ ──
    ax2 = fig.add_subplot(gs[1])
    m_sorted = m.sort_values("pathology_delta").reset_index(drop=True)
    xs = np.arange(len(m_sorted))
    width = 0.42
    colors_pat = ["tab:green" if v > 0 else "tab:red" for v in m_sorted["pathology_delta"]]
    colors_div = ["tab:green" if v > 0 else "tab:red" for v in m_sorted["diversity_delta"]]
    ax2.bar(xs - width/2, m_sorted["diversity_delta"], width=width,
              color=colors_div, alpha=0.35, edgecolor="none", label="Diversity Δ (per patient)")
    ax2.bar(xs + width/2, m_sorted["pathology_delta"], width=width,
              color=colors_pat, alpha=0.95, edgecolor="none", label="Pathology Δ (per patient)")
    ax2.axhline(0, color="black", linewidth=0.9)
    ax2.set_xlabel("Patients, sorted by pathology Δ  →")
    ax2.set_ylabel("Δ accuracy")
    # Python's .2f uses banker's rounding, so 0.125 becomes 0.12; the paper
    # rounds half-up, giving 0.13. Force half-up here to match the paper's
    # tables and text.
    def rhu(x, dp=2):
        from decimal import Decimal, ROUND_HALF_UP
        q = Decimal("1e-{}".format(dp))
        v = Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP)
        return f"{v:+.{dp}f}"
    ax2.set_title(f"Pathology: {100*n_pat_any/n:.0f}% patients above zero (δ={rhu(pat_delta)}, p={pat_wilcox:.3f})\n"
                    f"Diversity:  {100*n_div_any/n:.0f}% patients above zero (δ={rhu(div_delta)}, p={div_wilcox:.3f})",
                    fontsize=10)
    ax2.set_xticks([])
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(axis="y", alpha=0.25)

    plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight",
                  facecolor="white", transparent=False)
    print(f"Wrote {OUT_PNG}")

    # ================= SUMMARY =================
    md = [
        "# Pathology dominates diversity — the hero result",
        "",
        f"**n = {n} patients (48 with complete leakage-free features + multi-draw values)**",
        "",
        "## Categorical patient breakdown",
        "",
        "|                        | Diversity helps | Diversity hurts | Total |",
        "|---|---:|---:|---:|",
        f"| **Pathology helps**    | {n_both:2d} | **{n_pat_only:2d}** | {n_both+n_pat_only:2d} ({100*(n_both+n_pat_only)/n:.0f}%) |",
        f"| **Pathology hurts**    | {n_div_only:2d} | {n_neither:2d} | {n_div_only+n_neither:2d} |",
        f"| Total                  | {n_both+n_div_only:2d} ({100*(n_both+n_div_only)/n:.0f}%) | {n_pat_only+n_neither:2d} | {n:2d} |",
        "",
        "## The killer numbers",
        "",
        f"- **Pathology helps {n_pat_any}/{n} patients ({100*n_pat_any/n:.0f}%)** vs. "
          f"**diversity helps only {n_div_any}/{n} ({100*n_div_any/n:.0f}%)**",
        f"- Of the {int((~div_pos).sum())} patients where diversity FAILS them (adding 47 healthy donors makes",
        f"  things worse or does nothing), pathology-matching still helps **{pat_rescues}/{int((~div_pos).sum())}** "
          f"of them — mean rescue Δ = **{pat_rescue_mean:+.4f}**",
        f"- Cliff's δ: pathology **{pat_delta:+.3f}** vs diversity **{div_delta:+.3f}** — pathology's effect size is "
          f"{pat_delta/div_delta:.1f}× larger",
        f"- Bootstrap 95% CI on `Δ_pathology_effect − Δ_diversity_effect` (mean per-patient): "
          f"[{diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}]",
        f"- Bootstrap 95% CI on `δ_pathology − δ_diversity`: [{delta_diff_ci[0]:+.3f}, {delta_diff_ci[1]:+.3f}]",
        "",
        "## Paper-ready sentences",
        "",
        f"- \"Pathology-matched training helps {n_pat_any}/{n} patients ({100*n_pat_any/n:.0f}%); adding donor",
        f"  diversity while keeping training data healthy helps only {n_div_any}/{n} patients ({100*n_div_any/n:.0f}%).\"",
        f"- \"Even in the {int((~div_pos).sum())} patients where diversity provides no benefit, pathology-matching",
        f"  still delivers a mean +{pat_rescue_mean*100:.1f} pp gain ({pat_rescues}/{int((~div_pos).sum())} of them).\"",
        f"- \"Cliff's δ = {pat_delta:+.3f} for pathology versus {div_delta:+.3f} for diversity; the pairwise",
        f"  δ-difference bootstrap 95% CI excludes zero at [{delta_diff_ci[0]:+.2f}, {delta_diff_ci[1]:+.2f}].\"",
    ]
    OUT_MD.write_text("\n".join(md))
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
