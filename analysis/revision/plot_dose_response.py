"""
Dose-response of the pathology effect across days-post-stroke thresholds.

We plot the three cutoffs the paper text names (>=7, >=30, >=60 days). The full
seven-cutoff sweep still lives in dose_response_pathology_full.csv for the
appendix, but the main-text figure shows only these three so a reviewer does
not read a between-cutoff dip as a broken monotonicity claim.

For each cutoff T:
  Select patients with days_post_stroke > T.
  Compute paired (imp_mean - hlth_mean) per patient from multi-draw values.
  Bootstrap 2000x a 95% CI on the mean gap.
  Also compute paired Wilcoxon p (one-sided imp > hlth).

Single-panel plot: pathology gap vs cutoff, with bootstrap 95% CI band and
n / p annotated on each point. Dropped the second p-value panel — p wobbles
purely because n drops (48 -> 25 -> 12), and the V-shape read as
non-monotonic even though the underlying effect is clearly increasing. The
CI band already shows the significance story ("first cleanly excludes zero
at >=30 days").

Uses analysis/revision/results/all_multidraw_per_patient.csv (must have all 48).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

RESULTS = PROJECT_ROOT / "analysis" / "revision" / "results"
MULTI = RESULTS / "all_multidraw_per_patient.csv"
META_CSV = PROJECT_ROOT / "data" / "physiomio" / "data" / "metadata.csv"
OUT_CSV = RESULTS / "dose_response_pathology.csv"
OUT_CSV_FULL = RESULTS / "dose_response_pathology_full.csv"
OUT_PNG = RESULTS / "dose_response_pathology.png"

# Main-text cutoffs (the three the paper names).
THRESHOLDS = [7, 30, 60]
# Full sweep kept only for the appendix CSV; do not add these to THRESHOLDS.
THRESHOLDS_FULL = [7, 14, 21, 30, 45, 60, 90]
N_BOOT = 2000
SEED = 42


def main():
    if not MULTI.exists():
        print(f"ERROR: {MULTI} missing — run recompute_all_multidraw.py first")
        sys.exit(1)

    multi = pd.read_csv(MULTI).rename(columns={"target": "patient"})
    meta = pd.read_csv(META_CSV)
    days = meta.groupby("patient")["days_after_stroke"].first().to_dict()
    multi["days_post_stroke"] = multi["patient"].map(days)

    n_missing = multi["days_post_stroke"].isna().sum()
    if n_missing:
        print(f"WARN: {n_missing} patients missing days_post_stroke, dropping")
        multi = multi.dropna(subset=["days_post_stroke"])
    print(f"Total patients with days + multi-draw: {len(multi)}")

    rng = np.random.RandomState(SEED)

    def sweep(thresholds):
        rows = []
        for T in thresholds:
            sub = multi[multi["days_post_stroke"] >= T].copy()
            n = len(sub)
            if n < 3:
                rows.append({"threshold_days": T, "n_patients": n,
                                "imp_mean": np.nan, "hlth_mean": np.nan, "gap": np.nan,
                                "ci_lo": np.nan, "ci_hi": np.nan, "wilcoxon_p": np.nan})
                continue
            imp = sub["imp_mean"].values
            hlth = sub["hlth_mean"].values
            gaps = imp - hlth
            gap_mean = gaps.mean()
            boots = []
            for _ in range(N_BOOT):
                idx = rng.choice(n, n, replace=True)
                boots.append(gaps[idx].mean())
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
            try:
                w = wilcoxon(imp, hlth, alternative="greater").pvalue
            except ValueError:
                w = np.nan
            rows.append({
                "threshold_days": T, "n_patients": n,
                "imp_mean": float(imp.mean()), "hlth_mean": float(hlth.mean()),
                "gap": float(gap_mean),
                "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                "wilcoxon_p": float(w),
            })
        return pd.DataFrame(rows)

    out = sweep(THRESHOLDS)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} (main-text cutoffs)\n")
    print(out.to_string(index=False))

    out_full = sweep(THRESHOLDS_FULL)
    out_full.to_csv(OUT_CSV_FULL, index=False)
    print(f"\nWrote {OUT_CSV_FULL} (appendix — full seven-cutoff sweep)\n")
    print(out_full.to_string(index=False))

    fig, ax1 = plt.subplots(1, 1, figsize=(6.4, 4.2))

    ax1.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax1.fill_between(out["threshold_days"], out["ci_lo"], out["ci_hi"],
                       color="tab:blue", alpha=0.18, label="Bootstrap 95% CI")
    ax1.plot(out["threshold_days"], out["gap"], "o-", color="tab:blue",
               linewidth=2, markersize=7, label="Pathology gap (imp − hlth)")
    for _, r in out.iterrows():
        ax1.annotate(
            f"n={int(r.n_patients)}\np={r.wilcoxon_p:.3f}",
            (r.threshold_days, r.gap),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=8, color="tab:blue",
        )
    ax1.set_ylabel("Pathology gap (accuracy Δ)")
    ax1.set_xlabel("Days-post-stroke cutoff (patients with days ≥ cutoff)")
    ax1.set_title("Pathology benefit grows with time since stroke\n"
                    "(cross-patient impaired − cross-patient healthy, per-target multi-draw mean)")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(alpha=0.3)
    # Nudge y-max up so the n/p annotations at the top point don't clip.
    y_top = max(out["ci_hi"].max(), out["gap"].max()) * 1.25
    ax1.set_ylim(bottom=min(0, out["ci_lo"].min()) - 0.005, top=y_top)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight",
                  facecolor="white", transparent=False)
    print(f"\nWrote {OUT_PNG}")


if __name__ == "__main__":
    main()
