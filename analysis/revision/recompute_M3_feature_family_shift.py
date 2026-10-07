"""
Revision — M3: feature-family shift ranking.

Post-hoc analysis on existing feature_shift_ranked.csv + engineered feature list.
Groups the 370 features by base family (amplitude / envelope / spectral /
waveform-crossing / cross-channel / other), reports mean Wasserstein-1 shift
per family. Ties into the K=30 deployment finding: amplitude-family features
carry both the signal AND the shift.

Outputs:
  analysis/revision/results/M3_feature_family_shift_summary.md
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

SHIFT_CSV = PROJECT_ROOT / "analysis" / "mechanism" / "results" / "feature_shift_ranked.csv"
OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "M3_feature_family_shift_per_family.csv"
OUT_MD = OUT_DIR / "M3_feature_family_shift_summary.md"

AMPLITUDE_BASE = {"rms", "mav", "iemg", "var", "maxamp", "env_rms", "env_mean",
                   "env_max", "env_std", "wl"}
DEGENERATE_BASE = {"zc", "ssc", "wamp", "mean_freq", "median_freq"}


def feature_family(feature_name):
    """Assign a feature to a coarse family: amplitude / degenerate / cross-channel / other."""
    if "_ratio" in feature_name or "_diff" in feature_name:
        return "cross-channel"
    if "activity" in feature_name:
        return "other"
    # Strip temporal suffixes
    base = feature_name.split("_", 1)[1] if "_" in feature_name else feature_name
    for suffix in ["_prev2", "_prev", "_roll5", "_roll3", "_delta", "_accel", "_sess_norm"]:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    if base in AMPLITUDE_BASE:
        return "amplitude"
    if base in DEGENERATE_BASE:
        return "degenerate"
    return "other"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SHIFT_CSV.exists():
        print(f"Missing {SHIFT_CSV}. Aborting.")
        return

    shift = pd.read_csv(SHIFT_CSV)
    shift["family"] = shift["feature"].apply(feature_family)

    # Aggregate per family
    agg = shift.groupby("family")["w_grabmyo_vs_impaired"].agg(
        mean="mean", median="median", std="std", n="count"
    ).round(4).sort_values("mean", ascending=False)
    agg.to_csv(OUT_CSV)

    # Also compute top-K features and their family composition
    top10 = shift.nlargest(10, "w_grabmyo_vs_impaired")
    top30 = shift.nlargest(30, "w_grabmyo_vs_impaired")
    top10_families = top10["family"].value_counts()
    top30_families = top30["family"].value_counts()

    md = [
        "# M3 — feature-family shift ranking",
        "",
        f"Wasserstein-1 shift from GrabMyo (healthy) to PhysioMio impaired-arm, grouped by feature family.",
        f"Total features with shift measured: {len(shift)}.",
        "",
        "## Mean W₁ shift per family",
        "",
        "| Family | Mean shift | Median shift | Std | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, row in agg.iterrows():
        md.append(f"| {family} | {row['mean']:.3f} | {row['median']:.3f} | "
                  f"{row['std']:.3f} | {int(row['n'])} |")

    md += [
        "",
        "## Top-10 most-shifted features (family composition)",
        "",
        "| Family | Count in top-10 |",
        "|---|---:|",
    ]
    for f, c in top10_families.items():
        md.append(f"| {f} | {c} |")

    md += [
        "",
        "## Top-30 most-shifted features (family composition)",
        "",
        "| Family | Count in top-30 |",
        "|---|---:|",
    ]
    for f, c in top30_families.items():
        md.append(f"| {f} | {c} |")

    md += [
        "",
        "## Top 10 features by name",
        "",
    ]
    for _, r in top10.iterrows():
        md.append(f"- **{r['feature']}** (family: {r['family']}, W₁ = {r['w_grabmyo_vs_impaired']:.3f})")

    md += [
        "",
        "## Interpretation",
        "",
        "- If amplitude-family features dominate both the shift ranking AND the deployment",
        "  top-30 (from the feature audit), then the deployed signal path is exactly where",
        "  the healthy→stroke distribution shift is largest.",
        "- If crossing/frequency families dominate the shift, they're irrelevant at 20 Hz",
        "  deployment because they're already dead there (F ≈ 2 in the feature audit).",
        "- Ties into the K=30 finding: the features that matter at deployment are the same",
        "  features whose distribution differs most.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
