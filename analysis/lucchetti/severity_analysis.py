"""
Lucchetti severity stratification by L_CA (FMA-UE level 1-5).

Mirrors analysis/physiomio/severity_analysis.py adapted for Lucchetti's
per-subject single L_CA score (1-5 categorical, where 1 = severe / FMA 0-22
and 5 = mild / FMA 53-66).

Output:
  analysis/lucchetti/results/severity_summary.{md,json}
  analysis/lucchetti/results/severity_per_subject.csv
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import spearmanr

from analysis.seed import SEED, seed_everything

LUCCHETTI_DIR = PROJECT_ROOT / "data" / "lucchetti"
CAL_CSV = PROJECT_ROOT / "analysis" / "lucchetti" / "results" / "per_session_results.csv"
ZS_CSV = PROJECT_ROOT / "analysis" / "lucchetti" / "results" / "zero_shot_per_session.csv"
OUT_MD = PROJECT_ROOT / "analysis" / "lucchetti" / "results" / "severity_summary.md"
OUT_JSON = PROJECT_ROOT / "analysis" / "lucchetti" / "results" / "severity_summary.json"
OUT_CSV = PROJECT_ROOT / "analysis" / "lucchetti" / "results" / "severity_per_subject.csv"


def load_stroke_metadata():
    """Read L_CA + TimeFromEvent + HemiSide per stroke subject from .mat metadata."""
    meta_rows = []
    for i in range(1, 11):
        code = f"ST_{i:02d}"
        path = LUCCHETTI_DIR / "stroke" / code / f"ST{i:02d}.mat"
        if not path.exists():
            continue
        d = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        s = d["s"]
        meta_rows.append({
            "participant": code,
            "L_CA": int(s.L_CA),
            "TimeFromEvent_lo": int(np.atleast_1d(s.TimeFromEvent)[0]),
            "TimeFromEvent_hi": int(np.atleast_1d(s.TimeFromEvent)[-1]) if hasattr(s.TimeFromEvent, "__len__") else int(s.TimeFromEvent),
            "HemiSide": str(s.HemiSide),
            "Age_lo": int(np.atleast_1d(s.Age)[0]),
            "Gender": str(s.Gender),
        })
    return pd.DataFrame(meta_rows)


def main():
    seed_everything(SEED)
    meta = load_stroke_metadata()
    print(f"Loaded metadata for {len(meta)} stroke subjects")
    print(f"  L_CA distribution: {dict(meta['L_CA'].value_counts().sort_index())}")

    cal = pd.read_csv(CAL_CSV)
    cal = cal[cal["status"] == "ok"]
    zs = pd.read_csv(ZS_CSV)

    # Aggregate per-subject (impaired arm only — that's what we stratify on)
    def agg(df, prefix):
        imp = df[df["arm"] == "impaired"]
        return imp.groupby("participant").agg(**{
            f"{prefix}_acc_mean": ("acc", "mean"),
            f"{prefix}_acc_std": ("acc", "std"),
            f"{prefix}_f1_macro": ("f1_macro", "mean"),
            f"{prefix}_n_sessions": ("acc", "count"),
        }).reset_index()

    cal_imp = agg(cal, "cal")
    zs_imp = agg(zs, "zs")
    merged = meta.merge(cal_imp, on="participant", how="left").merge(zs_imp, on="participant", how="left")
    merged["cal_minus_zs"] = merged["cal_acc_mean"] - merged["zs_acc_mean"]
    merged.to_csv(OUT_CSV, index=False)
    print(f"\nMerged per-subject:")
    print(merged[["participant", "L_CA", "zs_acc_mean", "cal_acc_mean", "cal_minus_zs"]].to_string(index=False))

    # Spearman correlation: L_CA vs calibration benefit
    rho, p = spearmanr(merged["L_CA"].values, merged["cal_minus_zs"].values)
    # And L_CA vs absolute calibrated accuracy
    rho_abs, p_abs = spearmanr(merged["L_CA"].values, merged["cal_acc_mean"].values)

    # Tertile / level stratification
    level_stats = {}
    for lvl in sorted(merged["L_CA"].unique()):
        sub = merged[merged["L_CA"] == lvl]
        level_stats[int(lvl)] = {
            "n_subjects": int(len(sub)),
            "zs_acc_mean": float(sub["zs_acc_mean"].mean()),
            "zs_acc_std": float(sub["zs_acc_mean"].std()) if len(sub) > 1 else float("nan"),
            "cal_acc_mean": float(sub["cal_acc_mean"].mean()),
            "cal_acc_std": float(sub["cal_acc_mean"].std()) if len(sub) > 1 else float("nan"),
            "delta_mean": float(sub["cal_minus_zs"].mean()),
        }

    summary = {
        "n_subjects": int(len(merged)),
        "L_CA_distribution": dict(merged["L_CA"].value_counts().sort_index().to_dict()),
        "by_level": level_stats,
        "spearman_L_CA_vs_cal_benefit": {"rho": float(rho), "p": float(p)},
        "spearman_L_CA_vs_calibrated_acc": {"rho": float(rho_abs), "p": float(p_abs)},
        "scale_meaning": "L_CA = Fugl-Meyer UE level: 1 = 0-22 (severe), 2 = 23-31, 3 = 32-42, 4 = 43-52, 5 = 53-66 (mild)",
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))

    md = [
        "# Lucchetti severity stratification",
        "",
        f"n = {len(merged)} stroke subjects, scored by **L_CA = Fugl-Meyer UE level (1-5)**.",
        f"  - Level 1: FMA 0-22 (severe)  · Level 2: 23-31  · Level 3: 32-42  · Level 4: 43-52  · Level 5: 53-66 (mild)",
        f"  - L_CA distribution in cohort: {summary['L_CA_distribution']}",
        "",
        "## Per-level accuracy (impaired arm only)",
        "",
        "| L_CA | n | Zero-shot | + Calibration | Δ |",
        "|---:|---:|---:|---:|---:|",
    ]
    for lvl, s in level_stats.items():
        md.append(f"| {lvl} | {s['n_subjects']} | {s['zs_acc_mean']:.4f} ± {s['zs_acc_std']:.4f} | **{s['cal_acc_mean']:.4f} ± {s['cal_acc_std']:.4f}** | +{s['delta_mean']:.4f} |")

    md += [
        "",
        "## Correlation with severity",
        "",
        f"- Spearman ρ(L_CA, calibration benefit Δacc) = **{rho:+.3f}**, p = {p:.3f}",
        f"- Spearman ρ(L_CA, calibrated accuracy)     = **{rho_abs:+.3f}**, p = {p_abs:.3f}",
        "",
        "Positive ρ would mean less-impaired patients (higher L_CA) benefit more from calibration. Near-zero ρ would mean calibration helps regardless of severity (same null finding as PhysioMio).",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(f"\nKey: Spearman ρ(L_CA, cal benefit) = {rho:+.3f}, p = {p:.3f}")


if __name__ == "__main__":
    main()
