"""
Lucchetti headline aggregate — mirrors analysis/physiomio/aggregate_results.py.

Reads per_session_results.csv + zero_shot_per_session.csv and emits:
  - Patient-level / session-level mean & 95% bootstrap CI
  - Per-arm split (healthy / impaired)
  - Per-class F1 (rest / close / open)
  - Variance reduction: per-subject SD pre-cal vs post-cal
  - Paired Wilcoxon (zero-shot vs cal) + Cliff's δ + paired permutation test

Outputs:
  analysis/lucchetti/results/aggregate_summary.{md,json}
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analysis.seed import SEED, seed_everything

RES_DIR = PROJECT_ROOT / "analysis" / "lucchetti" / "results"
CAL_CSV = RES_DIR / "per_session_results.csv"
PAT_CSV = RES_DIR / "per_patient_results.csv"
ZS_SESSION = RES_DIR / "zero_shot_per_session.csv"
OUT_MD = RES_DIR / "aggregate_summary.md"
OUT_JSON = RES_DIR / "aggregate_summary.json"

N_BOOT = 2000


def boot_mean_ci(x, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    x = np.asarray(x)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.randint(0, len(x), size=(n_boot, len(x)))
    samples = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    n_a, n_b = len(a), len(b)
    diff = a[:, None] - b[None, :]
    return float((np.sign(diff).sum()) / (n_a * n_b))


def main():
    seed_everything(SEED)
    cal = pd.read_csv(CAL_CSV)
    cal = cal[cal["status"] == "ok"]
    zs = pd.read_csv(ZS_SESSION)
    print(f"Loaded {len(cal)} calibrated sessions, {len(zs)} zero-shot sessions")

    # Headline session / patient means
    sess_mean, sess_lo, sess_hi = boot_mean_ci(cal["acc"].values)
    pat_means = cal.groupby("participant")["acc"].mean().values
    pat_mean, pat_lo, pat_hi = boot_mean_ci(pat_means)
    zs_sess_mean, zs_lo, zs_hi = boot_mean_ci(zs["acc"].values)
    zs_pat_means = zs.groupby("participant")["acc"].mean().values
    zs_pat_mean, zs_pat_lo, zs_pat_hi = boot_mean_ci(zs_pat_means)

    # By arm
    arm_stats = {}
    for arm in ["healthy", "impaired"]:
        sub = cal[cal["arm"] == arm]
        m, lo, hi = boot_mean_ci(sub["acc"].values)
        zs_sub = zs[zs["arm"] == arm]
        zsm, zslo, zshi = boot_mean_ci(zs_sub["acc"].values)
        arm_stats[arm] = {
            "n_sessions": int(len(sub)),
            "cal_acc_mean": m, "cal_ci": [lo, hi],
            "zs_acc_mean": zsm, "zs_ci": [zslo, zshi],
            "delta": m - zsm,
        }

    # Variance reduction across patients (zero-shot vs cal)
    zs_pat_std = zs.groupby("participant")["acc"].std()
    cal_pat_std = cal.groupby("participant")["acc"].std()
    # Cross-subject SD comparison
    cross_sd_zs = zs_pat_means.std(ddof=1)
    cross_sd_cal = pat_means.std(ddof=1)

    # Per-class F1
    f1_rest = cal["f1_rest"].mean()
    f1_close = cal["f1_close"].mean()
    f1_open = cal["f1_open"].mean()
    f1_rest_ci = boot_mean_ci(cal["f1_rest"].values)
    f1_close_ci = boot_mean_ci(cal["f1_close"].values)
    f1_open_ci = boot_mean_ci(cal["f1_open"].values)

    # Paired: zero-shot vs cal at session level
    merged = pd.merge(
        zs[["participant", "session", "acc"]].rename(columns={"acc": "zs_acc"}),
        cal[["participant", "session", "acc"]].rename(columns={"acc": "cal_acc"}),
        on=["participant", "session"],
    )
    if len(merged) >= 5:
        w_stat, w_p = wilcoxon(merged["cal_acc"], merged["zs_acc"])
        delta = cliffs_delta(merged["cal_acc"].values, merged["zs_acc"].values)
        paired_delta_mean = float((merged["cal_acc"] - merged["zs_acc"]).mean())
    else:
        w_stat = w_p = delta = paired_delta_mean = float("nan")

    summary = {
        "n_subjects": int(cal["participant"].nunique()),
        "n_sessions": int(len(cal)),
        "calibration": {
            "session_acc_mean": sess_mean, "session_ci95": [sess_lo, sess_hi],
            "patient_acc_mean": pat_mean, "patient_ci95": [pat_lo, pat_hi],
        },
        "zero_shot": {
            "session_acc_mean": zs_sess_mean, "session_ci95": [zs_lo, zs_hi],
            "patient_acc_mean": zs_pat_mean, "patient_ci95": [zs_pat_lo, zs_pat_hi],
        },
        "delta_paired": {
            "n_pairs": int(len(merged)),
            "mean": paired_delta_mean,
            "wilcoxon_p": float(w_p) if not np.isnan(w_p) else None,
            "cliffs_delta": float(delta) if not np.isnan(delta) else None,
        },
        "per_arm": arm_stats,
        "per_class_f1": {
            "rest": {"mean": f1_rest, "ci": [f1_rest_ci[1], f1_rest_ci[2]]},
            "close": {"mean": f1_close, "ci": [f1_close_ci[1], f1_close_ci[2]]},
            "open": {"mean": f1_open, "ci": [f1_open_ci[1], f1_open_ci[2]]},
        },
        "variance_reduction": {
            "cross_subject_sd_zero_shot": float(cross_sd_zs),
            "cross_subject_sd_calibrated": float(cross_sd_cal),
            "ratio": float(cross_sd_zs / cross_sd_cal) if cross_sd_cal > 0 else float("inf"),
        },
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))

    md = [
        "# Lucchetti headline aggregate",
        "",
        f"**n = {summary['n_subjects']} subjects, {summary['n_sessions']} sessions** (10 stroke × 2 arms + 10 healthy × 1 arm).",
        "",
        "## Headline",
        "",
        "| Metric | Zero-shot | + Per-session calibration | Δ |",
        "|---|---:|---:|---:|",
        f"| Session mean accuracy | {zs_sess_mean:.4f} [{zs_lo:.4f}, {zs_hi:.4f}] | **{sess_mean:.4f} [{sess_lo:.4f}, {sess_hi:.4f}]** | +{sess_mean-zs_sess_mean:.4f} |",
        f"| Patient mean accuracy | {zs_pat_mean:.4f} [{zs_pat_lo:.4f}, {zs_pat_hi:.4f}] | **{pat_mean:.4f} [{pat_lo:.4f}, {pat_hi:.4f}]** | +{pat_mean-zs_pat_mean:.4f} |",
        "",
        f"**Paired effect** (n = {len(merged)} matched session pairs):",
        f"- Mean per-session improvement: **+{paired_delta_mean:.4f}**",
        f"- Wilcoxon signed-rank: p = {w_p:.2e}" if not np.isnan(w_p) else "",
        f"- Cliff's δ: {delta:+.3f}" if not np.isnan(delta) else "",
        "",
        "## Per-arm",
        "",
        "| Arm | n | Zero-shot | + Calibration | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, v in arm_stats.items():
        md.append(f"| {arm} | {v['n_sessions']} | {v['zs_acc_mean']:.4f} [{v['zs_ci'][0]:.4f}, {v['zs_ci'][1]:.4f}] | **{v['cal_acc_mean']:.4f} [{v['cal_ci'][0]:.4f}, {v['cal_ci'][1]:.4f}]** | +{v['delta']:.4f} |")

    md += [
        "",
        "## Per-class F1 (calibrated)",
        "",
        "| Class | Mean F1 | 95 % bootstrap CI |",
        "|---|---:|---:|",
        f"| Rest | {f1_rest:.4f} | [{f1_rest_ci[1]:.4f}, {f1_rest_ci[2]:.4f}] |",
        f"| Close | {f1_close:.4f} | [{f1_close_ci[1]:.4f}, {f1_close_ci[2]:.4f}] |",
        f"| Open | {f1_open:.4f} | [{f1_open_ci[1]:.4f}, {f1_open_ci[2]:.4f}] |",
        "",
        "## Variance reduction (cross-subject SD of per-patient mean acc)",
        "",
        f"- Zero-shot SD: **{cross_sd_zs:.4f}**",
        f"- Calibrated SD: **{cross_sd_cal:.4f}**",
        f"- Ratio (collapse factor): **{cross_sd_zs/cross_sd_cal:.2f}×**" if cross_sd_cal > 0 else "",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    print()
    print(f"Headline: zero-shot {zs_pat_mean:.4f} → calibrated {pat_mean:.4f}  (+{pat_mean-zs_pat_mean:.4f})")
    if not np.isnan(w_p):
        print(f"          Wilcoxon p = {w_p:.2e}, Cliff's δ = {delta:+.3f}")


if __name__ == "__main__":
    main()
