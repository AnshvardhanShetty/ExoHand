"""
PhysioMio aggregate analysis — produces the paper-ready headline statistics.

Inputs:
    analysis/physiomio/results/per_session_results.csv   # 329 sessions × 14 cols (with cal)
    analysis/physiomio/results/per_patient_results.csv   # 48 patients × 11 cols (with cal, aggregated)
    analysis/physiomio/results/zero_shot_per_patient.csv # 48 patients × 14 cols (no cal)

Outputs:
    analysis/physiomio/results/aggregate_summary.md   # paper-ready markdown
    analysis/physiomio/results/aggregate_summary.json # downstream machine-readable

Statistical methods:
    - Bootstrap 95% CIs (2000 resamples) on:
        * patient-level mean accuracy with cal
        * session-level mean accuracy with cal
        * per-class F1 means
        * cross-subject std (with cal)
        * calibration-improvement delta (paired)
        * variance-reduction ratio (zero-shot std / with-cal std)
    - Paired Wilcoxon signed-rank (H1: cal > zero-shot) per patient
    - Cliff's delta effect size on the cal vs zero-shot paired distribution
    - Per-arm paired Wilcoxon (healthy contralateral vs impaired arm, per patient)
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from analysis.seed import SEED, seed_everything


RESULTS_DIR = PROJECT_ROOT / "analysis" / "physiomio" / "results"
PER_SESSION_CSV = RESULTS_DIR / "per_session_results.csv"
PER_PATIENT_CSV = RESULTS_DIR / "per_patient_results.csv"
ZERO_SHOT_CSV = RESULTS_DIR / "zero_shot_per_patient.csv"
OUT_MD = RESULTS_DIR / "aggregate_summary.md"
OUT_JSON = RESULTS_DIR / "aggregate_summary.json"


def bootstrap_stat(values, stat_fn, n_bootstrap, rng):
    point = stat_fn(values)
    n = len(values)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        samples[i] = stat_fn(values[idx])
    return point, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def bootstrap_paired_diff(a, b, n_bootstrap, rng):
    """Bootstrap CI for mean(b - a). a and b are paired by index."""
    diff = b - a
    point = float(np.mean(diff))
    n = len(diff)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        samples[i] = np.mean(diff[idx])
    return point, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def bootstrap_ratio(a, b, n_bootstrap, rng):
    """Bootstrap CI for std(a) / std(b)."""
    sa, sb = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    point = sa / max(sb, 1e-12)
    n = len(a)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        sai = np.std(a[idx], ddof=1)
        sbi = np.std(b[idx], ddof=1)
        samples[i] = sai / max(sbi, 1e-12)
    return point, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def cliffs_delta(a, b):
    """Cliff's delta for paired (b - a) sense. Positive = b > a typically."""
    n = len(a)
    n_ab = sum(bi > ai for ai, bi in zip(a, b))
    n_ba = sum(ai > bi for ai, bi in zip(a, b))
    return (n_ab - n_ba) / n


def main():
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    n_bootstrap = 2000

    # --- Load all three CSVs ---
    per_sess = pd.read_csv(PER_SESSION_CSV)
    per_sess = per_sess[per_sess["status"] == "ok"].copy()
    per_pat_cal = pd.read_csv(PER_PATIENT_CSV)
    per_pat_zs = pd.read_csv(ZERO_SHOT_CSV)

    print(f"Loaded:")
    print(f"  per_session (with cal): {per_sess.shape}")
    print(f"  per_patient (with cal): {per_pat_cal.shape}")
    print(f"  zero_shot (per patient): {per_pat_zs.shape}")

    # --- Patient-level and session-level CIs (with cal) ---
    pat_acc = per_pat_cal["acc_mean"].values
    sess_acc = per_sess["acc"].values
    sess_f1 = per_sess["f1_macro"].values

    mean_pat_acc = bootstrap_stat(pat_acc, np.mean, n_bootstrap, rng)
    std_pat_acc = bootstrap_stat(pat_acc, lambda x: float(np.std(x, ddof=1)), n_bootstrap, rng)
    mean_sess_acc = bootstrap_stat(sess_acc, np.mean, n_bootstrap, rng)
    std_sess_acc = bootstrap_stat(sess_acc, lambda x: float(np.std(x, ddof=1)), n_bootstrap, rng)
    mean_sess_f1 = bootstrap_stat(sess_f1, np.mean, n_bootstrap, rng)

    # --- Per-class F1 CIs ---
    per_class = {}
    for cls in ["rest", "close", "open"]:
        v = per_sess[f"f1_{cls}"].values
        per_class[cls] = bootstrap_stat(v, np.mean, n_bootstrap, rng)

    # --- Paired zero-shot vs with-cal (per patient) ---
    # join on participant
    paired = per_pat_zs[["participant", "acc_no_cal"]].merge(
        per_pat_cal[["participant", "acc_mean"]].rename(columns={"acc_mean": "acc_with_cal"}),
        on="participant", how="inner",
    )
    assert len(paired) == 48, f"Expected 48 paired patients, got {len(paired)}"
    no_cal = paired["acc_no_cal"].values
    with_cal = paired["acc_with_cal"].values
    delta_paired = bootstrap_paired_diff(no_cal, with_cal, n_bootstrap, rng)

    wilcoxon_res = stats.wilcoxon(with_cal, no_cal, alternative="greater")
    cliffs = float(cliffs_delta(no_cal, with_cal))

    # Variance ratio: no-cal std / with-cal std (how much variance collapse)
    var_ratio = bootstrap_ratio(no_cal, with_cal, n_bootstrap, rng)

    # --- Per-arm paired Wilcoxon (healthy contralateral vs impaired, per patient) ---
    arm_paired = per_pat_cal[["participant", "acc_healthy_mean", "acc_impaired_mean"]].dropna()
    h_arm = arm_paired["acc_healthy_mean"].values
    i_arm = arm_paired["acc_impaired_mean"].values
    arm_wilcoxon = stats.wilcoxon(h_arm, i_arm, alternative="greater")
    arm_delta = bootstrap_paired_diff(i_arm, h_arm, n_bootstrap, rng)   # (healthy - impaired)

    # ============================================================
    # Console output
    # ============================================================
    def fmt(t):
        v, lo, hi = t
        return f"{v:.4f}  [{lo:.4f}, {hi:.4f}]"

    print()
    print("=" * 75)
    print(f"PhysioMio per-session calibration — paper-ready aggregate (n={len(per_pat_cal)} patients, {len(per_sess)} sessions)")
    print("=" * 75)
    print()
    print(f"WITH-CAL HEADLINE  (per-session 30s recal, weight 100x, single-tier)")
    print(f"  Patient-level mean acc:   {fmt(mean_pat_acc)}    [n=48]")
    print(f"  Patient-level std:        {fmt(std_pat_acc)}")
    print(f"  Session-level mean acc:   {fmt(mean_sess_acc)}    [n=329]")
    print(f"  Session-level std:        {fmt(std_sess_acc)}")
    print(f"  Session-level macro-F1:   {fmt(mean_sess_f1)}")
    print()
    print(f"PER-CLASS F1 (mean across sessions)")
    for cls, t in per_class.items():
        print(f"  {cls:5s}:  {fmt(t)}")
    print()
    print(f"PAIRED ZERO-SHOT vs WITH-CAL  (per patient, n=48)")
    print(f"  Zero-shot mean acc:       {no_cal.mean():.4f}")
    print(f"  With-cal mean acc:        {with_cal.mean():.4f}")
    print(f"  Δacc (cal - no cal):      {fmt(delta_paired)}")
    print(f"  Wilcoxon signed-rank (H1: cal > zero-shot):")
    print(f"    statistic = {wilcoxon_res.statistic:.4f}")
    print(f"    p-value   = {wilcoxon_res.pvalue:.4g}")
    print(f"  Cliff's delta:            {cliffs:+.4f}")
    print(f"  Variance ratio no-cal/with-cal: {fmt(var_ratio)} (>1 means cal reduces variance)")
    print()
    print(f"PER-ARM PAIRED  (healthy contralateral vs impaired arm, per patient)")
    print(f"  Healthy arm mean: {h_arm.mean():.4f}  std={h_arm.std(ddof=1):.4f}")
    print(f"  Impaired arm mean: {i_arm.mean():.4f}  std={i_arm.std(ddof=1):.4f}")
    print(f"  Δ (healthy - impaired):   {fmt(arm_delta)}")
    print(f"  Wilcoxon (H1: healthy > impaired):")
    print(f"    statistic = {arm_wilcoxon.statistic:.4f}")
    print(f"    p-value   = {arm_wilcoxon.pvalue:.4g}")
    print()

    # ============================================================
    # Markdown summary
    # ============================================================
    md = [
        f"# PhysioMio per-session calibration — aggregate summary",
        "",
        f"**n = 48 patients · 329 sessions · seed {SEED} · bootstrap resamples = {n_bootstrap}**",
        f"Protocol: GrabMyo (1.14 M windows, w=1) + 30 s per-session stratified recal (w=100×), --fast HGB, per-gesture temporal split with 3-window buffer (no signal leakage).",
        "",
        "## Headline numbers",
        "",
        "| Metric | Value | 95% CI |",
        "|---|---|---|",
        f"| **Patient-level mean accuracy** (n=48) | **{mean_pat_acc[0]:.4f}** | [{mean_pat_acc[1]:.4f}, {mean_pat_acc[2]:.4f}] |",
        f"| **Session-level mean accuracy** (n=329) | **{mean_sess_acc[0]:.4f}** | [{mean_sess_acc[1]:.4f}, {mean_sess_acc[2]:.4f}] |",
        f"| Session-level macro-F1 (n=329) | {mean_sess_f1[0]:.4f} | [{mean_sess_f1[1]:.4f}, {mean_sess_f1[2]:.4f}] |",
        f"| Patient-level cross-subject std | {std_pat_acc[0]:.4f} | [{std_pat_acc[1]:.4f}, {std_pat_acc[2]:.4f}] |",
        f"| Session-level cross-session std | {std_sess_acc[0]:.4f} | [{std_sess_acc[1]:.4f}, {std_sess_acc[2]:.4f}] |",
        "",
        "## Per-class F1 (session-level means, n=329)",
        "",
        "| Class | Mean F1 | 95% CI |",
        "|---|---|---|",
    ]
    for cls, t in per_class.items():
        md.append(f"| {cls} | {t[0]:.4f} | [{t[1]:.4f}, {t[2]:.4f}] |")

    md += [
        "",
        "## Paired zero-shot vs calibration (per patient, n=48)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean acc zero-shot (no cal) | {no_cal.mean():.4f} |",
        f"| Mean acc with cal | {with_cal.mean():.4f} |",
        f"| **Δaccuracy (cal − no cal), bootstrap CI** | **+{delta_paired[0]:.4f} [{delta_paired[1]:+.4f}, {delta_paired[2]:+.4f}]** |",
        f"| Paired Wilcoxon signed-rank, H₁: cal > no-cal | W = {wilcoxon_res.statistic:.4f}, **p = {wilcoxon_res.pvalue:.4g}** |",
        f"| Cliff's δ | **{cliffs:+.4f}** |",
        f"| Variance ratio (no-cal std / with-cal std) | {var_ratio[0]:.2f}× [{var_ratio[1]:.2f}×, {var_ratio[2]:.2f}×] |",
        "",
        "## Per-arm paired comparison (healthy contralateral vs paretic, per patient)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Healthy-arm mean acc | {h_arm.mean():.4f}  (std {h_arm.std(ddof=1):.4f}) |",
        f"| Impaired-arm mean acc | {i_arm.mean():.4f}  (std {i_arm.std(ddof=1):.4f}) |",
        f"| Δ (healthy − impaired) | +{arm_delta[0]:.4f} [{arm_delta[1]:+.4f}, {arm_delta[2]:+.4f}] |",
        f"| Wilcoxon, H₁: healthy > impaired | W = {arm_wilcoxon.statistic:.4f}, **p = {arm_wilcoxon.pvalue:.4g}** |",
        "",
        "## Cross-comparison with GrabMyo headline",
        "",
        "| Protocol | Mean acc | Cross-subject std |",
        "|---|---|---|",
        f"| GrabMyo within-population LOSO (healthy, variant e) | 0.9732 | 0.0207 |",
        f"| **PhysioMio per-session cal (stroke, this work)** | **{mean_pat_acc[0]:.4f}** | **{std_pat_acc[0]:.4f}** |",
        f"| PhysioMio zero-shot (no cal, this work) | {no_cal.mean():.4f} | {np.std(no_cal, ddof=1):.4f} |",
        "",
        f"Calibration recovers **+{delta_paired[0]:.2%}** on stroke patients from a zero-shot baseline of {no_cal.mean():.2%}.",
        f"Residual gap to GrabMyo within-population: ~{0.9732 - mean_pat_acc[0]:.2%}.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")

    # ============================================================
    # JSON for downstream
    # ============================================================
    summary = {
        "n_patients": int(len(per_pat_cal)),
        "n_sessions": int(len(per_sess)),
        "seed": SEED,
        "n_bootstrap": n_bootstrap,
        "headline": {
            "patient_mean_acc": {"value": mean_pat_acc[0], "ci_lo": mean_pat_acc[1], "ci_hi": mean_pat_acc[2]},
            "session_mean_acc": {"value": mean_sess_acc[0], "ci_lo": mean_sess_acc[1], "ci_hi": mean_sess_acc[2]},
            "session_mean_f1": {"value": mean_sess_f1[0], "ci_lo": mean_sess_f1[1], "ci_hi": mean_sess_f1[2]},
            "patient_std": {"value": std_pat_acc[0], "ci_lo": std_pat_acc[1], "ci_hi": std_pat_acc[2]},
            "session_std": {"value": std_sess_acc[0], "ci_lo": std_sess_acc[1], "ci_hi": std_sess_acc[2]},
        },
        "per_class_f1": {cls: {"value": t[0], "ci_lo": t[1], "ci_hi": t[2]} for cls, t in per_class.items()},
        "paired_zero_vs_cal": {
            "zero_shot_mean": float(no_cal.mean()),
            "with_cal_mean": float(with_cal.mean()),
            "delta": {"value": delta_paired[0], "ci_lo": delta_paired[1], "ci_hi": delta_paired[2]},
            "wilcoxon_statistic": float(wilcoxon_res.statistic),
            "wilcoxon_p_value": float(wilcoxon_res.pvalue),
            "wilcoxon_alternative": "greater (cal > zero-shot)",
            "cliffs_delta": cliffs,
            "variance_ratio_no_over_with": {"value": var_ratio[0], "ci_lo": var_ratio[1], "ci_hi": var_ratio[2]},
        },
        "per_arm_paired": {
            "healthy_mean": float(h_arm.mean()),
            "impaired_mean": float(i_arm.mean()),
            "healthy_std": float(h_arm.std(ddof=1)),
            "impaired_std": float(i_arm.std(ddof=1)),
            "delta_healthy_minus_impaired": {"value": arm_delta[0], "ci_lo": arm_delta[1], "ci_hi": arm_delta[2]},
            "wilcoxon_statistic": float(arm_wilcoxon.statistic),
            "wilcoxon_p_value": float(arm_wilcoxon.pvalue),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
