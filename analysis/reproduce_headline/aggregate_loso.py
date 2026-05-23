"""
Bootstrap CIs and paired statistics for the full LOSO run.

Reads loso_results_full43.csv (or any per-subject results CSV) and produces:
  - Mean no-cal / with-cal accuracy with bootstrap 95% CI
  - Mean Δacc with bootstrap 95% CI
  - Cross-subject std for no-cal / with-cal with bootstrap CI
  - Paired Wilcoxon signed-rank test on no-cal vs with-cal accuracy
  - Cliff's delta effect size
  - Variance-reduction ratio (no-cal std / with-cal std) with bootstrap CI

Outputs:
  - Console table for quick inspection
  - aggregate_summary.md   — paper-ready Markdown summary
  - aggregate_summary.json — machine-readable for downstream scripts

Usage:
    python analysis/reproduce_headline/aggregate_loso.py                # default: loso_results_full43.csv
    python analysis/reproduce_headline/aggregate_loso.py --in <csv>
    python analysis/reproduce_headline/aggregate_loso.py --in <csv> --n-bootstrap 5000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_stat(values: np.ndarray, stat_fn, n_bootstrap: int, rng: np.random.RandomState) -> tuple:
    """Return (point estimate, ci_lo_95, ci_hi_95) for any 1D-array statistic."""
    point = stat_fn(values)
    n = len(values)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        samples[i] = stat_fn(values[idx])
    return point, np.percentile(samples, 2.5), np.percentile(samples, 97.5)


def bootstrap_paired_diff(a: np.ndarray, b: np.ndarray, n_bootstrap: int, rng: np.random.RandomState) -> tuple:
    """Bootstrap CI for paired mean(b - a)."""
    diff = b - a
    point = float(np.mean(diff))
    n = len(diff)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        samples[i] = np.mean(diff[idx])
    return point, np.percentile(samples, 2.5), np.percentile(samples, 97.5)


def bootstrap_ratio(a: np.ndarray, b: np.ndarray, n_bootstrap: int, rng: np.random.RandomState) -> tuple:
    """Bootstrap CI for std(a) / std(b). Paired by subject; resamples subject indices."""
    sa, sb = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    point = sa / max(sb, 1e-12)
    n = len(a)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        sai = np.std(a[idx], ddof=1)
        sbi = np.std(b[idx], ddof=1)
        samples[i] = sai / max(sbi, 1e-12)
    return point, np.percentile(samples, 2.5), np.percentile(samples, 97.5)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size for two paired distributions (b - a sense)."""
    n = len(a)
    n_ab = sum(bi > ai for ai, bi in zip(a, b))
    n_ba = sum(ai > bi for ai, bi in zip(a, b))
    return (n_ab - n_ba) / n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_csv",
                        default="analysis/reproduce_headline/loso_results_full43.csv")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-md", default="analysis/reproduce_headline/aggregate_summary.md")
    parser.add_argument("--out-json", default="analysis/reproduce_headline/aggregate_summary.json")
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    if not in_path.exists():
        print(f"Input CSV not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(in_path)
    df["pnum"] = df["participant"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("pnum").reset_index(drop=True)

    n = len(df)
    rng = np.random.RandomState(args.seed)

    no_cal = df["acc_no_cal"].values
    with_cal = df["acc_with_cal"].values
    delta = df["delta_acc"].values
    f1_no_cal = df["f1_no_cal"].values
    f1_with_cal = df["f1_with_cal"].values

    print(f"Loaded {n} folds from {in_path}")
    print(f"Bootstrap resamples: {args.n_bootstrap}")
    print(f"Seed: {args.seed}\n")

    # --- Bootstrap CIs ---
    mean_no_cal = bootstrap_stat(no_cal, np.mean, args.n_bootstrap, rng)
    mean_with_cal = bootstrap_stat(with_cal, np.mean, args.n_bootstrap, rng)
    mean_delta = bootstrap_paired_diff(no_cal, with_cal, args.n_bootstrap, rng)
    std_no_cal = bootstrap_stat(no_cal, lambda x: np.std(x, ddof=1), args.n_bootstrap, rng)
    std_with_cal = bootstrap_stat(with_cal, lambda x: np.std(x, ddof=1), args.n_bootstrap, rng)
    var_ratio = bootstrap_ratio(no_cal, with_cal, args.n_bootstrap, rng)
    mean_f1_no_cal = bootstrap_stat(f1_no_cal, np.mean, args.n_bootstrap, rng)
    mean_f1_with_cal = bootstrap_stat(f1_with_cal, np.mean, args.n_bootstrap, rng)

    # --- Paired statistics ---
    wilcoxon_res = stats.wilcoxon(with_cal, no_cal, alternative="greater")
    cliffs = cliffs_delta(no_cal, with_cal)

    # --- Console output ---
    def fmt(t):
        v, lo, hi = t
        return f"{v:.4f} [{lo:.4f}, {hi:.4f}]"

    print("=" * 70)
    print(f"AGGREGATE (n={n} participants, bootstrap 95% CI)")
    print("=" * 70)
    print(f"  Accuracy (no-cal):       {fmt(mean_no_cal)}")
    print(f"  Accuracy (with-cal):     {fmt(mean_with_cal)}")
    print(f"  Δaccuracy (paired):      {fmt(mean_delta)}")
    print(f"  Macro-F1 (no-cal):       {fmt(mean_f1_no_cal)}")
    print(f"  Macro-F1 (with-cal):     {fmt(mean_f1_with_cal)}")
    print()
    print(f"  Cross-subject std (no-cal):   {fmt(std_no_cal)}")
    print(f"  Cross-subject std (with-cal): {fmt(std_with_cal)}")
    print(f"  Variance ratio (no/with):     {fmt(var_ratio)}")
    print()
    print(f"  Paired Wilcoxon signed-rank (H1: with-cal > no-cal):")
    print(f"    statistic = {wilcoxon_res.statistic:.4f}")
    print(f"    p-value   = {wilcoxon_res.pvalue:.6g}")
    print(f"  Cliff's delta (effect size):  {cliffs:+.4f}")
    print()

    # --- Per-fold table (sorted by no-cal difficulty) ---
    print("=" * 70)
    print("PER-FOLD TABLE (sorted ascending by no-cal — hardest folds first)")
    print("=" * 70)
    df_sorted = df.sort_values("acc_no_cal").reset_index(drop=True)
    print(df_sorted[["participant", "acc_no_cal", "acc_with_cal", "delta_acc",
                     "f1_no_cal", "f1_with_cal"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Markdown summary for paper ---
    md_lines = [
        f"# LOSO aggregate (n={n}, variant_e protocol: 1200 windows + weight 100×)",
        "",
        f"**Source:** `{in_path.name}` · **Bootstrap resamples:** {args.n_bootstrap} · **Seed:** {args.seed}",
        "",
        "## Headline numbers",
        "",
        "| Metric | Mean | 95% CI |",
        "|---|---|---|",
        f"| Accuracy (no-cal) | {mean_no_cal[0]:.4f} | [{mean_no_cal[1]:.4f}, {mean_no_cal[2]:.4f}] |",
        f"| Accuracy (with-cal) | {mean_with_cal[0]:.4f} | [{mean_with_cal[1]:.4f}, {mean_with_cal[2]:.4f}] |",
        f"| **Δaccuracy (paired)** | **{mean_delta[0]:+.4f}** | **[{mean_delta[1]:+.4f}, {mean_delta[2]:+.4f}]** |",
        f"| Macro-F1 (no-cal) | {mean_f1_no_cal[0]:.4f} | [{mean_f1_no_cal[1]:.4f}, {mean_f1_no_cal[2]:.4f}] |",
        f"| Macro-F1 (with-cal) | {mean_f1_with_cal[0]:.4f} | [{mean_f1_with_cal[1]:.4f}, {mean_f1_with_cal[2]:.4f}] |",
        "",
        "## Cross-subject variance",
        "",
        "| Metric | Value | 95% CI |",
        "|---|---|---|",
        f"| std (no-cal) | {std_no_cal[0]:.4f} | [{std_no_cal[1]:.4f}, {std_no_cal[2]:.4f}] |",
        f"| std (with-cal) | {std_with_cal[0]:.4f} | [{std_with_cal[1]:.4f}, {std_with_cal[2]:.4f}] |",
        f"| Variance ratio (no/with) | {var_ratio[0]:.2f}× | [{var_ratio[1]:.2f}×, {var_ratio[2]:.2f}×] |",
        "",
        "## Statistical test",
        "",
        f"- Paired Wilcoxon signed-rank, H₁: with-cal > no-cal",
        f"- statistic = {wilcoxon_res.statistic:.4f}, **p = {wilcoxon_res.pvalue:.4g}**",
        f"- Cliff's delta = {cliffs:+.4f}",
        "",
        "## Per-fold (sorted by difficulty)",
        "",
        "| participant | no-cal | with-cal | Δ | F1 no-cal | F1 with-cal |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in df_sorted.iterrows():
        md_lines.append(
            f"| {r['participant']} | {r['acc_no_cal']:.4f} | {r['acc_with_cal']:.4f} | "
            f"{r['delta_acc']:+.4f} | {r['f1_no_cal']:.4f} | {r['f1_with_cal']:.4f} |"
        )

    Path(args.out_md).write_text("\n".join(md_lines))
    print(f"\nWrote {args.out_md}")

    # --- JSON for downstream ---
    summary = {
        "input_csv": str(in_path),
        "n_folds": n,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "metrics": {
            "acc_no_cal": {"mean": mean_no_cal[0], "ci_lo": mean_no_cal[1], "ci_hi": mean_no_cal[2]},
            "acc_with_cal": {"mean": mean_with_cal[0], "ci_lo": mean_with_cal[1], "ci_hi": mean_with_cal[2]},
            "delta_acc": {"mean": mean_delta[0], "ci_lo": mean_delta[1], "ci_hi": mean_delta[2]},
            "f1_no_cal": {"mean": mean_f1_no_cal[0], "ci_lo": mean_f1_no_cal[1], "ci_hi": mean_f1_no_cal[2]},
            "f1_with_cal": {"mean": mean_f1_with_cal[0], "ci_lo": mean_f1_with_cal[1], "ci_hi": mean_f1_with_cal[2]},
            "std_no_cal": {"value": std_no_cal[0], "ci_lo": std_no_cal[1], "ci_hi": std_no_cal[2]},
            "std_with_cal": {"value": std_with_cal[0], "ci_lo": std_with_cal[1], "ci_hi": std_with_cal[2]},
            "variance_ratio": {"value": var_ratio[0], "ci_lo": var_ratio[1], "ci_hi": var_ratio[2]},
        },
        "wilcoxon": {"statistic": float(wilcoxon_res.statistic), "p_value": float(wilcoxon_res.pvalue),
                     "alternative": "greater"},
        "cliffs_delta": cliffs,
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
