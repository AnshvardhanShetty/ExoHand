"""
Compare calibration-protocol variants on the same held-out folds.

Loads multiple per-variant CSVs produced by loso_eval.py, restricts to the
participants present in all variants, prints a per-fold comparison table and
aggregate stats.

Usage:
    python analysis/reproduce_headline/compare_variants.py \
        --baseline loso_results_baseline.csv \
        --variants variant_a:loso_results_variant_a.csv variant_b:loso_results_variant_b.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_variants(baseline_path: Path, variant_specs: list) -> tuple:
    """variant_specs: list of 'name:path' strings. Returns dict {name: df}."""
    variants = {}
    if baseline_path.exists():
        variants["baseline"] = pd.read_csv(baseline_path)
    for spec in variant_specs:
        if ":" not in spec:
            print(f"Skipping malformed variant spec: {spec}", file=sys.stderr)
            continue
        name, path = spec.split(":", 1)
        p = Path(path)
        if not p.exists():
            print(f"Variant {name} not found: {p}", file=sys.stderr)
            continue
        variants[name] = pd.read_csv(p)
    return variants


def common_folds(variants: dict) -> list:
    if not variants:
        return []
    sets = [set(df["participant"].tolist()) for df in variants.values()]
    common = set.intersection(*sets) if sets else set()
    return sorted(common, key=lambda s: int("".join(c for c in s if c.isdigit())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="analysis/reproduce_headline/loso_results_baseline.csv")
    parser.add_argument("--variants", nargs="*", default=[],
                        help="Pairs of name:path (e.g. variant_a:loso_results_variant_a.csv)")
    args = parser.parse_args()

    variants = load_variants(Path(args.baseline), args.variants)
    if not variants:
        print("No variants loaded.")
        return

    folds = common_folds(variants)
    print(f"Variants: {list(variants.keys())}")
    print(f"Common folds: {folds}\n")

    if not folds:
        print("No folds present in all variants.")
        return

    # Per-fold table
    rows = []
    for fold in folds:
        row = {"participant": fold}
        for vname, df in variants.items():
            r = df[df["participant"] == fold].iloc[0]
            row[f"{vname}_no_cal"] = r["acc_no_cal"]
            row[f"{vname}_with_cal"] = r["acc_with_cal"]
            row[f"{vname}_delta"] = r["delta_acc"]
        rows.append(row)
    table = pd.DataFrame(rows)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("=== PER-FOLD COMPARISON ===")
    print(table.to_string(index=False))
    print()

    # Aggregate
    print("=== AGGREGATE (mean ± std across common folds) ===")
    agg = []
    for vname, df in variants.items():
        sub = df[df["participant"].isin(folds)]
        agg.append({
            "variant": vname,
            "n_folds": len(sub),
            "acc_no_cal_mean": sub["acc_no_cal"].mean(),
            "acc_no_cal_std": sub["acc_no_cal"].std(),
            "acc_with_cal_mean": sub["acc_with_cal"].mean(),
            "acc_with_cal_std": sub["acc_with_cal"].std(),
            "delta_mean": sub["delta_acc"].mean(),
            "delta_std": sub["delta_acc"].std(),
        })
    print(pd.DataFrame(agg).to_string(index=False))
    print()

    # Decision-tree shortcut
    print("=== HEADLINE COMPARISON ===")
    if "baseline" in variants:
        base_delta = variants["baseline"][variants["baseline"]["participant"].isin(folds)]["delta_acc"].mean()
        for vname, df in variants.items():
            if vname == "baseline":
                continue
            sub = df[df["participant"].isin(folds)]
            v_delta = sub["delta_acc"].mean()
            v_no_cal = sub["acc_no_cal"].mean()
            v_with_cal = sub["acc_with_cal"].mean()
            shift = v_delta - base_delta
            marker = "✓ CLOSES GAP" if v_delta >= 0.06 else ("→ marginal" if shift > 0.01 else "→ no help")
            print(f"  {vname:15s}  no_cal={v_no_cal:.4f}  with_cal={v_with_cal:.4f}  Δ={v_delta:+.4f}  (vs baseline Δ={base_delta:+.4f}, shift {shift:+.4f})  {marker}")


if __name__ == "__main__":
    main()
