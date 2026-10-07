"""
Merge multi-draw chronic results into the leakage-free ladder CSV.

Adds two new columns (non-destructive — original single-draw values stay):
  row3_vm_lopo_multidraw       : chronic multi-draw mean for 47 imp donors → target imp
  row_exp1_hlth_multidraw      : chronic multi-draw mean for 47 hlth donors → target imp

Both are NaN for acute patients (multi-draw only covered the 25 chronic).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

RESULTS = PROJECT_ROOT / "analysis" / "revision" / "results"
LADDER = RESULTS / "leakage_free_ladder_per_patient.csv"
MULTI = RESULTS / "chronic_multidraw_per_patient.csv"

def main():
    ladder = pd.read_csv(LADDER)
    multi = pd.read_csv(MULTI)

    m = multi[["target", "imp_mean", "hlth_mean"]].rename(columns={
        "target": "patient",
        "imp_mean": "row3_vm_lopo_multidraw",
        "hlth_mean": "row_exp1_hlth_multidraw",
    })

    # Drop old multi-draw columns if a prior run left them
    for col in ("row3_vm_lopo_multidraw", "row_exp1_hlth_multidraw"):
        if col in ladder.columns:
            ladder = ladder.drop(columns=col)

    out = ladder.merge(m, on="patient", how="left")
    out.to_csv(LADDER, index=False)

    n_chronic = out.row3_vm_lopo_multidraw.notna().sum()
    print(f"Wrote {LADDER}")
    print(f"  {len(out)} patients total, {n_chronic} chronic with multi-draw values")
    print(f"  chronic mean row3_vm_lopo_multidraw:  {out.row3_vm_lopo_multidraw.mean():.4f}")
    print(f"  chronic mean row_exp1_hlth_multidraw: {out.row_exp1_hlth_multidraw.mean():.4f}")
    print(f"  chronic mean gap:                     {(out.row3_vm_lopo_multidraw - out.row_exp1_hlth_multidraw).mean():+.4f}")

if __name__ == "__main__":
    main()
