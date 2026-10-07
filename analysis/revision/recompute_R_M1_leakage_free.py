"""
R-M1 — within-patient vs across-patient Wasserstein-1, leakage-free features.

For each patient:
  d_within = mean_features W₁(own healthy_01 cal, own impaired_01 cal)
  d_across = mean_features W₁(own impaired_01 cal, pooled other patients' impaired_01 cal)

Uses frozen splits + engineer_features_leakage_free (z-score μ/σ fit on cal rows only).
Reports leakage-free + legacy side-by-side. Wilcoxon + Cliff's δ preserved.

Outputs:
  analysis/revision/results/R_M1_leakage_free_per_patient.csv
  analysis/revision/results/R_M1_leakage_free_summary.md
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, wilcoxon

from analysis.seed import SEED, seed_everything
from analysis.physiomio.per_session_eval import PHYSIOMIO_PKL, GRABMYO_META
from analysis.revision.recompute_leakage_free_zscore import engineer_features_leakage_free

FROZEN = PROJECT_ROOT / "analysis" / "revision" / "frozen_splits.parquet"
LEGACY_CSV = PROJECT_ROOT / "analysis" / "revision" / "results" / "M1_within_vs_across_wasserstein_per_patient.csv"
OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "R_M1_leakage_free_per_patient.csv"
OUT_MD = OUT_DIR / "R_M1_leakage_free_summary.md"


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    gt = sum((x[i] > y[i]) for i in range(n))
    lt = sum((x[i] < y[i]) for i in range(n))
    return (gt - lt) / n


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    t0 = time.time()

    print("Loading raw PhysioMio...")
    df = pd.read_pickle(PHYSIOMIO_PKL).reset_index(drop=True)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]

    print("Loading frozen splits...")
    frozen = pd.read_parquet(FROZEN)
    cal_mask = pd.Series(False, index=df.index)
    per_patient = {}
    for _, r in frozen.iterrows():
        cal_mask.loc[r["cal_idx"]] = True
        per_patient.setdefault(r["patient"], {})[r["session"]] = list(r["cal_idx"])
    keep_patients = [p for p in per_patient if
                     "impaired_01" in per_patient[p] and "healthy_01" in per_patient[p]]
    print(f"  cal_mask True: {int(cal_mask.sum())} rows, {len(keep_patients)} patients with both arms")

    print("Engineering features (leakage-free)...")
    df_eng = engineer_features_leakage_free(df.copy(), cal_mask)

    print("Computing per-patient Wasserstein distances...")
    patient_list = sorted(keep_patients, key=lambda s: int(s.replace("patient", "")))

    # Pre-extract per-patient cal blocks
    blocks = {}
    for p in patient_list:
        blocks[p] = {
            "imp": df_eng.loc[per_patient[p]["impaired_01"], feature_cols].fillna(0).values.astype(np.float32),
            "hlth": df_eng.loc[per_patient[p]["healthy_01"], feature_cols].fillna(0).values.astype(np.float32),
        }

    rows = []
    for i, patient in enumerate(patient_list, 1):
        own_imp = blocks[patient]["imp"]
        own_hlth = blocks[patient]["hlth"]
        others_imp = np.vstack([blocks[p]["imp"] for p in patient_list if p != patient])
        d_within_per_feat = []
        d_across_per_feat = []
        for fi in range(len(feature_cols)):
            try:
                d_w = wasserstein_distance(own_hlth[:, fi], own_imp[:, fi])
                d_a = wasserstein_distance(own_imp[:, fi], others_imp[:, fi])
                if np.isfinite(d_w) and np.isfinite(d_a):
                    d_within_per_feat.append(d_w)
                    d_across_per_feat.append(d_a)
            except Exception:
                continue
        d_within = float(np.mean(d_within_per_feat))
        d_across = float(np.mean(d_across_per_feat))
        rows.append({
            "patient": patient,
            "n_features": len(d_within_per_feat),
            "d_within_own_hlth_vs_own_imp": d_within,
            "d_across_own_imp_vs_others_imp": d_across,
            "diff_within_minus_across": d_within - d_across,
            "leakage_free": True,
        })
        elapsed = time.time() - t0
        print(f"[{i}/{len(patient_list)}] {patient}  d_within={d_within:.4f}  d_across={d_across:.4f}  "
              f"diff={d_within-d_across:+.4f}  [{elapsed/60:.1f}min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    w = wilcoxon(out.d_within_own_hlth_vs_own_imp, out.d_across_own_imp_vs_others_imp,
                  alternative="greater")
    d = cliffs_delta(out.d_within_own_hlth_vs_own_imp.values,
                      out.d_across_own_imp_vs_others_imp.values)
    ratio = out.d_within_own_hlth_vs_own_imp.mean() / out.d_across_own_imp_vs_others_imp.mean()

    # Load legacy for side-by-side
    legacy_summary = "no legacy CSV found"
    if LEGACY_CSV.exists():
        leg = pd.read_csv(LEGACY_CSV)
        leg_ratio = leg.d_within_own_hlth_vs_own_imp.mean() / leg.d_across_own_imp_vs_others_imp.mean()
        legacy_summary = (f"legacy d_within = {leg.d_within_own_hlth_vs_own_imp.mean():.4f}, "
                          f"d_across = {leg.d_across_own_imp_vs_others_imp.mean():.4f}, "
                          f"ratio = {leg_ratio:.3f}×")

    md = [
        "# R-M1 — within-vs-across Wasserstein (leakage-free)",
        "",
        f"n = {len(out)} patients. Leakage-free features via engineer_features_leakage_free.",
        "",
        "## Side-by-side (leakage-free vs legacy)",
        "",
        "| Metric | Leakage-free | Legacy | Δ |",
        "|---|---:|---:|---:|",
        f"| d_within (own hlth ↔ own imp) | {out.d_within_own_hlth_vs_own_imp.mean():.4f} | 0.7363 | {out.d_within_own_hlth_vs_own_imp.mean() - 0.7363:+.4f} |",
        f"| d_across (own imp ↔ others imp) | {out.d_across_own_imp_vs_others_imp.mean():.4f} | 0.3321 | {out.d_across_own_imp_vs_others_imp.mean() - 0.3321:+.4f} |",
        f"| Ratio d_within / d_across | {ratio:.3f}× | 2.217× | {ratio - 2.217:+.3f}× |",
        "",
        f"**Paired Wilcoxon (d_within > d_across), leakage-free: p = {w.pvalue:.4e}**",
        f"**Cliff's δ: {d:+.3f}**",
        f"Patients where within > across: {(out['diff_within_minus_across'] > 0).sum()}/{len(out)}",
        "",
        f"Legacy: {legacy_summary}",
        "",
        "## Gate decision (pre-registered)",
        "",
        "- If ratio drops below ~1.5× OR loses significance → rewrite the mechanism section.",
        "- If ratio and significance survive → geometry claim holds under clean features.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(f"Ratio: {ratio:.3f}×  Wilcoxon p: {w.pvalue:.3e}  Cliff's δ: {d:+.3f}")


if __name__ == "__main__":
    main()
