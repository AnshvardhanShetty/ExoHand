"""
Revision — M1: within-patient vs across-patient Wasserstein distance.

The geometric explanation for the cross-arm result:
  is W1(own-healthy, own-impaired) > W1(own-impaired, other patients' impaired)?

For each patient:
  d_within = mean_features W1(patient's healthy_01 features, patient's impaired_01 features)
  d_across = mean_features W1(patient's impaired_01 features, pooled other patients' impaired_01)

Paired Wilcoxon + Cliff's δ on d_within vs d_across.

If within > across → within-patient cross-limb distance exceeds across-patient within-pathology
distance. Geometric confirmation of the cross-arm result.

Outputs:
  analysis/revision/results/M1_within_vs_across_wasserstein_per_patient.csv
  analysis/revision/results/M1_within_vs_across_wasserstein_summary.md
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
from ml.train_hgb_v2 import engineer_features
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, GRABMYO_META, TEST_PER_CLASS, split_session,
)

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "M1_within_vs_across_wasserstein_per_patient.csv"
OUT_MD = OUT_DIR / "M1_within_vs_across_wasserstein_summary.md"


def cliffs_delta(x, y):
    """Cliff's δ between two paired arrays."""
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    gt = sum((x[i] > y[i]) for i in range(n))
    lt = sum((x[i] < y[i]) for i in range(n))
    return (gt - lt) / n


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    t0 = time.time()

    print("Loading + engineering PhysioMio...")
    df = pd.read_pickle(PHYSIOMIO_PKL)
    eng = engineer_features(df)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]
    print(f"  features: {len(feature_cols)}")

    # Extract cal feature matrices per patient (both arms)
    patients = sorted(eng["participant"].unique(), key=lambda s: int(s.replace("patient", "")))
    per_patient = {}
    for patient in patients:
        s_imp = eng[(eng.participant == patient) & (eng.session == "impaired_01")]
        s_hlth = eng[(eng.participant == patient) & (eng.session == "healthy_01")]
        if len(s_imp) == 0 or len(s_hlth) == 0:
            continue
        try:
            rng = np.random.RandomState(SEED)
            _, imp_cal_idx, _ = split_session(s_imp, TEST_PER_CLASS, rng)
            rng2 = np.random.RandomState(SEED + 1)
            _, hlth_cal_idx, _ = split_session(s_hlth, TEST_PER_CLASS, rng2)
        except Exception:
            continue
        if len(imp_cal_idx) < 6 or len(hlth_cal_idx) < 6:
            continue
        per_patient[patient] = {
            "imp": s_imp.loc[imp_cal_idx, feature_cols].fillna(0).values.astype(np.float32),
            "hlth": s_hlth.loc[hlth_cal_idx, feature_cols].fillna(0).values.astype(np.float32),
        }
    print(f"Kept {len(per_patient)} patients with both arms.")

    patient_list = sorted(per_patient.keys(), key=lambda s: int(s.replace("patient", "")))
    rows = []
    for i, patient in enumerate(patient_list, 1):
        own_imp = per_patient[patient]["imp"]
        own_hlth = per_patient[patient]["hlth"]

        # Pool other patients' impaired
        others_imp = np.vstack([per_patient[p]["imp"] for p in patient_list if p != patient])

        # For each feature, W1 within (own-healthy vs own-impaired) and W1 across (own-impaired vs others'-impaired)
        d_within_per_feat = []
        d_across_per_feat = []
        for fi in range(len(feature_cols)):
            try:
                d_w = wasserstein_distance(own_hlth[:, fi], own_imp[:, fi])
                d_a = wasserstein_distance(own_imp[:, fi], others_imp[:, fi])
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
        })
        elapsed = time.time() - t0
        print(f"[{i}/{len(patient_list)}] {patient}  "
              f"d_within={d_within:.4f}  d_across={d_across:.4f}  "
              f"diff={d_within - d_across:+.4f}  [{elapsed/60:.1f}min]",
              flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    valid = out.dropna(subset=["diff_within_minus_across"])
    w = wilcoxon(valid.d_within_own_hlth_vs_own_imp, valid.d_across_own_imp_vs_others_imp,
                  alternative="greater")
    d = cliffs_delta(valid.d_within_own_hlth_vs_own_imp.values,
                      valid.d_across_own_imp_vs_others_imp.values)
    n_within_greater = int((valid["diff_within_minus_across"] > 0).sum())

    md = [
        "# M1 — within-patient vs across-patient Wasserstein distance",
        "",
        f"n = {len(valid)} patients with both healthy_01 and impaired_01 sessions.",
        "",
        "For each patient:",
        "- **d_within**: mean W₁(own_healthy features, own_impaired features) over 370 features",
        "- **d_across**: mean W₁(own_impaired features, pooled_other_patients_impaired features)",
        "",
        "## Headline",
        "",
        "| Distance | Mean | Median |",
        "|---|---:|---:|",
        f"| d_within (own healthy ↔ own impaired) | {valid.d_within_own_hlth_vs_own_imp.mean():.4f} | {valid.d_within_own_hlth_vs_own_imp.median():.4f} |",
        f"| d_across (own impaired ↔ others' impaired) | {valid.d_across_own_imp_vs_others_imp.mean():.4f} | {valid.d_across_own_imp_vs_others_imp.median():.4f} |",
        "",
        f"**Paired Wilcoxon (d_within > d_across): p = {w.pvalue:.4e}**",
        f"**Cliff's δ: {d:+.3f}**",
        f"Patients where within > across: {n_within_greater} / {len(valid)}",
        "",
        "## Interpretation",
        "",
        "If d_within > d_across (p<0.05, δ>0.2), the healthy-vs-impaired distance within",
        "one person is larger than the impaired-vs-impaired distance across people. That's",
        "the geometric explanation for why cross-arm PO underperforms LOPO: pathology",
        "puts stroke EMG in a distinct region of feature space that healthy data doesn't",
        "sample, and this region is more shared across patients than between arms of one.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
