"""
Montage-plumbing floor test — GrabMyo zero-shot on PhysioMio HEALTHY arms.

Answers the specific reviewer objection: "how much of the 0.360 zero-shot failure
on impaired arms is montage-alignment loss vs. actual healthy→impaired transfer failure?"

Setup mirrors the headline zero-shot test in recompute_leakage_free_ladder.py:
  - Same GrabMyo classifier (fit once on 200k GrabMyo windows)
  - Same StandardScaler on GrabMyo features
  - Same leakage-free feature engineering pipeline
  - Only difference: test on PhysioMio patients' HEALTHY-arm test sets (not impaired)

Interpretation:
  - Acc ~ 0.90 → montage plumbing works, 0.360 impaired failure is pathology-specific
  - Acc ~ 0.36 → montage is the bottleneck, transfer story reframes
  - Anything intermediate → partial plumbing loss; report honestly

Outputs:
  analysis/revision/results/montage_floor_per_patient.csv
  analysis/revision/results/montage_floor_summary.md
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

from analysis.seed import SEED, seed_everything
from analysis.physiomio.per_session_eval import PHYSIOMIO_PKL, GRABMYO_META, GRABMYO_CACHE
from analysis.revision.recompute_leakage_free_zscore import engineer_features_leakage_free

FROZEN = PROJECT_ROOT / "analysis" / "revision" / "frozen_splits.parquet"
LADDER_CSV = PROJECT_ROOT / "analysis" / "revision" / "results" / "leakage_free_ladder_per_patient.csv"
OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "montage_floor_per_patient.csv"
OUT_MD = OUT_DIR / "montage_floor_summary.md"

GM_SUBSAMPLE = 200_000


def make_hgb():
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=SEED,
        early_stopping=False, class_weight="balanced",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    t0 = time.time()

    print("Loading + engineering leakage-free...")
    df = pd.read_pickle(PHYSIOMIO_PKL).reset_index(drop=True)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]

    frozen = pd.read_parquet(FROZEN)
    cal_mask = pd.Series(False, index=df.index)
    per_patient = {}
    for _, r in frozen.iterrows():
        cal_mask.loc[r["cal_idx"]] = True
        per_patient.setdefault(r["patient"], {})[r["session"]] = {
            "cal_idx": list(r["cal_idx"]),
            "test_idx": list(r["test_idx"]),
        }
    df_eng = engineer_features_leakage_free(df.copy(), cal_mask)

    print(f"Loading GrabMyo cache ({GM_SUBSAMPLE//1000}k subsample)...")
    gm = pd.read_pickle(GRABMYO_CACHE).sample(GM_SUBSAMPLE, random_state=SEED)
    gm_X = gm[feature_cols].values.astype(np.float32)
    gm_y = gm["intent_idx"].values.astype(np.int64)

    print("Fitting GrabMyo-only classifier ONCE...")
    sc_gm = StandardScaler().fit(gm_X)
    clf_gm = make_hgb().fit(sc_gm.transform(gm_X), gm_y)

    # For each patient with a healthy-arm test set, run zero-shot GrabMyo → healthy
    patients_with_hlth = sorted([p for p in per_patient if "healthy_01" in per_patient[p]],
                                     key=lambda s: int(s.replace("patient", "")))
    print(f"Patients with healthy_01: {len(patients_with_hlth)}")

    rows = []
    for i, patient in enumerate(patients_with_hlth, 1):
        # Healthy sessions in frozen splits have no test_idx (healthy arms were only
        # ever used as donors). For zero-shot we can safely use the healthy CAL windows
        # as the test set — the GrabMyo classifier has never seen any PhysioMio data,
        # so there is no leak.
        hlth_idx = per_patient[patient]["healthy_01"]["cal_idx"]
        X_hlth_test = df_eng.loc[hlth_idx, feature_cols].fillna(0).values.astype(np.float32)
        y_hlth_test = df_eng.loc[hlth_idx, "intent_idx"].values.astype(np.int64)

        if len(np.unique(y_hlth_test)) < 2:
            acc = np.nan
        else:
            acc = float(accuracy_score(y_hlth_test, clf_gm.predict(sc_gm.transform(X_hlth_test))))

        rows.append({
            "patient": patient,
            "n_test": len(y_hlth_test),
            "zero_shot_healthy_acc": acc,
        })
        elapsed = time.time() - t0
        print(f"[{i}/{len(patients_with_hlth)}] {patient}  n_test={len(y_hlth_test)}  "
              f"gm→hlth={acc:.4f}  [{elapsed/60:.1f}min]", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    # Compare against headline: GrabMyo → impaired
    ladder = pd.read_csv(LADDER_CSV)[["patient", "row4_grabmyo_zero_shot"]]
    ladder = ladder.rename(columns={"row4_grabmyo_zero_shot": "zero_shot_impaired_acc"})
    m = out.merge(ladder, on="patient", how="inner")

    hlth_mean = m["zero_shot_healthy_acc"].mean()
    imp_mean = m["zero_shot_impaired_acc"].mean()

    # Paired Wilcoxon: does healthy target beat impaired target?
    both = m.dropna(subset=["zero_shot_healthy_acc", "zero_shot_impaired_acc"])
    w = wilcoxon(both["zero_shot_healthy_acc"], both["zero_shot_impaired_acc"],
                    alternative="greater")

    # Bootstrap CI on healthy-mean
    rng = np.random.RandomState(SEED)
    boots = []
    for _ in range(5000):
        boots.append(m["zero_shot_healthy_acc"].sample(len(m), replace=True, random_state=rng.randint(1e9)).mean())
    hlth_ci = np.percentile(boots, [2.5, 97.5])

    CHANCE = 1/3

    md = [
        "# Montage-plumbing floor test — GrabMyo zero-shot on healthy arms",
        "",
        f"n = {len(m)} patients with both healthy and impaired test sets.",
        "Same classifier, same alignment, same features as headline zero-shot;",
        "only test-target arm differs.",
        "",
        "## The comparison",
        "",
        "| Target arm | Zero-shot mean acc | Median | vs chance (0.333) |",
        "|---|---:|---:|---:|",
        f"| Healthy arm | **{hlth_mean:.4f}** | {m['zero_shot_healthy_acc'].median():.4f} | {hlth_mean/CHANCE:.2f}× |",
        f"| Impaired arm (headline) | {imp_mean:.4f} | {m['zero_shot_impaired_acc'].median():.4f} | {imp_mean/CHANCE:.2f}× |",
        f"| Δ (healthy − impaired) | **{hlth_mean-imp_mean:+.4f}** | | |",
        "",
        f"- Bootstrap 95% CI on healthy-target mean: [{hlth_ci[0]:.4f}, {hlth_ci[1]:.4f}]",
        f"- Paired Wilcoxon (H1: healthy > impaired): p = {w.pvalue:.4f}",
        "",
        "## Interpretation",
        "",
        f"- If healthy ~ 0.90: montage plumbing works, impaired 0.360 failure is pathology-specific",
        f"- If healthy ~ 0.36: montage is the bottleneck, transfer story reframes",
        f"- Observed healthy mean = {hlth_mean:.3f}",
        "",
    ]
    if hlth_mean > 0.7:
        md.append("**Verdict: montage plumbing works.** Zero-shot healthy-target accuracy is ")
        md.append("substantially above chance and above the impaired-target zero-shot, so the ")
        md.append("headline failure on impaired arms is not attributable to alignment loss alone. ")
        md.append("The transfer failure is stroke-specific.")
    elif hlth_mean < 0.42:
        md.append("**Verdict: montage plumbing is largely broken.** Zero-shot healthy-target ")
        md.append("accuracy is near chance, meaning the alignment pipeline itself loses most of the signal. ")
        md.append("The 'healthy→impaired' framing needs to be replaced with 'we cannot cross-transfer ")
        md.append("between EMG montages at any distance.'")
    else:
        md.append("**Verdict: partial plumbing loss.** Zero-shot healthy-target accuracy sits ")
        md.append("between chance and full transfer. Some of the impaired-target failure is ")
        md.append("attributable to montage mismatch, but a substantial pathology-specific component remains.")

    OUT_MD.write_text("\n".join(md))
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
