"""
Revision — C4: GrabMyo weight sweep.

Rules the possibility that our specific 100× cal weighting is why GrabMyo doesn't help.

For each patient, at cal_per_gesture=36 (the paper's operating point):
  - Train HGB on GrabMyo (300k) + own impaired cal, at weights {0, 1, 10, 100, 1000}×
  - Test on balanced 39/39/39 held-out impaired test
  - Compare each weight to cal-only baseline

Kill rule: if any weight beats cal-only by >1 pp with paired Wilcoxon p<0.05, the null
result claim ("GrabMyo doesn't help") dies. Paper becomes "GrabMyo helps only at weight X."

Resumable.

Outputs:
  analysis/revision/results/C4_grabmyo_weight_sweep_per_patient.csv
  analysis/revision/results/C4_grabmyo_weight_sweep_summary.md
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
from ml.train_hgb_v2 import engineer_features
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, GRABMYO_META, GRABMYO_CACHE, TEST_PER_CLASS, split_session,
)

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "C4_grabmyo_weight_sweep_per_patient.csv"
OUT_MD = OUT_DIR / "C4_grabmyo_weight_sweep_summary.md"

WEIGHTS = [0.0, 1.0, 10.0, 100.0, 1000.0]   # 0 = cal-only baseline


def make_hgb(seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def fit_score(X_gm, y_gm, X_cal, y_cal, X_test, y_test, weight):
    """Train HGB with GrabMyo at given cal weight ratio (weight×) relative to GrabMyo (1×)."""
    if weight == 0.0:
        # Cal-only, no GrabMyo
        if len(np.unique(y_cal)) < 2:
            return np.nan
        sc = StandardScaler().fit(X_cal)
        clf = make_hgb().fit(sc.transform(X_cal), y_cal)
    else:
        X_all = np.vstack([X_gm, X_cal])
        y_all = np.concatenate([y_gm, y_cal])
        w = np.ones(len(X_all), dtype=np.float32)
        w[len(X_gm):] = float(weight)
        sc = StandardScaler().fit(X_all)
        clf = make_hgb().fit(sc.transform(X_all), y_all, sample_weight=w)
    return float(accuracy_score(y_test, clf.predict(sc.transform(X_test))))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    t0 = time.time()

    print("Loading + engineering PhysioMio...")
    df = pd.read_pickle(PHYSIOMIO_PKL)
    eng = engineer_features(df)

    with open(GRABMYO_META) as f:
        gm_features = json.load(f)["feature_cols"]

    print("Loading GrabMyo (300k subsample)...")
    gm = pd.read_pickle(GRABMYO_CACHE).sample(300_000, random_state=SEED)
    gm_X = gm[gm_features].values.astype(np.float32)
    gm_y = gm["intent_idx"].values.astype(np.int64)

    # Resume: skip patients whose full weight ladder is already recorded
    rows = []
    done = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        counts = existing.groupby("patient").size()
        done = set(counts[counts >= len(WEIGHTS)].index)
        rows = existing[existing.patient.isin(done)].to_dict("records")
        print(f"Resume: {len(rows)} rows, {len(done)} patients fully done")

    patients = sorted(eng["participant"].unique(), key=lambda s: int(s.replace("patient", "")))
    for pi, patient in enumerate(patients, 1):
        if patient in done:
            continue
        s01 = eng[(eng.participant == patient) & (eng.session == "impaired_01")]
        if len(s01) == 0:
            continue
        try:
            test_idx, cal_idx, _ = split_session(s01, TEST_PER_CLASS, rng)
        except Exception:
            continue
        if len(test_idx) < 15 or len(cal_idx) < 6:
            continue

        X_cal = s01.loc[cal_idx, gm_features].fillna(0).values.astype(np.float32)
        y_cal = s01.loc[cal_idx, "intent_idx"].values.astype(np.int64)
        X_test = s01.loc[test_idx, gm_features].fillna(0).values.astype(np.float32)
        y_test = s01.loc[test_idx, "intent_idx"].values.astype(np.int64)

        for w in WEIGHTS:
            try:
                acc = fit_score(gm_X, gm_y, X_cal, y_cal, X_test, y_test, w)
            except Exception as e:
                print(f"  {patient} w={w}: {e}", flush=True)
                acc = np.nan
            rows.append({"patient": patient, "weight": w, "acc": acc,
                          "n_cal": len(X_cal), "n_test": len(X_test)})

        elapsed = time.time() - t0
        eta = elapsed / max(1, pi - len(done)) * (len(patients) - pi)
        accs_this_patient = [r["acc"] for r in rows if r["patient"] == patient]
        print(f"[{pi}/{len(patients)}] {patient}: "
              f"cal-only={accs_this_patient[0]:.4f}  100x={accs_this_patient[3]:.4f}  "
              f"[{elapsed/60:.1f}min, eta {eta/60:.0f}min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return

    # Aggregate per weight
    pivot = out.pivot(index="patient", columns="weight", values="acc")
    means = pivot.mean().to_dict()
    medians = pivot.median().to_dict()

    md = [
        "# C4 — GrabMyo weight sweep",
        "",
        f"n = {len(pivot)} patients, cal_per_gesture=36 (paper operating point).",
        "Weight = cal-weight multiplier in the joint training (1× = GrabMyo weight; 0 = cal-only, no GrabMyo).",
        "",
        "## Ladder",
        "",
        "| Weight (× GrabMyo) | Mean acc | Median acc | vs cal-only Δ | Wilcoxon p |",
        "|---:|---:|---:|---:|---:|",
    ]

    baseline = pivot[0.0].dropna()
    for w in WEIGHTS:
        col = pivot[w].dropna()
        if w == 0.0:
            md.append(f"| **0 (cal-only)** | {col.mean():.4f} | {col.median():.4f} | — | — |")
        else:
            paired = pivot[[0.0, w]].dropna()
            delta = paired[w].mean() - paired[0.0].mean()
            try:
                wtest = wilcoxon(paired[w], paired[0.0], alternative="greater")
                pval = f"{wtest.pvalue:.3e}"
            except Exception:
                pval = "—"
            md.append(f"| {w:g} | {col.mean():.4f} | {col.median():.4f} | "
                      f"{delta:+.4f} | {pval} |")

    md += [
        "",
        "## Interpretation (pre-registered decision rule)",
        "",
        "- If ANY weight ≠ 100× beats cal-only by > 1 pp with paired Wilcoxon p < 0.05,",
        "  the null result headline dies. Paper becomes 'GrabMyo helps only at weight X'.",
        "- If no weight beats cal-only meaningfully, the null result claim survives across",
        "  the full ladder — pretraining doesn't help regardless of weighting scheme.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
