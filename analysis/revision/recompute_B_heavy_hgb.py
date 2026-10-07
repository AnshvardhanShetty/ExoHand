"""
Revision recompute B — heavier HGB config to push absolute accuracy.

The paper uses a light HGB (max_iter=100, max_depth=10) so the sweep fits
finish quickly. The shipped adapted model uses (max_iter=2500, max_depth=18)
which is much more expressive. This script reruns per_session_eval on all
48 patients with the heavier config to see if absolute accuracy improves.

Realistic upside: +2-4 pp, taking 0.86 → 0.88-0.90.

Outputs:
  analysis/revision/results/heavy_hgb_per_session.csv
  analysis/revision/results/heavy_hgb_summary.md
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

from analysis.seed import SEED, seed_everything
from ml.train_hgb_v2 import engineer_features
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, GRABMYO_META, GRABMYO_CACHE, TEST_PER_CLASS,
    split_session, CLASSES, CAL_WEIGHT,
)

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "heavy_hgb_per_session.csv"
OUT_MD = OUT_DIR / "heavy_hgb_summary.md"


def make_heavy_hgb(seed=SEED):
    """Match the shipped adapted_hgb config (max_iter=2500, max_depth=18)."""
    return HistGradientBoostingClassifier(
        learning_rate=0.03,           # slower LR to make more iters count
        max_leaf_nodes=255,           # up from 63
        max_iter=2500,                # up from 100 (early stopping will cut it short)
        min_samples_leaf=20,
        l2_regularization=0.01,
        max_depth=18,                 # up from 10
        random_state=seed,
        early_stopping=True,          # allow it to stop early
        validation_fraction=0.1,
        n_iter_no_change=50,
        class_weight="balanced",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)

    print("Loading + engineering PhysioMio...")
    df = pd.read_pickle(PHYSIOMIO_PKL)
    eng = engineer_features(df)

    with open(GRABMYO_META) as f:
        gm_meta = json.load(f)
    gm_features = gm_meta["feature_cols"]

    print("Loading GrabMyo cache (300k subsample — full 1.14M was too slow with heavy HGB)...")
    gm = pd.read_pickle(GRABMYO_CACHE).sample(300_000, random_state=SEED)
    gm_X = gm[gm_features].values.astype(np.float32)
    gm_y = gm["intent_idx"].values.astype(np.int64)
    print(f"  GrabMyo: {len(gm_X):,} × {len(gm_features)}")

    print("\nPer-session eval with HEAVY HGB (max_iter=2500, max_depth=18)...")

    # ── Resume support: skip (participant, session) already in the CSV ──
    rows = []
    done = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done = set(zip(existing["participant"], existing["session"]))
        print(f"Resume: {len(rows)} existing rows, {len(done)} sessions already done")

    t_start = time.time()
    for pi, ((patient, sess), s_data) in enumerate(eng.groupby(["participant", "session"]), 1):
        if not sess.startswith("impaired_"):
            continue
        if (patient, sess) in done:
            continue
        try:
            test_idx, cal_idx, _ = split_session(s_data, TEST_PER_CLASS, rng)
        except Exception:
            continue
        if len(test_idx) < 15 or len(cal_idx) < 6:
            continue
        X_cal = s_data.loc[cal_idx, gm_features].fillna(0).values.astype(np.float32)
        y_cal = s_data.loc[cal_idx, "intent_idx"].values.astype(np.int64)
        X_test = s_data.loc[test_idx, gm_features].fillna(0).values.astype(np.float32)
        y_test = s_data.loc[test_idx, "intent_idx"].values.astype(np.int64)

        X_all = np.vstack([gm_X, X_cal])
        y_all = np.concatenate([gm_y, y_cal])
        w = np.ones(len(X_all), dtype=np.float32)
        w[len(gm_X):] = CAL_WEIGHT

        sc = StandardScaler().fit(X_all)
        clf = make_heavy_hgb()
        t_fit = time.time()
        clf.fit(sc.transform(X_all), y_all, sample_weight=w)
        fit_time = time.time() - t_fit
        preds = clf.predict(sc.transform(X_test))
        acc = float(accuracy_score(y_test, preds))
        f1m = float(f1_score(y_test, preds, average="macro", zero_division=0))
        cls_f1 = f1_score(y_test, preds, average=None, labels=CLASSES, zero_division=0)

        rows.append({
            "participant": patient, "session": sess,
            "n_cal": len(X_cal), "n_test": len(X_test),
            "acc": acc, "f1_macro": f1m,
            "f1_rest": float(cls_f1[0]), "f1_close": float(cls_f1[1]), "f1_open": float(cls_f1[2]),
            "n_iter_used": int(clf.n_iter_),
            "fit_time_s": fit_time,
        })
        elapsed = time.time() - t_start
        eta = elapsed / len(rows) * 238 - elapsed if len(rows) > 0 else 0
        print(f"  [{len(rows)}/238] {patient}/{sess}: acc={acc:.4f} f1={f1m:.4f} "
              f"n_iter={clf.n_iter_}  fit={fit_time:.0f}s  [{elapsed/60:.1f}min, eta {eta/60:.0f}min]",
              flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    pat_mean = out.groupby("participant").acc.mean()
    print(f"\nRESULT: heavy HGB patient-mean acc = {pat_mean.mean():.4f}")
    print(f"        (light HGB baseline: 0.860; upside vs baseline: {pat_mean.mean() - 0.860:+.3f})")

    # Compare paired to the shipped light-HGB numbers.
    light = pd.read_csv(PROJECT_ROOT / "analysis/physiomio/results/per_session_results.csv")
    if "status" in light.columns:
        light = light[light.status == "ok"]
    light = light[light.arm == "impaired"]
    light_pat = light.groupby("participant").acc.mean()

    common = pat_mean.index.intersection(light_pat.index)
    delta = pat_mean.loc[common] - light_pat.loc[common]
    n_improve = int((delta > 0).sum())
    w = wilcoxon(pat_mean.loc[common], light_pat.loc[common], alternative="greater")

    md = [
        "# Experiment B — heavy HGB config for absolute accuracy",
        "",
        f"Config: max_iter=2500 (from 100), max_depth=18 (from 10),",
        f"max_leaf_nodes=255 (from 63), learning_rate=0.03 (from 0.1),",
        f"early_stopping enabled.",
        "",
        "## Results",
        "",
        f"- **Heavy-HGB patient-mean:** {pat_mean.mean():.4f}  (n={len(pat_mean)})",
        f"- **Light-HGB (paper baseline):** {light_pat.mean():.4f}",
        f"- **Δ:** {pat_mean.mean() - light_pat.mean():+.4f}",
        f"- Patients improved: {n_improve}/{len(common)}",
        f"- Paired Wilcoxon p (heavy > light): {w.pvalue:.2e}",
        "",
        "## Reading the table",
        "",
        "If Δ > +0.02 and p < 0.05, the heavy config is worth adopting for the",
        "paper — the abstract number moves from 0.86 to ~0.88-0.90, widening",
        "the gap over LDA-Hudgins (0.83).",
        "",
        "If Δ < +0.02, the light and heavy configs are indistinguishable — we're",
        "at the ceiling for this task/data combination, and pushing absolute",
        "accuracy further needs a different lever (better cal protocol, more",
        "features, or accepting the ceiling).",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
