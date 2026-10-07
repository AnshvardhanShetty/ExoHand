"""
Revision recompute #2 — refit per-participant z-score on CAL WINDOWS ONLY.

The submitted paper's engineer_features runs per-participant z-score using
ALL windows for each patient, including test windows. That is a form of
test-set peeking. Reviewer #2 flagged it.

This script:
  1. Replicates engineer_features but with per-patient (and per-session)
     z-score stats fit on cal windows only, applied to cal + test.
  2. Reruns zero-shot, calibration-only, and GrabMyo+cal on the same
     balanced 39/39/39 test set as recompute #1.
  3. Reports how much the numbers move.

Only PhysioMio (n=48) — the largest cohort where the leakage-vs-no-leakage
delta is most measurable.

Outputs:
  analysis/revision/results/leakage_free_zscore_per_session.csv
  analysis/revision/results/leakage_free_zscore_summary.md
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

from analysis.seed import SEED, seed_everything
from ml.train_hgb_v2 import (
    engineer_features,        # used for the LEAKY baseline (compares against)
    add_temporal_features,
    add_cross_channel_features,
    add_temporal_on_interactions,
    add_rank_features,
    add_within_trial_position,
    META_COLS,
)
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, GRABMYO_META, GRABMYO_CACHE, TEST_PER_CLASS,
    split_session, CLASSES, CAL_WEIGHT,
)

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "leakage_free_zscore_per_session.csv"
OUT_MD = OUT_DIR / "leakage_free_zscore_summary.md"


def engineer_features_leakage_free(df, cal_mask, feature_cols_after_expansion=None):
    """Like engineer_features but fits per-participant and per-participant-session
    z-score stats from cal_mask==True rows only, then applies to all rows.

    Non-normalising steps (temporal lags, cross-channel, rank percentiles,
    within-trial position) are applied globally — they don't leak test info
    because they operate on per-row features within the same session.
    """
    # 1. Base features already present. Apply the non-normalising expansions first.
    df = add_temporal_features(df)
    df = add_cross_channel_features(df)
    df = add_temporal_on_interactions(df)
    df = add_rank_features(df)
    df = add_within_trial_position(df)

    # 2. Now select numeric feature columns for z-scoring.
    feature_cols = [c for c in df.columns if c not in META_COLS
                    and not c.endswith("_sess_norm")]

    # 3. Per-participant z-score: fit μ/σ on cal_mask==True per patient.
    cal_df = df[cal_mask]
    for patient in df["participant"].unique():
        p_all = df["participant"] == patient
        p_cal = p_all & cal_mask
        if not p_cal.any():
            continue
        stats = cal_df[cal_df["participant"] == patient][feature_cols].agg(["mean", "std"])
        mu = stats.loc["mean"]
        sigma = stats.loc["std"].replace(0, 1e-8) + 1e-8
        df.loc[p_all, feature_cols] = (df.loc[p_all, feature_cols] - mu) / sigma

    # 4. Per-participant-session z-score for _sess_norm columns.
    for col in feature_cols[:45]:  # match train_hgb_v2's convention
        norm_col = f"{col}_sess_norm"
        vals = np.zeros(len(df), dtype=np.float32)
        for (patient, sess), grp in df.groupby(["participant", "session"], sort=False):
            grp_idx = grp.index
            cal_grp = grp_idx[cal_mask.reindex(grp_idx, fill_value=False).values]
            if len(cal_grp) < 2:
                vals[grp_idx] = 0.0
                continue
            cal_vals = df.loc[cal_grp, col].values
            mu = float(np.mean(cal_vals))
            sigma = float(np.std(cal_vals)) + 1e-8
            vals[grp_idx] = (df.loc[grp_idx, col].values - mu) / sigma
        df[norm_col] = vals

    return df


def make_clf(seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)

    print("Loading data…")
    physiomio = pd.read_pickle(PHYSIOMIO_PKL)
    print(f"  PhysioMio raw shape: {physiomio.shape}")

    # First pass: figure out cal/test split per session so we can build the mask.
    print("Computing per-session cal/test splits…")
    split_map = {}     # (participant, session) -> (cal_idx, test_idx)
    idx_col = "row_idx_orig"
    physiomio = physiomio.reset_index(drop=True)
    physiomio[idx_col] = physiomio.index
    for (patient, sess), s_data in physiomio.groupby(["participant", "session"]):
        if not sess.startswith("impaired_"):
            continue
        test_idx, cal_idx, _ = split_session(s_data, TEST_PER_CLASS, rng)
        split_map[(patient, sess)] = (set(cal_idx), set(test_idx))

    cal_mask = pd.Series(False, index=physiomio.index)
    test_mask = pd.Series(False, index=physiomio.index)
    for (patient, sess), (cal_idx, test_idx) in split_map.items():
        cal_mask.iloc[list(cal_idx)] = True
        test_mask.iloc[list(test_idx)] = True
    print(f"  cal windows: {cal_mask.sum():,}  test windows: {test_mask.sum():,}")

    print("Engineering features (LEAKAGE-FREE per-participant z-score)…")
    t0 = time.time()
    eng = engineer_features_leakage_free(physiomio.copy(), cal_mask)
    print(f"  done in {time.time()-t0:.1f}s  new shape: {eng.shape}")

    feature_cols = [c for c in eng.columns if c not in META_COLS]
    print(f"  {len(feature_cols)} feature columns")

    # Load GrabMyo base features (already engineered — using its own per-participant
    # z-score on GrabMyo participants, which is not a leakage concern for us since
    # GrabMyo is training data, not test).
    print("Loading GrabMyo cache…")
    grabmyo_eng = pd.read_pickle(GRABMYO_CACHE)
    with open(GRABMYO_META) as f:
        gm_meta = json.load(f)
    gm_feature_cols = gm_meta["feature_cols"]
    # Match columns between the two frames.
    common_cols = [c for c in gm_feature_cols if c in feature_cols]
    print(f"  {len(common_cols)} shared feature columns between GrabMyo and PhysioMio")
    grabmyo_X = grabmyo_eng[common_cols].values.astype(np.float32)
    grabmyo_y = grabmyo_eng["intent_idx"].values.astype(np.int64)

    # Per-session eval: three arms.
    print("Per-session eval: zero-shot, calibration-only, GrabMyo+cal…")
    rows = []
    for (patient, sess), (cal_idx, test_idx) in split_map.items():
        s_data = eng[(eng["participant"] == patient) & (eng["session"] == sess)]
        cal_rows = s_data[s_data.index.isin(cal_idx)]
        test_rows = s_data[s_data.index.isin(test_idx)]
        if len(test_rows) < 15:
            continue
        X_cal = cal_rows[common_cols].values.astype(np.float32)
        y_cal = cal_rows["intent_idx"].values.astype(np.int64)
        X_test = test_rows[common_cols].values.astype(np.float32)
        y_test = test_rows["intent_idx"].values.astype(np.int64)

        result = {"participant": patient, "session": sess,
                  "n_cal": len(X_cal), "n_test": len(X_test)}

        # --- Zero-shot: fit only on GrabMyo (needs to load pretrained model).
        # Easiest: retrain HGB on GrabMyo only, apply to test.
        # (Note: this is not the same as the shipped GrabMyo model since features
        # are z-scored with the leakage-free stats; that's the point.)
        # For efficiency: fit GrabMyo-only model ONCE outside the loop.
        # We'll do that below.

        # --- Calibration-only: fit on cal only, predict on test.
        if len(np.unique(y_cal)) >= 2:
            sc_po = StandardScaler().fit(X_cal)
            clf_po = make_clf().fit(sc_po.transform(X_cal), y_cal)
            preds_po = clf_po.predict(sc_po.transform(X_test))
            result["po_acc"] = float(accuracy_score(y_test, preds_po))
        else:
            result["po_acc"] = np.nan

        # --- GrabMyo + cal
        X_all = np.vstack([grabmyo_X, X_cal])
        y_all = np.concatenate([grabmyo_y, y_cal])
        w_all = np.ones(len(X_all), dtype=np.float32)
        w_all[len(grabmyo_X):] = CAL_WEIGHT
        sc_gm = StandardScaler().fit(X_all)
        clf_gm = make_clf().fit(sc_gm.transform(X_all), y_all, sample_weight=w_all)
        preds_gm = clf_gm.predict(sc_gm.transform(X_test))
        result["gm_cal_acc"] = float(accuracy_score(y_test, preds_gm))
        rows.append(result)
        print(f"  {patient}/{sess}: PO={result['po_acc']:.4f} GM+cal={result['gm_cal_acc']:.4f}", flush=True)

    # Now the zero-shot: fit GrabMyo-only classifier once, predict on all test sets.
    print("\nFitting GrabMyo-only classifier for zero-shot…")
    sc_zs = StandardScaler().fit(grabmyo_X)
    clf_zs = make_clf().fit(sc_zs.transform(grabmyo_X), grabmyo_y)
    for r in rows:
        s_data = eng[(eng["participant"] == r["participant"]) & (eng["session"] == r["session"])]
        cal_idx, test_idx = split_map[(r["participant"], r["session"])]
        test_rows = s_data[s_data.index.isin(test_idx)]
        X_test = test_rows[common_cols].values.astype(np.float32)
        y_test = test_rows["intent_idx"].values.astype(np.int64)
        preds_zs = clf_zs.predict(sc_zs.transform(X_test))
        r["zs_acc"] = float(accuracy_score(y_test, preds_zs))

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)

    # Patient-level aggregation and comparison with the LEAKY numbers.
    zs_pat = df_out.groupby("participant").zs_acc.mean()
    po_pat = df_out.groupby("participant").po_acc.mean()
    gm_pat = df_out.groupby("participant").gm_cal_acc.mean()

    print("\n" + "=" * 65)
    print("LEAKAGE-FREE Z-SCORE RESULTS (PhysioMio impaired-arm)")
    print("=" * 65)
    print(f"  zero-shot:         {zs_pat.mean():.4f}  (leaky: 0.3462)")
    print(f"  calibration-only:  {po_pat.mean():.4f}  (leaky: 0.8777)")
    print(f"  GrabMyo + cal:     {gm_pat.mean():.4f}  (leaky: 0.8603)")
    print(f"  Δ (GM+cal − PO):   {(gm_pat - po_pat).mean():+.4f}")

    md = [
        "# Leakage-free per-participant z-score — revision recompute #2",
        "",
        "Reviewer #2 flagged that `add_per_participant_normalisation` computes",
        "per-patient μ/σ across ALL windows (including test). This recompute",
        "refits those stats from cal windows only.",
        "",
        "## Result (PhysioMio impaired-arm, n=48, balanced 39/39/39 test)",
        "",
        "| arm | leaky (submitted) | leakage-free | Δ |",
        "|---|---:|---:|---:|",
        f"| zero-shot | 0.346 | **{zs_pat.mean():.3f}** | {zs_pat.mean() - 0.3462:+.3f} |",
        f"| calibration-only | 0.878 | **{po_pat.mean():.3f}** | {po_pat.mean() - 0.8777:+.3f} |",
        f"| GrabMyo + cal | 0.860 | **{gm_pat.mean():.3f}** | {gm_pat.mean() - 0.8603:+.3f} |",
        "",
        "## Interpretation",
        "",
        "If |Δ| < 0.02 everywhere: leakage was not material; the submitted",
        "numbers stand. If |Δ| > 0.02 somewhere: report leakage-free numbers.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
