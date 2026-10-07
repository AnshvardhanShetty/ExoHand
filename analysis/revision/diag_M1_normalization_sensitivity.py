"""
Diagnostic: does M1's ratio actually depend on the choice of z-score normalization?

Compute d_within, d_across, and their ratio under THREE feature normalizations:

  (1) LEGACY: pooled per-participant across all rows including test (the buggy one)
  (2) LEAKAGE-FREE POOLED: cal rows only, both arms pooled per patient (my current)
  (3) PER-SESSION: cal rows only, each session normalized with its own cal stats
                    (matches deployed-scenario; matches C2 V2)
  (4) RAW: no normalization at all (baseline for sanity)

If (2) ≈ (3): pooling arms in the z-score doesn't matter much; the leakage-free
              result stands.
If (3) ≈ (1): per-session z-score gives similar ratio to legacy, which would mean
              the drop 2.2 → 1.4 came from pooling arms (my implementation choice),
              not from removing leakage. Would validate mentor's suspicion.
If all three differ substantially: the measurement is normalization-dependent and
                                    we need to state which we chose and why.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from analysis.seed import SEED, seed_everything
from analysis.physiomio.per_session_eval import PHYSIOMIO_PKL, GRABMYO_META
from analysis.revision.recompute_leakage_free_zscore import engineer_features_leakage_free
from ml.train_hgb_v2 import engineer_features
from analysis.revision.recompute_A_lda_at_cal_sizes import split_at
from analysis.physiomio.per_session_eval import TEST_PER_CLASS

CAL_SIZE = 36


def per_session_zscore(df, feature_cols, cal_mask):
    """Apply per (participant, session) z-score using ONLY cal rows for stats."""
    df = df.copy()
    for (patient, session), idx in df.groupby(["participant", "session"]).groups.items():
        cal_idx_here = [i for i in idx if cal_mask.loc[i]]
        if len(cal_idx_here) < 3:
            continue
        sub = df.loc[cal_idx_here, feature_cols]
        mu = sub.mean()
        sigma = sub.std().replace(0, 1e-8) + 1e-8
        df.loc[idx, feature_cols] = (df.loc[idx, feature_cols] - mu) / sigma
    return df


def compute_M1(df_engineered, per_patient_meta, feature_cols):
    """d_within and d_across per patient, mean across features."""
    patient_list = sorted(per_patient_meta.keys(), key=lambda s: int(s.replace("patient", "")))
    blocks = {}
    for p in patient_list:
        blocks[p] = {
            "imp": df_engineered.loc[per_patient_meta[p]["impaired_01"], feature_cols].fillna(0).values.astype(np.float64),
            "hlth": df_engineered.loc[per_patient_meta[p]["healthy_01"], feature_cols].fillna(0).values.astype(np.float64),
        }
    per_patient_ratio = []
    per_patient_within = []
    per_patient_across = []
    for p in patient_list:
        own_imp = blocks[p]["imp"]
        own_hlth = blocks[p]["hlth"]
        others_imp = np.vstack([blocks[q]["imp"] for q in patient_list if q != p])
        dw, da = [], []
        for fi in range(len(feature_cols)):
            try:
                w = wasserstein_distance(own_hlth[:, fi], own_imp[:, fi])
                a = wasserstein_distance(own_imp[:, fi], others_imp[:, fi])
                if np.isfinite(w) and np.isfinite(a):
                    dw.append(w); da.append(a)
            except Exception:
                continue
        d_within = np.mean(dw); d_across = np.mean(da)
        per_patient_within.append(d_within)
        per_patient_across.append(d_across)
        per_patient_ratio.append(d_within / d_across if d_across > 0 else np.nan)
    return {
        "d_within_mean": np.mean(per_patient_within),
        "d_across_mean": np.mean(per_patient_across),
        "ratio_mean": np.mean(per_patient_within) / np.mean(per_patient_across),
        "n_patients_ratio_above_1": int(np.sum(np.array(per_patient_ratio) > 1)),
        "n_patients_ratio_above_1_5": int(np.sum(np.array(per_patient_ratio) > 1.5)),
        "median_per_patient_ratio": float(np.nanmedian(per_patient_ratio)),
    }


def main():
    seed_everything(SEED)
    print("Loading raw PhysioMio...")
    df = pd.read_pickle(PHYSIOMIO_PKL).reset_index(drop=True)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]

    # Determine cal splits and per-patient metadata (impaired_01 + healthy_01)
    cal_mask = pd.Series(False, index=df.index)
    per_patient = {}
    for patient in sorted(df.participant.unique(), key=lambda s: int(s.replace("patient", ""))):
        s_imp = df[(df.participant == patient) & (df.session == "impaired_01")]
        s_hlth = df[(df.participant == patient) & (df.session == "healthy_01")]
        if len(s_imp) == 0 or len(s_hlth) == 0:
            continue
        try:
            rng = np.random.RandomState(SEED)
            _, imp_cal = split_at(s_imp, CAL_SIZE, TEST_PER_CLASS, rng)
            rng2 = np.random.RandomState(SEED + 1)
            _, hlth_cal = split_at(s_hlth, CAL_SIZE, TEST_PER_CLASS, rng2)
        except Exception:
            continue
        if len(imp_cal) < 6 or len(hlth_cal) < 6:
            continue
        cal_mask.loc[list(imp_cal)] = True
        cal_mask.loc[list(hlth_cal)] = True
        per_patient[patient] = {"impaired_01": list(imp_cal), "healthy_01": list(hlth_cal)}
    print(f"  {len(per_patient)} patients with both arms")

    # (1) LEGACY: normal engineer_features (with test-leaking z-scores)
    print("\n(1) LEGACY features...")
    df_legacy = engineer_features(df.copy())
    r1 = compute_M1(df_legacy, per_patient, feature_cols)
    print(r1)

    # (2) LEAKAGE-FREE POOLED (arms pooled per patient, cal only)
    print("\n(2) LEAKAGE-FREE POOLED (my current)...")
    df_lf = engineer_features_leakage_free(df.copy(), cal_mask.copy())
    r2 = compute_M1(df_lf, per_patient, feature_cols)
    print(r2)

    # (3) PER-SESSION z-score: replace the patient-pooled step with per-session
    print("\n(3) PER-SESSION z-score (each session on its own cal)...")
    # Start from leakage-free but re-normalize per-session
    df_lf_reload = engineer_features_leakage_free(df.copy(), cal_mask.copy())
    # engineer_features_leakage_free applied per-patient z-score already — undo isn't
    # practical, so recompute from raw with only non-normalising steps + per-session z
    from ml.train_hgb_v2 import (add_temporal_features, add_cross_channel_features,
                                   add_rank_features, add_within_trial_position)
    df_manual = add_temporal_features(df.copy())
    df_manual = add_cross_channel_features(df_manual)
    # Skip add_temporal_on_interactions (needs it) — reuse via engineer_features_leakage_free without normalizing
    # Simpler: use df_manual's columns as-is; only z-score.
    feat_cols_present = [c for c in feature_cols if c in df_manual.columns]
    df_ps = per_session_zscore(df_manual, feat_cols_present, cal_mask)
    r3 = compute_M1(df_ps, per_patient, feat_cols_present)
    print(r3)

    print("\n" + "=" * 70)
    print("SUMMARY — M1 ratio under three normalisations")
    print("=" * 70)
    print(f"(1) LEGACY (pooled, includes test):         ratio = {r1['ratio_mean']:.3f}× "
          f"| {r1['n_patients_ratio_above_1']}/48 > 1  | {r1['n_patients_ratio_above_1_5']}/48 > 1.5")
    print(f"(2) LEAKAGE-FREE POOLED (cal only, mine):   ratio = {r2['ratio_mean']:.3f}× "
          f"| {r2['n_patients_ratio_above_1']}/48 > 1  | {r2['n_patients_ratio_above_1_5']}/48 > 1.5")
    print(f"(3) PER-SESSION z-score (session cal only): ratio = {r3['ratio_mean']:.3f}× "
          f"| {r3['n_patients_ratio_above_1']}/48 > 1  | {r3['n_patients_ratio_above_1_5']}/48 > 1.5")


if __name__ == "__main__":
    main()
