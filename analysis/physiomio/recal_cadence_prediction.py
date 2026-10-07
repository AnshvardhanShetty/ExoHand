"""
Recal-cadence prediction: predict per-patient longitudinal decay rate from
session-1 features.

This turns the longitudinal-degradation finding (Stream 5) from a passive
characterisation into an actionable method:

    *Given a patient's first session, predict how rapidly their per-session
    calibration accuracy will decay across subsequent sessions, so the
    clinic can schedule recalibration cadence per-patient.*

The contribution is a clean, small methodological proposal that's original
to this paper (not in ReactEMG or prior EMG calibration work).

Protocol:
  1. Target = per-patient linear slope of impaired-arm accuracy vs session
     distance, fitted from `longitudinal_per_session.csv` on patients with
     ≥ 3 sessions. Negative slope = faster decay.
  2. Features = session-1-only (impaired_01) summaries:
       a) FMA-equivalent severity score (from severity_per_patient.csv)
       b) Within-session calibrated accuracy (dist = 0)
       c) Per-class F1 at dist = 0 (rest, close, open)
       d) Per-class F1 *imbalance* = max(F1) − min(F1) at dist = 0
       e) Cross-arm transfer accuracy (impaired-cal → healthy arm at dist = 0)
  3. Regressor = HistGradientBoostingRegressor with leave-one-patient-out CV.
     LOPO is the natural protocol because the test "patient" matches the
     deployment scenario (predict for a new patient given only their first
     session).
  4. Report: LOPO R², RMSE, MAE, mean baseline (predict-the-mean) RMSE,
     and per-feature permutation importance.

Output:
  analysis/physiomio/results/recal_cadence_prediction.{csv,md,json}
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from analysis.seed import SEED, seed_everything

LONG_CSV   = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "longitudinal_per_session.csv"
PSESS_CSV  = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "per_session_results.csv"
SEV_CSV    = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "severity_per_patient.csv"

OUT_CSV    = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "recal_cadence_prediction.csv"
OUT_JSON   = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "recal_cadence_prediction.json"
OUT_MD     = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "recal_cadence_prediction.md"

MIN_SESSIONS_FOR_FIT = 3   # need ≥ 3 distances to fit a slope


def fit_per_patient_decay_slopes() -> pd.DataFrame:
    """Return per-patient linear-fit slope of impaired-arm acc vs session distance."""
    df = pd.read_csv(LONG_CSV)
    imp = df[df["arm"] == "impaired"].copy()
    imp["impaired_session_distance"] = imp["impaired_session_distance"].astype(int)
    rows = []
    for pat, g in imp.groupby("patient"):
        g = g.sort_values("impaired_session_distance")
        if len(g) < MIN_SESSIONS_FOR_FIT:
            continue
        x = g["impaired_session_distance"].values
        y = g["acc"].values
        slope, intercept = np.polyfit(x, y, 1)
        # Also store first-vs-last drop (an alternative target)
        rows.append({
            "patient": pat,
            "n_sessions_in_fit": int(len(g)),
            "max_distance": int(x.max()),
            "acc_dist0": float(y[x == 0][0]) if 0 in x else float("nan"),
            "decay_slope": float(slope),   # acc per session-distance, negative = degrading
            "decay_first_minus_last": float(y[0] - y[-1]),
        })
    return pd.DataFrame(rows)


def build_session1_features() -> pd.DataFrame:
    """Per-patient features extracted from impaired_01 (session 1) only.

    These are the variables a clinic *would* have available after the patient's
    first visit — so the regression respects deployment semantics.
    """
    sess = pd.read_csv(PSESS_CSV)
    sess = sess[sess["status"] == "ok"]
    sev = pd.read_csv(SEV_CSV)
    long_df = pd.read_csv(LONG_CSV)

    # Session-1 features per patient: from impaired_01 entry in per_session_results
    s1 = sess[sess["session"] == "impaired_01"].copy()
    feats = s1[["participant", "acc", "f1_macro", "f1_rest", "f1_close", "f1_open"]].rename(
        columns={"participant": "patient", "acc": "s1_acc", "f1_macro": "s1_f1_macro",
                 "f1_rest": "s1_f1_rest", "f1_close": "s1_f1_close", "f1_open": "s1_f1_open"}
    )
    # Per-class F1 imbalance — large spread suggests one class dominates the cal
    feats["s1_f1_imbalance"] = feats[["s1_f1_rest", "s1_f1_close", "s1_f1_open"]].max(axis=1) - \
                               feats[["s1_f1_rest", "s1_f1_close", "s1_f1_open"]].min(axis=1)
    # Cross-arm transfer at dist=0 — from longitudinal eval, healthy-arm session 1
    cross_arm = long_df[
        (long_df["arm"] == "healthy") & (long_df["cal_session"] == "impaired_01")
    ].groupby("patient")["acc"].mean().rename("s1_crossarm_acc").reset_index()
    cross_arm = cross_arm.rename(columns={"patient": "patient"})
    feats = feats.merge(cross_arm, on="patient", how="left")
    # FMA (impaired-arm mean per patient)
    sev_r = sev[["participant", "impaired_fma_mean"]].rename(
        columns={"participant": "patient", "impaired_fma_mean": "fma_impaired"}
    )
    feats = feats.merge(sev_r, on="patient", how="left")
    return feats


def main():
    seed_everything(SEED)
    t0 = time.time()

    print("[1/4] Fitting per-patient decay slopes...")
    targets = fit_per_patient_decay_slopes()
    print(f"  {len(targets)} patients with ≥ {MIN_SESSIONS_FOR_FIT} session distances")
    print(f"  decay_slope: mean={targets['decay_slope'].mean():+.4f} std={targets['decay_slope'].std():.4f} "
          f"range=[{targets['decay_slope'].min():+.4f}, {targets['decay_slope'].max():+.4f}]")

    print("\n[2/4] Building session-1 features...")
    features = build_session1_features()
    print(f"  {len(features)} patients with features")

    merged = targets.merge(features, on="patient").dropna()
    print(f"  {len(merged)} patients after joining targets and features")

    feature_cols = [
        "s1_acc", "s1_f1_macro",
        "s1_f1_rest", "s1_f1_close", "s1_f1_open",
        "s1_f1_imbalance",
        "s1_crossarm_acc",
        "fma_impaired",
    ]
    X = merged[feature_cols].values
    y = merged["decay_slope"].values

    print("\n[3/4] Leave-one-patient-out CV...")
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    actuals = y.copy()
    rng = np.random.RandomState(SEED)
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        reg = HistGradientBoostingRegressor(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=5, l2_regularization=0.1, random_state=SEED,
        )
        reg.fit(Xtr, y[train_idx])
        preds[test_idx[0]] = reg.predict(Xte)[0]

    # LOPO metrics
    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mae = float(mean_absolute_error(actuals, preds))
    r2 = float(r2_score(actuals, preds))
    # Baseline: predict the mean
    baseline_rmse = float(np.sqrt(mean_squared_error(actuals, np.full_like(actuals, actuals.mean()))))
    # Spearman correlation (rank agreement is what a clinic actually uses)
    rho, p_rho = spearmanr(actuals, preds)

    print(f"  LOPO R² = {r2:+.4f}")
    print(f"  LOPO RMSE = {rmse:.4f} (vs predict-the-mean baseline {baseline_rmse:.4f}; ratio {rmse/baseline_rmse:.3f})")
    print(f"  LOPO MAE = {mae:.4f}")
    print(f"  Spearman ρ(predicted, actual decay slope) = {rho:+.4f}, p = {p_rho:.4f}")

    # Final-model permutation importance (fit on full data once)
    print("\n[4/4] Permutation importance (fit on all patients)...")
    scaler_full = StandardScaler()
    X_full = scaler_full.fit_transform(X)
    reg_full = HistGradientBoostingRegressor(
        max_iter=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=5, l2_regularization=0.1, random_state=SEED,
    )
    reg_full.fit(X_full, y)
    pi = permutation_importance(reg_full, X_full, y, n_repeats=30,
                                random_state=SEED, scoring="neg_mean_squared_error")
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    print(imp_df.to_string(index=False))

    # ── Save outputs ──
    per_patient_out = merged[["patient"] + feature_cols + ["decay_slope"]].copy()
    per_patient_out["predicted_decay_slope"] = preds
    per_patient_out["residual"] = actuals - preds
    per_patient_out.to_csv(OUT_CSV, index=False)

    summary = {
        "n_patients": int(len(merged)),
        "lopo_metrics": {"r2": r2, "rmse": rmse, "mae": mae,
                         "baseline_rmse": baseline_rmse,
                         "rmse_ratio_vs_baseline": rmse / baseline_rmse},
        "spearman": {"rho": float(rho), "p": float(p_rho)},
        "target_summary": {
            "decay_slope_mean": float(actuals.mean()),
            "decay_slope_std": float(actuals.std()),
            "decay_slope_min": float(actuals.min()),
            "decay_slope_max": float(actuals.max()),
        },
        "feature_importance": imp_df.to_dict("records"),
        "features_used": feature_cols,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))

    md = [
        "# Recal-cadence prediction",
        "",
        f"**Task:** predict per-patient longitudinal accuracy decay rate from "
        f"session-1 features. Decay rate = linear slope of impaired-arm accuracy "
        f"vs session distance from cal, fitted on patients with ≥ {MIN_SESSIONS_FOR_FIT} sessions.",
        "",
        f"**Cohort:** n = {len(merged)} stroke patients (out of 48 PhysioMio total) — "
        f"the subset with enough longitudinal sessions to fit a per-patient decay slope.",
        "",
        f"**Model:** HistGradientBoostingRegressor (max_iter=200, max_depth=4, "
        f"l2_reg=0.1), leave-one-patient-out cross-validation.",
        "",
        "## Headline result",
        "",
        "| Metric | Value | Interpretation |",
        "|---|---:|---|",
        f"| **LOPO R²** | **{r2:+.4f}** | Fraction of decay-rate variance explained |",
        f"| LOPO RMSE | {rmse:.4f} | acc / session-distance |",
        f"| LOPO MAE | {mae:.4f} | |",
        f"| Predict-the-mean RMSE baseline | {baseline_rmse:.4f} | |",
        f"| RMSE ratio (model / baseline) | **{rmse/baseline_rmse:.3f}** | < 1.0 = model beats predict-the-mean |",
        f"| Spearman ρ(predicted, actual) | **{rho:+.4f}** (p = {p_rho:.3f}) | Rank agreement — relevant for prioritising patients |",
        "",
        "## Feature importance (permutation, on full-data fit)",
        "",
        "| Feature | Permutation importance (mean ± std) |",
        "|---|---:|",
    ]
    for _, r in imp_df.iterrows():
        md.append(f"| `{r['feature']}` | {r['importance_mean']:.4f} ± {r['importance_std']:.4f} |")

    md += [
        "",
        "## How this enters the paper",
        "",
        "**One paragraph in §4.3 (within-subject temporal shift):**",
        "",
        "> *Beyond characterising the cross-session degradation, we ask whether decay "
        "rate is predictable from session-1 features — a question with direct "
        "deployment relevance, since a clinic can use such a prediction to schedule "
        "per-patient recalibration cadence. Fitting HistGradientBoosting regression "
        f"on {len(merged)} patients with leave-one-patient-out cross-validation, we "
        f"achieve R² = {r2:+.3f} (RMSE {rmse/baseline_rmse:.2f}× the predict-the-mean "
        f"baseline) and Spearman ρ = {rho:+.3f} between predicted and actual decay "
        "slopes. The most predictive features are [see table]. This converts the "
        "longitudinal-degradation finding from a passive characterisation into an "
        "actionable per-patient deployment knob.*",
        "",
        "## Caveats",
        "",
        "- Sample size is n = "
        f"{len(merged)}, modest for a regression problem; LOPO CV is the right "
        "protocol given the constraint.",
        "- Decay slope is a single-number summary of a curve that's often noisy; "
        "results would tighten with a more robust target (e.g., median decay or "
        "robust regression on the curve).",
        "- The protocol applies to the specific calibration scheme studied here; "
        "different calibration designs (e.g., different cal-weight or feature pipeline) "
        "would have different decay characteristics.",
    ]
    OUT_MD.write_text("\n".join(md))

    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"Total wall: {(time.time()-t0):.1f}s")


if __name__ == "__main__":
    main()
