"""
Revision — mechanism analysis at scale (238 sessions, 2 models).

Extends `analysis/mechanism/feature_importance.py` from n=10 to all impaired sessions
and adds a GrabMyo-only control model so the falsification test is crisp.

Question tested: does the calibrated model's feature importance track distribution
shift (GrabMyo → PhysioMio-impaired) — the standard transfer-learning assumption?

For each impaired session:
  1. Load the cached CALIBRATED HGB (GrabMyo + per-session cal)
     → compute permutation importance on that session's balanced test set
     → correlate importance rank vs Wasserstein-shift rank (Spearman ρ)
  2. Apply the fixed GRABMYO-ONLY HGB (trained once, no session cal)
     → compute permutation importance on the SAME balanced test set
     → correlate importance rank vs Wasserstein-shift rank (Spearman ρ)

Predicted outcome (from the n=10 pilot at ρ=+0.05, p=0.32):
  - Calibrated ρ ≈ 0 (calibration is diffuse, not shift-driven)
  - GrabMyo-only ρ meaningfully positive (pretraining-only model DOES weight shifted features)

If confirmed at n≈238 with tight CIs, this **falsifies the standard TL framing** for stroke
EMG. That's the reason-for-failure story for ICBINB and the mechanism paragraph for TS-LIMITS.

Resumable: skips (participant, session) already in the CSV.

Outputs:
  analysis/revision/results/mechanism_at_scale_per_session.csv
  analysis/revision/results/mechanism_at_scale_summary.md
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from analysis.seed import SEED, seed_everything
from ml.train_hgb_v2 import engineer_features
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, GRABMYO_META, GRABMYO_CACHE, MODEL_CACHE_DIR,
    TEST_PER_CLASS, split_session,
)

SHIFT_CSV = PROJECT_ROOT / "analysis" / "mechanism" / "results" / "feature_shift_ranked.csv"
OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "mechanism_at_scale_per_session.csv"
OUT_MD = OUT_DIR / "mechanism_at_scale_summary.md"
# Per-session per-feature importance vectors (for pairwise similarity analysis)
IMP_PARQUET = OUT_DIR / "mechanism_at_scale_importance_vectors.parquet"

N_REPEATS = 3   # reduced from pilot's 5 for speed at 238 sessions × 2 models


def make_hgb(seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def bootstrap_ci(x, n=2000, seed=SEED):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    samples = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    t0 = time.time()

    print("Loading + engineering PhysioMio...")
    df = pd.read_pickle(PHYSIOMIO_PKL)
    eng = engineer_features(df)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]
    print(f"  engineered in {time.time()-t0:.0f}s")

    # Load Wasserstein-1 shift table
    shift_df = pd.read_csv(SHIFT_CSV)[["feature", "w_grabmyo_vs_impaired"]]
    shift_df["shift_rank"] = shift_df["w_grabmyo_vs_impaired"].rank(ascending=False).astype(int)
    shift_map = dict(zip(shift_df["feature"], shift_df["shift_rank"]))
    features_with_shift = [f for f in feature_cols if f in shift_map]
    print(f"Features with shift measurement: {len(features_with_shift)} / {len(feature_cols)}")

    # ── Train GrabMyo-only classifier ONCE ──
    print("Training GrabMyo-only classifier (once, 300k subsample)...")
    gm = pd.read_pickle(GRABMYO_CACHE).sample(300_000, random_state=SEED)
    gm_X = gm[feature_cols].values.astype(np.float32)
    gm_y = gm["intent_idx"].values.astype(np.int64)
    gm_scaler = StandardScaler().fit(gm_X)
    gm_clf = make_hgb().fit(gm_scaler.transform(gm_X), gm_y)
    del gm, gm_X, gm_y
    print(f"  GrabMyo-only classifier ready (in {time.time()-t0:.0f}s)")

    # ── Resume support ──
    rows = []
    done = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done = set(zip(existing["participant"], existing["session"]))
        print(f"Resume: {len(rows)} rows, {len(done)} sessions already done")

    # Per-session importance vectors (loaded if partial exists, extended each session)
    imp_records = []
    if IMP_PARQUET.exists():
        imp_records = pd.read_parquet(IMP_PARQUET).to_dict("records")
        print(f"  {len(imp_records)} existing importance-vector rows")

    # ── Iterate over all impaired sessions ──
    all_sessions = sorted(
        [(p, s) for (p, s), _ in eng.groupby(["participant", "session"])
         if s.startswith("impaired_")],
        key=lambda ps: (int(ps[0].replace("patient", "")), ps[1]),
    )
    remaining = [ps for ps in all_sessions if ps not in done]
    print(f"Sessions total: {len(all_sessions)}, remaining: {len(remaining)}")

    for i, (participant, session) in enumerate(all_sessions, 1):
        if (participant, session) in done:
            continue
        cache_path = MODEL_CACHE_DIR / f"{participant}__{session}.joblib"
        if not cache_path.exists():
            continue

        s_data = eng[(eng["participant"] == participant) & (eng["session"] == session)]
        local_rng = np.random.RandomState(SEED)
        try:
            test_idx, _, _ = split_session(s_data, TEST_PER_CLASS, local_rng)
        except Exception:
            continue
        if len(test_idx) < 15:
            continue

        X_test_raw = s_data.loc[test_idx, feature_cols].values.astype(np.float32)
        y_test = s_data.loc[test_idx, "intent_idx"].values.astype(np.int64)

        # ── Calibrated model importance (cached per-session model) ──
        try:
            bundle = joblib.load(cache_path)
            cal_clf, cal_scaler = bundle["clf"], bundle["scaler"]
            X_test_cal = cal_scaler.transform(X_test_raw)
            cal_result = permutation_importance(
                cal_clf, X_test_cal, y_test, n_repeats=N_REPEATS,
                random_state=SEED, n_jobs=1, scoring="accuracy",
            )
            cal_imp = pd.Series(cal_result.importances_mean, index=feature_cols)
        except Exception as e:
            print(f"  {participant}/{session}: cal importance failed ({e})", flush=True)
            continue

        # ── GrabMyo-only model importance (SAME test set) ──
        try:
            X_test_gm = gm_scaler.transform(X_test_raw)
            gm_result = permutation_importance(
                gm_clf, X_test_gm, y_test, n_repeats=N_REPEATS,
                random_state=SEED, n_jobs=1, scoring="accuracy",
            )
            gm_imp = pd.Series(gm_result.importances_mean, index=feature_cols)
        except Exception as e:
            print(f"  {participant}/{session}: GM importance failed ({e})", flush=True)
            continue

        # Per-session Spearman ρ against fixed shift ranking (restrict to features with shift)
        cal_ranks = cal_imp[features_with_shift].rank(ascending=False)
        gm_ranks = gm_imp[features_with_shift].rank(ascending=False)
        shift_ranks = pd.Series(
            [shift_map[f] for f in features_with_shift], index=features_with_shift,
        )
        cal_rho, cal_p = spearmanr(cal_ranks.values, shift_ranks.values)
        gm_rho, gm_p = spearmanr(gm_ranks.values, shift_ranks.values)

        rows.append({
            "participant": participant,
            "session": session,
            "cal_rho": float(cal_rho),
            "cal_p": float(cal_p),
            "gm_rho": float(gm_rho),
            "gm_p": float(gm_p),
            "n_test": len(y_test),
        })

        # Save per-session importance vector (for post-hoc pairwise-similarity analysis)
        imp_records.append({
            "participant": participant,
            "session": session,
            "model": "calibrated",
            **{f: float(v) for f, v in cal_imp.items()},
        })
        imp_records.append({
            "participant": participant,
            "session": session,
            "model": "grabmyo_only",
            **{f: float(v) for f, v in gm_imp.items()},
        })

        elapsed = time.time() - t0
        n_done = len(rows)
        remaining_count = len(all_sessions) - n_done
        eta = (elapsed / n_done) * remaining_count if n_done > 0 else 0
        print(f"[{i}/{len(all_sessions)}] {participant}/{session}: "
              f"cal_ρ={cal_rho:+.3f} (p={cal_p:.2f})  gm_ρ={gm_rho:+.3f} (p={gm_p:.2f})  "
              f"[{elapsed/60:.1f}min, eta {eta/60:.0f}min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        pd.DataFrame(imp_records).to_parquet(IMP_PARQUET, index=False)

    # ── Summary ──
    out = pd.DataFrame(rows)
    if len(out) == 0:
        print("No rows collected. Aborting.")
        return

    cal_mean, cal_med = out["cal_rho"].mean(), out["cal_rho"].median()
    gm_mean, gm_med = out["gm_rho"].mean(), out["gm_rho"].median()
    cal_lo, cal_hi = bootstrap_ci(out["cal_rho"].values)
    gm_lo, gm_hi = bootstrap_ci(out["gm_rho"].values)

    # Paired test: cal_rho vs gm_rho within-session
    valid = ~out["cal_rho"].isna() & ~out["gm_rho"].isna()
    diff = out.loc[valid, "gm_rho"].values - out.loc[valid, "cal_rho"].values
    w = wilcoxon(diff, alternative="greater")   # GM > cal → mechanism differs

    md = [
        "# Mechanism at scale — falsifying distribution-shift-correction as the mechanism",
        "",
        f"Permutation importance vs Wasserstein-1 shift on {len(out)} impaired-arm ",
        f"PhysioMio sessions, with two classifiers evaluated on each session's balanced test set:",
        "",
        "- **Calibrated HGB** — GrabMyo + per-session cal (cached per-session model)",
        "- **GrabMyo-only HGB** — trained once on 300k GrabMyo subsample, no session cal",
        "",
        "For each session and each classifier, permutation importance is ranked and correlated",
        "(Spearman ρ) against the fixed GrabMyo→PhysioMio-impaired Wasserstein-1 shift ranking.",
        "",
        "## Headline",
        "",
        "| Model | Mean ρ | Median ρ | 95% bootstrap CI | Std |",
        "|---|---:|---:|---:|---:|",
        f"| **Calibrated HGB** | {cal_mean:+.3f} | {cal_med:+.3f} | [{cal_lo:+.3f}, {cal_hi:+.3f}] | {out['cal_rho'].std():.3f} |",
        f"| **GrabMyo-only HGB** | {gm_mean:+.3f} | {gm_med:+.3f} | [{gm_lo:+.3f}, {gm_hi:+.3f}] | {out['gm_rho'].std():.3f} |",
        "",
        f"**Paired Wilcoxon (GM ρ > Cal ρ, one-sided): p = {w.pvalue:.3e}**",
        "",
        "## Interpretation",
        "",
        "If **Cal ρ ≈ 0** while **GM ρ > 0** (and Wilcoxon p is small), then:",
        "",
        "- The GrabMyo-only classifier *does* weight the shifted features (as standard TL",
        "  theory predicts — its decisions rely on features that differ between healthy and",
        "  stroke distributions).",
        "- The calibrated classifier does *not* — calibration overrides that shift-driven",
        "  prior with per-patient-specific features.",
        "",
        "This falsifies the standard TL mechanism (\"calibration corrects distribution shift\")",
        "for stroke EMG. Per-patient decision boundaries are effectively independent tasks;",
        "healthy-subject pretraining does not inform them regardless of how well distributions",
        "align. This is the mechanistic reason large-scale healthy-EMG pretraining fails to",
        "improve stroke EMG classification at deployment (§4).",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(f"cal_ρ = {cal_mean:+.3f} [{cal_lo:+.3f}, {cal_hi:+.3f}]")
    print(f"gm_ρ  = {gm_mean:+.3f} [{gm_lo:+.3f}, {gm_hi:+.3f}]")
    print(f"Paired Wilcoxon (gm > cal): p = {w.pvalue:.3e}")


if __name__ == "__main__":
    main()
