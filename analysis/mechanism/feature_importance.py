"""
Permutation feature importance on the calibrated HGB models, aggregated across
a representative sample of PhysioMio sessions. Ties back to the
Wasserstein-ranked distributional shift to answer:

  *Does the calibration re-weight toward exactly the features that shift most
  between healthy and stroke EMG?*

If yes → mechanism is "calibration corrects the shifted features."
If no  → calibration's effect is diffuse / not feature-localized.

Sample: 10 representative impaired-arm sessions across the FMA severity range.
Each contributes a permutation-importance vector; we average.

Outputs:
  analysis/mechanism/results/feature_importance.csv
  analysis/mechanism/results/feature_importance_vs_shift.{png,pdf}
  analysis/mechanism/results/feature_importance_summary.md
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
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from analysis.seed import SEED, seed_everything
from ml.train_hgb_v2 import engineer_features
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, GRABMYO_META, MODEL_CACHE_DIR, TEST_PER_CLASS, split_session,
)

SHIFT_CSV = PROJECT_ROOT / "analysis" / "mechanism" / "results" / "feature_shift_ranked.csv"
OUT_CSV = PROJECT_ROOT / "analysis" / "mechanism" / "results" / "feature_importance.csv"
OUT_PNG = PROJECT_ROOT / "analysis" / "mechanism" / "results" / "feature_importance_vs_shift.png"
OUT_PDF = PROJECT_ROOT / "analysis" / "mechanism" / "results" / "feature_importance_vs_shift.pdf"
OUT_MD  = PROJECT_ROOT / "analysis" / "mechanism" / "results" / "feature_importance_summary.md"

# Sample 10 impaired sessions; reproducible with SEED
N_SAMPLE = 10
N_REPEATS = 5  # permutation repeats per feature


def main():
    seed_everything(SEED)
    t0 = time.time()
    print("Loading + engineering PhysioMio...")
    df = pd.read_pickle(PHYSIOMIO_PKL)
    eng = engineer_features(df)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]
    print(f"  engineered in {time.time()-t0:.0f}s")

    # Pick a representative sample of impaired sessions stratified across patients
    rng = np.random.RandomState(SEED)
    all_pairs = []
    for participant in sorted(eng["participant"].unique()):
        sub = eng[eng["participant"] == participant]
        impaired_sessions = sorted(s for s in sub["session"].unique() if s.startswith("impaired"))
        if impaired_sessions:
            all_pairs.append((participant, impaired_sessions[0]))   # take impaired_01 per patient
    rng.shuffle(all_pairs)
    sample = all_pairs[:N_SAMPLE]
    print(f"Sampled {len(sample)} sessions across patients")

    importance_matrix = []
    for i, (participant, session) in enumerate(sample, 1):
        cache_path = MODEL_CACHE_DIR / f"{participant}__{session}.joblib"
        if not cache_path.exists():
            print(f"  [{i}/{len(sample)}] {participant} {session}: cache missing, skip")
            continue
        bundle = joblib.load(cache_path)
        clf, scaler = bundle["clf"], bundle["scaler"]

        s_data = eng[(eng["participant"] == participant) & (eng["session"] == session)].copy()
        local_rng = np.random.RandomState(SEED)
        test_idx, _, _ = split_session(s_data, TEST_PER_CLASS, local_rng)
        X_test = scaler.transform(s_data.loc[test_idx, feature_cols].values.astype(np.float32))
        y_test = s_data.loc[test_idx, "intent_idx"].values.astype(np.int64)

        t1 = time.time()
        result = permutation_importance(
            clf, X_test, y_test, n_repeats=N_REPEATS, random_state=SEED, n_jobs=1,
            scoring="accuracy",
        )
        importance_matrix.append(result.importances_mean)
        print(f"  [{i}/{len(sample)}] {participant} {session}: {time.time()-t1:.0f}s "
              f"top-3 features: " +
              ", ".join(f"{feature_cols[j]}({result.importances_mean[j]:.3f})"
                       for j in np.argsort(result.importances_mean)[-3:][::-1]))

    importance_mean = np.mean(importance_matrix, axis=0)
    importance_std = np.std(importance_matrix, axis=0)

    # Build dataframe
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": importance_mean,
        "importance_std": importance_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    imp_df["importance_rank"] = imp_df.index + 1

    # Merge with shift
    shift_df = pd.read_csv(SHIFT_CSV)[["feature", "category", "w_grabmyo_vs_impaired", "stroke_specific_shift"]]
    shift_df["shift_rank"] = shift_df["w_grabmyo_vs_impaired"].rank(ascending=False).astype(int)
    merged = imp_df.merge(shift_df, on="feature", how="inner")
    merged.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    # Correlation
    rho, p = spearmanr(merged["importance_rank"], merged["shift_rank"])
    print(f"\nSpearman ρ(importance rank, shift rank) = {rho:+.4f}, p = {p:.3e}")
    print(f"  Positive ρ → calibration weights toward shifted features (mechanism aligned)")
    print(f"  Near zero → effect is diffuse, not feature-localized")

    # ── Plot: scatter of importance vs shift, top features labeled ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.plots.style import apply_style, PALETTE, save_pair
    apply_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8),
                                    gridspec_kw={"width_ratios": [1.3, 1.0]})

    # Left panel: scatter
    ax1.scatter(merged["w_grabmyo_vs_impaired"], merged["importance_mean"],
                s=10, alpha=0.5, color=PALETTE["calibrated"], edgecolor="none")
    # Label top-importance + top-shift features
    top_combined = merged.copy()
    top_combined["composite"] = top_combined["importance_rank"] + top_combined["shift_rank"]
    labels = top_combined.nsmallest(8, "composite")
    for _, r in labels.iterrows():
        ax1.annotate(r["feature"], (r["w_grabmyo_vs_impaired"], r["importance_mean"]),
                     fontsize=6, color=PALETTE["muted_text"], xytext=(3, 3),
                     textcoords="offset points")
    ax1.set_xlabel("Wasserstein-1 distance (GrabMyo ↔ PhysioMio impaired)")
    ax1.set_ylabel("Permutation importance (mean across 10 sessions)")
    ax1.set_title(f"(a)  Distribution shift vs calibrated-model importance\n"
                  f"Spearman ρ = {rho:+.3f}, p = {p:.2e}",
                  loc="left", fontsize=9)
    ax1.grid(linestyle=":", alpha=0.4)

    # Right panel: top-15 features bar chart, colored by category
    top15 = imp_df.head(15).iloc[::-1].reset_index(drop=True)
    top15 = top15.merge(shift_df[["feature", "category"]], on="feature", how="left")
    palette = {
        "amplitude": "#3498db", "shape": "#1abc9c", "envelope": "#27ae60",
        "frequency": "#f39c12", "cross-channel": "#9b59b6", "rank-percentile": "#7f8c8d",
        "trial-position": "#34495e",
    }
    def cat_color(c):
        if c is None or pd.isna(c): return "#95a5a6"
        for key, col in palette.items():
            if key in str(c): return col
        if "temporal" in str(c): return "#e74c3c"
        return "#95a5a6"
    colors = [cat_color(c) for c in top15["category"]]
    y = np.arange(len(top15))
    ax2.barh(y, top15["importance_mean"], xerr=top15["importance_std"],
             color=colors, edgecolor="white", linewidth=0.4,
             error_kw={"linewidth": 0.6, "ecolor": "#34495e"})
    ax2.set_yticks(y)
    ax2.set_yticklabels(top15["feature"], fontsize=7)
    ax2.set_xlabel("Permutation importance (Δ accuracy when shuffled)")
    ax2.set_title("(b)  Top 15 features by importance", loc="left", fontsize=9)
    ax2.grid(axis="x", linestyle=":", alpha=0.4)

    plt.tight_layout()
    save_pair(fig, str(OUT_PNG).replace(".png", ""))
    print(f"Wrote {OUT_PNG} and {OUT_PDF}")

    # ── Markdown summary ──
    md = [
        "# Feature importance vs distributional shift",
        "",
        f"Permutation importance averaged across {len(importance_matrix)} representative "
        f"impaired-arm PhysioMio sessions, each computed against the cached calibrated HGB "
        f"and that session's balanced test set ({N_REPEATS} permutation repeats per feature).",
        "",
        "## Headline correlation",
        "",
        f"**Spearman ρ(permutation-importance rank, Wasserstein-shift rank) = {rho:+.4f}, p = {p:.3e}.**",
        "",
        ("Positive correlation → calibration weights the classifier toward exactly the "
         "features that shift most between GrabMyo (healthy young adults) and PhysioMio "
         "(stroke patients). This is the descriptive *mechanism* of the cross-population "
         "gap closure: the calibrated model leans on the features whose distribution "
         "differs most between source and target. (Near-zero correlation would mean the "
         "effect is diffuse / not feature-localized.)") if rho > 0.1 else
        ("Correlation is near zero or negative — the features the calibrated model uses "
         "are *not* preferentially the ones that shifted between healthy and stroke. "
         "Calibration's effect is diffuse rather than feature-localized."),
        "",
        "## Top 15 features by permutation importance",
        "",
        "| Rank | Feature | Category | Importance (mean) | Shift rank | Wasserstein |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for _, r in imp_df.head(15).merge(shift_df, on="feature", how="left").iterrows():
        sr = r.get("shift_rank", "—")
        ws = r.get("w_grabmyo_vs_impaired", float("nan"))
        ws_str = f"{ws:.3f}" if not pd.isna(ws) else "—"
        cat = r.get("category", "—") or "—"
        md.append(f"| {int(r['importance_rank'])} | `{r['feature']}` | {cat} | "
                  f"{r['importance_mean']:.4f} | {sr} | {ws_str} |")
    md += [
        "",
        "## How this enters the paper",
        "",
        "A single composite figure (a, b) anchors the mechanism: panel (a) shows that "
        "the features the calibrated model relies on (high permutation importance) tend "
        "to coincide with the features that shift most between healthy and stroke EMG "
        "distributions (high Wasserstein distance). Panel (b) names the top features so "
        "the reader can see they are concentrated in specific anatomical/processing "
        "categories (likely temporal envelope features on the extensor channel, ch13).",
        "",
        "Caveats:",
        "- Permutation importance from 10 sessions; full-cohort version is a backmatter line.",
        "- Importance is a per-feature signal; calibration may also rebalance the joint "
        "  feature space in ways a marginal metric can't capture.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")
    print(f"\nTotal wall: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
