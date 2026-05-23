"""
PhysioMio severity stratification — Stream 2.

Aggregates per-patient FMA (Fugl-Meyer-style 0/1/2 score) from PhysioMio's
per-gesture metadata, then tests whether calibration effectiveness scales with
patient severity.

Hypotheses tested:
    1. More-impaired patients (lower FMA on paretic arm) → lower with-cal accuracy
       (Spearman + Pearson, FMA vs acc_with_cal)
    2. More-impaired patients → bigger calibration improvement (Δacc = cal − no_cal)
       This is the "calibration helps most where it's needed most" narrative
    3. Healthy-arm FMA should be near-ceiling and uncorrelated with anything

Aggregation:
    Per patient, impaired-arm FMA = mean across all (impaired session × gesture)
    pairs that have an FMA value (Rest excluded since it has no FMA). 15 gestures
    × N impaired sessions per patient = per-patient impaired FMA mean.

Outputs:
    analysis/physiomio/results/severity_per_patient.csv
    analysis/physiomio/results/severity_summary.md
    analysis/physiomio/results/severity_summary.json
    analysis/physiomio/results/severity_scatter.png
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analysis.seed import SEED, seed_everything


PHYSIOMIO_ROOT = PROJECT_ROOT / "data" / "physiomio" / "data"
RESULTS_DIR = PROJECT_ROOT / "analysis" / "physiomio" / "results"
PER_PATIENT_CAL = RESULTS_DIR / "per_patient_results.csv"
ZERO_SHOT_CSV = RESULTS_DIR / "zero_shot_per_patient.csv"
OUT_CSV = RESULTS_DIR / "severity_per_patient.csv"
OUT_MD = RESULTS_DIR / "severity_summary.md"
OUT_JSON = RESULTS_DIR / "severity_summary.json"
OUT_PNG = RESULTS_DIR / "severity_scatter.png"


def compute_patient_fma(patient_dir: Path) -> dict:
    """Aggregate FMA across all sessions for one patient.

    Returns dict with:
        - healthy_fma_mean: mean FMA across all healthy-arm sessions × gestures
        - impaired_fma_mean: mean FMA across all impaired-arm sessions × gestures
        - impaired_fma_min: per-session worst session's mean (most-severe session)
        - n_healthy_sessions, n_impaired_sessions
        - n_fma_values_healthy/impaired (total FMA-bearing observations)
    """
    out = {"patient": patient_dir.name}
    for arm in ["healthy_arm", "impaired_arm"]:
        sessions = sorted((patient_dir / arm).glob("*.parquet"))
        all_fma = []
        per_session_means = []
        for parq in sessions:
            df = pd.read_parquet(parq, columns=["movement_type", "fma"])
            # First FMA value per gesture (constant within gesture segment)
            per_g = df.groupby("movement_type")["fma"].first().dropna()
            all_fma.extend(per_g.values.tolist())
            if len(per_g) > 0:
                per_session_means.append(float(per_g.mean()))
        prefix = "healthy" if arm == "healthy_arm" else "impaired"
        out[f"{prefix}_fma_mean"] = float(np.mean(all_fma)) if all_fma else np.nan
        out[f"{prefix}_fma_min_session_mean"] = float(min(per_session_means)) if per_session_means else np.nan
        out[f"n_{prefix}_sessions"] = len(sessions)
        out[f"n_{prefix}_fma_values"] = len(all_fma)
    return out


def main():
    seed_everything(SEED)

    # --- Load classification results ---
    cal_df = pd.read_csv(PER_PATIENT_CAL)
    zs_df = pd.read_csv(ZERO_SHOT_CSV)
    print(f"Loaded cal results: {cal_df.shape}")
    print(f"Loaded zero-shot results: {zs_df.shape}")

    # --- Compute per-patient FMA ---
    print("\nAggregating FMA across patients...")
    patient_dirs = sorted(
        [p for p in PHYSIOMIO_ROOT.iterdir() if p.is_dir() and p.name.startswith("patient")],
        key=lambda p: int(p.name.replace("patient", "")),
    )
    fma_records = []
    for pd_dir in patient_dirs:
        rec = compute_patient_fma(pd_dir)
        fma_records.append(rec)
        print(f"  {pd_dir.name}: healthy_fma={rec['healthy_fma_mean']:.3f}  "
              f"impaired_fma={rec['impaired_fma_mean']:.3f}  "
              f"impaired_worst={rec['impaired_fma_min_session_mean']:.3f}")
    fma_df = pd.DataFrame(fma_records)

    # --- Join with classification results ---
    df = fma_df.merge(
        cal_df[["participant", "acc_mean", "acc_healthy_mean", "acc_impaired_mean"]].rename(
            columns={"acc_mean": "acc_with_cal", "acc_healthy_mean": "acc_with_cal_healthy",
                     "acc_impaired_mean": "acc_with_cal_impaired"}),
        left_on="patient", right_on="participant", how="inner",
    ).merge(
        zs_df[["participant", "acc_no_cal"]].rename(columns={"acc_no_cal": "acc_zero_shot"}),
        on="participant", how="inner",
    )
    df["delta_acc"] = df["acc_with_cal"] - df["acc_zero_shot"]
    df.to_csv(OUT_CSV, index=False)
    print(f"\nMerged severity + accuracy table: {df.shape}")

    # --- Correlation tests ---
    def corr_test(x, y, name_x, name_y):
        """Spearman + Pearson with p-values, plus 95% bootstrap CI on Spearman ρ."""
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
        rho_s, p_s = stats.spearmanr(x, y)
        r_p, p_p = stats.pearsonr(x, y)
        # Bootstrap CI on Spearman ρ
        rng = np.random.RandomState(SEED)
        boot = []
        for _ in range(2000):
            idx = rng.randint(0, len(x), size=len(x))
            boot.append(stats.spearmanr(x[idx], y[idx])[0])
        ci_lo = float(np.percentile(boot, 2.5))
        ci_hi = float(np.percentile(boot, 97.5))
        return {
            "name_x": name_x,
            "name_y": name_y,
            "n": int(len(x)),
            "spearman_rho": float(rho_s),
            "spearman_p": float(p_s),
            "spearman_ci_lo": ci_lo,
            "spearman_ci_hi": ci_hi,
            "pearson_r": float(r_p),
            "pearson_p": float(p_p),
        }

    print("\n" + "=" * 70)
    print("CORRELATIONS (severity vs calibration metrics)")
    print("=" * 70)

    tests = [
        corr_test(df["impaired_fma_mean"].values, df["acc_with_cal"].values,
                  "impaired_fma_mean", "acc_with_cal"),
        corr_test(df["impaired_fma_mean"].values, df["acc_with_cal_impaired"].values,
                  "impaired_fma_mean", "acc_with_cal_impaired"),
        corr_test(df["impaired_fma_mean"].values, df["acc_zero_shot"].values,
                  "impaired_fma_mean", "acc_zero_shot"),
        corr_test(df["impaired_fma_mean"].values, df["delta_acc"].values,
                  "impaired_fma_mean", "delta_acc"),
        corr_test(df["impaired_fma_min_session_mean"].values, df["acc_with_cal_impaired"].values,
                  "impaired_worst_session_fma", "acc_with_cal_impaired"),
        corr_test(df["healthy_fma_mean"].values, df["acc_with_cal_healthy"].values,
                  "healthy_fma_mean", "acc_with_cal_healthy"),
    ]
    for t in tests:
        print(f"\n  {t['name_x']}  vs  {t['name_y']}   (n={t['n']})")
        print(f"    Spearman ρ = {t['spearman_rho']:+.4f}  [95% CI {t['spearman_ci_lo']:+.4f}, {t['spearman_ci_hi']:+.4f}]  p={t['spearman_p']:.4g}")
        print(f"    Pearson r  = {t['pearson_r']:+.4f}  p={t['pearson_p']:.4g}")

    # --- Severity-stratified accuracy ---
    print("\n" + "=" * 70)
    print("STRATIFIED BY IMPAIRED-ARM SEVERITY")
    print("=" * 70)
    # Tertile split
    df["severity_tertile"] = pd.qcut(df["impaired_fma_mean"], 3,
                                     labels=["severe", "moderate", "mild"], duplicates="drop")
    strat = df.groupby("severity_tertile", observed=True).agg(
        n=("patient", "count"),
        acc_with_cal=("acc_with_cal", "mean"),
        acc_with_cal_std=("acc_with_cal", "std"),
        acc_zero_shot=("acc_zero_shot", "mean"),
        delta=("delta_acc", "mean"),
        delta_std=("delta_acc", "std"),
        fma_range_lo=("impaired_fma_mean", "min"),
        fma_range_hi=("impaired_fma_mean", "max"),
    ).round(4)
    print(strat.to_string())

    # --- Scatter plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(df["impaired_fma_mean"], df["acc_zero_shot"], alpha=0.6, color="C3", s=50)
    axes[0].set_xlabel("Impaired-arm mean FMA (1.0–2.0)")
    axes[0].set_ylabel("Zero-shot accuracy")
    axes[0].set_title("Zero-shot (no calibration)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    axes[1].scatter(df["impaired_fma_mean"], df["acc_with_cal_impaired"], alpha=0.6, color="C2", s=50)
    axes[1].set_xlabel("Impaired-arm mean FMA (1.0–2.0)")
    axes[1].set_ylabel("With-cal accuracy (impaired arm only)")
    axes[1].set_title("With per-session calibration")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    axes[2].scatter(df["impaired_fma_mean"], df["delta_acc"], alpha=0.6, color="C0", s=50)
    axes[2].set_xlabel("Impaired-arm mean FMA (1.0–2.0)")
    axes[2].set_ylabel("Δ accuracy (cal − no cal)")
    axes[2].set_title("Calibration improvement")
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(0, color="black", linewidth=0.5, alpha=0.5)

    fig.suptitle("Calibration effectiveness vs patient severity (FMA)", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print(f"\nWrote {OUT_PNG}")

    # --- Markdown ---
    md = [
        "# PhysioMio severity stratification — Stream 2",
        "",
        f"n = {len(df)} patients · severity = Fugl-Meyer-style 0/1/2 score, aggregated per-patient",
        "",
        "## Severity distribution",
        "",
        f"- Healthy-arm FMA mean (across patients): {df['healthy_fma_mean'].mean():.3f} ± {df['healthy_fma_mean'].std():.3f} (range {df['healthy_fma_mean'].min():.2f} - {df['healthy_fma_mean'].max():.2f})",
        f"- Impaired-arm FMA mean (across patients): {df['impaired_fma_mean'].mean():.3f} ± {df['impaired_fma_mean'].std():.3f} (range {df['impaired_fma_mean'].min():.2f} - {df['impaired_fma_mean'].max():.2f})",
        "",
        "## Correlations",
        "",
        "| Severity metric | Outcome | n | Spearman ρ [95% CI] | Pearson r | p (Spearman) |",
        "|---|---|---|---|---|---|",
    ]
    for t in tests:
        md.append(f"| {t['name_x']} | {t['name_y']} | {t['n']} | "
                  f"{t['spearman_rho']:+.4f} [{t['spearman_ci_lo']:+.4f}, {t['spearman_ci_hi']:+.4f}] | "
                  f"{t['pearson_r']:+.4f} | {t['spearman_p']:.4g} |")
    md += [
        "",
        "## Severity-stratified accuracy (impaired-arm FMA tertiles)",
        "",
        "| Tertile | n | FMA range | Zero-shot acc | With-cal acc | Δ acc |",
        "|---|---|---|---|---|---|",
    ]
    for label, row in strat.iterrows():
        md.append(f"| {label} | {int(row['n'])} | "
                  f"{row['fma_range_lo']:.2f}–{row['fma_range_hi']:.2f} | "
                  f"{row['acc_zero_shot']:.4f} | "
                  f"{row['acc_with_cal']:.4f} | "
                  f"+{row['delta']:.4f} |")
    md += [
        "",
        f"![scatter]({OUT_PNG.name})",
        "",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")

    # --- JSON ---
    summary = {
        "n_patients": int(len(df)),
        "severity_distribution": {
            "healthy_fma_mean_across_patients": float(df["healthy_fma_mean"].mean()),
            "healthy_fma_std_across_patients": float(df["healthy_fma_mean"].std()),
            "impaired_fma_mean_across_patients": float(df["impaired_fma_mean"].mean()),
            "impaired_fma_std_across_patients": float(df["impaired_fma_mean"].std()),
            "impaired_fma_min": float(df["impaired_fma_mean"].min()),
            "impaired_fma_max": float(df["impaired_fma_mean"].max()),
        },
        "correlations": tests,
        "stratified": strat.reset_index().astype(object).to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
