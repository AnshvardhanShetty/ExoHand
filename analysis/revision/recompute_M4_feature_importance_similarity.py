"""
Revision — M4: pairwise feature-importance similarity across patients/sessions.

Post-hoc analysis on mechanism_at_scale_importance_vectors.parquet.
For each pair of sessions (calibrated model), compute Spearman ρ between
their feature-importance rankings. Low mean pairwise ρ → each session's model
depends on a different feature subset → per-patient/per-session specificity in
feature space.

Also computes the same for GrabMyo-only model as a control (should be highly
similar across sessions, since the model is the same).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
IN = OUT_DIR / "mechanism_at_scale_importance_vectors.parquet"
OUT_MD = OUT_DIR / "M4_feature_importance_similarity_summary.md"


def pairwise_spearman_matrix(matrix):
    """matrix: (N, F). Return upper-triangle Spearman ρ between rows."""
    N = matrix.shape[0]
    rhos = []
    for i in range(N):
        for j in range(i + 1, N):
            rho, _ = spearmanr(matrix[i], matrix[j])
            if np.isfinite(rho):
                rhos.append(rho)
    return np.array(rhos)


def main():
    if not IN.exists():
        print(f"Missing {IN}")
        return
    df = pd.read_parquet(IN)
    feature_cols = [c for c in df.columns if c not in ("participant", "session", "model")]

    md = ["# M4 — pairwise feature-importance similarity across sessions", ""]

    for model_name in df.model.unique():
        sub = df[df.model == model_name]
        matrix = sub[feature_cols].values
        rhos = pairwise_spearman_matrix(matrix)
        md += [
            f"## Model: {model_name}",
            "",
            f"- Sessions: {len(sub)}, feature dim: {len(feature_cols)}",
            f"- Pairwise ρ (n_pairs = {len(rhos)}): mean = {rhos.mean():+.3f}, median = {np.median(rhos):+.3f}, std = {rhos.std():.3f}",
            f"- Fraction of pairs with ρ > 0.5: {(rhos > 0.5).mean():.2%}",
            f"- Fraction of pairs with ρ > 0.8: {(rhos > 0.8).mean():.2%}",
            "",
        ]

    md += [
        "## Interpretation",
        "",
        "- Calibrated model: low pairwise ρ → different sessions rely on different feature",
        "  subsets → per-patient/per-session specificity confirmed in feature space.",
        "- GrabMyo-only model: same model applied to different test sets. Higher pairwise",
        "  ρ expected. If similar to calibrated, importance patterns are noise-dominated.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(open(OUT_MD).read())


if __name__ == "__main__":
    main()
