"""
R-C1 — channel permutation on leakage-free features.

For each patient, try all 24 channel permutations + medial-lateral mirror on the
healthy-arm cal features → score on impaired-arm test (identity permutation).

Uses frozen splits + engineer_features_leakage_free. Also computes:
  - oracle-best-perm vs VM-LOPO paired Wilcoxon (retires channel objection)

Outputs:
  analysis/revision/results/R_C1_leakage_free_per_patient.csv
  analysis/revision/results/R_C1_leakage_free_summary.md
"""

import json
import sys
import time
from itertools import permutations
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
from analysis.physiomio.per_session_eval import PHYSIOMIO_PKL, GRABMYO_META
from analysis.revision.recompute_leakage_free_zscore import engineer_features_leakage_free

FROZEN = PROJECT_ROOT / "analysis" / "revision" / "frozen_splits.parquet"
LADDER_CSV = PROJECT_ROOT / "analysis" / "revision" / "results" / "leakage_free_ladder_per_patient.csv"
LEGACY_CSV = PROJECT_ROOT / "analysis" / "revision" / "results" / "C1_channel_permutation_per_patient.csv"
OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "R_C1_leakage_free_per_patient.csv"
OUT_MD = OUT_DIR / "R_C1_leakage_free_summary.md"

CANONICAL = [0, 4, 9, 13]


def make_hgb(seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def permute_channels(X_df, feature_cols, perm):
    rename = {CANONICAL[i]: perm[i] for i in range(len(CANONICAL))}
    column_set = set(X_df.columns)

    def _remap_one(col):
        parts = col.split("_")
        if (len(parts) >= 3 and parts[0].startswith("ch") and parts[1].startswith("ch")):
            try:
                i_id = int(parts[0][2:]); j_id = int(parts[1][2:])
            except ValueError:
                pass
            else:
                new_i = rename.get(i_id, i_id); new_j = rename.get(j_id, j_id)
                cand = f"ch{new_i}_ch{new_j}_" + "_".join(parts[2:])
                return cand if cand in column_set else col
        if len(parts) >= 2 and parts[0].startswith("ch"):
            try:
                ch_id = int(parts[0][2:])
            except ValueError:
                return col
            new_id = rename.get(ch_id, ch_id)
            cand = f"ch{new_id}_" + "_".join(parts[1:])
            return cand if cand in column_set else col
        return col

    new_cols = [_remap_one(c) for c in feature_cols]
    return X_df[new_cols].values.astype(np.float32)


def fit_and_score(X_train, y_train, X_test, y_test):
    if len(np.unique(y_train)) < 2:
        return np.nan
    sc = StandardScaler().fit(X_train)
    clf = make_hgb().fit(sc.transform(X_train), y_train)
    return float(accuracy_score(y_test, clf.predict(sc.transform(X_test))))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    t0 = time.time()

    print("Loading raw PhysioMio...")
    df = pd.read_pickle(PHYSIOMIO_PKL).reset_index(drop=True)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]

    print("Loading frozen splits...")
    frozen = pd.read_parquet(FROZEN)
    cal_mask = pd.Series(False, index=df.index)
    per_patient = {}
    for _, r in frozen.iterrows():
        cal_mask.loc[r["cal_idx"]] = True
        per_patient.setdefault(r["patient"], {})[r["session"]] = {
            "cal_idx": list(r["cal_idx"]),
            "test_idx": list(r["test_idx"]),
        }
    keep_patients = [p for p in per_patient if
                     "impaired_01" in per_patient[p] and "healthy_01" in per_patient[p]]

    print("Engineering features (leakage-free)...")
    df_eng = engineer_features_leakage_free(df.copy(), cal_mask)

    all_perms = list(permutations(CANONICAL))
    identity_perm = tuple(CANONICAL)
    mirror_perm = tuple([CANONICAL[1], CANONICAL[0], CANONICAL[3], CANONICAL[2]])

    rows = []
    done = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done = set(existing.patient.tolist())
        print(f"Resume: {len(rows)} rows, {len(done)} done")

    patient_list = sorted(keep_patients, key=lambda s: int(s.replace("patient", "")))
    for pi, patient in enumerate(patient_list, 1):
        if patient in done:
            continue

        test_idx = per_patient[patient]["impaired_01"]["test_idx"]
        hlth_cal_idx = per_patient[patient]["healthy_01"]["cal_idx"]

        X_test = df_eng.loc[test_idx, feature_cols].fillna(0).values.astype(np.float32)
        y_test = df_eng.loc[test_idx, "intent_idx"].values.astype(np.int64)
        y_hlth = df_eng.loc[hlth_cal_idx, "intent_idx"].values.astype(np.int64)
        X_hlth_df = df_eng.loc[hlth_cal_idx, feature_cols].fillna(0)

        perm_accs = []
        for perm in all_perms:
            X_perm = permute_channels(X_hlth_df, feature_cols, perm)
            perm_accs.append((perm, fit_and_score(X_perm, y_hlth, X_test, y_test)))

        acc_by = {p: a for p, a in perm_accs}
        acc_id = acc_by[identity_perm]
        valid_accs = [a for _, a in perm_accs if not np.isnan(a)]
        acc_best = max(valid_accs)
        acc_worst = min(valid_accs)
        best_perm = [p for p, a in perm_accs if a == acc_best][0]
        acc_mirror = acc_by.get(mirror_perm, np.nan)

        rows.append({
            "patient": patient,
            "acc_identity": acc_id, "acc_best_perm": acc_best,
            "acc_worst_perm": acc_worst, "acc_mirror_perm": acc_mirror,
            "best_perm": str(best_perm),
            "leakage_free": True,
        })
        elapsed = time.time() - t0
        eta = elapsed / max(1, pi - len(done)) * (len(patient_list) - pi)
        print(f"[{pi}/{len(patient_list)}] {patient}: id={acc_id:.4f} best={acc_best:.4f} "
              f"mirror={acc_mirror:.4f}  [{elapsed/60:.1f}min, eta {eta/60:.0f}min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    imp_own_ref = 0.896   # leakage-free own-cal from ladder
    id_mean = out.acc_identity.mean()
    best_mean = out.acc_best_perm.mean()
    gap_baseline = imp_own_ref - id_mean
    gap_after_best = imp_own_ref - best_mean
    gap_recovered_frac = ((id_mean - best_mean) if best_mean < id_mean
                           else (best_mean - id_mean)) / gap_baseline if gap_baseline > 0 else 0

    # Also compute oracle-vs-VM-LOPO Wilcoxon on leakage-free numbers
    ladder = pd.read_csv(LADDER_CSV)
    m = out.merge(ladder[["patient", "row3_vm_lopo"]], on="patient", how="inner")
    w_oracle = wilcoxon(m.row3_vm_lopo, m.acc_best_perm, alternative="greater")

    md = [
        "# R-C1 — channel permutation (leakage-free)",
        "",
        f"n = {len(out)} patients. All 24 permutations of the 4 canonical channels.",
        "",
        "## Results (leakage-free)",
        "",
        "| Config | Mean acc | Median acc |",
        "|---|---:|---:|",
        f"| Identity (baseline) | {id_mean:.4f} | {out.acc_identity.median():.4f} |",
        f"| Best permutation (per patient, oracle) | {best_mean:.4f} | {out.acc_best_perm.median():.4f} |",
        f"| Worst permutation | {out.acc_worst_perm.mean():.4f} | {out.acc_worst_perm.median():.4f} |",
        f"| Medial-lateral mirror | {out.acc_mirror_perm.mean():.4f} | {out.acc_mirror_perm.median():.4f} |",
        f"| Impaired-arm own cal (leakage-free ref) | {imp_own_ref:.4f} | — |",
        "",
        f"Gap (own cal − identity): {gap_baseline:+.4f}",
        f"Gap (own cal − best perm): {gap_after_best:+.4f}",
        f"Fraction of gap recovered: {gap_recovered_frac:.2%}",
        "",
        "## VM-LOPO vs cross-arm-oracle (leakage-free)",
        "",
        f"VM-LOPO (leakage-free): 0.752",
        f"Cross-arm best-perm oracle (this run): {best_mean:.4f}",
        f"Paired Wilcoxon (VM-LOPO > cross-arm-oracle): p = {w_oracle.pvalue:.4e}",
        f"Patients where VM-LOPO > oracle: {(m.row3_vm_lopo > m.acc_best_perm).sum()}/{len(m)}",
        "",
        "## Decision (pre-registered)",
        "",
        "- If recovery > ⅓ of the gap → channel mounting is a real confound. Report",
        "  corrected best-perm number as headline.",
        "- If recovery < ⅓ → mounting is not the confound; original claim survives.",
        "- Legacy (leaky) recovery: 21.11%, below threshold.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
