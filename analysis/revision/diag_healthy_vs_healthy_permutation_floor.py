"""
Diagnostic: what does oracle channel permutation recover when there's no biology?

Train on half of healthy_01 cal, test on the other half. Same arm, same session,
no motor reorganization possible — any accuracy the "best permutation" beats
the identity permutation by is 100% noise (oracle selecting against test set).

That number is the floor for interpreting R-C1's 42% recovery on cross-arm.

If the healthy-vs-healthy floor is small (~5-10%), R-C1's 42% is signal (biology
or mismatch). If it's close to 42%, R-C1's 42% is mostly noise.
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

from analysis.seed import SEED, seed_everything
from analysis.physiomio.per_session_eval import PHYSIOMIO_PKL, GRABMYO_META, TEST_PER_CLASS
from analysis.revision.recompute_leakage_free_zscore import engineer_features_leakage_free
from analysis.revision.recompute_A_lda_at_cal_sizes import split_at

CAL_SIZE = 36
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
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    t0 = time.time()

    print("Loading raw PhysioMio...")
    df = pd.read_pickle(PHYSIOMIO_PKL).reset_index(drop=True)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]

    # Determine cal_mask from healthy_01 splits (train half only for stats)
    print("Building splits: healthy_01 → cal (train half) + test (other half)...")
    cal_mask = pd.Series(False, index=df.index)
    per_patient = {}
    for patient in sorted(df.participant.unique(), key=lambda s: int(s.replace("patient", ""))):
        s_hlth = df[(df.participant == patient) & (df.session == "healthy_01")]
        if len(s_hlth) == 0:
            continue
        try:
            rng_p = np.random.RandomState(SEED)
            # Use split_at exactly as in cross-arm: cal_per_gesture=36 (train half), test_per_class=39
            test_idx, cal_idx = split_at(s_hlth, CAL_SIZE, TEST_PER_CLASS, rng_p)
        except Exception:
            continue
        if len(test_idx) < 15 or len(cal_idx) < 6:
            continue
        cal_mask.loc[cal_idx] = True
        per_patient[patient] = {"train": list(cal_idx), "test": list(test_idx)}
    print(f"  {len(per_patient)} patients with valid healthy_01 splits")

    print("Engineering features (leakage-free)...")
    df_eng = engineer_features_leakage_free(df.copy(), cal_mask)

    all_perms = list(permutations(CANONICAL))
    identity_perm = tuple(CANONICAL)

    rows = []
    patient_list = sorted(per_patient.keys(), key=lambda s: int(s.replace("patient", "")))
    for pi, patient in enumerate(patient_list, 1):
        train_idx = per_patient[patient]["train"]
        test_idx = per_patient[patient]["test"]

        y_train = df_eng.loc[train_idx, "intent_idx"].values.astype(np.int64)
        y_test = df_eng.loc[test_idx, "intent_idx"].values.astype(np.int64)
        X_train_df = df_eng.loc[train_idx, feature_cols].fillna(0)
        X_test_df = df_eng.loc[test_idx, feature_cols].fillna(0)
        X_test = X_test_df.values.astype(np.float32)

        perm_accs = []
        for perm in all_perms:
            X_perm = permute_channels(X_train_df, feature_cols, perm)
            perm_accs.append((perm, fit_and_score(X_perm, y_train, X_test, y_test)))

        acc_by = {p: a for p, a in perm_accs}
        acc_id = acc_by[identity_perm]
        valid = [a for _, a in perm_accs if not np.isnan(a)]
        acc_best = max(valid)
        acc_worst = min(valid)
        rows.append({
            "patient": patient,
            "acc_identity": acc_id,
            "acc_best_perm": acc_best,
            "acc_worst_perm": acc_worst,
            "delta_best_minus_id": acc_best - acc_id,
        })
        elapsed = time.time() - t0
        print(f"[{pi}/{len(patient_list)}] {patient}  id={acc_id:.4f}  best={acc_best:.4f}  "
              f"delta={acc_best - acc_id:+.4f}  [{elapsed/60:.1f}min]", flush=True)

    out = pd.DataFrame(rows)
    id_mean = out.acc_identity.mean()
    best_mean = out.acc_best_perm.mean()
    delta_mean = out.delta_best_minus_id.mean()

    # Interpret against R-C1 numbers
    # R-C1 cross-arm: identity 0.639, best 0.746, delta 0.107, gap-to-own-cal 0.257, recovery 42%
    rc1_delta = 0.107
    print(f"\n=== HEALTHY-vs-HEALTHY PERMUTATION FLOOR (n={len(out)}) ===")
    print(f"Identity permutation acc:  {id_mean:.4f}")
    print(f"Best permutation acc:      {best_mean:.4f}")
    print(f"Delta (best − identity):   {delta_mean:+.4f}")
    print(f"")
    print(f"For comparison, R-C1 (cross-arm) delta was: {rc1_delta:+.4f}")
    print(f"Ratio floor / R-C1 delta:  {delta_mean / rc1_delta:.2%}")
    print(f"")
    print(f"Interpretation:")
    print(f"- If floor ≈ 0.01-0.02: R-C1's 0.107 delta is real signal (biology / mismatch)")
    print(f"- If floor ≈ 0.05+: some of R-C1's recovery is oracle noise")
    print(f"- If floor ≈ 0.107: R-C1's 'recovery' is essentially all noise")


if __name__ == "__main__":
    main()
