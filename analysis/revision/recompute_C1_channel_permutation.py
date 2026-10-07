"""
Revision — C1: channel-permutation cross-arm test.

Rules out the "same electrode indices ≠ same anatomical location across arms"
confound. For each patient, try all 4! = 24 permutations of which chosen channel
maps to which canonical role (ch0, ch4, ch9, ch13) in the healthy-arm training
data. Feature columns get renamed accordingly. Then train cross-arm PO. Also
tests medial-lateral mirror (swap flexor/extensor pairs).

Baseline: current cross-arm PO with identity permutation.
Report: best permutation accuracy, worst permutation, ID accuracy, gap recovered.

Decision rule (pre-registered): if best permutation recovers >⅓ of the 33 pp
gap → mounting is partly the confound; report corrected number as headline.

Outputs:
  analysis/revision/results/C1_channel_permutation_per_patient.csv
  analysis/revision/results/C1_channel_permutation_summary.md
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
from ml.train_hgb_v2 import engineer_features
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, GRABMYO_META, TEST_PER_CLASS, split_session,
)

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "C1_channel_permutation_per_patient.csv"
OUT_MD = OUT_DIR / "C1_channel_permutation_summary.md"

CANONICAL = [0, 4, 9, 13]   # ch0, ch4, ch9, ch13


def make_hgb(seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def permute_channels(X_df, feature_cols, perm):
    """Permute the columns so that features under ch{CANONICAL[i]}_* pick up values
    from ch{perm[i]}_* in the underlying dataframe.

    Handles both single-channel features (`chN_feat`) and cross-channel features
    (`chI_chJ_feat`) — both channel IDs are remapped consistently.
    """
    rename = {CANONICAL[i]: perm[i] for i in range(len(CANONICAL))}

    def _remap_one(col):
        # Try two-channel prefix first: ch<I>_ch<J>_<rest>
        parts = col.split("_")
        if (len(parts) >= 3 and parts[0].startswith("ch") and parts[1].startswith("ch")):
            try:
                i_id = int(parts[0][2:])
                j_id = int(parts[1][2:])
            except ValueError:
                pass
            else:
                new_i = rename.get(i_id, i_id)
                new_j = rename.get(j_id, j_id)
                candidate = f"ch{new_i}_ch{new_j}_" + "_".join(parts[2:])
                return candidate if candidate in _column_set else col
        # Single-channel prefix: ch<I>_<rest>
        if len(parts) >= 2 and parts[0].startswith("ch"):
            try:
                ch_id = int(parts[0][2:])
            except ValueError:
                return col
            new_id = rename.get(ch_id, ch_id)
            candidate = f"ch{new_id}_" + "_".join(parts[1:])
            return candidate if candidate in _column_set else col
        return col

    _column_set = set(X_df.columns)
    new_cols = [_remap_one(c) for c in feature_cols]
    return X_df[new_cols].values.astype(np.float32)


def fit_score(X_train, y_train, X_test, y_test):
    if len(np.unique(y_train)) < 2:
        return np.nan
    sc = StandardScaler().fit(X_train)
    clf = make_hgb().fit(sc.transform(X_train), y_train)
    return float(accuracy_score(y_test, clf.predict(sc.transform(X_test))))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    t0 = time.time()

    print("Loading + engineering PhysioMio...")
    df = pd.read_pickle(PHYSIOMIO_PKL)
    eng = engineer_features(df)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]

    # Enumerate all 24 permutations of the 4 canonical channels
    all_perms = list(permutations(CANONICAL))
    # Also add medial-lateral mirror (swap flexor pair with extensor pair):
    # canonical order [ch0=flex1, ch4=ext1, ch9=flex2, ch13=ext2] → mirror = [ch4, ch0, ch13, ch9]
    mirror_perm = tuple([CANONICAL[1], CANONICAL[0], CANONICAL[3], CANONICAL[2]])
    identity_perm = tuple(CANONICAL)

    # Resume
    rows = []
    done = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done = set(existing["patient"].tolist())
        print(f"Resume: {len(rows)} rows, {len(done)} done")

    patients = sorted(eng["participant"].unique(), key=lambda s: int(s.replace("patient", "")))
    for pi, patient in enumerate(patients, 1):
        if patient in done:
            continue

        s_imp = eng[(eng.participant == patient) & (eng.session == "impaired_01")]
        s_hlth = eng[(eng.participant == patient) & (eng.session == "healthy_01")]
        if len(s_imp) == 0 or len(s_hlth) == 0:
            continue
        try:
            rng_p = np.random.RandomState(SEED)
            test_idx, _, _ = split_session(s_imp, TEST_PER_CLASS, rng_p)
            rng_p2 = np.random.RandomState(SEED + 1)
            _, hlth_cal_idx, _ = split_session(s_hlth, TEST_PER_CLASS, rng_p2)
        except Exception:
            continue
        if len(test_idx) < 15 or len(hlth_cal_idx) < 6:
            continue

        # Test set: impaired arm, NEVER permuted
        X_test = s_imp.loc[test_idx, feature_cols].fillna(0).values.astype(np.float32)
        y_test = s_imp.loc[test_idx, "intent_idx"].values.astype(np.int64)
        y_hlth = s_hlth.loc[hlth_cal_idx, "intent_idx"].values.astype(np.int64)
        X_hlth_df = s_hlth.loc[hlth_cal_idx, feature_cols].fillna(0)

        # Sweep all permutations, score each
        perm_accs = []
        for perm in all_perms:
            X_perm = permute_channels(X_hlth_df, feature_cols, perm)
            acc = fit_score(X_perm, y_hlth, X_test, y_test)
            perm_accs.append((perm, acc))

        # Identify best/worst/identity/mirror
        acc_by_perm = {p: a for p, a in perm_accs}
        acc_id = acc_by_perm[identity_perm]
        acc_best = max(a for _, a in perm_accs if not np.isnan(a))
        acc_worst = min(a for _, a in perm_accs if not np.isnan(a))
        best_perm = [p for p, a in perm_accs if a == acc_best][0]
        acc_mirror = acc_by_perm.get(mirror_perm, np.nan)

        rows.append({
            "patient": patient,
            "acc_identity": acc_id,
            "acc_best_perm": acc_best,
            "acc_worst_perm": acc_worst,
            "acc_mirror_perm": acc_mirror,
            "best_perm": str(best_perm),
            "n_test": len(y_test),
        })

        elapsed = time.time() - t0
        eta = elapsed / max(1, pi - len(done)) * (len(patients) - pi)
        print(f"[{pi}/{len(patients)}] {patient}: id={acc_id:.4f}  best={acc_best:.4f}  "
              f"mirror={acc_mirror:.4f}  [{elapsed/60:.1f}min eta {eta/60:.0f}min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return

    # Reference: cross-arm own-cal baseline (identity permutation should reproduce it)
    imp_own_ref = 0.875   # from cross-arm same-patient results

    id_mean = out.acc_identity.mean()
    best_mean = out.acc_best_perm.mean()
    mirror_mean = out.acc_mirror_perm.mean()
    gap_baseline = imp_own_ref - id_mean            # ~33 pp
    gap_after_best = imp_own_ref - best_mean
    recovered_pp = (id_mean - best_mean) if best_mean < id_mean else (best_mean - id_mean)
    gap_recovered_frac = recovered_pp / gap_baseline if gap_baseline > 0 else 0

    md = [
        "# C1 — channel-permutation cross-arm test",
        "",
        f"n = {len(out)} patients. Cross-arm PO tried across all 24 channel permutations",
        "of the 4 canonical channels (ch0, ch4, ch9, ch13). Baseline = identity permutation.",
        "",
        "## Results",
        "",
        "| Config | Mean acc | Median acc |",
        "|---|---:|---:|",
        f"| Identity (baseline) | {id_mean:.4f} | {out.acc_identity.median():.4f} |",
        f"| Best permutation (per patient) | {best_mean:.4f} | {out.acc_best_perm.median():.4f} |",
        f"| Worst permutation | {out.acc_worst_perm.mean():.4f} | {out.acc_worst_perm.median():.4f} |",
        f"| Medial-lateral mirror | {mirror_mean:.4f} | {out.acc_mirror_perm.median():.4f} |",
        f"| Impaired-arm own cal (reference) | 0.875 | — |",
        "",
        f"Cross-arm identity baseline gap from own cal: {gap_baseline:+.4f}",
        f"Cross-arm best-perm gap from own cal: {gap_after_best:+.4f}",
        f"Fraction of gap recovered by best permutation: {gap_recovered_frac:.2%}",
        "",
        "## Decision (pre-registered)",
        "",
        "- If best permutation recovers > ⅓ of the 33 pp gap (i.e., > 11 pp):",
        "  the gap is partly a channel-mounting artefact.",
        "  Report corrected best-perm number as the headline.",
        "- If recovery < ⅓: mounting is not the confound; original claim survives.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
