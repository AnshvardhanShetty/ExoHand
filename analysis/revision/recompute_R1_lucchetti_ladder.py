"""
Revision — R1: Lucchetti replication of the training-source ladder.

Replicates rows 1, 3, 4 of the ICBINB ladder on Lucchetti (n=10 stroke).
Row 2 (cross-arm same-patient) is not replicable — Lucchetti's healthy cohort
(HS_XX) is separate subjects from the stroke cohort (ST_XX), not same-patient
healthy arms.

Rows:
  1. Own impaired arm 22 s cal → own impaired test          (per-session baseline)
  2. [not replicable in Lucchetti]
  3. Other stroke patients' impaired arms cal → held-out    (LOPO)
  3b. Volume-matched LOPO variant                            (matches per-session cal size)
  4. GrabMyo (43 healthy) zero-shot                          (already computed)

Outputs:
  analysis/revision/results/R1_lucchetti_ladder_per_patient.csv
  analysis/revision/results/R1_lucchetti_ladder_summary.md
"""

import json
import sys
import time
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
from ml.train_hgb_v2 import engineer_features
from analysis.lucchetti.per_session_eval import (
    LUCCHETTI_PKL, GRABMYO_META, TEST_PER_CLASS,
    split_session_lucchetti,
)

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "R1_lucchetti_ladder_per_patient.csv"
OUT_MD = OUT_DIR / "R1_lucchetti_ladder_summary.md"


def make_hgb(seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def stratified_subsample(X, y, n_target, rng):
    classes = np.unique(y)
    n_per_class = max(1, n_target // len(classes))
    keep = []
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) <= n_per_class:
            keep.extend(idx)
        else:
            keep.extend(rng.choice(idx, n_per_class, replace=False))
    return X[np.array(keep)], y[np.array(keep)]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    t0 = time.time()

    print("Loading Lucchetti + engineering...")
    df = pd.read_pickle(LUCCHETTI_PKL)
    eng = engineer_features(df)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]

    # Load pre-computed row 1 and row 4 baselines
    row1_csv = PROJECT_ROOT / "analysis" / "lucchetti" / "results" / "per_session_results.csv"
    row4_csv = PROJECT_ROOT / "analysis" / "lucchetti" / "results" / "zero_shot_per_session.csv"
    row1_df = None
    if row1_csv.exists():
        r1 = pd.read_csv(row1_csv)
        if "arm" in r1.columns:
            r1 = r1[r1.arm == "impaired"]
        row1_df = r1.groupby("participant").acc.mean().reset_index().rename(
            columns={"acc": "own_cal_acc"}
        )
    row4_df = None
    if row4_csv.exists():
        r4 = pd.read_csv(row4_csv)
        if "arm" in r4.columns:
            r4 = r4[r4.arm == "impaired"]
        row4_df = r4.groupby("participant").acc.mean().reset_index().rename(
            columns={"acc": "zero_shot_acc"}
        )

    # Extract per-patient impaired-arm cal + test blocks
    patients = sorted(eng["participant"].unique())
    stroke_patients = [p for p in patients if p.startswith("ST_")]
    per_patient = {}
    for patient in stroke_patients:
        s = eng[(eng.participant == patient) & (eng.session == "impaired_01")]
        if len(s) == 0:
            continue
        try:
            rng_p = np.random.RandomState(SEED)
            test_idx, cal_idx, _ = split_session_lucchetti(s, TEST_PER_CLASS, rng_p)
        except Exception as e:
            print(f"  {patient}: split failed ({e})")
            continue
        if len(test_idx) < 15 or len(cal_idx) < 6:
            continue
        per_patient[patient] = {
            "X_cal": s.loc[cal_idx, feature_cols].fillna(0).values.astype(np.float32),
            "y_cal": s.loc[cal_idx, "intent_idx"].values.astype(np.int64),
            "X_test": s.loc[test_idx, feature_cols].fillna(0).values.astype(np.float32),
            "y_test": s.loc[test_idx, "intent_idx"].values.astype(np.int64),
        }
    print(f"Kept {len(per_patient)} stroke patients with valid impaired_01 splits")

    rows = []
    patient_list = sorted(per_patient.keys())
    for i, held_out in enumerate(patient_list, 1):
        pat = per_patient[held_out]

        # Row 3: LOPO cal-only (pool others' cal, full volume)
        others_X = np.vstack([per_patient[p]["X_cal"] for p in patient_list if p != held_out])
        others_y = np.concatenate([per_patient[p]["y_cal"] for p in patient_list if p != held_out])
        try:
            sc = StandardScaler().fit(others_X)
            clf = make_hgb().fit(sc.transform(others_X), others_y)
            row3_acc = float(accuracy_score(pat["y_test"], clf.predict(sc.transform(pat["X_test"]))))
        except Exception as e:
            row3_acc = np.nan

        # Row 3b: volume-matched LOPO
        n_target = len(pat["X_cal"])
        rng_p = np.random.RandomState(abs(hash(held_out)) & 0xffffffff)
        X_sub, y_sub = stratified_subsample(others_X, others_y, n_target, rng_p)
        try:
            sc = StandardScaler().fit(X_sub)
            clf = make_hgb().fit(sc.transform(X_sub), y_sub)
            row3b_acc = float(accuracy_score(pat["y_test"], clf.predict(sc.transform(pat["X_test"]))))
        except Exception as e:
            row3b_acc = np.nan

        rows.append({
            "patient": held_out,
            "n_cal": len(pat["X_cal"]),
            "n_test": len(pat["y_test"]),
            "row3_lopo_full_acc": row3_acc,
            "row3b_lopo_volumematched_acc": row3b_acc,
        })

        elapsed = time.time() - t0
        print(f"[{i}/{len(patient_list)}] {held_out}  "
              f"row3 (full LOPO)={row3_acc:.4f}  row3b (VM-LOPO)={row3b_acc:.4f}  "
              f"[{elapsed/60:.1f}min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    # Merge with row 1 (own cal) and row 4 (zero-shot) if available
    if row1_df is not None:
        out = out.merge(row1_df, left_on="patient", right_on="participant", how="left").drop(
            columns=["participant"], errors="ignore"
        )
    if row4_df is not None:
        out = out.merge(row4_df, left_on="patient", right_on="participant", how="left").drop(
            columns=["participant"], errors="ignore"
        )
    out.to_csv(OUT_CSV, index=False)

    md = [
        "# R1 — Lucchetti replication of the training-source ladder",
        "",
        f"n = {len(out)} stroke patients. Rows 1, 3, 3b replicated. Row 2 (cross-arm same-patient) not applicable.",
        "",
        "## Results",
        "",
        "| Row | Training source | Mean acc | Median acc |",
        "|---|---|---:|---:|",
    ]
    if "own_cal_acc" in out.columns:
        md.append(f"| 1 | Own impaired-arm 22s cal | {out.own_cal_acc.mean():.4f} | {out.own_cal_acc.median():.4f} |")
    if "row3_lopo_full_acc" in out.columns:
        md.append(f"| 3 | LOPO (9 other stroke patients, full pool) | {out.row3_lopo_full_acc.mean():.4f} | {out.row3_lopo_full_acc.median():.4f} |")
    if "row3b_lopo_volumematched_acc" in out.columns:
        md.append(f"| 3b | LOPO volume-matched to per-session cal size | {out.row3b_lopo_volumematched_acc.mean():.4f} | {out.row3b_lopo_volumematched_acc.median():.4f} |")
    if "zero_shot_acc" in out.columns:
        md.append(f"| 4 | Zero-shot GrabMyo (43 healthy) | {out.zero_shot_acc.mean():.4f} | {out.zero_shot_acc.median():.4f} |")
    md.append("")
    md += [
        "## Comparison to PhysioMio (n=48)",
        "",
        "| Row | PhysioMio | Lucchetti | Ordering matches? |",
        "|---|---:|---:|---:|",
        f"| 1 own cal | 0.88 | {out.own_cal_acc.mean():.4f} if avail | — |",
        f"| 3 LOPO full | 0.63 | {out.row3_lopo_full_acc.mean():.4f} | — |",
        f"| 3b LOPO VM | (running) | {out.row3b_lopo_volumematched_acc.mean():.4f} | — |",
        f"| 4 zero-shot | 0.35 | {out.zero_shot_acc.mean() if 'zero_shot_acc' in out.columns else 'NA'} | — |",
        "",
        "## Decision (pre-registered)",
        "",
        "- If the qualitative ordering (per-session cal >> VM-LOPO >> zero-shot) holds on",
        "  Lucchetti, replication is claimed. Independent-cohort confirmation.",
        "- If ordering differs, replication is conditional and stated explicitly.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
