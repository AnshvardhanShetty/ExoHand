"""
Revision — leave-one-patient-out (LOPO) cross-patient generalization test.

Direct test of the "per-patient EMG is an independent task" hypothesis:

For each of 48 PhysioMio patients:
  - Pool cal data from the OTHER 47 patients (all their impaired_01 cal windows)
  - Train HGB on that pooled cal:
      Arm A: cal-only (no GrabMyo)
      Arm B: GrabMyo + pooled cal
  - Test on the held-out patient's balanced test set

Compare against:
  - Per-session baseline: this patient's own cal → their own test (~0.86)
  - Zero-shot baseline: GrabMyo only, no per-patient cal (~0.35 on balanced test)

Predicted outcome (if per-patient specificity is the mechanism):
  - LOPO cal-only << per-session cal-only (huge drop, near zero-shot)
  - LOPO GM+cal << per-session GM+cal (also huge drop)
  → Confirms per-patient decision boundaries are largely independent tasks.

If LOPO is close to per-session, per-patient specificity is WRONG and we need to revise.

Resumable: skips patients already in the CSV.

Outputs:
  analysis/revision/results/lopo_cross_patient_per_patient.csv
  analysis/revision/results/lopo_cross_patient_summary.md
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
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, GRABMYO_META, GRABMYO_CACHE, TEST_PER_CLASS,
    CAL_WEIGHT, CLASSES, split_session,
)

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "lopo_cross_patient_per_patient.csv"
OUT_MD = OUT_DIR / "lopo_cross_patient_summary.md"


def make_hgb(seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    t0 = time.time()

    print("Loading + engineering PhysioMio...")
    df = pd.read_pickle(PHYSIOMIO_PKL)
    eng = engineer_features(df)

    with open(GRABMYO_META) as f:
        gm_meta = json.load(f)
    gm_features = gm_meta["feature_cols"]

    print("Loading GrabMyo (300k subsample)...")
    gm = pd.read_pickle(GRABMYO_CACHE).sample(300_000, random_state=SEED)
    gm_X = gm[gm_features].values.astype(np.float32)
    gm_y = gm["intent_idx"].values.astype(np.int64)
    print(f"  GrabMyo: {len(gm_X):,} × {len(gm_features)}")

    # ── Collect per-patient cal + test blocks (impaired_01 for consistency) ──
    print("\nExtracting per-patient cal + test blocks (impaired_01)...")
    patients = sorted(eng["participant"].unique(), key=lambda s: int(s.replace("patient", "")))
    per_patient = {}
    for patient in patients:
        s01 = eng[(eng.participant == patient) & (eng.session == "impaired_01")]
        if len(s01) == 0:
            continue
        try:
            test_idx, cal_idx, _ = split_session(s01, TEST_PER_CLASS, rng)
        except Exception:
            continue
        if len(test_idx) < 15 or len(cal_idx) < 6:
            continue
        per_patient[patient] = {
            "X_cal": s01.loc[cal_idx, gm_features].fillna(0).values.astype(np.float32),
            "y_cal": s01.loc[cal_idx, "intent_idx"].values.astype(np.int64),
            "X_test": s01.loc[test_idx, gm_features].fillna(0).values.astype(np.float32),
            "y_test": s01.loc[test_idx, "intent_idx"].values.astype(np.int64),
        }
    print(f"  Kept {len(per_patient)} patients with valid impaired_01 splits")

    # ── Resume support ──
    rows = []
    done = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done = set(existing["held_out_patient"].tolist())
        print(f"Resume: {len(rows)} rows, {len(done)} patients done")

    # ── Iterate: hold each patient out, train on the others' cal ──
    patient_list = sorted(per_patient.keys(), key=lambda s: int(s.replace("patient", "")))
    for i, held_out in enumerate(patient_list, 1):
        if held_out in done:
            continue

        # Pool cal from other 47 patients
        X_cal_pool = np.vstack([per_patient[p]["X_cal"] for p in patient_list if p != held_out])
        y_cal_pool = np.concatenate([per_patient[p]["y_cal"] for p in patient_list if p != held_out])

        # Test on held-out patient
        X_test = per_patient[held_out]["X_test"]
        y_test = per_patient[held_out]["y_test"]

        # ── Arm A: LOPO cal-only ──
        try:
            sc_a = StandardScaler().fit(X_cal_pool)
            clf_a = make_hgb().fit(sc_a.transform(X_cal_pool), y_cal_pool)
            po_acc = float(accuracy_score(y_test, clf_a.predict(sc_a.transform(X_test))))
        except Exception as e:
            print(f"  {held_out}: LOPO cal-only failed ({e})", flush=True)
            po_acc = np.nan

        # ── Arm B: LOPO GrabMyo + cal ──
        try:
            X_all = np.vstack([gm_X, X_cal_pool])
            y_all = np.concatenate([gm_y, y_cal_pool])
            w = np.ones(len(X_all), dtype=np.float32)
            w[len(gm_X):] = CAL_WEIGHT
            sc_b = StandardScaler().fit(X_all)
            clf_b = make_hgb().fit(sc_b.transform(X_all), y_all, sample_weight=w)
            gm_acc = float(accuracy_score(y_test, clf_b.predict(sc_b.transform(X_test))))
        except Exception as e:
            print(f"  {held_out}: LOPO GM+cal failed ({e})", flush=True)
            gm_acc = np.nan

        rows.append({
            "held_out_patient": held_out,
            "n_cal_pool": len(X_cal_pool),
            "n_test": len(X_test),
            "lopo_po_acc": po_acc,
            "lopo_gm_acc": gm_acc,
        })
        elapsed = time.time() - t0
        eta = elapsed / max(1, len(rows) - len(done)) * (len(patient_list) - len(rows))
        print(f"[{i}/{len(patient_list)}] held-out {held_out}: "
              f"LOPO PO={po_acc:.4f}  LOPO GM+cal={gm_acc:.4f}  "
              f"[{elapsed/60:.1f}min, eta {eta/60:.0f}min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        print("No rows collected. Aborting.")
        return

    # ── Reference numbers from prior recomputes ──
    zero_shot_ref = 0.346          # zero_shot_balanced_summary.md (PhysioMio)
    persession_po_ref = 0.878      # cal-size sweep at cal=36, PO
    persession_gm_ref = 0.860      # per_session_results.csv, GM+cal

    lopo_po_mean = out["lopo_po_acc"].mean()
    lopo_gm_mean = out["lopo_gm_acc"].mean()
    lopo_po_med = out["lopo_po_acc"].median()
    lopo_gm_med = out["lopo_gm_acc"].median()

    # Paired Wilcoxon: does GM+cal help vs cal-only in the LOPO regime?
    valid = ~out["lopo_po_acc"].isna() & ~out["lopo_gm_acc"].isna()
    w_lopo = wilcoxon(out.loc[valid, "lopo_gm_acc"], out.loc[valid, "lopo_po_acc"],
                       alternative="greater")

    md = [
        "# LOPO cross-patient generalization — testing per-patient specificity",
        "",
        f"Leave-one-patient-out on {len(out)} PhysioMio impaired_01 sessions.",
        "For each held-out patient, pool cal data from the other 47 patients and train HGB.",
        "Two arms: cal-only (Arm A), GrabMyo+pooled_cal (Arm B). Test on held-out patient.",
        "",
        "## Results",
        "",
        "| Regime | Mean acc | Median acc | Reference |",
        "|---|---:|---:|---|",
        f"| **Zero-shot** (no per-patient data at all) | — | — | 0.346 (recompute #1) |",
        f"| **LOPO cal-only** (47 other patients' cal, no GrabMyo) | **{lopo_po_mean:.4f}** | {lopo_po_med:.4f} | — |",
        f"| **LOPO GrabMyo + pooled cal** (47 other patients' cal + GrabMyo) | **{lopo_gm_mean:.4f}** | {lopo_gm_med:.4f} | — |",
        f"| **Per-session cal-only** (this patient's own cal) | — | — | 0.878 (cal-size sweep) |",
        f"| **Per-session GrabMyo+cal** (this patient's own cal) | — | — | 0.860 (per_session_results) |",
        "",
        f"**Paired Wilcoxon (LOPO GM+cal > LOPO cal-only): p = {w_lopo.pvalue:.3e}**",
        "",
        "## Interpretation",
        "",
        "**If LOPO is close to zero-shot** (say ≤0.45), per-patient specificity is confirmed:",
        "even 47 other patients' worth of cal data doesn't help patient N, because their",
        "decision boundary is idiosyncratic. This is a direct test of 'per-patient EMG is",
        "an independent task' — the mechanistic reason GrabMyo pretraining doesn't help.",
        "",
        "**If LOPO is close to per-session** (say ≥0.75), per-patient specificity is wrong:",
        "cross-patient data DOES generalize, and the reason GrabMyo doesn't help must be",
        "something else (GrabMyo distribution too different from PhysioMio impaired-arm,",
        "GrabMyo sample-size saturation, etc.).",
        "",
        "**Intermediate** (0.45-0.75) suggests partial per-patient specificity plus a",
        "usable-but-limited universal component.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(f"LOPO cal-only:    {lopo_po_mean:.4f}")
    print(f"LOPO GrabMyo+cal: {lopo_gm_mean:.4f}")
    print(f"Reference: zero-shot 0.35, per-session cal 0.88")


if __name__ == "__main__":
    main()
