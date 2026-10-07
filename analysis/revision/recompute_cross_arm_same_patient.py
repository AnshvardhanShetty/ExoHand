"""
Revision — same-patient cross-arm test.

Sharpens the mechanism story: is "per-patient" the axis of specificity, or is
"impaired-arm" the axis? Every PhysioMio patient has both healthy and impaired
arm sessions. This test probes:

For each patient:
  - Train HGB on the target patient's OWN healthy-arm cal (healthy_01)
  - Test on their impaired-arm test set (impaired_01, balanced 39/39/39)

Compare against three references:
  - Zero-shot (GrabMyo only, no per-patient data)                  ~0.35
  - LOPO (47 other patients' impaired-arm cal, no target patient)  ~0.67 (running)
  - Per-session cal (this patient's own impaired-arm cal)          ~0.88

Predicted outcomes and interpretations:

  If cross-arm ≈ per-session (~0.80+):
    Per-patient info transfers across arms. "Patient" is the specificity axis.
    Strongest support for per-patient-independent-task story.

  If cross-arm ≈ LOPO (~0.65):
    Patient-specific info transfers partially; arm-specific factors matter as
    much as patient-specific ones. Nuanced.

  If cross-arm ≈ zero-shot (~0.35):
    Impaired-arm EMG is a fundamentally different distribution from healthy-arm,
    even in the SAME patient. "Arm" is the specificity axis, not "patient".
    Would revise mechanism story: "impaired-arm EMG is a distinct distribution,
    healthy-population priors (GrabMyo) cannot inform it."

Resumable: skips patients already in the CSV.

Outputs:
  analysis/revision/results/cross_arm_same_patient_per_patient.csv
  analysis/revision/results/cross_arm_same_patient_summary.md
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

from analysis.seed import SEED, seed_everything
from ml.train_hgb_v2 import engineer_features
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, GRABMYO_META, GRABMYO_CACHE, TEST_PER_CLASS,
    CAL_WEIGHT, split_session,
)

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "cross_arm_same_patient_per_patient.csv"
OUT_MD = OUT_DIR / "cross_arm_same_patient_summary.md"


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

    # ── Find patients with both healthy_01 and impaired_01 ──
    session_pairs = (
        eng[eng.session.isin(["healthy_01", "impaired_01"])]
        .groupby("participant").session.nunique()
        .pipe(lambda s: s[s == 2].index.tolist())
    )
    patients = sorted(session_pairs, key=lambda s: int(s.replace("patient", "")))
    print(f"Patients with both healthy_01 and impaired_01: {len(patients)}")

    # ── Resume ──
    rows = []
    done = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done = set(existing["patient"].tolist())
        print(f"Resume: {len(rows)} rows, {len(done)} patients done")

    for pi, patient in enumerate(patients, 1):
        if patient in done:
            continue

        # Impaired arm (target)
        s_imp = eng[(eng.participant == patient) & (eng.session == "impaired_01")]
        rng_local = np.random.RandomState(SEED)
        try:
            test_idx, imp_cal_idx, _ = split_session(s_imp, TEST_PER_CLASS, rng_local)
        except Exception:
            continue
        if len(test_idx) < 15 or len(imp_cal_idx) < 6:
            continue

        X_test = s_imp.loc[test_idx, gm_features].fillna(0).values.astype(np.float32)
        y_test = s_imp.loc[test_idx, "intent_idx"].values.astype(np.int64)
        # Impaired-arm own cal (baseline reference)
        X_imp_cal = s_imp.loc[imp_cal_idx, gm_features].fillna(0).values.astype(np.float32)
        y_imp_cal = s_imp.loc[imp_cal_idx, "intent_idx"].values.astype(np.int64)

        # Healthy arm (source of interest)
        s_hlth = eng[(eng.participant == patient) & (eng.session == "healthy_01")]
        rng_local2 = np.random.RandomState(SEED + 1)
        try:
            _, hlth_cal_idx, _ = split_session(s_hlth, TEST_PER_CLASS, rng_local2)
        except Exception:
            continue
        if len(hlth_cal_idx) < 6:
            continue

        X_hlth_cal = s_hlth.loc[hlth_cal_idx, gm_features].fillna(0).values.astype(np.float32)
        y_hlth_cal = s_hlth.loc[hlth_cal_idx, "intent_idx"].values.astype(np.int64)

        # ── Arm A: cross-arm cal-only (healthy-arm cal → impaired-arm test) ──
        try:
            if len(np.unique(y_hlth_cal)) < 2:
                cross_arm_po = np.nan
            else:
                sc_a = StandardScaler().fit(X_hlth_cal)
                clf_a = make_hgb().fit(sc_a.transform(X_hlth_cal), y_hlth_cal)
                cross_arm_po = float(accuracy_score(y_test, clf_a.predict(sc_a.transform(X_test))))
        except Exception as e:
            print(f"  {patient}: cross-arm PO failed ({e})", flush=True)
            cross_arm_po = np.nan

        # ── Arm B: cross-arm GrabMyo + healthy-arm cal → impaired-arm test ──
        try:
            X_all = np.vstack([gm_X, X_hlth_cal])
            y_all = np.concatenate([gm_y, y_hlth_cal])
            w = np.ones(len(X_all), dtype=np.float32)
            w[len(gm_X):] = CAL_WEIGHT
            sc_b = StandardScaler().fit(X_all)
            clf_b = make_hgb().fit(sc_b.transform(X_all), y_all, sample_weight=w)
            cross_arm_gm = float(accuracy_score(y_test, clf_b.predict(sc_b.transform(X_test))))
        except Exception as e:
            print(f"  {patient}: cross-arm GM+cal failed ({e})", flush=True)
            cross_arm_gm = np.nan

        # ── Reference: impaired-arm own cal (baseline; expected ~0.88) ──
        try:
            if len(np.unique(y_imp_cal)) < 2:
                imp_own_acc = np.nan
            else:
                sc_c = StandardScaler().fit(X_imp_cal)
                clf_c = make_hgb().fit(sc_c.transform(X_imp_cal), y_imp_cal)
                imp_own_acc = float(accuracy_score(y_test, clf_c.predict(sc_c.transform(X_test))))
        except Exception:
            imp_own_acc = np.nan

        rows.append({
            "patient": patient,
            "n_hlth_cal": len(X_hlth_cal),
            "n_imp_cal": len(X_imp_cal),
            "n_test": len(X_test),
            "cross_arm_po_acc": cross_arm_po,           # healthy-arm cal → impaired-arm test
            "cross_arm_gm_acc": cross_arm_gm,           # healthy-arm cal + GrabMyo → impaired-arm test
            "imp_own_cal_acc": imp_own_acc,             # baseline: impaired-arm own cal
        })

        elapsed = time.time() - t0
        eta = elapsed / (pi - len(done) if pi > len(done) else 1) * (len(patients) - pi)
        print(f"[{pi}/{len(patients)}] {patient}: "
              f"cross-arm PO={cross_arm_po:.4f}  cross-arm GM+cal={cross_arm_gm:.4f}  "
              f"imp-own={imp_own_acc:.4f}  [{elapsed/60:.1f}min, eta {eta/60:.0f}min]",
              flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        print("No rows collected. Aborting.")
        return

    cross_po_mean = out["cross_arm_po_acc"].mean()
    cross_gm_mean = out["cross_arm_gm_acc"].mean()
    imp_own_mean = out["imp_own_cal_acc"].mean()

    # Reference numbers from prior recomputes
    zero_shot_ref = 0.346          # zero_shot_balanced_summary.md (PhysioMio)
    lopo_ref = 0.67                # from LOPO prediction / early data

    md = [
        "# Cross-arm same-patient generalization — sharpening the mechanism story",
        "",
        f"For each of {len(out)} PhysioMio patients with both healthy_01 and impaired_01:",
        "  - Train HGB on OWN healthy-arm cal, test on impaired-arm balanced test set",
        "  - Two arms: cal-only (Arm A), GrabMyo + healthy-arm cal (Arm B)",
        "  - Baseline: own impaired-arm cal (Arm C, expected ~0.88)",
        "",
        "## Results",
        "",
        "| Regime | Mean accuracy | Reference |",
        "|---|---:|---|",
        f"| **Zero-shot** (GrabMyo only, no per-patient data) | — | 0.346 (recompute #1) |",
        f"| **LOPO** (47 other patients' impaired-arm cal) | — | ~0.67 (early LOPO data) |",
        f"| **Cross-arm PO** (this patient's healthy-arm cal → impaired-arm test) | **{cross_po_mean:.4f}** | this experiment |",
        f"| **Cross-arm GM+cal** (this patient's healthy-arm cal + GrabMyo) | **{cross_gm_mean:.4f}** | this experiment |",
        f"| **Impaired-arm own cal** (baseline) | **{imp_own_mean:.4f}** | this experiment |",
        f"| **Per-session cal** (impaired arm, cal-size sweep reference) | — | 0.878 |",
        "",
        "## Interpretation",
        "",
        "**If cross-arm ≈ per-session cal (~0.80+):** 'Patient' is the specificity axis.",
        "Per-patient info transfers across arms. Strongest support for per-patient-",
        "independent-task story.",
        "",
        "**If cross-arm ≈ LOPO (~0.65):** Both patient and arm axes matter about equally.",
        "The healthy arm carries some patient-specific info but not enough to fully",
        "substitute for impaired-arm calibration.",
        "",
        "**If cross-arm ≈ zero-shot (~0.35):** 'Arm' is the specificity axis, not patient.",
        "Impaired-arm EMG is a distinct distribution even from the SAME patient's healthy",
        "arm. Would revise mechanism story: it's impaired-arm-specific, not per-patient",
        "specificity per se.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(f"Cross-arm PO:     {cross_po_mean:.4f}")
    print(f"Cross-arm GM+cal: {cross_gm_mean:.4f}")
    print(f"Imp-arm own cal:  {imp_own_mean:.4f}")
    print(f"References: zero-shot 0.35, LOPO ~0.67, per-session 0.88")


if __name__ == "__main__":
    main()
