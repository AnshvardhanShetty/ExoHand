"""
Revision — C2: per-limb normalisation cross-arm test.

Rules out the "healthy vs impaired arm have different amplitude scales" confound.

For each patient, cross-arm PO with three normalization variants:
  V1 (baseline): StandardScaler fit on healthy-arm cal only, applied to both cal and test.
                  This is what our current cross-arm PO does.
  V2 (per-limb z-score): Fit separate StandardScaler on healthy-arm cal AND on impaired-arm test.
                          Both are transformed by their own scaler → both are unit-variance
                          around 0 but in each limb's own reference frame.
  V3 (amplitude-equalised): Rescale healthy-arm feature values so per-feature mean amplitude
                             matches impaired-arm test's per-feature mean. Then fit scaler on
                             rescaled healthy → apply to test.

Decision (pre-registered): if the gap closes below 20 pp under V2 or V3, the story
becomes "scale/SNR mismatch" not "pathology". Report both regardless.

Outputs:
  analysis/revision/results/C2_per_limb_normalisation_per_patient.csv
  analysis/revision/results/C2_per_limb_normalisation_summary.md
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
    PHYSIOMIO_PKL, GRABMYO_META, TEST_PER_CLASS, split_session,
)

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "C2_per_limb_normalisation_per_patient.csv"
OUT_MD = OUT_DIR / "C2_per_limb_normalisation_summary.md"


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
        feature_cols = json.load(f)["feature_cols"]

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
            test_idx, imp_cal_idx, _ = split_session(s_imp, TEST_PER_CLASS, rng_p)
            rng_p2 = np.random.RandomState(SEED + 1)
            _, hlth_cal_idx, _ = split_session(s_hlth, TEST_PER_CLASS, rng_p2)
        except Exception:
            continue
        if len(test_idx) < 15 or len(hlth_cal_idx) < 6 or len(imp_cal_idx) < 6:
            continue

        X_hlth = s_hlth.loc[hlth_cal_idx, feature_cols].fillna(0).values.astype(np.float32)
        y_hlth = s_hlth.loc[hlth_cal_idx, "intent_idx"].values.astype(np.int64)
        X_imp_cal = s_imp.loc[imp_cal_idx, feature_cols].fillna(0).values.astype(np.float32)
        X_test = s_imp.loc[test_idx, feature_cols].fillna(0).values.astype(np.float32)
        y_test = s_imp.loc[test_idx, "intent_idx"].values.astype(np.int64)

        # ── V1: baseline (fit scaler on healthy-arm cal, apply to test) ──
        try:
            sc = StandardScaler().fit(X_hlth)
            clf = make_hgb().fit(sc.transform(X_hlth), y_hlth)
            v1_acc = float(accuracy_score(y_test, clf.predict(sc.transform(X_test))))
        except Exception:
            v1_acc = np.nan

        # ── V2: per-limb z-score (fit separate scalers, keep coordinate frames aligned only in shape) ──
        try:
            sc_h = StandardScaler().fit(X_hlth)
            sc_t = StandardScaler().fit(X_imp_cal)   # fit on impaired-arm cal (proxy for test's limb frame)
            X_hlth_z = sc_h.transform(X_hlth)
            X_test_z = sc_t.transform(X_test)
            clf = make_hgb().fit(X_hlth_z, y_hlth)
            v2_acc = float(accuracy_score(y_test, clf.predict(X_test_z)))
        except Exception:
            v2_acc = np.nan

        # ── V3: amplitude-equalise healthy-arm features to impaired-arm means, then normal pipeline ──
        try:
            # Per-feature scale factor: impaired_cal mean / healthy_cal mean
            hlth_mean = X_hlth.mean(axis=0) + 1e-8
            imp_mean = X_imp_cal.mean(axis=0)
            scale = imp_mean / hlth_mean
            # Clip extreme scaling to avoid instability
            scale = np.clip(scale, 0.1, 10.0)
            X_hlth_eq = X_hlth * scale
            sc = StandardScaler().fit(X_hlth_eq)
            clf = make_hgb().fit(sc.transform(X_hlth_eq), y_hlth)
            v3_acc = float(accuracy_score(y_test, clf.predict(sc.transform(X_test))))
        except Exception:
            v3_acc = np.nan

        rows.append({
            "patient": patient,
            "v1_baseline_acc": v1_acc,
            "v2_per_limb_z_acc": v2_acc,
            "v3_amplitude_eq_acc": v3_acc,
            "n_test": len(y_test),
        })

        elapsed = time.time() - t0
        eta = elapsed / max(1, pi - len(done)) * (len(patients) - pi)
        print(f"[{pi}/{len(patients)}] {patient}: V1={v1_acc:.4f}  V2={v2_acc:.4f}  V3={v3_acc:.4f}  "
              f"[{elapsed/60:.1f}min, eta {eta/60:.0f}min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return

    imp_own_ref = 0.875
    v1_mean = out.v1_baseline_acc.mean()
    v2_mean = out.v2_per_limb_z_acc.mean()
    v3_mean = out.v3_amplitude_eq_acc.mean()

    md = [
        "# C2 — per-limb normalisation cross-arm test",
        "",
        f"n = {len(out)} patients. Three normalization variants for cross-arm PO.",
        "",
        "## Results",
        "",
        "| Variant | Description | Mean acc | Gap from own-cal (0.875) |",
        "|---|---|---:|---:|",
        f"| **V1 baseline** | scaler on healthy-arm cal only | {v1_mean:.4f} | {imp_own_ref - v1_mean:+.4f} |",
        f"| **V2 per-limb z** | separate scalers per limb | {v2_mean:.4f} | {imp_own_ref - v2_mean:+.4f} |",
        f"| **V3 amplitude-eq** | rescale healthy to impaired-mean | {v3_mean:.4f} | {imp_own_ref - v3_mean:+.4f} |",
        "",
        "## Decision (pre-registered)",
        "",
        "- If V2 or V3 closes the gap to < 20 pp (i.e., mean acc > 0.675): story is",
        "  'scale/SNR mismatch', not 'pathology'. Report both numbers regardless.",
        "- If both stay near V1 baseline: pathology story holds.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
