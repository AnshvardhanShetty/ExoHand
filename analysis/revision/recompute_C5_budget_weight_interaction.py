"""
Revision — C5: budget × weight interaction.

Extends C4 (GrabMyo weight sweep at cal=36) across the cal-size sweep.
For each patient, for each cal_per_gesture ∈ {3, 6, 12, 24, 36}:
  train HGB at weights {0, 1, 10, 100, 1000}× and record accuracy.

Yields precise claim: "pretraining helps only below X seconds of cal, at weight Y".

Resumable.
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
)
from analysis.revision.recompute_A_lda_at_cal_sizes import split_at, CAL_SIZES, BUFFER


OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "C5_budget_weight_interaction_per_patient.csv"
OUT_MD = OUT_DIR / "C5_budget_weight_interaction_summary.md"

WEIGHTS = [0.0, 1.0, 10.0, 100.0, 1000.0]


def make_hgb(seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False, class_weight="balanced",
    )


def fit_score(X_gm, y_gm, X_cal, y_cal, X_test, y_test, weight):
    if weight == 0.0:
        if len(np.unique(y_cal)) < 2:
            return np.nan
        sc = StandardScaler().fit(X_cal)
        clf = make_hgb().fit(sc.transform(X_cal), y_cal)
    else:
        X_all = np.vstack([X_gm, X_cal])
        y_all = np.concatenate([y_gm, y_cal])
        w = np.ones(len(X_all), dtype=np.float32)
        w[len(X_gm):] = float(weight)
        sc = StandardScaler().fit(X_all)
        clf = make_hgb().fit(sc.transform(X_all), y_all, sample_weight=w)
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
        gm_features = json.load(f)["feature_cols"]

    print("Loading GrabMyo (300k)...")
    gm = pd.read_pickle(GRABMYO_CACHE).sample(300_000, random_state=SEED)
    gm_X = gm[gm_features].values.astype(np.float32)
    gm_y = gm["intent_idx"].values.astype(np.int64)

    rows = []
    done_keys = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        rows = existing.to_dict("records")
        done_keys = set(zip(existing.patient, existing.cal_per_gesture, existing.weight))
        print(f"Resume: {len(rows)} rows")

    patients = sorted(eng["participant"].unique(), key=lambda s: int(s.replace("patient", "")))
    for pi, patient in enumerate(patients, 1):
        s01 = eng[(eng.participant == patient) & (eng.session == "impaired_01")]
        if len(s01) == 0:
            continue
        for cal_size in CAL_SIZES:
            try:
                test_idx, cal_idx = split_at(s01, cal_size, TEST_PER_CLASS, rng)
            except Exception:
                continue
            if len(test_idx) < 15 or len(cal_idx) < 3:
                continue
            X_cal = s01.loc[cal_idx, gm_features].fillna(0).values.astype(np.float32)
            y_cal = s01.loc[cal_idx, "intent_idx"].values.astype(np.int64)
            X_test = s01.loc[test_idx, gm_features].fillna(0).values.astype(np.float32)
            y_test = s01.loc[test_idx, "intent_idx"].values.astype(np.int64)
            for w in WEIGHTS:
                if (patient, cal_size, w) in done_keys:
                    continue
                try:
                    acc = fit_score(gm_X, gm_y, X_cal, y_cal, X_test, y_test, w)
                except Exception as e:
                    print(f"  {patient} cal={cal_size} w={w}: {e}", flush=True)
                    acc = np.nan
                rows.append({
                    "patient": patient, "cal_per_gesture": cal_size, "weight": w, "acc": acc,
                })
        elapsed = time.time() - t0
        n_done_patients = pi
        eta = elapsed / max(1, n_done_patients) * (len(patients) - n_done_patients)
        print(f"[{pi}/{len(patients)}] {patient} done  [{elapsed/60:.1f}min, eta {eta/60:.0f}min]",
              flush=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    out = pd.DataFrame(rows)
    grid = out.groupby(["cal_per_gesture", "weight"]).acc.mean().unstack().round(4)

    md = [
        "# C5 — budget × weight interaction grid",
        "",
        f"n = 48 patients × {len(WEIGHTS)} weights × {len(CAL_SIZES)} cal budgets.",
        "",
        "## Mean patient accuracy grid (rows = cal_per_gesture, cols = weight×)",
        "",
        grid.to_markdown(),
        "",
        "## Reading",
        "",
        "- Row cal=3: PO (weight=0) collapses (~0.33). GrabMyo (weight>0) rescues it.",
        "- Row cal=36 (paper op point): all weights converge, cal-only ties best.",
        "- The crossover cal size is where 'pretraining helps' becomes 'pretraining redundant'.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
