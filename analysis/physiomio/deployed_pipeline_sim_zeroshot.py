"""
Zero-shot on the simulated deployed (20 Hz P-P envelope) pipeline.

Both endpoints of the lift (zero-shot → calibrated) should live on the same
signal pipeline for the paper to describe one system. This script computes
the zero-shot endpoint on the envelope pipeline.

Pipeline:
  1. Process ALL of GrabMyo (43 participants × 3 sessions) through the 20 Hz
     P-P envelope pipeline. Cached to grabmyo_features_370_20hz_full.pkl
     after first run.
  2. Re-extract PhysioMio impaired_01 envelope features for 48 patients.
  3. Train HGB on GrabMyo envelope features only (no patient cal).
  4. For each patient, evaluate on the same balanced 117-window test split
     used in per_session_eval / deployed_pipeline_sim.

Output:
  analysis/physiomio/results/deployed_pipeline_sim_zeroshot.csv
  analysis/physiomio/results/deployed_pipeline_sim_zeroshot_summary.md
"""

import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from analysis.seed import SEED, seed_everything
from ml.train_hgb_v2 import engineer_features
from analysis.physiomio.deployed_pipeline_sim import (
    extract_session_features_20hz, split_session, make_clf,
    CAL_PER_GESTURE, BUFFER, TEST_PER_CLASS, CLASSES,
)
from analysis.physiomio.deployed_pipeline_sim_gmcal import (
    build_grabmyo_20hz_subset, N_GRABMYO_PARTICIPANTS,
)

# Patch the subset cache path so a full 43-participant cache is built/loaded separately.
import analysis.physiomio.deployed_pipeline_sim_gmcal as gmcal_mod
gmcal_mod.GM_20HZ_CACHE = PROJECT_ROOT / "analysis" / ".cache" / "grabmyo_features_370_20hz_full.pkl"
gmcal_mod.N_GRABMYO_PARTICIPANTS = 43

PHYSIOMIO_ROOT = PROJECT_ROOT / "data" / "physiomio" / "data"
CHANNEL_PICKS = PROJECT_ROOT / "data" / "physiomio_channel_picks.csv"

OUT_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "deployed_pipeline_sim_zeroshot.csv"
OUT_MD = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "deployed_pipeline_sim_zeroshot_summary.md"


def main():
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    t0 = time.time()

    # ── GrabMyo envelope features (full 43 participants × 3 sessions) ──
    print("Building/loading full GrabMyo envelope cache (43 participants × 3 sessions)...")
    gm = build_grabmyo_20hz_subset(n_participants=43)
    print(f"  {len(gm)} windows from {gm.participant.nunique()} participant-sessions")

    # ── PhysioMio envelope features (48 patients × impaired_01) ──
    print("\nRebuilding PhysioMio envelope features for 48 patients...")
    picks = pd.read_csv(CHANNEL_PICKS)
    picks["chosen_channels"] = picks["chosen_channels"].apply(json.loads)
    pick_map = dict(zip(picks["patient"], picks["chosen_channels"]))
    patients = sorted([p for p in PHYSIOMIO_ROOT.iterdir() if p.is_dir() and p.name.startswith("patient")],
                      key=lambda p: int(p.name.replace("patient", "")))
    pm_dfs = []
    for pi, pdir in enumerate(patients, 1):
        patient = pdir.name
        impaired_01 = pdir / "impaired_arm" / "01.parquet"
        channels = pick_map.get(patient)
        if not impaired_01.exists() or channels is None:
            continue
        try:
            session_df = extract_session_features_20hz(impaired_01, channels)
        except Exception as e:
            print(f"  [err] {patient}: {e}"); continue
        session_df["participant"] = patient
        session_df["session"] = "impaired_01"
        pm_dfs.append(session_df)
        if pi % 12 == 0:
            print(f"  [{pi:>2d}/{len(patients)}]  ({time.time()-t0:.0f}s)", flush=True)
    pm = pd.concat(pm_dfs, ignore_index=True)
    print(f"  PhysioMio: {len(pm)} windows across {pm.participant.nunique()} patients")

    # ── Engineer features ──
    print("\nEngineer features (60 → 370) on combined frame...")
    combined = pd.concat([pm, gm], ignore_index=True)
    eng = engineer_features(combined)
    meta_cols = {"participant", "session", "gesture_name", "trial", "t_rel_s", "intent", "intent_idx"}
    feature_cols = [c for c in eng.columns if c not in meta_cols]
    print(f"  features: {len(feature_cols)}")

    gm_eng = eng[eng.participant.isin(gm.participant.unique())].copy()
    pm_eng = eng[eng.participant.isin(pm.participant.unique())].copy()
    gm_X = gm_eng[feature_cols].values.astype(np.float32)
    gm_y = gm_eng["intent_idx"].values.astype(np.int64)
    print(f"  GrabMyo training rows: {len(gm_X):,}")
    print(f"  PhysioMio rows: {len(pm_eng):,}")

    # ── Train HGB on GrabMyo only (zero-shot) ──
    print("\nFitting HGB on GrabMyo envelope features only (zero-shot)...")
    scaler = StandardScaler()
    gm_X_s = scaler.fit_transform(gm_X)
    clf = make_clf(SEED)
    t_fit = time.time()
    clf.fit(gm_X_s, gm_y)
    print(f"  fit_time: {time.time()-t_fit:.1f}s")

    # ── Evaluate zero-shot on each PhysioMio session (using same balanced test split) ──
    print("\nEvaluating zero-shot on 48 patients...")
    rows = []
    for patient in sorted(pm_eng.participant.unique(), key=lambda s: int(s.replace("patient", ""))):
        s_data = pm_eng[(pm_eng.participant == patient) & (pm_eng.session == "impaired_01")].copy()
        test_idx, cal_idx = split_session(s_data, CAL_PER_GESTURE, TEST_PER_CLASS, rng)
        if len(test_idx) == 0:
            continue
        X_test = s_data.loc[test_idx, feature_cols].values.astype(np.float32)
        y_test = s_data.loc[test_idx, "intent_idx"].values.astype(np.int64)
        X_test_s = scaler.transform(X_test).astype(np.float32)
        preds = clf.predict(X_test_s)
        acc = accuracy_score(y_test, preds)
        f1m = f1_score(y_test, preds, average="macro", zero_division=0)
        cls_f1 = f1_score(y_test, preds, average=None, labels=CLASSES, zero_division=0)
        rows.append({
            "participant": patient,
            "n_test": int(len(test_idx)),
            "acc": acc, "f1_macro": f1m,
            "f1_rest": cls_f1[0], "f1_close": cls_f1[1], "f1_open": cls_f1[2],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    md = [
        "# Deployed-pipeline simulation — zero-shot endpoint",
        "",
        f"GrabMyo training: {len(gm_X):,} envelope windows from {gm.participant.nunique()} participant-sessions",
        f"Patients evaluated: {len(df)}",
        "",
        "## Result",
        "",
        f"- **Zero-shot at simulated 20 Hz P-P pipeline:** {df.acc.mean():.4f} patient-mean",
        f"  (median {df.acc.median():.4f}, std {df.acc.std():.4f})",
        f"- **Reference zero-shot at raw 2 kHz (zero_shot_per_session.csv):** 0.188 patient-mean",
        f"- **Δ:** {df.acc.mean() - 0.188:+.4f}",
        "",
        "## Per-class F1",
        "",
        f"- rest:  {df.f1_rest.mean():.4f}",
        f"- close: {df.f1_close.mean():.4f}",
        f"- open:  {df.f1_open.mean():.4f}",
        "",
        "## Implication",
        "",
        "If this number lands near 0.19, the headline lift `0.19 → 0.86` is fully on the",
        "deployed envelope pipeline. Both endpoints reported on the same regime → paper",
        "describes one system, one pipeline, one set of numbers.",
    ]
    OUT_MD.write_text("\n".join(md))

    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(f"\n=== RESULT ===")
    print(f"  Zero-shot at envelope pipeline:  {df.acc.mean():.4f}  (n={len(df)})")
    print(f"  Zero-shot at raw 2 kHz (reference): 0.188")
    print(f"  Δ:                                  {df.acc.mean() - 0.188:+.4f}")


if __name__ == "__main__":
    main()
