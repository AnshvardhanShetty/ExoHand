"""
GM+cal arm of the deployed-pipeline simulation.

Pairs the simulated 20 Hz P-P PhysioMio (patient-side, same as patient-only
arm) with a 20 Hz P-P GrabMyo subsample, to test whether the GM+cal arm
behaves like the patient-only arm did (0.865) when both train and test are
in the deployed signal regime.

Pipeline:
  1. Re-process PhysioMio impaired_01 to 20 Hz P-P features (same as
     deployed_pipeline_sim.py).
  2. Process a GrabMyo subsample (10 participants × 3 sessions) through the
     same 20 Hz P-P pipeline.
  3. For each PhysioMio session: train HGB on (GrabMyo + cal × 100 weight),
     test on the held-out 117-window balanced split.
  4. Patient-mean accuracy reported.

For speed we subsample GrabMyo to a representative 10-participant subset
(~70 K windows). The population prior is still a useful regulariser at
that size, and the goal here is just to see whether GM+cal ≈ patient-only
at the simulated deployed pipeline (which the raw 2 kHz sweep showed to be
the case at cal=36).
"""

import os
import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from analysis.seed import SEED, seed_everything
from analysis.emgbench.feature_extraction import _features_for_one_window
from ml.train_hgb_v2 import engineer_features
from analysis.physiomio.deployed_pipeline_sim import (
    extract_session_features_20hz, split_session, make_clf,
    TEENSY_PP_WINDOW_MS, TEENSY_OUTPUT_HZ, INFERENCE_WINDOW_MS, INFERENCE_STRIDE_MS,
    CANONICAL_NAMES, CAL_PER_GESTURE, BUFFER, TEST_PER_CLASS, CLASSES,
)
from ml.preprocessing_grabmyo import (
    CHANNELS_TO_USE as GRABMYO_CHANNELS, LOWCUT as GM_LOW, HIGHCUT as GM_HIGH,
    FILTER_ORDER as GM_FILT_ORD,
)

PHYSIOMIO_ROOT = PROJECT_ROOT / "data" / "physiomio" / "data"
GRABMYO_ROOT = PROJECT_ROOT / "grabmyo"
CHANNEL_PICKS = PROJECT_ROOT / "data" / "physiomio_channel_picks.csv"
GM_20HZ_CACHE = PROJECT_ROOT / "analysis" / ".cache" / "grabmyo_features_370_20hz_subset.pkl"

OUT_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "deployed_pipeline_sim_gmcal.csv"
OUT_MD = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "deployed_pipeline_sim_gmcal_summary.md"

CAL_WEIGHT = 100.0
N_GRABMYO_PARTICIPANTS = 10  # subsample size

# Gesture → intent mapping for GrabMyo (1-indexed gesture numbers)
GRABMYO_GESTURE_TO_INTENT = {
    15: "open",    # Hand Open
    16: "close",   # Hand Close
    17: "rest",    # Rest
}
INTENT_TO_IDX = {"rest": 0, "close": 1, "open": 2}


def _bandpass(x, fs):
    nyq = 0.5 * fs
    b, a = butter(GM_FILT_ORD, [GM_LOW / nyq, min(GM_HIGH / nyq, 0.99)], btype="band")
    return filtfilt(b, a, x)


def _pp_downsample(x, samples_per_pp):
    n = (len(x) // samples_per_pp) * samples_per_pp
    return x[:n].reshape(-1, samples_per_pp).ptp(axis=1)


def process_grabmyo_trial(dat_path, fs_assume=2048):
    """Returns (pp_4ch, env_4ch) at 20 Hz, plus metadata from filename."""
    rec = wfdb.rdrecord(str(dat_path).replace(".dat", ""))
    data = rec.p_signal.T.astype(np.float64)  # (32, N)
    fs = rec.fs
    pp_per_window = int(TEENSY_PP_WINDOW_MS / 1000.0 * fs)  # ~100 at 2048 Hz

    pp_streams = []
    for ch in GRABMYO_CHANNELS:
        x = data[ch] - data[ch].mean()
        xf = _bandpass(x, fs)
        pp = _pp_downsample(xf, pp_per_window)
        pp_streams.append(pp)
    min_n = min(len(s) for s in pp_streams)
    pp_4ch = np.stack([s[:min_n] for s in pp_streams], axis=1)  # (n_pp, 4)

    env_win = max(1, int(200 / 1000.0 * TEENSY_OUTPUT_HZ))
    env_kernel = np.ones(env_win) / env_win
    env_4ch = np.stack([
        np.convolve(np.abs(pp_4ch[:, c]), env_kernel, mode="same")
        for c in range(4)
    ], axis=1)
    return pp_4ch, env_4ch


def _windows_to_features(pp_4ch, env_4ch, meta):
    win_samples_20hz = int(INFERENCE_WINDOW_MS / 1000.0 * TEENSY_OUTPUT_HZ)
    stride_samples_20hz = int(INFERENCE_STRIDE_MS / 1000.0 * TEENSY_OUTPUT_HZ)
    feature_names = [
        "rms", "mav", "var", "wl", "maxamp", "zc", "ssc", "wamp", "iemg",
        "mean_freq", "median_freq", "env_mean", "env_max", "env_std", "env_rms",
    ]
    rows = []
    n_pp = pp_4ch.shape[0]
    for start in range(0, n_pp - win_samples_20hz + 1, stride_samples_20hz):
        end = start + win_samples_20hz
        row = dict(meta)
        row["t_rel_s"] = (start + win_samples_20hz / 2) / TEENSY_OUTPUT_HZ
        for ci, canon in enumerate(CANONICAL_NAMES):
            w = pp_4ch[start:end, ci]
            env = env_4ch[start:end, ci]
            feats = _features_for_one_window(w, env, TEENSY_OUTPUT_HZ)
            for fname in feature_names:
                row[f"ch{canon}_{fname}"] = feats[fname]
        rows.append(row)
    return rows


def build_grabmyo_20hz_subset(n_participants=N_GRABMYO_PARTICIPANTS):
    """Process a small GrabMyo subset through the 20 Hz P-P pipeline."""
    if GM_20HZ_CACHE.exists():
        print(f"Loading cached GrabMyo 20 Hz subset from {GM_20HZ_CACHE}")
        return pd.read_pickle(GM_20HZ_CACHE)

    GM_20HZ_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building GrabMyo 20 Hz cache (subset, n={n_participants} participants × 3 sessions)...")
    t0 = time.time()
    all_rows = []
    # Iterate sessions and participants
    participants_used = set()
    sessions_to_walk = ["Session1", "Session2", "Session3"]
    for sess in sessions_to_walk:
        sess_dir = GRABMYO_ROOT / sess
        if not sess_dir.exists():
            print(f"  [skip] {sess_dir} missing"); continue
        # Find participant folders (sorted) and take the first n
        ppl = sorted([p for p in sess_dir.iterdir() if p.is_dir()],
                     key=lambda p: int(p.name.split("participant")[-1]))[:n_participants]
        for pdir in ppl:
            pid = pdir.name
            participants_used.add(pid.split("_participant")[-1] if "_participant" in pid else pid)
            for dat in sorted(pdir.glob("*.dat")):
                # Parse e.g. session1_participant1_gesture15_trial3.dat
                stem = dat.stem
                parts = stem.split("_")
                try:
                    gesture = int([p for p in parts if p.startswith("gesture")][0].replace("gesture", ""))
                    trial = int([p for p in parts if p.startswith("trial")][0].replace("trial", ""))
                except (IndexError, ValueError):
                    continue
                if gesture not in GRABMYO_GESTURE_TO_INTENT:
                    continue
                intent = GRABMYO_GESTURE_TO_INTENT[gesture]
                meta = {
                    "participant": pid,
                    "session": sess.lower(),
                    "gesture_name": str(gesture),
                    "trial": trial,
                    "intent": intent,
                    "intent_idx": INTENT_TO_IDX[intent],
                }
                try:
                    pp_4ch, env_4ch = process_grabmyo_trial(dat)
                    rows = _windows_to_features(pp_4ch, env_4ch, meta)
                    all_rows.extend(rows)
                except Exception as e:
                    print(f"  [err] {dat.name}: {e}")
        print(f"  {sess}: cumulative {len(all_rows)} windows  ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(all_rows)
    print(f"  GrabMyo 20 Hz subset: {len(df)} windows from {df.participant.nunique()} participant-sessions")
    df.to_pickle(GM_20HZ_CACHE)
    return df


def main():
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)

    # ── PhysioMio side: rebuild the simulated 20 Hz P-P features (same as deployed_pipeline_sim) ──
    print("Rebuilding PhysioMio 20 Hz P-P features for the 48-patient cohort...")
    picks = pd.read_csv(CHANNEL_PICKS)
    picks["chosen_channels"] = picks["chosen_channels"].apply(json.loads)
    pick_map = dict(zip(picks["patient"], picks["chosen_channels"]))
    patients = sorted([p for p in PHYSIOMIO_ROOT.iterdir() if p.is_dir() and p.name.startswith("patient")],
                      key=lambda p: int(p.name.replace("patient", "")))
    pm_dfs = []
    t0 = time.time()
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

    # ── GrabMyo side: process small subset through 20 Hz P-P pipeline ──
    gm = build_grabmyo_20hz_subset()

    # Concatenate, engineer features
    print("\nConcatenating PhysioMio + GrabMyo for engineer_features...")
    combined = pd.concat([pm, gm], ignore_index=True)
    print(f"  combined: {len(combined)} windows")
    eng = engineer_features(combined)
    meta_cols = {"participant", "session", "gesture_name", "trial", "t_rel_s", "intent", "intent_idx"}
    feature_cols = [c for c in eng.columns if c not in meta_cols]
    print(f"  features: {len(feature_cols)}")

    # Split into GrabMyo half + PhysioMio half
    gm_eng = eng[eng.participant.isin(gm.participant.unique())].copy()
    pm_eng = eng[eng.participant.isin(pm.participant.unique())].copy()
    print(f"  GrabMyo engineered rows: {len(gm_eng)}")
    print(f"  PhysioMio engineered rows: {len(pm_eng)}")
    gm_X = gm_eng[feature_cols].values.astype(np.float32)
    gm_y = gm_eng["intent_idx"].values.astype(np.int64)

    # Per-session GM+cal eval
    print("\nPer-session GM+cal evaluation on PhysioMio...")
    rows = []
    for patient in sorted(pm_eng.participant.unique(), key=lambda s: int(s.replace("patient", ""))):
        s_data = pm_eng[(pm_eng.participant == patient) & (pm_eng.session == "impaired_01")].copy()
        test_idx, cal_idx = split_session(s_data, CAL_PER_GESTURE, TEST_PER_CLASS, rng)
        if len(test_idx) == 0 or len(cal_idx) == 0:
            continue
        X_cal = s_data.loc[cal_idx, feature_cols].values.astype(np.float32)
        y_cal = s_data.loc[cal_idx, "intent_idx"].values.astype(np.int64)
        X_test = s_data.loc[test_idx, feature_cols].values.astype(np.float32)
        y_test = s_data.loc[test_idx, "intent_idx"].values.astype(np.int64)
        if len(np.unique(y_cal)) < 2:
            continue

        X_train = np.vstack([gm_X, X_cal])
        y_train = np.concatenate([gm_y, y_cal])
        sw = np.ones(len(X_train), dtype=np.float32)
        sw[len(gm_X):] = CAL_WEIGHT
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test).astype(np.float32)
        clf = make_clf(SEED)
        t_fit = time.time()
        try:
            clf.fit(X_train_s, y_train, sample_weight=sw)
        except Exception as e:
            print(f"  [skip] {patient}: fit failed ({e})"); continue
        preds = clf.predict(X_test_s)
        acc = accuracy_score(y_test, preds)
        f1m = f1_score(y_test, preds, average="macro", zero_division=0)
        cls_f1 = f1_score(y_test, preds, average=None, labels=CLASSES, zero_division=0)
        rows.append({
            "participant": patient, "n_cal": int(len(cal_idx)), "n_test": int(len(test_idx)),
            "fit_time_s": time.time() - t_fit,
            "acc": acc, "f1_macro": f1m,
            "f1_rest": cls_f1[0], "f1_close": cls_f1[1], "f1_open": cls_f1[2],
        })
        print(f"  {patient}: acc={acc:.4f}  fit={time.time()-t_fit:.1f}s  ({len(rows)}/48 done)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    baseline_po_2khz = 0.8777
    baseline_po_20hz = 0.8649
    md = [
        "# Deployed-pipeline simulation — GM+cal arm",
        "",
        f"GrabMyo subsample: {len(gm_eng)} windows from {gm.participant.nunique()} participant-sessions",
        f"Patients evaluated: {len(df)}",
        "",
        "## Result",
        "",
        f"- **GM+cal at simulated 20 Hz P-P pipeline:** {df.acc.mean():.4f} patient-mean",
        f"  (median {df.acc.median():.4f}, std {df.acc.std():.4f})",
        f"- **Patient-only at simulated 20 Hz P-P (deployed_pipeline_sim.csv):** {baseline_po_20hz:.4f}",
        f"- **Patient-only at raw 2 kHz (cal_size_sweep_v1.csv):** {baseline_po_2khz:.4f}",
        "",
        "## Deltas",
        "",
        f"- GM+cal 20 Hz P-P vs PO 20 Hz P-P: {df.acc.mean() - baseline_po_20hz:+.4f}",
        f"- GM+cal 20 Hz P-P vs PO 2 kHz raw:  {df.acc.mean() - baseline_po_2khz:+.4f}",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(f"\n=== RESULT ===")
    print(f"  GM+cal at simulated 20 Hz P-P pipeline: {df.acc.mean():.4f}  (n={len(df)})")
    print(f"  PO at simulated 20 Hz P-P:               {baseline_po_20hz:.4f}")
    print(f"  PO at raw 2 kHz:                          {baseline_po_2khz:.4f}")


if __name__ == "__main__":
    main()
