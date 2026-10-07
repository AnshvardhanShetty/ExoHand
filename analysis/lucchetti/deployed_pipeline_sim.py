"""
Lucchetti on the simulated deployed (20 Hz P-P envelope) pipeline.

Replicates Lucchetti's three headline numbers on the envelope pipeline:
  1. Zero-shot 3-class accuracy on stroke (n=10 impaired arm).
  2. Calibrated 3-class accuracy on stroke (n=10 impaired arm).
  3. Calibrated binary (movement vs rest) accuracy on stroke (n=10 impaired arm).

Pipeline:
  - Load each stroke subject's .mat file (1 kHz raw EMG, 12 channels).
  - Apply notch + bandpass at native 1 kHz.
  - Compute 50 ms-window P-P amplitude per channel → 20 Hz stream.
  - Build per-sample labels from kinematic events (at 125 Hz → upsampled to 20 Hz).
  - Window into 200 ms (4 P-P samples) at 50 ms (1 sample) stride.
  - Compute 60 base features per window at fs=20 Hz.
  - Engineer features → 370.
  - Use the full GrabMyo envelope cache for zero-shot + calibrated training.

Output:
  analysis/lucchetti/results/deployed_pipeline_sim.csv
  analysis/lucchetti/results/deployed_pipeline_sim_summary.md
"""

import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from analysis.seed import SEED, seed_everything
from analysis.emgbench.feature_extraction import _features_for_one_window
from ml.train_hgb_v2 import engineer_features
from ml.preprocessing_lucchetti import (
    PICKS_LUCCHETTI, CANONICAL_GRABMYO_NAMES, LUCCHETTI_EMG_FS, KIN_FS,
    LOW_HZ, HIGH_HZ, BP_ORDER, NOTCH_HZ, NOTCH_Q,
    TASK_INTENT_MAP, INTENT_TO_IDX,
)
from analysis.physiomio.deployed_pipeline_sim import (
    TEENSY_PP_WINDOW_MS, TEENSY_OUTPUT_HZ, INFERENCE_WINDOW_MS, INFERENCE_STRIDE_MS,
    make_clf,
)

LUCCHETTI_ROOT = PROJECT_ROOT / "data" / "lucchetti"
GM_20HZ_CACHE = PROJECT_ROOT / "analysis" / ".cache" / "grabmyo_features_370_20hz_full.pkl"

OUT_DIR = PROJECT_ROOT / "analysis" / "lucchetti" / "results"
OUT_CSV = OUT_DIR / "deployed_pipeline_sim.csv"
OUT_MD = OUT_DIR / "deployed_pipeline_sim_summary.md"

CAL_WEIGHT = 100.0
TEST_PER_CLASS = 39
CLASSES = [0, 1, 2]

FEATURE_NAMES_PER_CHANNEL = [
    "rms", "mav", "var", "wl", "maxamp", "zc", "ssc", "wamp", "iemg",
    "mean_freq", "median_freq", "env_mean", "env_max", "env_std", "env_rms",
]


def _bandpass_notch_1khz(x):
    nyq = 0.5 * LUCCHETTI_EMG_FS
    bp_b, bp_a = butter(BP_ORDER, [LOW_HZ / nyq, min(HIGH_HZ / nyq, 0.99)], btype="band")
    n_b, n_a = iirnotch(NOTCH_HZ, NOTCH_Q, LUCCHETTI_EMG_FS)
    y = filtfilt(n_b, n_a, x)
    y = filtfilt(bp_b, bp_a, y)
    return y


def _pp_downsample(x, samples_per_pp):
    n = (len(x) // samples_per_pp) * samples_per_pp
    return x[:n].reshape(-1, samples_per_pp).ptp(axis=1)


def _label_stream_20hz(n_pp_samples, events_start_kin, events_end_kin, task_idx):
    """Build per-PP-sample labels. Kinematic events at 125 Hz → 20 Hz factor = 6.25."""
    labels = np.full(n_pp_samples, INTENT_TO_IDX["rest"], dtype=np.int8)
    movement_intent = TASK_INTENT_MAP.get(task_idx, "rest")
    movement_idx = INTENT_TO_IDX[movement_intent]
    factor = TEENSY_OUTPUT_HZ / KIN_FS    # 20 / 125 = 0.16
    for s_kin, e_kin in zip(events_start_kin, events_end_kin):
        s = int(s_kin * factor)
        e = int(e_kin * factor)
        s = max(0, min(s, n_pp_samples))
        e = max(s, min(e, n_pp_samples))
        if movement_intent != "rest":
            labels[s:e] = movement_idx
    return labels


def extract_session_features_20hz_lucchetti(subject_id, session_id, data_struct):
    pp_per_window = int(TEENSY_PP_WINDOW_MS / 1000.0 * LUCCHETTI_EMG_FS)  # 50
    win_samples_20hz = int(INFERENCE_WINDOW_MS / 1000.0 * TEENSY_OUTPUT_HZ)  # 4
    stride_samples_20hz = int(INFERENCE_STRIDE_MS / 1000.0 * TEENSY_OUTPUT_HZ)  # 1
    env_win = max(1, int(200 / 1000.0 * TEENSY_OUTPUT_HZ))
    env_kernel = np.ones(env_win) / env_win

    rows = []
    trial_counter = 0
    last_intent_idx = None
    for task_idx in range(len(data_struct)):
        task = data_struct[task_idx]
        emg = task.EMG    # (12, N) at 1 kHz
        events = task.Events
        starts = np.atleast_1d(events.Start).astype(int) if hasattr(events, "Start") else np.array([], dtype=int)
        ends = np.atleast_1d(events.End).astype(int) if hasattr(events, "End") else np.array([], dtype=int)

        emg_sel = emg[PICKS_LUCCHETTI, :].astype(np.float64)   # (4, N_1k)

        # Bandpass + notch + 50 ms P-P → 20 Hz stream per channel
        pp_streams = []
        for ch in range(4):
            x = emg_sel[ch] - np.mean(emg_sel[ch])
            xf = _bandpass_notch_1khz(x)
            pp = _pp_downsample(xf, pp_per_window)
            pp_streams.append(pp)
        min_n = min(len(s) for s in pp_streams)
        pp_4ch = np.stack([s[:min_n] for s in pp_streams], axis=1)   # (n_pp, 4)
        env_4ch = np.stack([
            np.convolve(np.abs(pp_4ch[:, c]), env_kernel, mode="same")
            for c in range(4)
        ], axis=1)

        # Labels per PP sample
        labels = _label_stream_20hz(min_n, starts, ends, task_idx)

        # Window into inference windows
        for w_start in range(0, min_n - win_samples_20hz + 1, stride_samples_20hz):
            w_end = w_start + win_samples_20hz
            win_labels = labels[w_start:w_end]
            unique, counts = np.unique(win_labels, return_counts=True)
            top_count = counts.max()
            if top_count / len(win_labels) < 0.95:
                continue   # mixed-label, skip
            intent_idx = int(unique[counts.argmax()])
            intent_name = {0: "rest", 1: "close", 2: "open"}[intent_idx]
            # Start a new trial only when the intent class changes (contiguous same-class runs).
            if last_intent_idx is None or intent_idx != last_intent_idx:
                trial_counter += 1
                last_intent_idx = intent_idx
            row = {
                "participant": subject_id,
                "session": session_id,
                "gesture_name": intent_name,
                "trial": trial_counter,
                "t_rel_s": (w_start + win_samples_20hz / 2) / TEENSY_OUTPUT_HZ,
                "intent": intent_name,
                "intent_idx": intent_idx,
            }
            for ci, canon in enumerate(CANONICAL_GRABMYO_NAMES):
                w = pp_4ch[w_start:w_end, ci]
                env = env_4ch[w_start:w_end, ci]
                feats = _features_for_one_window(w, env, TEENSY_OUTPUT_HZ)
                for fname in FEATURE_NAMES_PER_CHANNEL:
                    row[f"ch{canon}_{fname}"] = feats[fname]
            rows.append(row)
    return pd.DataFrame(rows)


def split_session_lucchetti_envelope(session_df, test_per_class, rng, buffer_windows=3):
    """Same per-trial half/half split as the original Lucchetti eval."""
    cal_idx = []
    test_pool_by_class = {0: [], 1: [], 2: []}
    class_counts = {0: 0, 1: 0, 2: 0}
    for _, g in session_df.groupby("trial", sort=True):
        cls = int(g["intent_idx"].iloc[0])
        sg = g.sort_values("t_rel_s")
        n = len(sg)
        class_counts[cls] += n
        if n < 8:
            cal_idx.extend(sg.index.tolist())
            continue
        cal_n = max(1, (n - buffer_windows) // 2)
        test_start = cal_n + buffer_windows
        cal_idx.extend(sg.index[:cal_n].tolist())
        test_pool_by_class[cls].extend(sg.index[test_start:].tolist())
    balanced_test = []
    for cls in CLASSES:
        pool = test_pool_by_class[cls]
        if len(pool) <= test_per_class:
            balanced_test.extend(pool)
        else:
            balanced_test.extend(rng.choice(pool, size=test_per_class, replace=False).tolist())
    return np.array(sorted(balanced_test)), np.array(sorted(cal_idx)), class_counts


def main():
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load GrabMyo envelope cache (built by deployed_pipeline_sim_zeroshot)
    print(f"Loading GrabMyo envelope cache from {GM_20HZ_CACHE}...")
    gm = pd.read_pickle(GM_20HZ_CACHE)
    print(f"  {len(gm)} GrabMyo windows  ({time.time()-t0:.0f}s)")

    # Process Lucchetti stroke subjects (impaired arm)
    print("\nExtracting Lucchetti stroke impaired-arm features at 20 Hz P-P...")
    stroke_dirs = sorted([d for d in (LUCCHETTI_ROOT / "stroke").iterdir() if d.is_dir()])
    luc_dfs = []
    for sd in stroke_dirs:
        subj = sd.name   # e.g. ST_01
        mat_path = sd / f"{subj.replace('_','')}.mat"
        if not mat_path.exists():
            # try alternate naming
            mats = list(sd.glob("*.mat"))
            if not mats:
                print(f"  [skip] {subj}: no .mat"); continue
            mat_path = mats[0]
        m = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
        s = m["s"]
        data = s.DataULpleg
        sess_df = extract_session_features_20hz_lucchetti(subj, "impaired_01", data)
        luc_dfs.append(sess_df)
        print(f"  {subj}: {len(sess_df)} windows  ({time.time()-t0:.0f}s)", flush=True)
    luc = pd.concat(luc_dfs, ignore_index=True)
    print(f"  Lucchetti stroke impaired: {len(luc)} windows across {luc.participant.nunique()} subjects")

    # Engineer features on combined (GrabMyo + Lucchetti)
    print("\nEngineer features (60 → 370)...")
    combined = pd.concat([luc, gm], ignore_index=True)
    eng = engineer_features(combined)
    meta_cols = {"participant", "session", "gesture_name", "trial", "t_rel_s", "intent", "intent_idx"}
    feature_cols = [c for c in eng.columns if c not in meta_cols]
    luc_eng = eng[eng.participant.isin(luc.participant.unique())].copy()
    gm_eng = eng[eng.participant.isin(gm.participant.unique())].copy()
    gm_X = gm_eng[feature_cols].values.astype(np.float32)
    gm_y = gm_eng["intent_idx"].values.astype(np.int64)

    # Fit GrabMyo-only model (zero-shot)
    print("\nFitting HGB on GrabMyo envelope features (zero-shot)...")
    scaler_zs = StandardScaler()
    gm_X_s = scaler_zs.fit_transform(gm_X)
    clf_zs = make_clf(SEED)
    t_fit = time.time()
    clf_zs.fit(gm_X_s, gm_y)
    print(f"  zero-shot fit_time: {time.time()-t_fit:.1f}s")

    # Per-session zero-shot + calibrated eval
    rows = []
    for subj in sorted(luc_eng.participant.unique()):
        s_data = luc_eng[(luc_eng.participant == subj) & (luc_eng.session == "impaired_01")].copy()
        test_idx, cal_idx, class_counts = split_session_lucchetti_envelope(s_data, TEST_PER_CLASS, rng)
        if len(test_idx) == 0:
            print(f"  [skip] {subj}: no test data"); continue
        X_test = s_data.loc[test_idx, feature_cols].values.astype(np.float32)
        y_test = s_data.loc[test_idx, "intent_idx"].values.astype(np.int64)

        # Zero-shot
        X_test_zs = scaler_zs.transform(X_test).astype(np.float32)
        preds_zs = clf_zs.predict(X_test_zs)
        acc_zs = accuracy_score(y_test, preds_zs)

        # Calibrated 3-class
        X_cal = s_data.loc[cal_idx, feature_cols].values.astype(np.float32)
        y_cal = s_data.loc[cal_idx, "intent_idx"].values.astype(np.int64)
        X_all = np.vstack([gm_X, X_cal])
        y_all = np.concatenate([gm_y, y_cal])
        w_all = np.ones(len(X_all), dtype=np.float32)
        w_all[len(gm_X):] = CAL_WEIGHT
        scaler_cal = StandardScaler()
        X_all_s = scaler_cal.fit_transform(X_all)
        X_test_cal = scaler_cal.transform(X_test).astype(np.float32)
        clf_cal = make_clf(SEED)
        t_fit = time.time()
        clf_cal.fit(X_all_s, y_all, sample_weight=w_all)
        cal_fit_time = time.time() - t_fit
        preds_cal = clf_cal.predict(X_test_cal)
        acc_cal_3 = accuracy_score(y_test, preds_cal)
        f1m_cal_3 = f1_score(y_test, preds_cal, average="macro", zero_division=0)
        cls_f1 = f1_score(y_test, preds_cal, average=None, labels=CLASSES, zero_division=0)

        # Binary (movement vs rest): collapse close+open → 1, rest → 0
        y_test_bin = (y_test != 0).astype(np.int64)
        preds_cal_bin = (preds_cal != 0).astype(np.int64)
        acc_cal_bin = accuracy_score(y_test_bin, preds_cal_bin)

        rows.append({
            "participant": subj, "session": "impaired_01", "arm": "impaired",
            "n_cal": int(len(cal_idx)), "n_test": int(len(test_idx)),
            "zs_acc": acc_zs,
            "cal_acc_3class": acc_cal_3,
            "cal_acc_binary": acc_cal_bin,
            "f1_macro_3class": f1m_cal_3,
            "f1_rest": cls_f1[0], "f1_close": cls_f1[1], "f1_open": cls_f1[2],
            "cal_fit_time_s": cal_fit_time,
        })
        print(f"  {subj}: zs={acc_zs:.3f}  cal3={acc_cal_3:.3f}  calBin={acc_cal_bin:.3f}  "
              f"fit={cal_fit_time:.1f}s ({len(rows)}/10)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    ref_zs_raw = 0.194
    ref_cal_3_raw = 0.795
    ref_cal_bin_raw = 0.960

    md = [
        "# Lucchetti — simulated deployed (20 Hz P-P envelope) pipeline",
        "",
        f"n stroke subjects: {len(df)}",
        f"GrabMyo training: {len(gm_X):,} envelope windows",
        "",
        "## Result",
        "",
        f"- **Zero-shot 3-class (envelope):**       {df.zs_acc.mean():.4f}  (raw 2 kHz ref: {ref_zs_raw:.3f})",
        f"- **Calibrated 3-class (envelope):**     {df.cal_acc_3class.mean():.4f}  (raw 2 kHz ref: {ref_cal_3_raw:.3f})",
        f"- **Calibrated binary (envelope):**      {df.cal_acc_binary.mean():.4f}  (raw 2 kHz ref: {ref_cal_bin_raw:.3f})",
        "",
        f"- median 3-class cal: {df.cal_acc_3class.median():.4f}  std: {df.cal_acc_3class.std():.4f}",
        f"- median binary cal:  {df.cal_acc_binary.median():.4f}",
        "",
        "## Per-class F1 (calibrated 3-class)",
        "",
        f"- rest:  {df.f1_rest.mean():.4f}",
        f"- close: {df.f1_close.mean():.4f}",
        f"- open:  {df.f1_open.mean():.4f}",
    ]
    OUT_MD.write_text("\n".join(md))

    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(f"\n=== RESULT (Lucchetti, envelope pipeline) ===")
    print(f"  Zero-shot 3-class:  {df.zs_acc.mean():.4f}  (raw 2 kHz: {ref_zs_raw:.3f})")
    print(f"  Calibrated 3-class: {df.cal_acc_3class.mean():.4f}  (raw 2 kHz: {ref_cal_3_raw:.3f})")
    print(f"  Calibrated binary:  {df.cal_acc_binary.mean():.4f}  (raw 2 kHz: {ref_cal_bin_raw:.3f})")


if __name__ == "__main__":
    main()
