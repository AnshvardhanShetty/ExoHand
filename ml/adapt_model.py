#!/usr/bin/env python3
"""
adapt_model.py — Patient-specific model adaptation.

Takes a recorded + labeled session and retrains the GrabMyo-pretrained HGB
model with the patient's data upweighted, producing a patient-specific model
that retains the GrabMyo generalisation backbone.

Usage:
    python adapt_model.py --session sessions/2026-02-13_xx-xx/ [--weight 10.0] [--no-bandpass]
"""

import argparse, os, sys, json, time

# Ensure project root is on sys.path so absolute imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore")

from ml.train_hgb_v2 import (
    add_temporal_features,
    add_cross_channel_features,
    add_temporal_on_interactions,
    add_rank_features,
    add_within_trial_position,
    add_per_participant_normalisation,
    add_per_session_normalisation,
    META_COLS,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRABMYO_DIR = os.path.join(ROOT, "grabmyo")
GRABMYO_CSV = os.path.join(GRABMYO_DIR, "grabmyo_intent_dataset.csv")
GRABMYO_META = os.path.join(GRABMYO_DIR, "improved_hgb_meta.json")

# --- Signal processing parameters ---
LOWCUT = 20.0
HIGHCUT = 450.0
FILTER_ORDER = 2          # lighter than GrabMyo 4th-order; MyoWare has analog frontend
WINDOW_S = 0.200          # 200 ms
STRIDE_S = 0.050          # 50 ms
ENV_SMOOTH_MS = 50.0      # envelope smoothing

# Default channel mapping: session port→GrabMyo channel
# port0→ch0 (F1 flexor), port1→ch4 (F5 extensor), port2→ch9 (F10 flexor), port3→ch13 (F14 extensor)
DEFAULT_CHANNEL_MAP = "0:0,1:4,2:9,3:13"

# Label remapping: session {close:0, open:1, rest:2} → GrabMyo {rest:0, close:1, open:2}
SESSION_TO_GRABMYO_LABEL = {0: 1, 1: 2, 2: 0}


# ------------------------------------------------------------------ signal processing

def bandpass_filter(data, fs, lowcut=LOWCUT, highcut=HIGHCUT, order=FILTER_ORDER):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype="band")
    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        x = data[:, ch] - np.mean(data[:, ch])
        filtered[:, ch] = filtfilt(b, a, x)
    return filtered


def compute_envelope(filtered, fs):
    env_win = max(1, int((ENV_SMOOTH_MS / 1000.0) * fs))
    kernel = np.ones(env_win) / env_win
    envelope = np.zeros_like(filtered)
    for ch in range(filtered.shape[1]):
        rect = np.abs(filtered[:, ch])
        envelope[:, ch] = np.convolve(rect, kernel, mode="same")
    return envelope


# ------------------------------------------------------------------ feature extraction

def extract_session_features(filtered, envelope, fs, labels, timestamps, ch_map):
    """Extract 15 features × 4 channels in sliding windows, matching GrabMyo exactly.

    ch_map: dict {session_port_idx: grabmyo_channel_num} e.g. {0:0, 1:4, 2:9, 3:13}
    """
    N = filtered.shape[0]
    win = int(WINDOW_S * fs)
    step = int(STRIDE_S * fs)

    rows = []
    for start in range(0, N - win + 1, step):
        end = start + win

        window_labels = labels[start:end]
        session_label = int(np.bincount(window_labels.astype(int), minlength=3).argmax())
        grabmyo_label = SESSION_TO_GRABMYO_LABEL[session_label]

        t_rel = (timestamps[start] + timestamps[end - 1]) / 2.0

        row = {
            "participant": "patient",
            "session": "session1",
            "gesture": grabmyo_label,
            "gesture_name": ["rest", "close", "open"][grabmyo_label],
            "trial": 1,
            "t_rel_s": t_rel,
            "intent": ["rest", "close", "open"][grabmyo_label],
            "intent_idx": grabmyo_label,
        }

        for sess_ch, grab_ch in ch_map.items():
            w = filtered[start:end, sess_ch]
            env = envelope[start:end, sess_ch]
            Nw = len(w)

            rms = np.sqrt(np.mean(w ** 2))
            mav = np.mean(np.abs(w))
            var = np.var(w)
            wl = np.sum(np.abs(np.diff(w)))
            maxamp = np.max(np.abs(w))
            thr = 0.01 * maxamp if maxamp > 0 else 0.0

            prod = w[:-1] * w[1:]
            zc = float(np.sum((prod < 0) & (np.abs(w[:-1] - w[1:]) > thr)))

            d1 = w[1:-1] - w[:-2]
            d2 = w[1:-1] - w[2:]
            ssc = float(np.sum((d1 * d2 > 0) & (np.abs(d1) > thr) & (np.abs(d2) > thr)))

            wamp = float(np.sum(np.abs(np.diff(w)) > thr))
            iemg = np.sum(np.abs(w))

            win_han = np.hanning(Nw)
            fft_vals = np.abs(rfft(w * win_han))
            freqs = rfftfreq(Nw, 1.0 / fs)
            psd = fft_vals ** 2
            psd_sum = psd.sum()
            if psd_sum > 0:
                mean_f = np.sum(freqs * psd) / psd_sum
                cum = np.cumsum(psd)
                median_f = freqs[np.searchsorted(cum, 0.5 * cum[-1])]
            else:
                mean_f = median_f = 0.0

            env_mean = np.mean(env)
            env_max = np.max(env)
            env_std = np.std(env)
            env_rms = np.sqrt(np.mean(env ** 2))

            pref = f"ch{grab_ch}_"
            row[pref + "rms"] = rms
            row[pref + "mav"] = mav
            row[pref + "var"] = var
            row[pref + "wl"] = wl
            row[pref + "maxamp"] = maxamp
            row[pref + "zc"] = zc
            row[pref + "ssc"] = ssc
            row[pref + "wamp"] = wamp
            row[pref + "iemg"] = iemg
            row[pref + "mean_freq"] = mean_f
            row[pref + "median_freq"] = median_f
            row[pref + "env_mean"] = env_mean
            row[pref + "env_max"] = env_max
            row[pref + "env_std"] = env_std
            row[pref + "env_rms"] = env_rms

        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ feature engineering

def engineer_features_for_saved_model(df):
    """Apply the full feature engineering pipeline matching train_hgb_v2.py."""
    df = add_temporal_features(df)
    df = add_cross_channel_features(df)
    df = add_temporal_on_interactions(df)
    df = add_rank_features(df)
    df = add_within_trial_position(df)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    df = add_per_participant_normalisation(df, feature_cols)
    base_features = [c for c in df.columns if c not in META_COLS
                     and "_prev" not in c and "_delta" not in c
                     and "_roll" not in c and "_accel" not in c
                     and "_ratio" not in c and "_diff" not in c
                     and "_pctile" not in c and "_sess_norm" not in c
                     and c != "trial_pos"]
    df = add_per_session_normalisation(df, [c for c in base_features if c in df.columns])
    return df


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(
        description="Adapt GrabMyo-pretrained model with patient-specific data"
    )
    parser.add_argument("--session", required=True, nargs="+",
                        help="Path(s) to session directories (each contains labeled_data.npz)")
    parser.add_argument("--weight", type=float, default=10.0,
                        help="Sample weight multiplier for patient data (default: 10.0)")
    parser.add_argument("--no-bandpass", action="store_true",
                        help="Skip bandpass filter (if MyoWare analog filtering is sufficient)")
    parser.add_argument("--channel-map", default=DEFAULT_CHANNEL_MAP,
                        help="Teensy port→GrabMyo channel mapping. Format: "
                             "port:ch,port:ch,port:ch,port:ch where ports are 0-3 (Teensy order) "
                             "and channels are GrabMyo names (0=F1 flexor, 4=F5 extensor, "
                             "9=F10 flexor, 13=F14 extensor). Default: 0:0,1:4,2:9,3:13")
    parser.add_argument("--output", default=None,
                        help="Output model path (default: exohand_adapted_model.pkl)")
    args = parser.parse_args()

    # Parse channel map
    ch_map = {}
    for pair in args.channel_map.split(","):
        port_s, ch_s = pair.split(":")
        ch_map[int(port_s)] = int(ch_s)
    if sorted(ch_map.keys()) != [0, 1, 2, 3]:
        print("ERROR: --channel-map must specify ports 0, 1, 2, and 3")
        sys.exit(1)
    expected_chs = {0, 4, 9, 13}
    if set(ch_map.values()) != expected_chs:
        print(f"ERROR: --channel-map channels must be {expected_chs} (F1, F5, F10, F14)")
        sys.exit(1)

    print("=" * 60)
    print("Patient-Specific Model Adaptation")
    print("=" * 60)
    print(f"  Channel map: port0→ch{ch_map[0]}  port1→ch{ch_map[1]}  port2→ch{ch_map[2]}  port3→ch{ch_map[3]}")

    # ---- 1. Load session data ----
    patient_dfs = []
    for sess_idx, sess_path in enumerate(args.session):
        npz_path = os.path.join(sess_path, "labeled_data.npz")
        info_path = os.path.join(sess_path, "session_info.json")
        if not os.path.exists(npz_path):
            print(f"ERROR: {npz_path} not found. Run label_session.py first.")
            sys.exit(1)

        print(f"\nLoading session {sess_idx+1}/{len(args.session)}: {sess_path}")
        npz = np.load(npz_path)
        X_raw = npz["X"]              # (N, 4)
        y_session = npz["y"]          # (N,) {close:0, open:1, rest:2}
        timestamps = npz["timestamps"]

        with open(info_path) as f:
            session_info = json.load(f)
        fs = session_info.get("approx_sample_rate_hz", 1000)

        print(f"  Samples: {len(X_raw):,}  Sample rate: {fs:.0f} Hz  "
              f"Duration: {len(X_raw)/fs:.1f}s")
        print(f"  Labels: close={np.sum(y_session==0):,}  "
              f"open={np.sum(y_session==1):,}  rest={np.sum(y_session==2):,}")

        # ---- 2. Bandpass filter ----
        if args.no_bandpass:
            print("  Skipping bandpass (--no-bandpass)")
            filtered = X_raw.copy()
            for ch in range(4):
                filtered[:, ch] -= np.mean(filtered[:, ch])
        else:
            print(f"  Bandpass {LOWCUT}-{HIGHCUT} Hz ({FILTER_ORDER}nd-order Butterworth)...")
            filtered = bandpass_filter(X_raw, fs)

        # ---- 3. Envelope ----
        print(f"  Envelope ({ENV_SMOOTH_MS}ms smoothing)...")
        envelope = compute_envelope(filtered, fs)

        # ---- 4. Extract features ----
        print(f"  Extracting features ({WINDOW_S*1000:.0f}ms window / {STRIDE_S*1000:.0f}ms stride)...")
        sess_df = extract_session_features(filtered, envelope, fs, y_session, timestamps, ch_map)
        sess_df["session"] = f"session{sess_idx+1}"
        patient_dfs.append(sess_df)
        print(f"  Windows: {len(sess_df):,}")
        lc = sess_df["intent_idx"].value_counts().sort_index()
        print(f"  Remapped labels: rest={lc.get(0,0)}  close={lc.get(1,0)}  open={lc.get(2,0)}")

    patient_df = pd.concat(patient_dfs, ignore_index=True)
    print(f"\n  Total patient windows: {len(patient_df):,}")

    # ---- 5. Load GrabMyo dataset ----
    print(f"\nLoading GrabMyo dataset...")
    t0 = time.time()
    grabmyo_df = pd.read_csv(GRABMYO_CSV)
    print(f"  {len(grabmyo_df):,} windows, {grabmyo_df['participant'].nunique()} participants "
          f"({time.time()-t0:.1f}s)")

    with open(GRABMYO_META) as f:
        model_meta = json.load(f)
    target_features = model_meta["feature_cols"]
    print(f"  Target feature set: {len(target_features)} features")

    # ---- 6. Combine datasets ----
    print(f"\nCombining (patient weight: {args.weight}x)...")
    combined = pd.concat([grabmyo_df, patient_df], ignore_index=True)

    # Split patient data into separate "trials" at label transitions
    patient_mask = combined["participant"] == "patient"
    patient_labels = combined.loc[patient_mask, "intent_idx"].values
    trial_num = 1
    trials = [trial_num]
    for i in range(1, len(patient_labels)):
        if patient_labels[i] != patient_labels[i - 1]:
            trial_num += 1
        trials.append(trial_num)
    combined.loc[patient_mask, "trial"] = trials

    # ---- 7. Feature engineering (matching saved 140-feature model) ----
    print("\nFeature engineering...")
    combined = engineer_features_for_saved_model(combined)

    missing = [f for f in target_features if f not in combined.columns]
    if missing:
        print(f"  WARNING: {len(missing)} missing features filled with 0: {missing[:5]}...")
        for f in missing:
            combined[f] = 0.0

    feature_cols = target_features

    # ---- 8. Patient normalisation stats (all 140 features, for real-time) ----
    patient_rows = combined[combined["participant"] == "patient"]
    patient_norm_stats = {
        "mean": patient_rows[feature_cols].mean().to_dict(),
        "std": (patient_rows[feature_cols].std() + 1e-8).to_dict(),
    }

    # ---- 9. Train ----
    X_all = combined[feature_cols].values
    y_all = combined["intent_idx"].values.astype(int)

    weights = np.ones(len(combined))
    weights[combined["participant"] == "patient"] = args.weight

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    print("\nTraining adapted HGB...")
    clf = HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_leaf_nodes=255,
        max_iter=2500,
        min_samples_leaf=20,
        l2_regularization=0.01,
        max_depth=18,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        class_weight="balanced",
    )
    t0 = time.time()
    clf.fit(X_scaled, y_all, sample_weight=weights)
    print(f"  {clf.n_iter_} iterations in {time.time()-t0:.1f}s")

    # ---- 10. Evaluate ----
    pm = (combined["participant"] == "patient").values
    patient_acc = accuracy_score(y_all[pm], clf.predict(X_scaled[pm]))

    grabmyo_parts = combined.loc[~pm, "participant"].unique()
    rng = np.random.RandomState(42)
    test_parts = rng.choice(grabmyo_parts, size=max(1, len(grabmyo_parts) // 10), replace=False)
    tm = combined["participant"].isin(test_parts).values & ~pm
    grabmyo_acc = accuracy_score(y_all[tm], clf.predict(X_scaled[tm]))

    print(f"\n  Patient accuracy:   {patient_acc:.1%}")
    print(f"  GrabMyo cross-subj: {grabmyo_acc:.1%}")
    print(classification_report(
        y_all[pm], clf.predict(X_scaled[pm]),
        target_names=["rest", "close", "open"],
    ))

    # ---- 11. Save ----
    output_path = args.output or os.path.join(ROOT, "exohand_adapted_model.pkl")
    model_data = {
        "model_type": "adapted_hgb",
        "model": clf,
        "scaler": scaler,
        "feature_names": feature_cols,
        "window_ms": int(WINDOW_S * 1000),
        "stride_ms": int(STRIDE_S * 1000),
        "label_names": ["close", "open", "rest"],
        "grabmyo_label_names": ["rest", "close", "open"],
        "channel_map": ch_map,
        "patient_norm_stats": patient_norm_stats,
        "bandpass_order": FILTER_ORDER,
        "bandpass_lowcut": LOWCUT,
        "bandpass_highcut": HIGHCUT,
        "env_smooth_ms": ENV_SMOOTH_MS,
        "sample_rate_hint": fs,
        "patient_accuracy": patient_acc,
        "grabmyo_accuracy": grabmyo_acc,
    }
    joblib.dump(model_data, output_path)
    print(f"\nSaved: {output_path}")
    print(f"  model_type=adapted_hgb  features={len(feature_cols)}  "
          f"patient={patient_acc:.1%}  grabmyo={grabmyo_acc:.1%}")


if __name__ == "__main__":
    main()
