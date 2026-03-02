import os
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq

# ============================================================
# CONFIG
# ============================================================
ROOT_DIR = "/Users/anshshetty/Library/Mobile Documents/com~apple~CloudDocs/ExoHand/grabmyo"

# Your selected 4 channels from 32-ch GrabMyo recording
# (indices are 0-based, matching WFDB's p_signal.T)
# F1 (idx 0) = flexor, F5 (idx 4) = extensor, F10 (idx 9) = flexor, F14 (idx 13) = extensor
CHANNELS_TO_USE = [0, 4, 9, 13]   # F1, F5, F10, F14

LOWCUT = 20.0
HIGHCUT = 450.0
FILTER_ORDER = 4

# Stroke-oriented windows: a bit longer, still responsive
WINDOW_LENGTH_S = 0.200   # 200 ms
STEP_LENGTH_S   = 0.050   # 50 ms

# Envelope smoothing window (ms)
ENV_SMOOTH_MS   = 50.0

# Gesture index → name mapping (1..17) from MotionSequence.txt
MOTION_NAMES = [
    "Lateral Prehension",                # 1
    "Thumb Adduction",                   # 2
    "Thumb and Little Finger Opposition",# 3
    "Thumb and Index Finger Opposition", # 4
    "Thumb and Index Finger Extension",  # 5
    "Thumb and Little Finger Extension", # 6
    "Index and Middle Finger Extension", # 7
    "Little Finger Extension",           # 8
    "Index Finger Extension",            # 9
    "Thumb Finger Extension",            # 10
    "Wrist Extension",                   # 11
    "Wrist Flexion",                     # 12
    "Forearm Supination",                # 13
    "Forearm Pronation",                 # 14
    "Hand Open",                         # 15
    "Hand Close",                        # 16
    "Rest"                               # 17
]
# ============================================================


# ============================================================
# UTILS
# ============================================================
def find_all_dat_files(root):
    """
    Walk GrabMyo directory and collect all .dat file paths.
    """
    files = []
    for dirpath, dirs, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".dat"):
                files.append(os.path.join(dirpath, f))
    return sorted(files)


def load_wfdb_trial(dat_path):
    """
    Loads a .dat/.hea pair using WFDB.
    Returns: (data, fs)
      data: (32, N)
      fs: sampling rate (2048 Hz)
    """
    rec = wfdb.rdrecord(dat_path.replace(".dat", ""))
    data = rec.p_signal.T.astype(np.float64)  # shape: (32, N)
    return data, rec.fs


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype="band")
    return b, a


def preprocess_4ch(emg32, fs):
    """
    Select 4 channels, bandpass, rectify, envelope.
    Returns:
        filtered: (C, N) bandpassed signals
        envelope: (C, N) smoothed rectified envelopes
    """
    chs = CHANNELS_TO_USE
    sig = emg32[chs, :]      # (C, N)
    C, N = sig.shape

    filtered = np.zeros_like(sig)
    envelope = np.zeros_like(sig)

    b, a = butter_bandpass(LOWCUT, HIGHCUT, fs, FILTER_ORDER)
    env_win = max(1, int((ENV_SMOOTH_MS / 1000.0) * fs))
    env_kernel = np.ones(env_win) / env_win

    for c in range(C):
        x = sig[c] - np.mean(sig[c])     # de-mean
        xf = filtfilt(b, a, x)          # bandpass
        rect = np.abs(xf)               # full-wave rectification
        env = np.convolve(rect, env_kernel, mode="same")  # smoothed envelope

        filtered[c] = xf
        envelope[c] = env

    return filtered, envelope


def extract_features(filtered, envelope, fs, meta):
    """
    Sliding window feature extraction.

    filtered: (C, N) bandpassed EMG
    envelope: (C, N) smoothed rectified EMG
    fs: sampling rate
    meta: dict with participant/session/gesture/trial info
    """
    C, N = filtered.shape
    win = int(WINDOW_LENGTH_S * fs)
    step = int(STEP_LENGTH_S * fs)

    rows = []

    for start in range(0, N - win + 1, step):
        end = start + win
        t_rel = (start + end) / 2.0 / fs

        row = {
            "participant": meta["participant"],
            "session": meta["session"],
            "gesture": meta["gesture"],          # numeric 1..17
            "gesture_name": meta["gesture_name"],
            "trial": meta["trial"],
            "t_rel_s": t_rel
        }

        for i, ch in enumerate(CHANNELS_TO_USE):
            w = filtered[i, start:end]
            env = envelope[i, start:end]
            Nw = len(w)

            # ---------- time-domain features ----------
            rms = np.sqrt(np.mean(w**2))
            mav = np.mean(np.abs(w))
            var = np.var(w)
            wl = np.sum(np.abs(np.diff(w)))
            maxamp = np.max(np.abs(w))
            thr = 0.01 * maxamp if maxamp > 0 else 0.0

            # zero crossings
            prod = w[:-1] * w[1:]
            zc = np.sum((prod < 0) & (np.abs(w[:-1] - w[1:]) > thr))

            # slope sign changes
            d1 = w[1:-1] - w[:-2]
            d2 = w[1:-1] - w[2:]
            ssc = np.sum(
                (d1 * d2 > 0)
                & (np.abs(d1) > thr)
                & (np.abs(d2) > thr)
            )

            # Willison amplitude
            wamp = np.sum(np.abs(np.diff(w)) > thr)

            # integrated EMG
            iemg = np.sum(np.abs(w))

            # ---------- frequency features ----------
            win_han = np.hanning(Nw)
            fft_vals = np.abs(rfft(w * win_han))
            freqs = rfftfreq(Nw, 1 / fs)
            psd = fft_vals**2
            psd_sum = psd.sum()

            if psd_sum > 0:
                mean_f = np.sum(freqs * psd) / psd_sum
                cum = np.cumsum(psd)
                median_f = freqs[np.searchsorted(cum, 0.5 * cum[-1])]
            else:
                mean_f = 0.0
                median_f = 0.0

            # ---------- envelope-centric features ----------
            env_mean = np.mean(env)
            env_max = np.max(env)
            env_std = np.std(env)
            env_rms = np.sqrt(np.mean(env**2))

            pref = f"ch{ch}_"
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

    return rows


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    dat_files = find_all_dat_files(ROOT_DIR)
    print(f"Found {len(dat_files)} trials.")

    all_rows = []

    for path in dat_files:
        print("Processing:", path)

        fname = os.path.basename(path)
        parts = fname.replace(".dat", "").split("_")

        # example: session1_subject5_gesture3_trial7.dat
        session = parts[0]          # "session1"
        participant = parts[1]      # "subject5"
        gesture_num = int(parts[2].replace("gesture", ""))  # 1..17
        trial = int(parts[3].replace("trial", ""))

        # map numeric gesture → name from MOTION_NAMES
        if 1 <= gesture_num <= len(MOTION_NAMES):
            gesture_name = MOTION_NAMES[gesture_num - 1]
        else:
            gesture_name = "Unknown"

        meta = {
            "session": session,
            "participant": participant,
            "gesture": gesture_num,
            "gesture_name": gesture_name,
            "trial": trial
        }

        emg32, fs = load_wfdb_trial(path)
        filtered, envelope = preprocess_4ch(emg32, fs)

        rows = extract_features(filtered, envelope, fs, meta)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print("Final dataset shape:", df.shape)

    out_csv = os.path.join(ROOT_DIR, "grabmyo_4ch_features_intentready.csv")
    df.to_csv(out_csv, index=False)

    print("Saved →", out_csv)


if __name__ == "__main__":
    main()
