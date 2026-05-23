"""
Per-window feature extraction for EMGBench-format EMG data.

Mirrors `ml/preprocessing_grabmyo.py:extract_features` semantics, but operates
on already-windowed arrays of shape (n_windows, n_channels, n_timesteps) — the
shape EMGBench's `Combined_Data.load_data` produces after pre-windowing.

Output: 15 features × 4 selected channels = 60 base features per window.
Feature names follow the same `ch<idx>_<name>` convention as preprocessing_grabmyo
so downstream `ml.train_hgb_v2.engineer_features` can be applied directly.
"""

from typing import Sequence

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt

LOWCUT = 20.0
HIGHCUT = 450.0
FILTER_ORDER = 4
ENV_SMOOTH_MS = 50.0


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    return butter(order, [low, high], btype="band")


def preprocess_windows(emg: np.ndarray, fs: float) -> tuple:
    """Bandpass + envelope on (n_windows, n_channels, n_timesteps).

    Returns (filtered, envelope) — both same shape as input.
    Filtering applied per-window to avoid leaking signal across window boundaries.
    """
    nw, nc, nt = emg.shape
    b, a = butter_bandpass(LOWCUT, HIGHCUT, fs, FILTER_ORDER)
    env_win = max(1, int((ENV_SMOOTH_MS / 1000.0) * fs))
    env_kernel = np.ones(env_win) / env_win

    filtered = np.zeros_like(emg, dtype=np.float64)
    envelope = np.zeros_like(emg, dtype=np.float64)

    for w in range(nw):
        for c in range(nc):
            x = emg[w, c].astype(np.float64)
            x = x - np.mean(x)
            xf = filtfilt(b, a, x, padtype="odd", padlen=min(3 * max(len(b), len(a)), nt - 1))
            filtered[w, c] = xf
            rect = np.abs(xf)
            envelope[w, c] = np.convolve(rect, env_kernel, mode="same")
    return filtered, envelope


def _features_for_one_window(w: np.ndarray, env: np.ndarray, fs: float) -> dict:
    """15-feature dict for one (n_timesteps,) signal + its envelope."""
    Nw = len(w)
    rms = float(np.sqrt(np.mean(w ** 2)))
    mav = float(np.mean(np.abs(w)))
    var = float(np.var(w))
    wl = float(np.sum(np.abs(np.diff(w))))
    maxamp = float(np.max(np.abs(w)))
    thr = 0.01 * maxamp if maxamp > 0 else 0.0
    prod = w[:-1] * w[1:]
    zc = float(np.sum((prod < 0) & (np.abs(w[:-1] - w[1:]) > thr)))
    d1 = w[1:-1] - w[:-2]
    d2 = w[1:-1] - w[2:]
    ssc = float(np.sum((d1 * d2 > 0) & (np.abs(d1) > thr) & (np.abs(d2) > thr)))
    wamp = float(np.sum(np.abs(np.diff(w)) > thr))
    iemg = float(np.sum(np.abs(w)))

    win_han = np.hanning(Nw)
    fft_vals = np.abs(rfft(w * win_han))
    freqs = rfftfreq(Nw, 1.0 / fs)
    psd = fft_vals ** 2
    psd_sum = psd.sum()
    if psd_sum > 0:
        mean_f = float(np.sum(freqs * psd) / psd_sum)
        cum = np.cumsum(psd)
        median_f = float(freqs[np.searchsorted(cum, 0.5 * cum[-1])])
    else:
        mean_f = 0.0
        median_f = 0.0

    return {
        "rms": rms, "mav": mav, "var": var, "wl": wl, "maxamp": maxamp,
        "zc": zc, "ssc": ssc, "wamp": wamp, "iemg": iemg,
        "mean_freq": mean_f, "median_freq": median_f,
        "env_mean": float(np.mean(env)),
        "env_max": float(np.max(env)),
        "env_std": float(np.std(env)),
        "env_rms": float(np.sqrt(np.mean(env ** 2))),
    }


def extract_60_features_per_subject(
    emg_subject: np.ndarray,
    selected_channels: Sequence[int],
    channel_name_map: Sequence[int],
    fs: float,
) -> np.ndarray:
    """Compute 15-feature × 4-channel = 60-feature matrix per window.

    Args:
        emg_subject: shape (n_windows, n_channels_total, n_timesteps), raw mV.
        selected_channels: indices into the dataset's channel array (length 4).
        channel_name_map: indices to use as ch<N>_<feat> suffixes in column names.
            For GrabMyo compatibility this is [0, 4, 9, 13] regardless of which
            actual dataset channels were selected — that's the canonical naming.
        fs: sampling rate in Hz.

    Returns:
        Feature matrix of shape (n_windows, 60) with column order matching
        the GrabMyo-format CSV (channels iterated in name-map order, each
        channel's 15 features in fixed sequence).
    """
    assert len(selected_channels) == 4 and len(channel_name_map) == 4
    nw = emg_subject.shape[0]
    emg_4 = emg_subject[:, selected_channels, :]    # (nw, 4, nt)
    filtered, envelope = preprocess_windows(emg_4, fs)

    feature_names_per_channel = [
        "rms", "mav", "var", "wl", "maxamp", "zc", "ssc", "wamp", "iemg",
        "mean_freq", "median_freq", "env_mean", "env_max", "env_std", "env_rms",
    ]
    out = np.zeros((nw, 60), dtype=np.float32)
    for w_idx in range(nw):
        col = 0
        for ch_idx, ch_name in enumerate(channel_name_map):
            feats = _features_for_one_window(filtered[w_idx, ch_idx], envelope[w_idx, ch_idx], fs)
            for fname in feature_names_per_channel:
                out[w_idx, col] = feats[fname]
                col += 1
    return out


def feature_column_names(channel_name_map: Sequence[int]) -> list:
    """The 60 column names in the same order as extract_60_features_per_subject."""
    feature_names_per_channel = [
        "rms", "mav", "var", "wl", "maxamp", "zc", "ssc", "wamp", "iemg",
        "mean_freq", "median_freq", "env_mean", "env_max", "env_std", "env_rms",
    ]
    cols = []
    for ch in channel_name_map:
        for f in feature_names_per_channel:
            cols.append(f"ch{ch}_{f}")
    return cols
