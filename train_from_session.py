#!/usr/bin/env python3
"""
train_from_session.py — Train ExoHand model from labeled session data.

Extracts windowed features (RMS, MAV, WL, ZC, SSC, ENV_RMS + temporal),
trains HistGradientBoostingClassifier, reports metrics.

Usage:
    python train_from_session.py --sessions sessions/2025-02-14_14-30/
    python train_from_session.py --sessions sessions/sess1/ sessions/sess2/
"""

import argparse
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


LABEL_NAMES = ["close", "open", "rest"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_session(session_dir):
    """Load labeled_data.npz and session_info.json from a session."""
    npz_path = os.path.join(session_dir, "labeled_data.npz")
    info_path = os.path.join(session_dir, "session_info.json")

    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found. Run label_session.py first.")
        sys.exit(1)

    d = np.load(npz_path)
    with open(info_path) as f:
        info = json.load(f)

    return d["X"], d["y"], d["timestamps"], info


def extract_window_features(window):
    """
    Extract features from a single window of shape (win_samples, n_channels).
    Returns a 1D feature vector.

    Per channel: RMS, MAV, WL, ZC, SSC, ENV_RMS (6 features × 4 channels = 24)
    """
    n_channels = window.shape[1]
    features = []

    for ch in range(n_channels):
        w = window[:, ch]
        n = len(w)

        # RMS
        rms = np.sqrt(np.mean(w ** 2))

        # MAV (mean absolute value)
        mav = np.mean(np.abs(w))

        # WL (waveform length)
        wl = np.sum(np.abs(np.diff(w)))

        # ZC (zero crossings) — relative to mean
        w_centered = w - np.mean(w)
        prod = w_centered[:-1] * w_centered[1:]
        zc = np.sum(prod < 0)

        # SSC (slope sign changes)
        if n >= 3:
            d1 = w[1:-1] - w[:-2]
            d2 = w[1:-1] - w[2:]
            ssc = np.sum(d1 * d2 > 0)
        else:
            ssc = 0

        # ENV_RMS (envelope RMS — RMS of absolute values, effectively same as RMS for EMG)
        env_rms = np.sqrt(np.mean(np.abs(w) ** 2))

        features.extend([rms, mav, wl, zc, ssc, env_rms])

    return np.array(features, dtype=np.float64)


def get_feature_names():
    """Return feature names for 4-channel setup."""
    names = []
    for ch in range(4):
        for feat in ["rms", "mav", "wl", "zc", "ssc", "env_rms"]:
            names.append(f"ch{ch}_{feat}")
    return names


def get_temporal_feature_names():
    """Return names including temporal features."""
    base = get_feature_names()
    temporal = []
    # Temporal features on envelope (env_rms) columns
    for ch in range(4):
        col = f"ch{ch}_env_rms"
        temporal.extend([f"{col}_prev", f"{col}_delta", f"{col}_roll3"])
    return base + temporal


def extract_features_from_session(X, y, timestamps, sample_rate,
                                   window_ms=50, stride_ms=10):
    """
    Extract windowed features with temporal context.

    Returns: (features, labels) — numpy arrays
    """
    win_samples = max(1, int(window_ms / 1000.0 * sample_rate))
    stride_samples = max(1, int(stride_ms / 1000.0 * sample_rate))

    print(f"  Window: {window_ms}ms = {win_samples} samples, "
          f"stride: {stride_ms}ms = {stride_samples} samples")

    base_features = []
    window_labels = []

    n = len(X)
    for start in range(0, n - win_samples + 1, stride_samples):
        end = start + win_samples
        window = X[start:end]

        # Label = majority label in window
        window_y = y[start:end]
        label = np.bincount(window_y, minlength=3).argmax()

        feat = extract_window_features(window)
        base_features.append(feat)
        window_labels.append(label)

    base_features = np.array(base_features)
    window_labels = np.array(window_labels)

    # Add temporal features (_prev, _delta, _roll3) on env_rms columns
    n_base = len(get_feature_names())
    env_rms_indices = []
    feat_names = get_feature_names()
    for i, name in enumerate(feat_names):
        if name.endswith("_env_rms"):
            env_rms_indices.append(i)

    temporal_cols = []
    for idx in env_rms_indices:
        col = base_features[:, idx]

        # _prev: shifted by 1, first = 0
        prev_col = np.zeros_like(col)
        prev_col[1:] = col[:-1]

        # _delta
        delta_col = col - prev_col

        # _roll3: rolling mean of 3
        roll3_col = np.zeros_like(col)
        for i in range(len(col)):
            start_r = max(0, i - 2)
            roll3_col[i] = col[start_r:i + 1].mean()

        temporal_cols.extend([prev_col, delta_col, roll3_col])

    temporal_array = np.column_stack(temporal_cols) if temporal_cols else np.empty((len(base_features), 0))
    all_features = np.hstack([base_features, temporal_array])

    return all_features, window_labels


def train_model(X_train, y_train, X_test, y_test):
    """Train HistGradientBoostingClassifier and evaluate."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=200,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    return model, scaler, y_pred, acc


def main():
    parser = argparse.ArgumentParser(description="Train ExoHand model from session data")
    parser.add_argument("--sessions", nargs="+", required=True,
                        help="Path(s) to session directories")
    parser.add_argument("--window-ms", type=int, default=50, help="Window size in ms (default 50)")
    parser.add_argument("--stride-ms", type=int, default=10, help="Stride in ms (default 10)")
    parser.add_argument("--output", default=None, help="Output model path (default: exohand_model.pkl)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(SCRIPT_DIR, "exohand_model.pkl")

    # Load all sessions
    all_features = []
    all_labels = []
    session_boundaries = []  # (start_idx, end_idx) per session for cross-session eval

    for sess_dir in args.sessions:
        print(f"\nLoading session: {sess_dir}")
        X, y, ts, info = load_session(sess_dir)
        sample_rate = info.get("approx_sample_rate_hz", 1000)

        print(f"  {len(X)} samples, ~{sample_rate} Hz")
        print(f"  Labels: close={np.sum(y==0)}, open={np.sum(y==1)}, rest={np.sum(y==2)}")

        features, labels = extract_features_from_session(
            X, y, ts, sample_rate, args.window_ms, args.stride_ms
        )
        print(f"  Extracted {len(features)} windows, {features.shape[1]} features")

        start = len(all_labels)
        all_features.append(features)
        all_labels.append(labels)
        session_boundaries.append((start, start + len(labels)))

    X_all = np.vstack(all_features)
    y_all = np.concatenate(all_labels)

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(X_all)} windows, {X_all.shape[1]} features")
    print(f"Class distribution:")
    for i, name in enumerate(LABEL_NAMES):
        count = (y_all == i).sum()
        print(f"  {name}: {count} ({100*count/len(y_all):.1f}%)")
    print(f"{'='*60}")

    # Train/test split: 80/20 stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    print(f"\nTraining: {len(X_train)} windows")
    print(f"Testing:  {len(X_test)} windows")

    model, scaler, y_pred, acc = train_model(X_train, y_train, X_test, y_test)

    print(f"\nAccuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Cross-session evaluation (if multiple sessions)
    cross_session_results = []
    if len(args.sessions) > 1:
        print(f"\n{'='*60}")
        print("CROSS-SESSION EVALUATION")
        print(f"{'='*60}")

        for i in range(len(args.sessions)):
            for j in range(len(args.sessions)):
                if i == j:
                    continue
                si_start, si_end = session_boundaries[i]
                sj_start, sj_end = session_boundaries[j]

                X_tr = X_all[si_start:si_end]
                y_tr = y_all[si_start:si_end]
                X_te = X_all[sj_start:sj_end]
                y_te = y_all[sj_start:sj_end]

                _, _, y_p, cross_acc = train_model(X_tr, y_tr, X_te, y_te)

                sess_i_name = os.path.basename(args.sessions[i].rstrip("/"))
                sess_j_name = os.path.basename(args.sessions[j].rstrip("/"))
                print(f"  Train on {sess_i_name} → Test on {sess_j_name}: {cross_acc:.4f}")
                cross_session_results.append((sess_i_name, sess_j_name, cross_acc))

    # Save model
    # Package model, scaler, and feature metadata together
    model_data = {
        "model": model,
        "scaler": scaler,
        "feature_names": get_temporal_feature_names(),
        "window_ms": args.window_ms,
        "stride_ms": args.stride_ms,
        "label_names": LABEL_NAMES,
    }
    joblib.dump(model_data, args.output)
    print(f"\nModel saved: {args.output}")

    # Save training report
    report_path = os.path.join(SCRIPT_DIR, "training_report.txt")
    with open(report_path, "w") as f:
        f.write("TRAINING REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Sessions: {args.sessions}\n")
        f.write(f"Total windows: {len(X_all)}\n")
        f.write(f"Features: {X_all.shape[1]}\n")
        f.write(f"Window: {args.window_ms}ms, Stride: {args.stride_ms}ms\n\n")
        f.write(f"Class distribution:\n")
        for i, name in enumerate(LABEL_NAMES):
            count = (y_all == i).sum()
            f.write(f"  {name}: {count} ({100*count/len(y_all):.1f}%)\n")
        f.write(f"\nTrain/Test: {len(X_train)}/{len(X_test)}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=LABEL_NAMES))
        f.write(f"\nConfusion Matrix:\n{cm}\n")
        if cross_session_results:
            f.write(f"\nCross-session results:\n")
            for si, sj, ca in cross_session_results:
                f.write(f"  {si} → {sj}: {ca:.4f}\n")
    print(f"Report saved: {report_path}")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix (Accuracy: {acc:.2%})")
    plt.tight_layout()
    cm_path = os.path.join(SCRIPT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Plot saved: {cm_path}")


if __name__ == "__main__":
    main()
