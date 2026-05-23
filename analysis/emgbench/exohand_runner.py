"""
ExoHand HGB runner against EMGBench datasets.

Hybrid integration:
  1. Use EMGBench's Setup.py for dataset download + utils selection
  2. Use EMGBench's Data_Initializer for raw EMG loading (X.data per subject)
  3. SKIP load_images() — substitute our 370-feature extraction:
     - select 4 channels per dataset adapter
     - extract 60 base features per window
     - synthesize META_COLS for the ml.train_hgb_v2 pipeline
     - call engineer_features() → 370 columns
  4. Use EMGBench's Data_Splitter for the LOSO split
  5. Train HistGradientBoostingClassifier on X.train
  6. Stratified slice of X.validation for calibration; refit with weight 100x
  7. Predict on X.test; emit per-fold metrics CSV mergeable with EMGBench baselines

PREREQUISITE: EMGBench must be cloned to a path on sys.path AND its datasets
downloaded. See analysis/emgbench/README.md for setup.

Usage (once setup is complete):
    python analysis/emgbench/exohand_runner.py \\
        --emgbench-root /path/to/emgbench \\
        --dataset capgmyo \\
        --leftout-subject 1 \\
        --calib-n-windows 1200 --calib-weight 100 --fast
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.seed import SEED, seed_everything
from analysis.emgbench.dataset_adapters import get_adapter, INTENT_TO_IDX
from analysis.emgbench.feature_extraction import (
    extract_60_features_per_subject, feature_column_names,
)
from ml.train_hgb_v2 import META_COLS, engineer_features


def import_emgbench(emgbench_root: str):
    """Make EMGBench's Setup / Data / Split_Strategies importable."""
    if not Path(emgbench_root).exists():
        raise FileNotFoundError(f"EMGBench root does not exist: {emgbench_root}")
    sys.path.insert(0, emgbench_root)
    # Re-export the few classes we need; do imports here so missing EMGBench
    # only errors at call-time, not module load time.
    from Setup.Setup import Setup                                       # noqa: F401
    from Data.X_Data import X_Data                                      # noqa: F401
    from Data.Y_Data import Y_Data                                      # noqa: F401
    from Data.Label_Data import Label_Data                              # noqa: F401
    from Data.Combined_Data import Combined_Data                        # noqa: F401
    from importlib import import_module
    return {
        "Setup": Setup,
        "X_Data": X_Data, "Y_Data": Y_Data, "Label_Data": Label_Data,
        "Combined_Data": Combined_Data,
        "load_split_strategy": lambda name: getattr(import_module(f"Split_Strategies.{name}"), name),
    }


def build_emgbench_env(EMG, dataset: str, leftout_subject: int, seed: int):
    """Run Setup.py's argparse + setup_for_dataset to produce an env object."""
    setup = EMG["Setup"]()
    # Synthesize CLI-equivalent args. We use only the safe subset.
    setup.args = argparse.Namespace(
        dataset=dataset,
        seed=seed,
        leftout_subject=leftout_subject,
        leave_one_subject_out=True,
        leave_one_session_out=False,
        force_regression=False,
        partial_dataset_ninapro=True,
        exercises=[1, 2],                       # ninapro default; ignored for other datasets
        turn_off_scaler_normalization=False,
        target_normalize=0.0, target_normalize_subject=0,
        include_transitions=False, transition_classifier=False, transition_classifier_arg=False,
        full_dataset_mcs=False,
        turn_on_unlabeled_domain_adaptation=False,
        domain_generalization=None,
        pretrain_and_finetune=False,
        transfer_learning=False,
        proportion_transfer_learning_from_leftout_subject=0.0,
        proportion_unlabeled_data_from_training_subjects=0.0,
        proportion_unlabeled_data_from_leftout_subject=0.0,
        proportion_data_from_training_subjects=1.0,
        reduce_training_data_size=False, reduced_training_data_size=0,
        reduce_data_for_transfer_learning=1,
        train_test_split_for_time_series=True,
        save_images=False,
        # CNN-specific flags that some Setup paths check; safe defaults
        turn_on_rms=False, rms_input_windowsize=0,
        turn_on_spectrogram=False, turn_on_phase_spectrogram=False,
        turn_on_cwt=False, turn_on_hht=False,
        model="HGB",                            # not in EMGBench's whitelist; we never reach Run_Model
        batch_size=64, epochs=1, gpu=0,
    )
    setup.setup_for_dataset()
    setup.set_exercise()
    env = setup.set_env()
    return env


def replace_emg_with_features(X_data, adapter, env):
    """Walk X.data list, replace each subject's raw windows with 370-feature rows.

    X.data goes from list of (n_windows, n_channels, n_timesteps) per subject
                  to list of (n_windows, 370)                     per subject.
    Y.data and label.data are left untouched at this stage; the dataframe
    constructed in build_engineered_dataframe pulls them from env.
    """
    n_subjects = len(X_data.data)
    feature_lists = []
    for subj_idx, emg in enumerate(X_data.data):
        emg = np.asarray(emg)
        if emg.ndim != 3:
            raise ValueError(
                f"Subject {subj_idx} EMG has shape {emg.shape}; expected 3D "
                "(n_windows, n_channels, n_timesteps)"
            )
        f60 = extract_60_features_per_subject(
            emg_subject=emg,
            selected_channels=adapter.selected_channels,
            channel_name_map=adapter.channel_name_map,
            fs=adapter.sample_rate_hz,
        )
        feature_lists.append(f60)
        print(f"  subj {subj_idx + 1}/{n_subjects}: extracted {f60.shape[0]} windows × 60 features")
    return feature_lists


def build_engineered_dataframe(feature_lists, label_arrays, adapter):
    """Construct the dataframe ml.train_hgb_v2.engineer_features() expects.

    Mirrors GrabMyo CSV schema:
      participant, session, gesture, gesture_name, trial, t_rel_s, intent, intent_idx,
      then 60 channel-feature columns.

    For EMGBench (no natural trial structure), we synthesize:
      session = 'session1'
      trial = window index within the subject
      t_rel_s = 0.0 (will be overwritten if engineer_features needs ordering)
      gesture/gesture_name = mapped 3-class intent (collapsed before z-score so the
        normalization sees gesture-collapsed groups; downstream pipeline doesn't
        care about the original 16-class id).

    Drops windows whose raw label has no mapping in adapter.gesture_to_intent.
    """
    base_cols = feature_column_names(adapter.channel_name_map)
    pieces = []
    for subj_idx, (f60, y_raw) in enumerate(zip(feature_lists, label_arrays)):
        # y_raw: shape (n_windows,) — depends on Y_Data encoding (one-hot vs class index)
        y_arr = np.asarray(y_raw)
        if y_arr.ndim > 1:
            y_arr = np.argmax(y_arr, axis=-1)

        # Map dataset labels → intent strings; drop unmapped windows
        gesture_labels_dataset = list(getattr(adapter, "raw_gesture_labels", []))
        # Fallback: if adapter doesn't enumerate raw labels, we use the indices directly
        # and the user's gesture_to_intent must be keyed by index.
        intents = []
        keep_mask = np.zeros(len(y_arr), dtype=bool)
        for w_idx, raw_label in enumerate(y_arr):
            key = gesture_labels_dataset[raw_label] if gesture_labels_dataset else int(raw_label)
            intent_str = adapter.gesture_to_intent.get(key)
            if intent_str is not None:
                intents.append(intent_str)
                keep_mask[w_idx] = True

        if not keep_mask.any():
            print(f"  subj {subj_idx + 1}: no windows match gesture mapping — skipping")
            continue

        f60_kept = f60[keep_mask]
        intents_arr = np.array(intents)
        intent_idx = np.array([INTENT_TO_IDX[s] for s in intents_arr], dtype=np.int64)

        df = pd.DataFrame(f60_kept, columns=base_cols)
        df["participant"] = f"subj{subj_idx + 1}"
        df["session"] = "session1"
        df["gesture"] = intent_idx                       # collapsed numeric — used for groupby
        df["gesture_name"] = intents_arr
        df["trial"] = np.arange(len(df))                  # synthetic — one window = one trial
        df["t_rel_s"] = 0.0
        df["intent"] = intents_arr
        df["intent_idx"] = intent_idx
        pieces.append(df)

    if not pieces:
        raise RuntimeError("No subjects survived gesture mapping. Check adapter.gesture_to_intent.")

    df = pd.concat(pieces, ignore_index=True)
    return df


def make_classifier(seed: int, fast: bool = False) -> HistGradientBoostingClassifier:
    if fast:
        return HistGradientBoostingClassifier(
            learning_rate=0.1, max_leaf_nodes=63, max_iter=300, min_samples_leaf=20,
            l2_regularization=0.01, max_depth=10, random_state=seed,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
            class_weight="balanced",
        )
    return HistGradientBoostingClassifier(
        learning_rate=0.03, max_leaf_nodes=255, max_iter=2500, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=18, random_state=seed,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
        class_weight="balanced",
    )


def stratified_sample(y: np.ndarray, n_target: int, rng: np.random.RandomState) -> np.ndarray:
    classes = np.unique(y)
    per_class = max(1, n_target // len(classes))
    picked = []
    for c in classes:
        idx = np.where(y == c)[0]
        take = min(per_class, len(idx))
        picked.extend(rng.choice(idx, size=take, replace=False))
    picked = list(set(picked))
    if len(picked) < n_target:
        leftover = np.setdiff1d(np.arange(len(y)), picked)
        topup = min(n_target - len(picked), len(leftover))
        if topup > 0:
            picked.extend(rng.choice(leftover, size=topup, replace=False))
    return np.array(sorted(picked))


def eval_fold(df_engineered, leftout_subject_code, calib_n_windows, calib_weight, fast, seed):
    """LOSO fold on the engineered 370-feature dataframe.

    Mirrors loso_eval.py's pattern: train on all subjects except leftout,
    eval on leftout, then stratified slice of leftout for calibration refit.
    """
    feature_cols = [c for c in df_engineered.columns if c not in META_COLS]
    train_mask = df_engineered["participant"] != leftout_subject_code
    test_mask = df_engineered["participant"] == leftout_subject_code

    X_train = df_engineered.loc[train_mask, feature_cols].values.astype(np.float32)
    y_train = df_engineered.loc[train_mask, "intent_idx"].values.astype(np.int64)
    X_test = df_engineered.loc[test_mask, feature_cols].values.astype(np.float32)
    y_test = df_engineered.loc[test_mask, "intent_idx"].values.astype(np.int64)

    if len(X_test) == 0:
        raise RuntimeError(f"Leftout subject {leftout_subject_code} has no test data after gesture mapping.")

    rng = np.random.RandomState(seed)
    calib_n = min(calib_n_windows, len(X_test) // 2)
    calib_idx = stratified_sample(y_test, calib_n, rng)
    eval_mask = np.ones(len(X_test), dtype=bool)
    eval_mask[calib_idx] = False
    X_calib, y_calib = X_test[calib_idx], y_test[calib_idx]
    X_eval, y_eval = X_test[eval_mask], y_test[eval_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_calib_s = scaler.transform(X_calib).astype(np.float32)
    X_eval_s = scaler.transform(X_eval).astype(np.float32)

    clf = make_classifier(seed, fast=fast)
    t0 = time.time()
    clf.fit(X_train_s, y_train)
    fit_time = time.time() - t0
    pred_no_cal = clf.predict(X_eval_s)
    acc_no_cal = accuracy_score(y_eval, pred_no_cal)
    f1_no_cal = f1_score(y_eval, pred_no_cal, average="macro")

    X_all = np.vstack([X_train_s, X_calib_s])
    y_all = np.concatenate([y_train, y_calib])
    w_all = np.ones(len(X_all), dtype=np.float32)
    w_all[len(X_train_s):] = calib_weight

    clf_cal = make_classifier(seed, fast=fast)
    t1 = time.time()
    clf_cal.fit(X_all, y_all, sample_weight=w_all)
    fit_time_cal = time.time() - t1
    pred_cal = clf_cal.predict(X_eval_s)
    acc_cal = accuracy_score(y_eval, pred_cal)
    f1_cal = f1_score(y_eval, pred_cal, average="macro")

    return {
        "leftout_subject": leftout_subject_code,
        "n_train_windows": int(len(X_train)),
        "n_calib_windows": int(calib_n),
        "n_eval_windows": int(eval_mask.sum()),
        "acc_no_cal": acc_no_cal,
        "f1_no_cal": f1_no_cal,
        "acc_with_cal": acc_cal,
        "f1_with_cal": f1_cal,
        "delta_acc": acc_cal - acc_no_cal,
        "fit_time_s_no_cal": fit_time,
        "fit_time_s_with_cal": fit_time_cal,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emgbench-root", required=True,
                        help="Path to a cloned jehanyang/emgbench checkout.")
    parser.add_argument("--dataset", required=True,
                        choices=["capgmyo", "hyser", "myoarmbanddataset",
                                 "ninapro-db5", "uciemg", "flexwear-hd"])
    parser.add_argument("--leftout-subject", type=int, default=1,
                        help="1-indexed leftout subject (matches EMGBench's leftout_subject arg).")
    parser.add_argument("--calib-n-windows", type=int, default=1200)
    parser.add_argument("--calib-weight", type=float, default=100.0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    seed_everything(SEED)
    EMG = import_emgbench(args.emgbench_root)
    adapter = get_adapter(args.dataset)
    print(f"=== EMGBench × ExoHand on {args.dataset} (leftout subject {args.leftout_subject}) ===")
    print(f"  fs={adapter.sample_rate_hz} Hz  n_channels_total={adapter.n_channels_total}")
    print(f"  selected_channels={list(adapter.selected_channels)}  name_map={list(adapter.channel_name_map)}")
    if adapter.notes:
        print(f"  NOTES: {adapter.notes}")

    # 1. EMGBench env
    print("\n[1/6] Building EMGBench env + dataset...")
    env = build_emgbench_env(EMG, args.dataset, args.leftout_subject, SEED)

    # 2. Load raw EMG
    print("[2/6] Loading raw EMG (this triggers dataset download if not cached)...")
    X = EMG["X_Data"](env)
    Y = EMG["Y_Data"](env)
    label = EMG["Label_Data"](env)
    combined = EMG["Combined_Data"](X, Y, label, env)
    combined.load_data()
    print(f"  n_subjects = {len(X.data)}  per-subject shape = {np.asarray(X.data[0]).shape}")

    # 3. Substitute load_images with our feature extraction
    print(f"\n[3/6] Extracting 60 base features per window per subject...")
    feature_lists = replace_emg_with_features(X, adapter, env)

    # 4. Build engineered dataframe and run engineer_features (370 cols)
    print(f"\n[4/6] Building dataframe + applying train_hgb_v2 feature engineering...")
    df = build_engineered_dataframe(feature_lists, Y.data, adapter)
    print(f"  pre-engineering shape: {df.shape}  participants: {df['participant'].nunique()}")
    df = engineer_features(df)
    print(f"  post-engineering shape: {df.shape}")

    # 5. LOSO eval — for now just the requested leftout subject
    print(f"\n[5/6] Training HGB + calibration refit for leftout={args.leftout_subject}...")
    leftout_code = f"subj{args.leftout_subject}"
    result = eval_fold(
        df, leftout_code,
        calib_n_windows=args.calib_n_windows,
        calib_weight=args.calib_weight,
        fast=args.fast,
        seed=SEED,
    )

    # 6. Output
    out_csv = args.out or f"analysis/emgbench/results_{args.dataset}_leftout{args.leftout_subject}.csv"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result | {"dataset": args.dataset}]).to_csv(out_csv, index=False)
    print(f"\n[6/6] Result:")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"  CSV: {out_csv}")


if __name__ == "__main__":
    main()
