"""
HGB v2 – Retrain HistGradientBoosting with aggressive temporal + normalisation
improvements to push cross-subject accuracy from ~64 % toward 90 %.

Key changes over train_improved.py:
1.  Per-participant z-score normalisation (removes absolute amplitude differences)
2.  Temporal features on ALL key signal columns (rms, mav, env_rms, wl)
3.  Deeper temporal context: 2 lags, 2 deltas, roll-3 and roll-5
4.  Cross-channel interaction features (pairwise ratios & differences)
5.  Temporal features on interaction features (ratio dynamics)
6.  Per-session normalisation layer on top of per-participant
7.  Tuned HGB hyperparameters (2500 iters, lower LR, deeper trees)
8.  Multiple random participant splits for robust evaluation

v2.1 changes:
 - max_iter 1000→2500, LR 0.05→0.03 (model was not early-stopping)
 - temporal features on cross-channel ratios (close/open discrimination)
 - per-participant+session normalisation for session drift
 - rank-percentile features for robustness to outliers
"""

import os, json, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings("ignore")

ROOT = "/Users/anshshetty/Library/Mobile Documents/com~apple~CloudDocs/ExoHand/grabmyo"
DATA_CSV = os.path.join(ROOT, "grabmyo_intent_dataset.csv")

META_COLS = [
    "participant", "session", "gesture", "gesture_name",
    "trial", "t_rel_s", "intent", "intent_idx",
]


# ------------------------------------------------------------------ helpers
def participant_split(parts, train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    parts = np.array(parts)
    rng.shuffle(parts)
    n = len(parts)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    return parts[:n_train], parts[n_train:n_train + n_val], parts[n_train + n_val:]


# ------------------------------------------------------------------ feature engineering
def add_per_participant_normalisation(df, feature_cols):
    """Z-score every feature column within each participant.
    This is the single biggest lever for cross-subject generalisation.
    """
    print("  Per-participant z-score normalisation …")
    normed = df.groupby("participant")[feature_cols].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    df[feature_cols] = normed
    return df


def add_per_session_normalisation(df, feature_cols):
    """Additional z-score within participant+session to handle session drift."""
    print("  Per-participant-session normalisation …")
    normed = df.groupby(["participant", "session"])[feature_cols].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    # Store as separate columns so the model gets both views
    for col in feature_cols[:45]:  # Only base features (not temporal) to limit bloat
        df[f"{col}_sess_norm"] = normed[col]
    return df


def add_temporal_features(df):
    """Add lag, delta, and rolling-mean features on key signal columns."""
    print("  Adding temporal features …")
    df = df.sort_values(["participant", "session", "trial", "t_rel_s"]).reset_index(drop=True)

    # Columns to add temporal context for
    key_cols = [c for c in df.columns if c.endswith(("_rms", "_mav", "_wl", "_env_rms"))
                and c not in META_COLS]
    key_cols = sorted(set(key_cols))

    grp = df.groupby(["participant", "session", "trial"])
    for col in key_cols:
        g = grp[col]
        # Lag features
        df[f"{col}_prev"]  = g.shift(1).fillna(0)
        df[f"{col}_prev2"] = g.shift(2).fillna(0)
        # Delta (velocity)
        df[f"{col}_delta"]  = df[col] - df[f"{col}_prev"]
        # Acceleration (delta of delta)
        df[f"{col}_accel"] = df[f"{col}_delta"] - g.shift(1).fillna(0).diff().fillna(0)
        # Rolling means
        df[f"{col}_roll3"] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f"{col}_roll5"] = g.transform(lambda x: x.rolling(5, min_periods=1).mean())

    return df


def add_cross_channel_features(df):
    """Add pairwise ratios and differences between channels."""
    print("  Adding cross-channel features …")
    channels = ["ch0", "ch4", "ch9", "ch13"]
    signals = ["rms", "mav", "env_rms"]

    for sig in signals:
        cols = {ch: f"{ch}_{sig}" for ch in channels if f"{ch}_{sig}" in df.columns}
        ch_list = list(cols.keys())
        for i in range(len(ch_list)):
            for j in range(i + 1, len(ch_list)):
                ci, cj = ch_list[i], ch_list[j]
                df[f"{ci}_{cj}_{sig}_ratio"] = df[cols[ci]] / (df[cols[cj]] + 1e-8)
                df[f"{ci}_{cj}_{sig}_diff"] = df[cols[ci]] - df[cols[cj]]

    # Aggregate activity features
    env_cols = [c for c in df.columns if c.endswith("env_rms") and "_prev" not in c
                and "_delta" not in c and "_roll" not in c and "_accel" not in c
                and "_ratio" not in c and "_diff" not in c]
    df["rest_activity"] = df[env_cols].sum(axis=1)

    rms_cols = {ch: f"{ch}_rms" for ch in channels if f"{ch}_rms" in df.columns}
    df["flexor_activity"]  = df[rms_cols.get("ch0", "ch0_rms")] + df[rms_cols.get("ch9", "ch9_rms")]
    df["extensor_activity"] = df[rms_cols.get("ch4", "ch4_rms")] + df[rms_cols.get("ch13", "ch13_rms")]
    df["flexor_extensor_ratio"] = df["flexor_activity"] / (df["extensor_activity"] + 1e-8)

    return df


def add_temporal_on_interactions(df):
    """Add temporal features on cross-channel ratios and differences.
    These dynamics are critical for close-vs-open discrimination."""
    print("  Adding temporal features on interaction columns …")
    interaction_cols = [c for c in df.columns if "_ratio" in c or "_diff" in c]
    # Also add temporal on flexor/extensor ratio
    interaction_cols += ["flexor_extensor_ratio", "flexor_activity", "extensor_activity", "rest_activity"]
    interaction_cols = [c for c in interaction_cols if c in df.columns]

    grp = df.groupby(["participant", "session", "trial"])
    for col in interaction_cols:
        g = grp[col]
        df[f"{col}_prev"]  = g.shift(1).fillna(0)
        df[f"{col}_delta"] = df[col] - df[f"{col}_prev"]
        df[f"{col}_roll3"] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())

    return df


def add_rank_features(df):
    """Add within-participant rank (percentile) features for key columns.
    More robust to outliers than z-scores for some features."""
    print("  Adding rank-percentile features …")
    rank_cols = ["ch0_env_rms", "ch4_env_rms", "ch9_env_rms", "ch13_env_rms",
                 "ch0_rms", "ch4_rms", "ch9_rms", "ch13_rms"]
    rank_cols = [c for c in rank_cols if c in df.columns]

    for col in rank_cols:
        df[f"{col}_pctile"] = df.groupby("participant")[col].rank(pct=True)

    return df


def add_within_trial_position(df):
    """Add normalised position within each trial (0→1)."""
    print("  Adding within-trial position …")
    df["trial_pos"] = df.groupby(["participant", "session", "trial"])["t_rel_s"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    )
    return df


# ------------------------------------------------------------------ main pipeline
def engineer_features(df):
    """Full feature-engineering pipeline."""
    # 1. Temporal features BEFORE normalisation (they need raw ordering)
    df = add_temporal_features(df)

    # 2. Cross-channel interaction features
    df = add_cross_channel_features(df)

    # 3. Temporal features on interaction columns (ratio dynamics)
    df = add_temporal_on_interactions(df)

    # 4. Rank-percentile features (outlier robust)
    df = add_rank_features(df)

    # 5. Within-trial position
    df = add_within_trial_position(df)

    # 6. Per-participant normalisation on ALL numeric feature columns
    feature_cols = [c for c in df.columns if c not in META_COLS]
    df = add_per_participant_normalisation(df, feature_cols)

    # 7. Per-session normalisation layer (base features only, adds _sess_norm cols)
    base_features = [c for c in df.columns if c not in META_COLS
                     and "_prev" not in c and "_delta" not in c
                     and "_roll" not in c and "_accel" not in c
                     and "_ratio" not in c and "_diff" not in c
                     and "_pctile" not in c and "_sess_norm" not in c
                     and c != "trial_pos"]
    df = add_per_session_normalisation(df, [c for c in base_features if c in df.columns])

    return df


def train_and_evaluate(df, seed=42):
    """Train a single HGB model on one participant split and return accuracy."""
    feature_cols = [c for c in df.columns if c not in META_COLS]
    parts = sorted(df["participant"].unique())
    train_p, val_p, test_p = participant_split(parts, seed=seed)

    train_mask = df.participant.isin(train_p)
    val_mask   = df.participant.isin(val_p)
    test_mask  = df.participant.isin(test_p)

    X_train, y_train = df.loc[train_mask, feature_cols].values, df.loc[train_mask, "intent_idx"].values
    X_val,   y_val   = df.loc[val_mask,   feature_cols].values, df.loc[val_mask,   "intent_idx"].values
    X_test,  y_test  = df.loc[test_mask,  feature_cols].values, df.loc[test_mask,  "intent_idx"].values

    # Global StandardScaler (per-participant normalisation already applied)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"\n  Features: {len(feature_cols)}")
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"  Train participants: {list(train_p[:5])} … ({len(train_p)} total)")
    print(f"  Test  participants: {list(test_p)}")

    clf = HistGradientBoostingClassifier(
        learning_rate=0.03,        # lower LR for finer convergence
        max_leaf_nodes=255,        # 127→255, allow more complex trees
        max_iter=2500,             # 1000→2500, model was still learning
        min_samples_leaf=20,       # slightly less regularisation
        l2_regularization=0.01,    # less L2 → let trees fit harder
        max_depth=18,              # 15→18, deeper trees for complex interactions
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,       # more patience with lower LR
        class_weight="balanced",
    )

    t0 = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s  ({clf.n_iter_} iterations)")

    val_acc  = accuracy_score(y_val,  clf.predict(X_val))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    y_pred   = clf.predict(X_test)

    print(f"\n  Val  Accuracy: {val_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    intent_names = ["rest", "close", "open"]
    print(classification_report(y_test, y_pred, target_names=intent_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return clf, scaler, feature_cols, test_acc, y_test, y_pred


def main():
    print("=" * 60)
    print("HGB v2.1 – Temporal + Normalisation + Interaction Dynamics")
    print("=" * 60)

    # ---- Load data ----
    print("\nLoading data …")
    df = pd.read_csv(DATA_CSV)
    print(f"  Raw shape: {df.shape}")
    print(f"  Participants: {df['participant'].nunique()}")
    print(f"  Intent distribution:\n{df['intent_idx'].value_counts().sort_index().to_string()}")

    # ---- Feature engineering ----
    print("\nFeature engineering …")
    df = engineer_features(df)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"  Total features: {len(feature_cols)}")

    # ---- Train on multiple random splits for robustness ----
    seeds = [42, 123, 7]
    all_accs = []

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Split {i+1}/{len(seeds)}  (seed={seed})")
        print(f"{'='*60}")
        clf, scaler, feat_cols, acc, y_test, y_pred = train_and_evaluate(df, seed=seed)
        all_accs.append(acc)

    print(f"\n{'='*60}")
    print("SUMMARY ACROSS SPLITS")
    print(f"{'='*60}")
    for i, (seed, acc) in enumerate(zip(seeds, all_accs)):
        print(f"  Split {i+1} (seed={seed}): {acc:.4f}")
    print(f"  Mean: {np.mean(all_accs):.4f}  Std: {np.std(all_accs):.4f}")

    # ---- Save best model (seed=42 split) ----
    print("\nRetraining final model on seed=42 split …")
    clf, scaler, feat_cols, final_acc, _, _ = train_and_evaluate(df, seed=42)

    out_model  = os.path.join(ROOT, "improved_hgb_model.pkl")
    out_scaler = os.path.join(ROOT, "improved_hgb_scaler.pkl")
    joblib.dump(clf, out_model)
    joblib.dump(scaler, out_scaler)

    # Save metadata
    intent_map = df[["intent", "intent_idx"]].drop_duplicates().sort_values("intent_idx")
    meta = {
        "feature_cols": feat_cols,
        "intent_to_idx": {row["intent"]: int(row["intent_idx"]) for _, row in intent_map.iterrows()},
        "idx_to_intent": {str(int(row["intent_idx"])): row["intent"] for _, row in intent_map.iterrows()},
        "accuracy": final_acc,
    }
    meta_path = os.path.join(ROOT, "improved_hgb_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: {out_model}")
    print(f"Saved: {out_scaler}")
    print(f"Saved: {meta_path}")
    print(f"\nFinal accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()
