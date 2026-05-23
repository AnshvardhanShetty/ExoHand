"""
Cal-data-size ablation: how many cal windows are needed?

Sweep cal data size {12, 24, 36, 60, 90, 120} cal windows per gesture
(≈ {600 ms, 1.2 s, 1.8 s, 3 s, 4.5 s, 6 s} of cued data per gesture × 12 gestures
= {7.2 s, 14 s, 22 s, 36 s, 54 s, 72 s} total cal duration).

Patient-only HGB protocol (no GrabMyo) — the baseline ablation makes more sense
given the patient-only finding from BASELINE_COMPARISON.md. Same balanced test
set per session.

Subset: by default runs all 48 patients (~30 min × 6 settings = 3 hours). With
--subset N runs first N patients only for a quick sensitivity curve.

Output:
  analysis/physiomio/results/cal_size_ablation.csv
  analysis/physiomio/results/cal_size_ablation_summary.md
"""

import argparse
import json
import sys
import time
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
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, GRABMYO_META, TEST_PER_CLASS, CLASSES,
)

OUT_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "cal_size_ablation.csv"
OUT_MD = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "cal_size_ablation_summary.md"

CAL_SIZES = [12, 24, 36, 60, 90, 120]   # windows per gesture
BUFFER_WINDOWS = 3
TEST_PER_CLASS_LOCAL = 39


def make_classifier(seed):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=300, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        class_weight="balanced",
    )


def split_with_cal_size(session_df, cal_per_gesture, test_per_class, rng):
    """Same temporal split as per_session_eval, but cal_per_gesture is variable."""
    cal_idx = []
    test_pool = {0: [], 1: [], 2: []}
    for _, g in session_df.groupby("trial", sort=True):
        cls = int(g["intent_idx"].iloc[0])
        sg = g.sort_values("t_rel_s")
        n = len(sg)
        # Ensure test set fits
        min_n = cal_per_gesture + BUFFER_WINDOWS + test_per_class
        if n >= min_n:
            cal_idx.extend(sg.index[:cal_per_gesture].tolist())
            test_pool[cls].extend(sg.index[cal_per_gesture + BUFFER_WINDOWS:cal_per_gesture + BUFFER_WINDOWS + test_per_class].tolist())
        else:
            # Use proportional split
            cal_n = max(1, (n - BUFFER_WINDOWS - test_per_class))
            cal_n = min(cal_n, cal_per_gesture)
            if cal_n + BUFFER_WINDOWS >= n:
                continue
            cal_idx.extend(sg.index[:cal_n].tolist())
            test_pool[cls].extend(sg.index[cal_n + BUFFER_WINDOWS:].tolist())
    balanced_test = []
    for cls in CLASSES:
        pool = test_pool[cls]
        if len(pool) <= test_per_class:
            balanced_test.extend(pool)
        else:
            balanced_test.extend(rng.choice(pool, size=test_per_class, replace=False).tolist())
    return np.array(sorted(balanced_test)), np.array(sorted(cal_idx))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only the first N patients (e.g. 12). Default: all 48.")
    args = parser.parse_args()

    seed_everything(SEED)
    rng_global = np.random.RandomState(SEED)
    t_start = time.time()

    print("Loading + engineering PhysioMio...")
    df = pd.read_pickle(PHYSIOMIO_PKL)
    eng = engineer_features(df)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]

    patients = sorted(eng["participant"].unique(),
                      key=lambda s: int(s.replace("patient", "")))
    if args.subset:
        patients = patients[:args.subset]
    print(f"Running cal-size ablation on {len(patients)} patients × {len(CAL_SIZES)} sizes")

    rows = []
    for cal_size in CAL_SIZES:
        print(f"\n=== cal_per_gesture = {cal_size} ({cal_size * 50} ms × 12 gestures = {cal_size * 50 * 12 / 1000:.1f} s total) ===")
        for pi, patient in enumerate(patients, 1):
            sessions = sorted(eng[eng["participant"] == patient]["session"].unique())
            for session in sessions:
                s_data = eng[(eng["participant"] == patient) & (eng["session"] == session)].copy()
                test_idx, cal_idx = split_with_cal_size(s_data, cal_size, TEST_PER_CLASS_LOCAL, rng_global)
                if len(test_idx) == 0 or len(cal_idx) == 0:
                    continue
                X_cal = s_data.loc[cal_idx, feature_cols].values.astype(np.float32)
                y_cal = s_data.loc[cal_idx, "intent_idx"].values.astype(np.int64)
                X_test = s_data.loc[test_idx, feature_cols].values.astype(np.float32)
                y_test = s_data.loc[test_idx, "intent_idx"].values.astype(np.int64)
                if len(np.unique(y_cal)) < 2:
                    continue
                scaler = StandardScaler()
                X_cal_s = scaler.fit_transform(X_cal)
                X_test_s = scaler.transform(X_test).astype(np.float32)
                clf = make_classifier(SEED)
                try:
                    clf.fit(X_cal_s, y_cal)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                preds = clf.predict(X_test_s)
                acc = accuracy_score(y_test, preds)
                f1m = f1_score(y_test, preds, average="macro", zero_division=0)
                cls_f1 = f1_score(y_test, preds, average=None, labels=CLASSES, zero_division=0)
                rows.append({
                    "cal_per_gesture": cal_size,
                    "cal_total_ms": cal_size * 50 * 12,
                    "participant": patient, "session": session,
                    "arm": session.split("_")[0],
                    "n_cal": int(len(cal_idx)), "n_test": int(len(test_idx)),
                    "acc": acc, "f1_macro": f1m,
                    "f1_rest": cls_f1[0], "f1_close": cls_f1[1], "f1_open": cls_f1[2],
                })
            if pi % 10 == 0:
                print(f"  patient {pi}/{len(patients)} done, elapsed={time.time()-t_start:.0f}s")
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    df_r = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_r.to_csv(OUT_CSV, index=False)

    # Summary
    summary = df_r.groupby("cal_per_gesture").agg(
        n_sessions=("acc", "count"),
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
        f1_macro_mean=("f1_macro", "mean"),
        f1_rest_mean=("f1_rest", "mean"),
        f1_close_mean=("f1_close", "mean"),
        f1_open_mean=("f1_open", "mean"),
    ).reset_index()
    summary["cal_total_s"] = summary["cal_per_gesture"] * 50 * 12 / 1000

    md = [
        "# Cal-data-size ablation",
        "",
        f"Patient-only HGB (no GrabMyo) on {len(patients)} PhysioMio patients × {len(CAL_SIZES)} cal sizes.",
        f"Cal windows per gesture sweep: {CAL_SIZES}.",
        "",
        "## Headline curve",
        "",
        "| cal/gest | Total cal | n sessions | Mean acc | F1 macro | F1 rest | F1 close | F1 open |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        md.append(f"| {int(r['cal_per_gesture'])} | {r['cal_total_s']:.1f} s | {int(r['n_sessions'])} | "
                  f"{r['acc_mean']:.4f} ± {r['acc_std']:.4f} | {r['f1_macro_mean']:.4f} | "
                  f"{r['f1_rest_mean']:.3f} | {r['f1_close_mean']:.3f} | {r['f1_open_mean']:.3f} |")

    md += [
        "",
        "## How to read",
        "",
        "Each row uses N cal windows per gesture × 12 gestures = total cal duration shown. "
        "60s of cued cal data (our main eval) = 60 windows/gesture (at 50 ms stride). "
        "If the curve saturates before 60 windows, the protocol could shorten the cal session. "
        "If it doesn't, we're already near the floor and longer cal would help.",
    ]
    OUT_MD.write_text("\n".join(md))

    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(f"\nSweep summary:")
    print(summary.to_string(index=False))
    print(f"\nTotal wall: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
