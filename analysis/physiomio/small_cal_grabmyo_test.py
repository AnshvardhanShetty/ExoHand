"""
The deciding test: does the GrabMyo prior earn its keep at very small cal sizes?

If foundation-model knowledge is necessary for stroke EMG classification, the
clearest evidence would be at the small-cal extreme — where 1-12 cal windows
per gesture is too few for HGB to fit a 3-class boundary alone, but the
GrabMyo prior should provide the structure that lets calibration work.

If patient-only and GrabMyo+cal track each other even at 1-3 windows/gesture,
then engineered features + per-participant z-scoring is doing the work the
foundation model would otherwise provide, and the paper's framing needs to
shift toward "no pretraining needed."

Protocol:
  - 8 representative patients × impaired_01 only (single session per patient)
  - 4 cal sizes per gesture × 12 gestures: {1, 3, 6, 12} cal windows / gesture
  - Test set: same balanced 117-window held-out (39/class) per session
  - Two configurations:
      • Patient-only HGB (no GrabMyo)
      • HGB + GrabMyo (weight 1) + cal (weight 100×)
  - Same fast HGB config (max_iter=300, max_depth=10, class_weight='balanced')

Output:
  analysis/physiomio/results/small_cal_grabmyo_test.csv
  analysis/physiomio/results/small_cal_grabmyo_test.{md,png}
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
    PHYSIOMIO_PKL, GRABMYO_CACHE, GRABMYO_META, TEST_PER_CLASS, CAL_WEIGHT, CLASSES,
)

OUT_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "small_cal_grabmyo_test.csv"
OUT_MD = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "small_cal_grabmyo_test.md"
OUT_PNG = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "small_cal_grabmyo_test.png"

CAL_SIZES = [1, 3, 6, 12, 24, 36]   # windows/gesture; 36 = current default for reference
BUFFER_WINDOWS = 3
N_PATIENTS_SUBSET = 8


def make_clf(seed):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=300, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        class_weight="balanced",
    )


def split_with_cal_size(session_df, cal_per_gesture, test_per_class, rng):
    """Per-gesture temporal split. First `cal_per_gesture` for cal, last 39 for test."""
    cal_idx = []
    test_pool = {0: [], 1: [], 2: []}
    for _, g in session_df.groupby("trial", sort=True):
        cls = int(g["intent_idx"].iloc[0])
        sg = g.sort_values("t_rel_s")
        n = len(sg)
        if n >= cal_per_gesture + BUFFER_WINDOWS + test_per_class:
            cal_idx.extend(sg.index[:cal_per_gesture].tolist())
            test_pool[cls].extend(
                sg.index[cal_per_gesture + BUFFER_WINDOWS:cal_per_gesture + BUFFER_WINDOWS + test_per_class].tolist()
            )
        else:
            # short gesture — proportional
            cal_n = max(1, min(cal_per_gesture, n - BUFFER_WINDOWS - 1))
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


def evaluate_one(config, X_cal, y_cal, X_test, y_test, grabmyo_X, grabmyo_y, seed):
    """Returns acc, f1_macro, per-class F1, fit_time."""
    if config == "patient_only":
        X_train = X_cal
        y_train = y_cal
        sample_weight = None
    elif config == "grabmyo_cal":
        X_train = np.vstack([grabmyo_X, X_cal])
        y_train = np.concatenate([grabmyo_y, y_cal])
        sample_weight = np.ones(len(X_train), dtype=np.float32)
        sample_weight[len(grabmyo_X):] = CAL_WEIGHT
    else:
        raise ValueError(config)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    clf = make_clf(seed)
    t0 = time.time()
    try:
        if sample_weight is None:
            clf.fit(X_train_s, y_train)
        else:
            clf.fit(X_train_s, y_train, sample_weight=sample_weight)
    except (ValueError, np.linalg.LinAlgError) as e:
        return None
    fit_time = time.time() - t0
    preds = clf.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    f1m = f1_score(y_test, preds, average="macro", zero_division=0)
    cls_f1 = f1_score(y_test, preds, average=None, labels=CLASSES, zero_division=0)
    return {
        "acc": acc, "f1_macro": f1m,
        "f1_rest": cls_f1[0], "f1_close": cls_f1[1], "f1_open": cls_f1[2],
        "fit_time_s": fit_time,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-patients", type=int, default=N_PATIENTS_SUBSET)
    args = parser.parse_args()

    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    t_start = time.time()

    print("Loading + engineering PhysioMio...")
    physiomio = pd.read_pickle(PHYSIOMIO_PKL)
    eng = engineer_features(physiomio)
    print(f"  shape: {eng.shape}")

    print("Loading GrabMyo cache...")
    grabmyo_eng = pd.read_pickle(GRABMYO_CACHE)
    with open(GRABMYO_META) as f:
        feature_cols = json.load(f)["feature_cols"]
    grabmyo_X = grabmyo_eng[feature_cols].values.astype(np.float32)
    grabmyo_y = grabmyo_eng["intent_idx"].values.astype(np.int64)
    print(f"  GrabMyo: {len(grabmyo_X):,} windows × {len(feature_cols)} features")

    # Pick representative patients — random subset of those with impaired_01
    all_patients = sorted(eng["participant"].unique(),
                          key=lambda s: int(s.replace("patient", "")))
    with_imp01 = [p for p in all_patients
                  if "impaired_01" in eng[eng["participant"] == p]["session"].unique()]
    rng.shuffle(with_imp01)
    patients = with_imp01[:args.n_patients]
    print(f"Patients sampled: {patients}")
    print(f"\nGrid: {len(patients)} patients × {len(CAL_SIZES)} cal sizes × 2 configs")
    print(f"Estimated compute: ~{len(patients) * len(CAL_SIZES) * 200 / 60:.0f} min for GrabMyo+cal arm\n")

    rows = []
    for cal_size in CAL_SIZES:
        for pi, patient in enumerate(patients, 1):
            s_data = eng[(eng["participant"] == patient) & (eng["session"] == "impaired_01")].copy()
            test_idx, cal_idx = split_with_cal_size(s_data, cal_size, TEST_PER_CLASS, rng)
            if len(test_idx) == 0 or len(cal_idx) == 0:
                continue
            X_cal = s_data.loc[cal_idx, feature_cols].values.astype(np.float32)
            y_cal = s_data.loc[cal_idx, "intent_idx"].values.astype(np.int64)
            X_test = s_data.loc[test_idx, feature_cols].values.astype(np.float32)
            y_test = s_data.loc[test_idx, "intent_idx"].values.astype(np.int64)
            if len(np.unique(y_cal)) < 2:
                continue
            this_run = {}
            for config in ["patient_only", "grabmyo_cal"]:
                res = evaluate_one(config, X_cal, y_cal, X_test, y_test, grabmyo_X, grabmyo_y, SEED)
                if res is None:
                    this_run[config] = None
                    continue
                this_run[config] = res
                rows.append({
                    "cal_per_gesture": cal_size,
                    "config": config,
                    "patient": patient,
                    "n_cal": int(len(cal_idx)),
                    "n_test": int(len(test_idx)),
                    **res,
                })
            po_acc = f"{this_run['patient_only']['acc']:.4f}" if this_run.get("patient_only") else "—"
            gm_acc = f"{this_run['grabmyo_cal']['acc']:.4f}" if this_run.get("grabmyo_cal") else "—"
            print(f"  cal_size={cal_size} pat={pi}/{len(patients)} {patient}: "
                  f"PO acc={po_acc} GM+cal acc={gm_acc}  "
                  f"elapsed={time.time()-t_start:.0f}s")
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # ── Aggregate + plot ──
    summary = df.groupby(["cal_per_gesture", "config"]).agg(
        n=("acc", "count"),
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
        f1_open_mean=("f1_open", "mean"),
        f1_close_mean=("f1_close", "mean"),
        f1_rest_mean=("f1_rest", "mean"),
    ).reset_index()
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.plots.style import apply_style, PALETTE, save_pair
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for config, label, color, marker in [
        ("patient_only", "Patient-only (no GrabMyo)", "#c0392b", "o"),
        ("grabmyo_cal", "GrabMyo + cal (weight 100×)", PALETTE["calibrated"], "s"),
    ]:
        sub = summary[summary["config"] == config].sort_values("cal_per_gesture")
        ax1.errorbar(sub["cal_per_gesture"], sub["acc_mean"], yerr=sub["acc_std"],
                     marker=marker, label=label, color=color, capsize=3, markersize=5,
                     linewidth=1.4)
        ax2.errorbar(sub["cal_per_gesture"], sub["f1_open_mean"], yerr=None,
                     marker=marker, label=label, color=color, markersize=5, linewidth=1.4)
    ax1.set_xscale("log")
    ax1.set_xticks(CAL_SIZES)
    ax1.set_xticklabels(CAL_SIZES)
    ax1.set_xlabel("Cal windows per gesture (log scale)")
    ax1.set_ylabel("Test accuracy")
    ax1.set_title("(a)  Accuracy vs cal size", loc="left")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(linestyle=":", alpha=0.4)
    ax1.set_ylim(0, 1.0)

    ax2.set_xscale("log")
    ax2.set_xticks(CAL_SIZES)
    ax2.set_xticklabels(CAL_SIZES)
    ax2.set_xlabel("Cal windows per gesture (log scale)")
    ax2.set_ylabel("Open-class F1")
    ax2.set_title("(b)  Open-class F1 vs cal size", loc="left")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(linestyle=":", alpha=0.4)
    ax2.set_ylim(0, 1.0)

    plt.tight_layout()
    save_pair(fig, str(OUT_PNG).replace(".png", ""))
    print(f"\nWrote {OUT_PNG}")

    # Markdown summary
    md = [
        "# Small-cal sweep: does GrabMyo earn its keep at low cal data?",
        "",
        f"n = {df['patient'].nunique()} patients × impaired_01 only · cal sizes {CAL_SIZES} windows/gesture · two configs.",
        "",
        "## Headline (paired by patient × cal size)",
        "",
        "| Cal/gesture | n | Patient-only acc | GrabMyo+cal acc | Δ (GM − PO) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for cal_size in CAL_SIZES:
        po_row = summary[(summary["cal_per_gesture"] == cal_size) & (summary["config"] == "patient_only")]
        gm_row = summary[(summary["cal_per_gesture"] == cal_size) & (summary["config"] == "grabmyo_cal")]
        if len(po_row) and len(gm_row):
            po_acc = po_row["acc_mean"].iloc[0]
            gm_acc = gm_row["acc_mean"].iloc[0]
            md.append(f"| {cal_size} | {int(po_row['n'].iloc[0])} | {po_acc:.4f} | {gm_acc:.4f} | {gm_acc - po_acc:+.4f} |")
    md += [
        "",
        "## Reading the result",
        "",
        "- If **GrabMyo+cal stays high while patient-only collapses** at small cal sizes "
        "(e.g., GM+cal at 0.85, PO at 0.60 at 1-3 windows/gesture), the foundation-model "
        "prior is doing real work at the low-data extreme. Paper keeps cross-population framing.",
        "- If **both configs track each other** even at 1-3 windows/gesture, engineered "
        "features + per-participant z-scoring is doing the work that GrabMyo would otherwise "
        "provide. Paper reframes as 'no pretraining needed at this feature richness'.",
        "- Per-class look matters: open is the rare class. If GrabMyo's contribution is "
        "specifically open-class F1 holding up at small cal, that's the cleanest mechanism statement.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")
    print(f"\nTotal wall: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
