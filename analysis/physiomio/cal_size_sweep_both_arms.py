"""
Cal-size sweep, both arms, both same-session and next-session test.

This is the load-bearing experiment for the GrabMyo-necessity claim. The math
argument (n_independent / n_features < 1 at 30 s re-cal) needs an empirical
backstop because the one existing data point (cal_per_gesture=36, ~432
windows) ties between patient-only and GrabMyo+cal. Without measuring the
small-cal end of the curve, the necessity claim is theoretical only and a
reviewer can dismiss it by pointing at the tied data point.

Protocol:
  - cal_per_gesture ∈ {3, 6, 12, 24, 36}    (cued windows per trial)
  - All 48 PhysioMio patients with impaired_01
  - Two arms:
      • patient_only: HGB(cal only)
      • grabmyo_cal:  HGB(GrabMyo_300K_subsample + cal × 100 weight)
  - Two test sets per (patient, cal_size, arm):
      • same_session: held-out test from impaired_01
      • next_session: full evaluation on impaired_02 (if it exists)

Speed optimizations (matter for the GrabMyo+cal arm only):
  - GrabMyo subsampled to 300 K rows with fixed seed (defensible: we're
    sweeping cal size, not GrabMyo size; sanity-check at cal=36 that the
    300 K curve matches the 1.14 M reference within ~1 pp).
  - max_iter=100, early_stopping=False
    (avoids the pathological case where val-split lands on GrabMyo and the
    loss decreases for all 300 iters without firing the early-stop)

Output:
  analysis/physiomio/results/cal_size_sweep.csv
  analysis/physiomio/results/cal_size_sweep_summary.md
  analysis/physiomio/results/cal_size_sweep.png
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

OUT_CSV = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "cal_size_sweep.csv"
OUT_MD = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "cal_size_sweep_summary.md"
OUT_PNG = PROJECT_ROOT / "analysis" / "physiomio" / "results" / "cal_size_sweep.png"

CAL_SIZES = [3, 6, 12, 24, 36]   # windows/gesture
BUFFER_WINDOWS = 3
GRABMYO_SUBSAMPLE = 300_000      # rows to subsample from 1.14 M


def make_clf(seed):
    return HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=100, min_samples_leaf=20,
        l2_regularization=0.01, max_depth=10, random_state=seed,
        early_stopping=False,
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


def fit_one(config, X_cal, y_cal, grabmyo_X, grabmyo_y, seed):
    """Fit one classifier and return (clf, scaler, fit_time)."""
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
    clf = make_clf(seed)
    t0 = time.time()
    try:
        if sample_weight is None:
            clf.fit(X_train_s, y_train)
        else:
            clf.fit(X_train_s, y_train, sample_weight=sample_weight)
    except (ValueError, np.linalg.LinAlgError):
        return None, None, None
    return clf, scaler, time.time() - t0


def eval_clf(clf, scaler, X_test, y_test):
    X_test_s = scaler.transform(X_test).astype(np.float32)
    preds = clf.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    f1m = f1_score(y_test, preds, average="macro", zero_division=0)
    cls_f1 = f1_score(y_test, preds, average=None, labels=CLASSES, zero_division=0)
    return {
        "acc": acc, "f1_macro": f1m,
        "f1_rest": cls_f1[0], "f1_close": cls_f1[1], "f1_open": cls_f1[2],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-patients", type=int, default=48, help="cap on patient count (default all)")
    parser.add_argument("--cal-sizes", type=str, default=None,
                        help="comma-separated cal sizes (default: 3,6,12,24,36)")
    parser.add_argument("--out-suffix", type=str, default="", help="suffix appended to output filename stems")
    args = parser.parse_args()
    cal_sizes = [int(x) for x in args.cal_sizes.split(",")] if args.cal_sizes else CAL_SIZES
    global OUT_CSV, OUT_MD, OUT_PNG
    if args.out_suffix:
        OUT_CSV = OUT_CSV.with_name(OUT_CSV.stem + args.out_suffix + OUT_CSV.suffix)
        OUT_MD = OUT_MD.with_name(OUT_MD.stem + args.out_suffix + OUT_MD.suffix)
        OUT_PNG = OUT_PNG.with_name(OUT_PNG.stem + args.out_suffix + OUT_PNG.suffix)
        print(f"Output suffix: writing to {OUT_CSV.name}")

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
    gm_X_full = grabmyo_eng[feature_cols].values.astype(np.float32)
    gm_y_full = grabmyo_eng["intent_idx"].values.astype(np.int64)
    print(f"  GrabMyo full: {len(gm_X_full):,} windows × {len(feature_cols)} features")

    # Stratified subsample to 300K (preserve class balance)
    sub_rng = np.random.RandomState(SEED)
    sub_idx = []
    for cls in np.unique(gm_y_full):
        cls_idx = np.where(gm_y_full == cls)[0]
        n_take = int(GRABMYO_SUBSAMPLE * (len(cls_idx) / len(gm_y_full)))
        chosen = sub_rng.choice(cls_idx, size=n_take, replace=False)
        sub_idx.extend(chosen)
    sub_idx = np.array(sorted(sub_idx))
    grabmyo_X = gm_X_full[sub_idx]
    grabmyo_y = gm_y_full[sub_idx]
    print(f"  GrabMyo subsample: {len(grabmyo_X):,} windows  "
          f"(class balance: {dict(zip(*np.unique(grabmyo_y, return_counts=True)))})")

    all_patients = sorted(eng["participant"].unique(), key=lambda s: int(s.replace("patient", "")))
    patients = [p for p in all_patients if "impaired_01" in eng[eng["participant"] == p]["session"].unique()]
    patients = patients[:args.n_patients]
    print(f"\nPatients (n={len(patients)}): {patients[:8]}...")
    print(f"Cal sizes: {cal_sizes}")
    n_fits = len(patients) * len(cal_sizes) * 2
    print(f"Total fits: {len(patients)} × {len(cal_sizes)} × 2 arms = {n_fits}")
    print(f"Estimated wall time: ~{len(patients) * len(cal_sizes) * 26 / 60:.0f} min (GrabMyo+cal arm dominates)\n")

    rows = []
    fit_count = 0
    for cal_size in cal_sizes:
        for pi, patient in enumerate(patients, 1):
            # impaired_01 data
            s01 = eng[(eng["participant"] == patient) & (eng["session"] == "impaired_01")].copy()
            if len(s01) == 0:
                continue
            test_idx, cal_idx = split_with_cal_size(s01, cal_size, TEST_PER_CLASS, rng)
            if len(test_idx) == 0 or len(cal_idx) == 0:
                continue
            X_cal = s01.loc[cal_idx, feature_cols].values.astype(np.float32)
            y_cal = s01.loc[cal_idx, "intent_idx"].values.astype(np.int64)
            X_same = s01.loc[test_idx, feature_cols].values.astype(np.float32)
            y_same = s01.loc[test_idx, "intent_idx"].values.astype(np.int64)
            if len(np.unique(y_cal)) < 2:
                continue

            # impaired_02 (if exists) — use ALL windows (no balanced subsample, just everything)
            s02 = eng[(eng["participant"] == patient) & (eng["session"] == "impaired_02")].copy()
            has_next = len(s02) > 0
            if has_next:
                X_next = s02[feature_cols].values.astype(np.float32)
                y_next = s02["intent_idx"].values.astype(np.int64)

            for config in ["patient_only", "grabmyo_cal"]:
                fit_count += 1
                clf, scaler, fit_time = fit_one(config, X_cal, y_cal, grabmyo_X, grabmyo_y, SEED)
                if clf is None:
                    print(f"  [skip] {config} cal={cal_size} {patient}: fit failed")
                    continue
                same = eval_clf(clf, scaler, X_same, y_same)
                next_metrics = eval_clf(clf, scaler, X_next, y_next) if has_next else None

                row = {
                    "cal_per_gesture": cal_size,
                    "config": config,
                    "patient": patient,
                    "n_cal": int(len(cal_idx)),
                    "n_test_same": int(len(test_idx)),
                    "n_test_next": int(len(s02)) if has_next else 0,
                    "has_next_session": has_next,
                    "fit_time_s": fit_time,
                    "same_acc": same["acc"], "same_f1m": same["f1_macro"],
                    "same_f1_rest": same["f1_rest"], "same_f1_close": same["f1_close"], "same_f1_open": same["f1_open"],
                }
                if next_metrics is not None:
                    row.update({
                        "next_acc": next_metrics["acc"], "next_f1m": next_metrics["f1_macro"],
                        "next_f1_rest": next_metrics["f1_rest"], "next_f1_close": next_metrics["f1_close"], "next_f1_open": next_metrics["f1_open"],
                    })
                rows.append(row)
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

            # Per-patient log
            last_two = rows[-2:]
            def fmt(v):
                return f"{v:.3f}" if v is not None else "—"
            po_acc = next((r["same_acc"] for r in last_two if r["config"] == "patient_only"), None)
            gm_acc = next((r["same_acc"] for r in last_two if r["config"] == "grabmyo_cal"), None)
            po_nxt = next((r.get("next_acc") for r in last_two if r["config"] == "patient_only"), None)
            gm_nxt = next((r.get("next_acc") for r in last_two if r["config"] == "grabmyo_cal"), None)
            elapsed = time.time() - t_start
            eta_min = (elapsed / fit_count * n_fits - elapsed) / 60 if fit_count > 0 else 0
            print(f"  cal={cal_size:>2d} pat={pi:>2d}/{len(patients)} {patient:<11} "
                  f"same: PO={fmt(po_acc)} GM={fmt(gm_acc)}  "
                  f"next: PO={fmt(po_nxt)} GM={fmt(gm_nxt)}  "
                  f"[{elapsed/60:.1f}min, eta {eta_min:.0f}min]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  rows={len(df)}")

    # ── Summary table ──
    def summarise(df, acc_col):
        if acc_col not in df.columns:
            return None
        sub = df.dropna(subset=[acc_col])
        agg = sub.groupby(["cal_per_gesture", "config"])[acc_col].agg(["mean", "std", "count"]).round(4)
        return agg

    same_summary = summarise(df, "same_acc")
    next_summary = summarise(df, "next_acc")

    md = ["# Cal-size sweep — both arms", "",
          f"n = {df['patient'].nunique()} patients · cal sizes {cal_sizes} windows/gesture · 2 arms.",
          f"GrabMyo subsample: {len(grabmyo_X):,} of {len(gm_X_full):,} ({100*len(grabmyo_X)/len(gm_X_full):.1f}%)",
          ""]
    md += ["## Same-session accuracy (impaired_01 held-out)", "", "```", str(same_summary), "```", ""]
    if next_summary is not None and len(next_summary) > 0:
        md += ["## Next-session accuracy (impaired_02 full)", "", "```", str(next_summary), "```", ""]
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")

    # ── Figure ──
    try:
        import matplotlib.pyplot as plt
        from analysis.plots.style import apply_style, PALETTE
        apply_style()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

        def plot_panel(ax, acc_col, title):
            if acc_col not in df.columns:
                ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes); return
            sub = df.dropna(subset=[acc_col])
            for config, color, label in [
                ("patient_only", "#c0392b", "patient-only"),
                ("grabmyo_cal", "#2980b9", "GrabMyo + cal × 100"),
            ]:
                d = sub[sub.config == config]
                grp = d.groupby("cal_per_gesture")[acc_col]
                m = grp.mean()
                # bootstrap 95% CI per cal size
                lo, hi = [], []
                bs_rng = np.random.RandomState(SEED)
                for cal, g in d.groupby("cal_per_gesture"):
                    v = g[acc_col].values
                    samples = [v[bs_rng.randint(0, len(v), size=len(v))].mean() for _ in range(1000)]
                    lo.append(np.percentile(samples, 2.5))
                    hi.append(np.percentile(samples, 97.5))
                ax.fill_between(m.index, lo, hi, color=color, alpha=0.15)
                ax.plot(m.index, m.values, "-o", color=color, label=label, markersize=5)
            ax.set_xscale("log")
            ax.set_xticks(cal_sizes); ax.set_xticklabels(cal_sizes)
            ax.set_xlabel("Cal windows per gesture")
            ax.set_title(title)
            ax.legend(loc="lower right", fontsize=9)

        plot_panel(axes[0], "same_acc", "(a) Same-session accuracy")
        axes[0].set_ylabel("Accuracy")
        plot_panel(axes[1], "next_acc", "(b) Next-session accuracy")
        axes[0].set_ylim(0.2, 1.0); axes[1].set_ylim(0.2, 1.0)
        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=150)
        print(f"Wrote {OUT_PNG}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
