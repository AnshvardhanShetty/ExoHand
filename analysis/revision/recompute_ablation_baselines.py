"""
Revision recompute #3 — formal ablation table.

Reviewer #5 asks for the baselines:
  - majority-class floor      (trivially 39/117 on balanced test → 0.333)
  - uniform-random floor      (also 0.333)
  - calibration-only HGB      (already in cal_size_sweep_v1.csv at cal_per_gesture=36)
  - GrabMyo-only + fixed norm (= zero-shot on balanced test, from recompute #1)
  - LDA on Hudgins 4-feature set   (MAV, WL, ZC, SSC per channel)
  - two-threshold envelope rule    (simple flexor/extensor thresholding)

Only for PhysioMio (n=48). Everything runs on the balanced 39/39/39 test set
so numbers are directly comparable to per_session_eval.

Outputs:
  analysis/revision/results/ablation_baselines_per_session.csv
  analysis/revision/results/ablation_baselines_summary.md
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from analysis.seed import SEED, seed_everything
from analysis.physiomio.per_session_eval import (
    PHYSIOMIO_PKL, TEST_PER_CLASS, split_session, CLASSES,
)

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_CSV = OUT_DIR / "ablation_baselines_per_session.csv"
OUT_MD = OUT_DIR / "ablation_baselines_summary.md"


# ---- Hudgins-style 4 features per channel (canonical myoelectric baseline) ----
HUDGINS_FEATURES = ["mav", "wl", "zc", "ssc"]
CHANNELS = [0, 4, 9, 13]  # canonical GrabMyo names in the PhysioMio schema


def hudgins_feature_cols(df):
    """Return the 16-column Hudgins subset (4 features × 4 channels)."""
    cols = []
    for ch in CHANNELS:
        for feat in HUDGINS_FEATURES:
            col = f"ch{ch}_{feat}"
            if col in df.columns:
                cols.append(col)
    return cols


def bootstrap_ci(x, n=2000, seed=SEED):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(x), size=(n, len(x)))
    samples = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def envelope_threshold_rule(X_cal, y_cal, X_test):
    """Two-threshold flexor/extensor envelope rule.

    Fit thresholds on cal data:
      - flexor_thr = mean(flexor_env_rms during CLOSE) * 0.6
      - extensor_thr = mean(extensor_env_rms during OPEN) * 0.6
    At test time:
      - if any flexor env_rms > flexor_thr → close
      - elif any extensor env_rms > extensor_thr → open
      - else → rest
    Expects columns ordered as [ch0_env_rms, ch4_env_rms, ch9_env_rms, ch13_env_rms].
    Convention: ch0/ch9 = flexor, ch4/ch13 = extensor (per GrabMyo canonical).
    """
    flexor_cols = [0, 2]   # ch0, ch9 in the 4-col array
    extensor_cols = [1, 3] # ch4, ch13
    close_mask = y_cal == 1
    open_mask = y_cal == 2
    if not close_mask.any() or not open_mask.any():
        return np.zeros(len(X_test), dtype=np.int64)  # all rest
    flex_thr = X_cal[close_mask][:, flexor_cols].mean() * 0.6
    ext_thr = X_cal[open_mask][:, extensor_cols].mean() * 0.6
    flex_test = X_test[:, flexor_cols].max(axis=1)
    ext_test = X_test[:, extensor_cols].max(axis=1)
    preds = np.zeros(len(X_test), dtype=np.int64)
    preds[flex_test > flex_thr] = 1
    preds[(ext_test > ext_thr) & (flex_test <= flex_thr)] = 2
    return preds


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)

    print("Loading PhysioMio…")
    df = pd.read_pickle(PHYSIOMIO_PKL)
    print(f"  shape: {df.shape}")

    hudgins_cols = hudgins_feature_cols(df)
    env_cols = [f"ch{ch}_env_rms" for ch in CHANNELS]
    print(f"  Hudgins cols ({len(hudgins_cols)}): {hudgins_cols[:6]}...")
    print(f"  Envelope cols: {env_cols}")

    rows = []
    df = df.reset_index(drop=True)

    for (patient, sess), s_data in df.groupby(["participant", "session"]):
        if not sess.startswith("impaired_"):
            continue
        try:
            test_idx, cal_idx, _ = split_session(s_data, TEST_PER_CLASS, rng)
        except Exception as e:
            continue
        if len(test_idx) < 15 or len(cal_idx) < 6:
            continue

        cal_rows = s_data.loc[cal_idx]
        test_rows = s_data.loc[test_idx]
        y_cal = cal_rows["intent_idx"].values.astype(np.int64)
        y_test = test_rows["intent_idx"].values.astype(np.int64)

        r = {"participant": patient, "session": sess,
             "n_cal": len(cal_idx), "n_test": len(test_idx)}

        # ── 1. Floors ──
        # majority-class prediction on balanced test = 1/3 by construction
        r["majority_class_acc"] = float(np.mean(y_test == np.bincount(y_test).argmax()))
        # uniform-random: expected 1/3 per class
        rng_local = np.random.RandomState(hash(patient + sess) & 0xffffffff)
        preds_rand = rng_local.choice(CLASSES, size=len(y_test), replace=True)
        r["uniform_random_acc"] = float(accuracy_score(y_test, preds_rand))

        # ── 2. LDA on Hudgins features ──
        if len(np.unique(y_cal)) >= 2 and len(hudgins_cols) > 0:
            X_cal = cal_rows[hudgins_cols].values.astype(np.float32)
            X_test = test_rows[hudgins_cols].values.astype(np.float32)
            try:
                sc = StandardScaler().fit(X_cal)
                clf = LinearDiscriminantAnalysis().fit(sc.transform(X_cal), y_cal)
                preds_lda = clf.predict(sc.transform(X_test))
                r["lda_hudgins_acc"] = float(accuracy_score(y_test, preds_lda))
            except Exception:
                r["lda_hudgins_acc"] = np.nan
        else:
            r["lda_hudgins_acc"] = np.nan

        # ── 3. Two-threshold envelope rule ──
        if all(c in df.columns for c in env_cols):
            X_cal_env = cal_rows[env_cols].values.astype(np.float32)
            X_test_env = test_rows[env_cols].values.astype(np.float32)
            preds_thr = envelope_threshold_rule(X_cal_env, y_cal, X_test_env)
            r["envelope_threshold_acc"] = float(accuracy_score(y_test, preds_thr))
        else:
            r["envelope_threshold_acc"] = np.nan

        rows.append(r)
        if len(rows) % 40 == 0:
            print(f"  {len(rows)} sessions done…", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    # Patient-level aggregation
    pat_means = out.groupby("participant")[[
        "majority_class_acc", "uniform_random_acc",
        "lda_hudgins_acc", "envelope_threshold_acc",
    ]].mean()

    print("\n" + "=" * 65)
    print("ABLATION BASELINES (PhysioMio impaired-arm, patient-mean, n=48)")
    print("=" * 65)
    md_rows = []
    for col, label in [
        ("majority_class_acc", "majority-class"),
        ("uniform_random_acc", "uniform-random"),
        ("lda_hudgins_acc", "LDA on Hudgins (MAV/WL/ZC/SSC)"),
        ("envelope_threshold_acc", "two-threshold envelope rule"),
    ]:
        v = pat_means[col].values
        m, lo, hi = bootstrap_ci(v)
        print(f"  {label:38s}  {m:.4f}  [{lo:.4f}, {hi:.4f}]")
        md_rows.append((label, m, lo, hi))

    # Reference numbers from other recomputes / existing results:
    reference = [
        ("zero-shot (balanced, recompute #1)",       0.346, 0.323, 0.370),
        ("calibration-only HGB (cal-size sweep)",   0.878, None, None),
        ("GrabMyo + cal (per_session_results.csv)", 0.860, None, None),
    ]

    md = [
        "# Ablation baselines — revision recompute #3",
        "",
        "Reviewer #5 asks for the canonical baselines on the same 48 patients",
        "× balanced 39/39/39 test set as the paper's headline number.",
        "",
        "## Baselines computed this run",
        "",
        "| baseline | patient-mean acc | 95% bootstrap CI |",
        "|---|---:|---:|",
    ]
    for label, m, lo, hi in md_rows:
        md.append(f"| {label} | {m:.3f} | [{lo:.3f}, {hi:.3f}] |")

    md += [
        "",
        "## Reference (from other files / recomputes)",
        "",
        "| method | patient-mean acc | source |",
        "|---|---:|---|",
    ]
    for label, m, lo, hi in reference:
        ci = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "—"
        md.append(f"| {label} | {m:.3f} {ci} | see notes |")

    md += [
        "",
        "## Interpretation",
        "",
        "- Chance / majority-class floors both sit at 0.333 on the balanced",
        "  test set (as required by construction).",
        "- LDA on 16 Hudgins features is the canonical myoelectric baseline.",
        "  If it lands near the paper's headline, the classical-ML+calibration",
        "  story is still ok but the specific 370-feature engineering",
        "  contributes less than the paper implies.",
        "- Two-threshold envelope rule is the simplest possible controller.",
        "  It should be far below any learned classifier; if it isn't, the",
        "  task is easier than the paper frames.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
