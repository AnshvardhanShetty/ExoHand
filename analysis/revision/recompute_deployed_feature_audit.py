"""
Revision — deployed-pipeline feature audit (reviewer concern #6).

The deployed pipeline runs at 20 Hz (Teensy emits 50 ms P-P amplitudes).
The ML inference window is 200 ms → 4 samples per feature computation.
Many of the 60 base features are mathematically degenerate on 4 samples:
  - zc / ssc / wamp — waveform-crossing counts (at most 3 events possible)
  - mean_freq / median_freq — 2-bin Hann FFT is noise

Question: which of the 60 base features actually carry class-discriminative
signal at deployment? If the top-K subset matches the top-60 accuracy, the
paper's "60-feature deployment" claim is defensible only in nominal sense.

Method:
  1. Reuse `extract_session_features_20hz` from deployed_pipeline_sim.py
     to obtain the 60-feature × N-window matrix per session for a small
     sample of patients (avoid re-running the full sim).
  2. Compute per-feature ANOVA F-statistic across (rest, close, open)
     within each session, then average across sessions.
  3. Rank features by F-stat. Identify the "dead zone" (F ≈ 0).
  4. Retrain HGB per session with only the top-K features. Report acc vs K.

Outputs:
  analysis/revision/results/deployed_feature_audit_per_feature.csv
  analysis/revision/results/deployed_feature_audit_topk_accuracy.csv
  analysis/revision/results/deployed_feature_audit_summary.md
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from analysis.seed import SEED, seed_everything
from analysis.physiomio.deployed_pipeline_sim import (
    PHYSIOMIO_ROOT, CHANNEL_PICKS, extract_session_features_20hz,
)

OUT_DIR = PROJECT_ROOT / "analysis" / "revision" / "results"
OUT_PER_FEATURE = OUT_DIR / "deployed_feature_audit_per_feature.csv"
OUT_TOPK = OUT_DIR / "deployed_feature_audit_topk_accuracy.csv"
OUT_MD = OUT_DIR / "deployed_feature_audit_summary.md"

AMPLITUDE_FEATS = {"rms", "mav", "var", "wl", "maxamp", "iemg",
                   "env_mean", "env_max", "env_std", "env_rms"}
DEGENERATE_FEATS = {"zc", "ssc", "wamp", "mean_freq", "median_freq"}

SAMPLE_PATIENTS = 10


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)

    picks = pd.read_csv(CHANNEL_PICKS)
    picks["chosen_channels"] = picks["chosen_channels"].apply(json.loads)
    pick_map = dict(zip(picks["patient"], picks["chosen_channels"]))

    all_patient_dirs = sorted(
        [p for p in PHYSIOMIO_ROOT.iterdir()
         if p.is_dir() and p.name.startswith("patient")],
        key=lambda p: int(p.name.replace("patient", "")),
    )
    patient_dirs = all_patient_dirs[:SAMPLE_PATIENTS]
    print(f"Auditing {len(patient_dirs)} patients (of {len(all_patient_dirs)}) at 20 Hz deployed rate")

    all_session_feats = []
    t_start = time.time()
    for pi, pdir in enumerate(patient_dirs, 1):
        patient = pdir.name
        channels = pick_map.get(patient)
        if channels is None:
            print(f"[{pi}] {patient} — no channel picks, skip")
            continue
        impaired_dir = pdir / "impaired_arm"
        if not impaired_dir.exists():
            print(f"[{pi}] {patient} — no impaired_arm dir, skip")
            continue
        sess_files = sorted(impaired_dir.glob("*.parquet"))
        if not sess_files:
            print(f"[{pi}] {patient} — no session parquets, skip")
            continue

        n_kept = 0
        for sess_file in sess_files:
            try:
                feats = extract_session_features_20hz(sess_file, channels)
            except Exception as e:
                print(f"    {patient}/{sess_file.name}: extract failed ({e})")
                continue
            if len(feats) < 50 or feats["intent_idx"].nunique() < 3:
                continue
            feats["patient"] = patient
            feats["session"] = f"impaired_{sess_file.stem}"
            all_session_feats.append(feats)
            n_kept += 1

        elapsed = time.time() - t_start
        eta = elapsed / pi * len(patient_dirs) - elapsed
        print(f"[{pi}/{len(patient_dirs)}] {patient}: {n_kept} sessions kept  "
              f"[{elapsed/60:.1f}min, eta {eta/60:.0f}min]", flush=True)

    if not all_session_feats:
        print("No sessions processed — aborting.")
        return
    full = pd.concat(all_session_feats, ignore_index=True)
    feature_cols = [c for c in full.columns
                    if c.startswith("ch") and c.rsplit("_", 1)[0] not in ("ch",)]
    # Reject non-feature columns (session, participant, etc are already not "ch*")
    feature_cols = [c for c in feature_cols if c not in ("gesture_name",)]
    print(f"\nCollected {len(full):,} windows × {len(feature_cols)} features "
          f"across {full.groupby(['patient','session']).ngroups} sessions.")

    # ── Per-feature ANOVA F across (rest, close, open) within each session ──
    print("\nComputing per-feature ANOVA F-stat within each session, averaging...")
    per_sess_records = []
    for (patient, sess), s_df in full.groupby(["patient", "session"]):
        if len(s_df) < 30:
            continue
        for col in feature_cols:
            vals = s_df[col].values.astype(np.float64)
            grouped = [s_df.loc[s_df.intent_idx == c, col].values.astype(np.float64)
                       for c in [0, 1, 2] if (s_df.intent_idx == c).any()]
            if len(grouped) < 2 or any(len(g) < 3 for g in grouped):
                f_stat = np.nan
            elif np.nanvar(vals) < 1e-12:
                f_stat = 0.0
            else:
                try:
                    f_stat, _ = f_oneway(*grouped)
                    if not np.isfinite(f_stat):
                        f_stat = 0.0
                except Exception:
                    f_stat = np.nan
            per_sess_records.append({
                "patient": patient, "session": sess, "feature": col,
                "f_stat": float(f_stat) if f_stat is not None else np.nan,
                "variance": float(np.nanvar(vals)),
            })
    per_sess = pd.DataFrame(per_sess_records)

    per_feat = per_sess.groupby("feature").agg(
        f_stat_mean=("f_stat", "mean"),
        f_stat_median=("f_stat", "median"),
        variance_mean=("variance", "mean"),
        n_sessions=("session", "count"),
    ).reset_index()
    per_feat["family"] = per_feat["feature"].str.rsplit("_", n=1).str[1]
    per_feat["expected"] = per_feat["family"].map(
        lambda f: "amplitude" if f in AMPLITUDE_FEATS
        else "degenerate" if f in DEGENERATE_FEATS
        else "other"
    )
    per_feat = per_feat.sort_values("f_stat_mean", ascending=False)
    per_feat.to_csv(OUT_PER_FEATURE, index=False)
    print(f"Wrote {OUT_PER_FEATURE}")

    # ── Top-K per-session accuracy sweep ──
    print("\nTop-K accuracy sweep on the audited patients (per-session, 70/30 split)...")
    ranked = per_feat.feature.tolist()
    K_VALUES = [5, 10, 20, 30, 40, 50, 60]

    topk_rows = []
    for k in K_VALUES:
        sel = ranked[:k]
        accs = []
        for (patient, sess), s_df in full.groupby(["patient", "session"]):
            if len(s_df) < 100:
                continue
            X = s_df[sel].fillna(0).values.astype(np.float32)
            y = s_df["intent_idx"].values.astype(np.int64)
            rng = np.random.RandomState(abs(hash((patient, sess, k))) & 0xffffffff)
            idx = np.arange(len(y))
            rng.shuffle(idx)
            n_train = int(0.7 * len(idx))
            tr, te = idx[:n_train], idx[n_train:]
            if len(np.unique(y[tr])) < 2 or len(te) < 10:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = HistGradientBoostingClassifier(
                max_iter=100, max_depth=10, learning_rate=0.1,
                random_state=SEED, class_weight="balanced",
            )
            clf.fit(sc.transform(X[tr]), y[tr])
            accs.append(accuracy_score(y[te], clf.predict(sc.transform(X[te]))))
        mean_acc = float(np.mean(accs)) if accs else np.nan
        topk_rows.append({"k": k, "mean_acc": mean_acc, "n_sessions": len(accs)})
        print(f"  k={k:3d}  mean_acc={mean_acc:.4f}  ({len(accs)} sessions)")

    topk = pd.DataFrame(topk_rows)
    topk.to_csv(OUT_TOPK, index=False)
    print(f"Wrote {OUT_TOPK}")

    # ── Summary ──
    amp_mean = per_feat[per_feat.expected == "amplitude"].f_stat_mean.mean()
    deg_mean = per_feat[per_feat.expected == "degenerate"].f_stat_mean.mean()
    dead_count = int((per_feat.f_stat_mean < 1.0).sum())

    md = [
        "# Deployed-pipeline feature audit — reviewer concern #6",
        "",
        f"Sampled {len(patient_dirs)} patients, "
        f"{full.groupby(['patient','session']).ngroups} sessions, "
        f"{len(full):,} 20 Hz windows.",
        "",
        "## Family-level F-statistics (mean across sessions)",
        "",
        f"- Amplitude features (rms, mav, var, wl, maxamp, iemg, env_*): mean F = **{amp_mean:.2f}**",
        f"- Degenerate features (zc, ssc, wamp, mean_freq, median_freq): mean F = **{deg_mean:.2f}**",
        f"- Features with F < 1 (essentially dead): **{dead_count} / {len(per_feat)}**",
        "",
        "## Accuracy vs feature count (top-K by F-stat)",
        "",
        "| top-K | patient-session mean acc | sessions |",
        "|---:|---:|---:|",
    ]
    for _, r in topk.iterrows():
        md.append(f"| {int(r.k)} | {r.mean_acc:.4f} | {int(r.n_sessions)} |")
    md += [
        "",
        "## Reading",
        "",
        "If amplitude features have F much greater than degenerate features, and if",
        "top-10 or top-20 accuracy matches top-60, the paper's '60-feature",
        "deployment' claim is nominal not effective. Honest revision options:",
        "",
        "  (a) Report the deployed feature set as the informative subset and re-run",
        "      headline numbers to confirm they hold. Cleanest fix.",
        "  (b) Keep 60 features but explicitly note that N carry no class signal",
        "      at 20 Hz — a documented protocol limitation.",
        "",
        "See `deployed_feature_audit_per_feature.csv` for the full ranked list.",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
