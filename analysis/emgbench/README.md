# EMGBench hybrid integration

The ExoHand HGB classifier runs against EMGBench's 6 supported datasets using the architecture confirmed in this session:

- **Use EMGBench's** dataset download (Setup.py) + raw-EMG loading (Combined_Data.load_data) + LOSO splits (Leave_One_Subject_Out).
- **Substitute** their lossy `X.load_images()` step with our 60→370-feature pipeline.
- **Run HGB** ourselves; emit per-fold metrics CSV mergeable with EMGBench's published baseline tables.

## Files

| File | Purpose |
|---|---|
| `feature_extraction.py` | (n_windows, n_channels, n_timesteps) → 60 base features per window. Mirrors `ml/preprocessing_grabmyo.py`. |
| `dataset_adapters.py` | Per-dataset channel selection + gesture→{close/open/rest} mapping. Many placeholders need verification. |
| `exohand_runner.py` | End-to-end pipeline: Setup → load → feature subst → engineer (370 cols) → LOSO + calibration → metrics. |

## Prerequisite setup (manual)

EMGBench needs to be cloned and its env installed before this runner works.

```bash
# 1. Clone EMGBench (somewhere outside the ExoHand repo)
git clone https://github.com/jehanyang/emgbench.git ~/emgbench
cd ~/emgbench
git lfs install

# 2. Install env (see EMGBench README — requires mamba/conda)
mamba env create -n emgbench -f environment.yml
conda activate emgbench

# 3. Datasets download on first run; capgmyo is the cheapest entry point
```

Pip-only install of EMGBench's deps is in principle possible but not tested
upstream; the conda path is the documented one.

## Running

```bash
# Smoke test on capgmyo, leftout subject 1
python analysis/emgbench/exohand_runner.py \
    --emgbench-root ~/emgbench \
    --dataset capgmyo \
    --leftout-subject 1 \
    --fast \
    --calib-n-windows 1200 --calib-weight 100
```

Wraps `exohand_runner.py` in a shell loop for all subjects of a given dataset
when you're ready to produce paper-table numbers.

## What needs verification before publication-quality runs

For each of the 6 datasets, **before** producing paper numbers:

1. **Channel selection.** Current adapters use evenly-spaced indices as a default. For datasets with high-density arrays (Hyser, FlexWear-HD), this throws away significant information. Recommended: replace with empirical activation-based selection — rank channels by mean envelope amplitude during close-class windows minus rest-class windows, take top 4.

2. **Gesture mapping.** Most adapters' `gesture_to_intent` dicts are placeholders. The exact label strings come from `emgbench/Setup/Utils/utils_<DATASET>.gesture_labels`. We need to read each utils file, list the labels, and decide each one's intent class explicitly. Don't ship default mappings without that check.

3. **Sample rate.** Each adapter declares a sample rate; verify against the utils file's frequency.txt or hardcoded fs. Mismatches will silently corrupt feature extraction (the bandpass/notch are fs-dependent).

4. **Synthetic trial structure.** Our dataframe construction sets `trial = window_idx` and `t_rel_s = 0.0`. This is fine for the engineer_features pipeline's groupby operations (they still discriminate subjects), but it means temporal features (`_prev`, `_delta`, `_roll3`) are computed across consecutive windows that may belong to different gestures in the raw data. If a dataset has natural trial/block structure exposed via `Y.data` or `label.data`, use it.

5. **Window length.** EMGBench's window length per dataset is set inside each utils_*.py. Our feature extraction is window-length-agnostic, but the `_features_for_one_window` function assumes the Nyquist allows 20-450 Hz bandpass. For datasets at < 1 kHz, the HIGHCUT will clamp automatically; verify the bandpass doesn't degenerate.

## Cross-references

- The architecture rationale for the hybrid integration is documented in Task #20 and Task #31 descriptions (see TaskList).
- The choice of `--calib-n-windows 1200 --calib-weight 100` matches `analysis/reproduce_headline/VARIANTS.md` variant (e) — the protocol that closes the GrabMyo calibration gap.
- The 60→370 feature pipeline reuses `ml.train_hgb_v2.engineer_features` unchanged; same code path that produced the canonical `improved_hgb_model.pkl`.
