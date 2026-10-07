# Paper — BrainBodyFM at NeurIPS 2026

**Accepted as a demonstration at BrainBodyFM — Foundation Models for the Brain and Body — at NeurIPS 2026.**

**Title:** *ExoHand: A £200 Closed-loop Hand Exoskeleton, Calibrated in 22 Seconds*

**Authors:** Anshvardhan Shetty · Adhiraiyan Sasikumar

The accepted demo trains its decoder from scratch on 22 seconds of the user's own signals during a guided session of about two minutes, and uses a two-servo tendon-driven exoskeleton. See the [main README](../README.md) for the current demonstration description.

This directory collects the supporting research and earlier submission drafts. Some documents retain the ICBINB framing, earlier hardware descriptions, or planning status from when they were written.

---

## Writing docs (live, active)

| File | What it is |
|---|---|
| [`WRITING_COMPANION.md`](WRITING_COMPANION.md) | Paste-at-start-of-session doc for any writing LLM. Everything about the paper: venue, claims, all numbers, framing decisions, deprecated numbers, section-by-section skeleton, drop-in sentences. **Single source of truth for what to write.** |
| [`FINAL_NUMBERS.md`](FINAL_NUMBERS.md) | Canonical numbers table. Every number in the paper must appear here. Deprecated-numbers table shows what to NOT use if found in older docs. |
| [`APPENDIX.md`](APPENDIX.md) | Compiled appendix (A–G): hardware BOM + latency, firmware-mirror equivalence, all pre-registered kill-or-confirm controls, full seven-cutoff dose-response, Lucchetti replication, Wasserstein-1 mechanism, and the verbatim pre-registration. Drop-in for the paper's appendix section. |
| [`ICBINB_SANITY_CHECK.md`](ICBINB_SANITY_CHECK.md) | Self-contained overview to paste into another LLM for an independent acceptance-probability review. |
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | Pre-registered decision rules (headline vs subgroup framing, thresholds). Reproduced verbatim as `APPENDIX.md` §G. |

---

## Figures (main-text-ready)

| File | Panel(s) | Lives in paper section |
|---|---|---|
| [`figures/F1_system_and_placement.png`](figures/F1_system_and_placement.png) (+ `.pdf`) | (a) pipeline block diagram (b) forearm electrode placement | §1 Problem |
| [`figures/F2_pathology_dominates.png`](figures/F2_pathology_dominates.png) | Per-patient scatter with rescue-quadrant callout + sorted waterfall | §4 Reason for failure |
| [`figures/F3_dose_response.png`](figures/F3_dose_response.png) | Pathology gap + CI band vs days-post-stroke cutoff (top) + log-p (bottom) | §4 Reason for failure |

Figures are **copies** — the originals live where their generating scripts write them:

- F1: `analysis/plots/figures/fig1_system_and_placement.png` (regenerate: `python3 analysis/plots/fig1_system_and_placement.py`)
- F2: `analysis/revision/results/pathology_dominates.png` (regenerate: `python3 analysis/revision/pathology_dominates_figure.py`)
- F3: `analysis/revision/results/dose_response_pathology.png` (regenerate: `python3 analysis/revision/plot_dose_response.py`)

If you regenerate any figure, re-copy it here.

---

## Handoffs to Adhi

| File | What it is |
|---|---|
| [`handoffs/ADHI_HANDOFF.md`](handoffs/ADHI_HANDOFF.md) (+ `.pdf`) | Original six-task priority list (A1 photos, A2 video, A3 latency, A4 live-n=3, A5 resource census, A6/T1 replay). Plain-language rewrite. |
| [`handoffs/A6_HANDOFF.md`](handoffs/A6_HANDOFF.md) (+ `.pdf`) | Standalone deep-dive on the T1 hardware-replay task with pre-committed pass criterion + safety loop. |
| [`handoffs/A6_INSTRUCTIONS_FOR_ADHI.md`](handoffs/A6_INSTRUCTIONS_FOR_ADHI.md) | Step-by-step (~5-year-old-friendly) instructions for the actual A6 run, including the Drive link. Sent to Adhi Aug 27. |

A6 is complete as of Aug 28 (48-patient replay + safety revert confirmed).

---

## Deprecated / archived

Older documents preserved for reference but superseded by the live docs above.

| File | Why deprecated |
|---|---|
| [`deprecated/PAPER_OVERVIEW_FOR_ASSESSMENT.md`](deprecated/PAPER_OVERVIEW_FOR_ASSESSMENT.md) | Early overview written before the ICBINB-BIO CFP structural requirements were confirmed. `WRITING_COMPANION.md` and `ICBINB_SANITY_CHECK.md` supersede it. |
| [`deprecated/PAPER_PLAN.md`](deprecated/PAPER_PLAN.md) | Early May 2026 planning doc from before the leakage-free re-analysis and multi-draw stability sweep. Numbers here are pre-multi-draw and should not be quoted. |

If in doubt: use the live docs, not the deprecated ones.

---

## Backing data (in the analysis tree — not moved)

The paper's numerical claims are backed by CSVs and parquets in `analysis/revision/results/` — not copied here to avoid drift. Full provenance table lives in `WRITING_COMPANION.md` §10 and `FINAL_NUMBERS.md` §7. Key files:

- `analysis/revision/results/leakage_free_ladder_per_patient.csv` — headline regime ladder per patient
- `analysis/revision/results/all_multidraw_per_patient.csv` — multi-draw pathology + diversity Δ per patient
- `analysis/revision/results/dose_response_pathology.csv` — 7-cutoff dose-response
- `analysis/revision/results/C4_leakage_free_summary.md` — GrabMyo weight sweep
- `analysis/revision/results/R_M1_leakage_free_summary.md` — Wasserstein-1 mechanism probe
- `analysis/revision/results/T1_deployed_stream_per_window.parquet` — A6 hardware replay per-window traces
- `analysis/revision/results/T1_deployed_accuracy_per_patient.csv` — A6 per-patient summary
- `analysis/revision/frozen_splits.parquet` — canonical (cal_idx, test_idx) per patient
- `data/physiomio_channel_picks.csv` — per-patient 4-channel selection (Cohen's d values)

---

## Quick navigation

- **Writing tonight?** Start in [`WRITING_COMPANION.md`](WRITING_COMPANION.md).
- **Need a specific number?** Look in [`FINAL_NUMBERS.md`](FINAL_NUMBERS.md).
- **Verifying a claim against source?** [`WRITING_COMPANION.md`](WRITING_COMPANION.md) §10 has the file-to-claim map.
- **Reviewer LLM check?** Paste [`ICBINB_SANITY_CHECK.md`](ICBINB_SANITY_CHECK.md).
- **Figure work?** [`figures/`](figures/) — but regenerate at the source path and re-copy here to stay in sync.
