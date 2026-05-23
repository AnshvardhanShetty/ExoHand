# The 1.0000 cross-session number is not generalization

## Finding

`training_report.txt` reports:
```
Cross-session results:
  2026-02-20_18-51 → 2026-02-20_19-09: 1.0000
  2026-02-20_19-09 → 2026-02-20_18-51: 1.0000
```

This was generated from `sessions/2026-02-20_18-51/` (started 18:51, ended 19:03) and `sessions/2026-02-20_19-09/` (started 19:09, ended 19:21) — **the same participant, same day, with an 18-minute gap between recordings.**

## Evidence of replicate-not-generalization

| Property | Session 1 (18-51) | Session 2 (19-09) |
|---|---|---|
| Date | 2026-02-20 | 2026-02-20 |
| Recording started | 19:03 (per `session_info.json`) | 19:21 |
| Duration | 733.89 s | 733.87 s |
| Sample count | 14,823 | 14,821 |
| Sample rate | 20.2 Hz | 20.2 Hz |
| Serial port | `/dev/cu.usbmodem176627901` | `/dev/cu.usbmodem176627901` |
| Cue count | 127 | 127 |
| Cue label distribution | 67 rest / 30 close / 30 open | 67 rest / 30 close / 30 open |
| Cue sequence (label + description) | identical | identical |
| Per-channel signal stats | ch0 μ=6.9 σ=8.2 · ch1 μ=25.6 σ=25.1 · ch2 μ=13.1 σ=17.1 · ch3 μ=17.5 σ=18.0 | ch0 μ=7.7 σ=13.9 · ch1 μ=25.1 σ=29.8 · ch2 μ=11.9 σ=17.6 · ch3 μ=17.5 σ=24.1 |
| Sample-by-sample correlation between sessions (paired by index) | — | **0.60** |

The two sessions are **the same protocol run twice on the same person with electrodes presumably still attached** (no evidence in the metadata of electrode reapplication between sessions). A correlation of 0.60 between paired raw samples is what one would expect from replicate recordings: not identical (some motor and ambient variability between executions), but far from independent.

The classifier trivially achieves 100% because the test set is functionally a held-out portion of the training distribution, not a new session distribution.

## What the number is NOT

It is not evidence of:
- Cross-day generalization
- Robustness to electrode reapplication
- Resilience to placement drift, skin-condition variation, or fatigue across sessions
- Longitudinal stability

It cannot be cited as any of those things in the paper.

## What a clean longitudinal protocol requires

For any longitudinal claim ("the model is stable across sessions / over time / with reapplied electrodes"), the protocol must include at least:

1. **Different days.** Session N and session N+1 separated by ≥ 24 hours, ideally ≥ 1 week, to span normal day-to-day variability in skin conductance, hydration, motor variability, etc.
2. **Electrode reapplication between sessions.** New electrode-skin interface each session. Even small position shifts (1-2 cm) substantively change EMG channel-to-muscle correspondence.
3. **Different time of day** when feasible — controls for diurnal effects on muscle activation and fatigue.
4. **No window overlap.** Windows from session N never used in evaluation of session N+1 and vice versa.
5. **Train calibration on session N's data only**, evaluate on session N+1. Do not concatenate sessions for training and split later — that defeats the purpose.

## Implication for the paper

- **Remove the 1.0000 cross-session line from `training_report.txt` or annotate it explicitly** as a replicate-measurement, not a generalization measurement.
- **The PhysioMio longitudinal evaluation (Stream 1.6 — now folded under Lucchetti / EMGBench dataset variability since PhysioMio is no longer in scope) is the right place for honest longitudinal claims.** Per the PhysioMio paper, recordings are collected at 3–14 separate sessions over each patient's inpatient stay, with implied electrode reapplication. That meets the criteria above.
- **Lucchetti has no longitudinal structure** — single session per arm. Cannot be used for cross-session generalization claims.
- **For EMGBench**, `--leave_one_session_out` works on the 2 datasets that have multi-session recordings (CapgMyo, others — TBD when we run the Stream 1.4 evaluation). Apply the criteria above when interpreting those numbers.

## Status

Task #4 closed. Stricter protocol documented; the 1.0000 number is identified as not citable. No further action needed until we have data with proper cross-session structure to evaluate.
