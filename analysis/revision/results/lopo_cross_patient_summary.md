# LOPO cross-patient generalization — testing per-patient specificity

Leave-one-patient-out on 48 PhysioMio impaired_01 sessions.
For each held-out patient, pool cal data from the other 47 patients and train HGB.
Two arms: cal-only (Arm A), GrabMyo+pooled_cal (Arm B). Test on held-out patient.

## Results

| Regime | Mean acc | Median acc | Reference |
|---|---:|---:|---|
| **Zero-shot** (no per-patient data at all) | — | — | 0.346 (recompute #1) |
| **LOPO cal-only** (47 other patients' cal, no GrabMyo) | **0.6287** | 0.6667 | — |
| **LOPO GrabMyo + pooled cal** (47 other patients' cal + GrabMyo) | **0.6002** | 0.6239 | — |
| **Per-session cal-only** (this patient's own cal) | — | — | 0.878 (cal-size sweep) |
| **Per-session GrabMyo+cal** (this patient's own cal) | — | — | 0.860 (per_session_results) |

**Paired Wilcoxon (LOPO GM+cal > LOPO cal-only): p = 9.942e-01**

## Interpretation

**If LOPO is close to zero-shot** (say ≤0.45), per-patient specificity is confirmed:
even 47 other patients' worth of cal data doesn't help patient N, because their
decision boundary is idiosyncratic. This is a direct test of 'per-patient EMG is
an independent task' — the mechanistic reason GrabMyo pretraining doesn't help.

**If LOPO is close to per-session** (say ≥0.75), per-patient specificity is wrong:
cross-patient data DOES generalize, and the reason GrabMyo doesn't help must be
something else (GrabMyo distribution too different from PhysioMio impaired-arm,
GrabMyo sample-size saturation, etc.).

**Intermediate** (0.45-0.75) suggests partial per-patient specificity plus a
usable-but-limited universal component.