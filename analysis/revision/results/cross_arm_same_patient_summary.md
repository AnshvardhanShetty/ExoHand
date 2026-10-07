# Cross-arm same-patient generalization — sharpening the mechanism story

For each of 48 PhysioMio patients with both healthy_01 and impaired_01:
  - Train HGB on OWN healthy-arm cal, test on impaired-arm balanced test set
  - Two arms: cal-only (Arm A), GrabMyo + healthy-arm cal (Arm B)
  - Baseline: own impaired-arm cal (Arm C, expected ~0.88)

## Results

| Regime | Mean accuracy | Reference |
|---|---:|---|
| **Zero-shot** (GrabMyo only, no per-patient data) | — | 0.346 (recompute #1) |
| **LOPO** (47 other patients' impaired-arm cal) | — | ~0.67 (early LOPO data) |
| **Cross-arm PO** (this patient's healthy-arm cal → impaired-arm test) | **0.5486** | this experiment |
| **Cross-arm GM+cal** (this patient's healthy-arm cal + GrabMyo) | **0.5132** | this experiment |
| **Impaired-arm own cal** (baseline) | **0.8752** | this experiment |
| **Per-session cal** (impaired arm, cal-size sweep reference) | — | 0.878 |

## Interpretation

**If cross-arm ≈ per-session cal (~0.80+):** 'Patient' is the specificity axis.
Per-patient info transfers across arms. Strongest support for per-patient-
independent-task story.

**If cross-arm ≈ LOPO (~0.65):** Both patient and arm axes matter about equally.
The healthy arm carries some patient-specific info but not enough to fully
substitute for impaired-arm calibration.

**If cross-arm ≈ zero-shot (~0.35):** 'Arm' is the specificity axis, not patient.
Impaired-arm EMG is a distinct distribution even from the SAME patient's healthy
arm. Would revise mechanism story: it's impaired-arm-specific, not per-patient
specificity per se.