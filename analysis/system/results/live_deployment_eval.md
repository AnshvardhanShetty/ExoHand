# Live deployment evidence — Adhi session, 2026-02-20

We deployed the integrated hardware (Teensy 4.0 + 4× MyoWare 2.0 + tendon-driven exoskeleton) on the project PI and recorded a 12.2-minute cued-gesture session at the integrated firmware rate (~20 Hz peak-to-peak per channel). The shipped model `exohand_adapted_model.pkl` — combining the GrabMyo base (43 healthy subjects, 1.14 M windows) with patient calibration data — was the model running live.

## Session summary

- **Duration:** 733.9 s (12.2 min)
- **Samples:** 14823 at 20.2 Hz (P-P amplitude per 50 ms window per channel, deployed firmware)
- **Cued events:** 127 total — rest=67, close=30, open=30
- **Hardware:** Teensy 4.0 + 4× MyoWare 2.0 + tendon-driven 3D-printed exoskeleton

## Deployment-time metrics (saved at calibration in the model bundle)

| Metric | Value |
|---|---:|
| Accuracy on Adhi (patient) held-out | **0.9917** (99.17%) |
| Accuracy on GrabMyo (base) held-out | **0.9998** (99.98%) |
| Feature count | 370 (engineered) |
| Window / stride | 200 ms / 50 ms |
| Mean cue-to-onset latency (post-hoc labelling) | 707 ms |

## What this number is and isn't

**Is:** the held-out accuracy of the deployed model on the patient's own session, computed at calibration time and stored in the model package. This anchors the deployment claim — *a real model, on real hardware, achieved this number on real device-acquired EMG.*

**Is NOT:** a cross-subject or cross-session generalisation metric. The model was calibrated for this specific patient using this specific session; the 99 % held-out is the within-session test split, not evidence the model generalises. Cross-population generalisation is the contribution of §4 (PhysioMio, n = 48) and §5 (Lucchetti, n = 10 stroke + 10 healthy). The deployment subsection's role is to confirm the pipeline operates end-to-end on real device hardware.

## How this enters the paper

One short paragraph in §3 (deployment characterisation), referenced by §6 (limitations):

> *We validate end-to-end deployment on a 12.2-minute live cued-gesture session recorded with the integrated hardware. The deployed model achieves **99.2% within-session held-out accuracy** on the patient and **100.0%** on the GrabMyo base; these are the metrics stored in the shipped model package and reflect the model's performance during the live recording. Cross-subject generalisation is characterised separately in §4 and §5.*