# Deployed-pipeline simulation on PhysioMio

Simulates the deployment (20 Hz P-P amplitude) pipeline on PhysioMio raw EMG,
to estimate the accuracy cost of giving up raw 2 kHz EMG.

- Patients evaluated: **48**
- Same per-session cal protocol (first 36 windows/gesture, balanced 117-window test)
- Patient-only HGB (no GrabMyo), 370 features computed at fs=20 Hz

## Result

- **20 Hz P-P simulated pipeline:** 0.8649 session-mean / patient-mean 0.8649
  (median 0.8675, std 0.1026, min 0.5726, max 1.0000)
- **Baseline (raw 2 kHz, patient-only):** 0.8777 (from cal_size_sweep_v1.csv, n=48)
- **Δ (deployed sim − baseline):** -0.0128

## Per-class F1

- rest:  0.9546
- close: 0.8534
- open:  0.7610