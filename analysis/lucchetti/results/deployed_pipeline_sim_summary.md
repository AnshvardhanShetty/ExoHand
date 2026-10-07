# Lucchetti — simulated deployed (20 Hz P-P envelope) pipeline

n stroke subjects: 10
GrabMyo training: 262,773 envelope windows

## Result

- **Zero-shot 3-class (envelope):**       0.3726  (raw 2 kHz ref: 0.194)
- **Calibrated 3-class (envelope):**     0.4923  (raw 2 kHz ref: 0.795)
- **Calibrated binary (envelope):**      0.6299  (raw 2 kHz ref: 0.960)

- median 3-class cal: 0.4701  std: 0.0840
- median binary cal:  0.6239

## Per-class F1 (calibrated 3-class)

- rest:  0.6190
- close: 0.3439
- open:  0.3648