# Cal-data-size ablation

Patient-only HGB (no GrabMyo) on 48 PhysioMio patients × 6 cal sizes.
Cal windows per gesture sweep: [12, 24, 36, 60, 90, 120].

## Headline curve

| cal/gest | Total cal | n sessions | Mean acc | F1 macro | F1 rest | F1 close | F1 open |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 7.2 s | 328 | 0.8318 ± 0.1043 | 0.8140 | 0.946 | 0.821 | 0.675 |
| 24 | 14.4 s | 328 | 0.8626 ± 0.1026 | 0.8487 | 0.974 | 0.846 | 0.726 |
| 36 | 21.6 s | 328 | 0.8698 ± 0.1049 | 0.8586 | 0.969 | 0.852 | 0.755 |
| 60 | 36.0 s | 328 | 0.8703 ± 0.1044 | 0.8590 | 0.970 | 0.853 | 0.755 |
| 90 | 54.0 s | 328 | 0.8698 ± 0.1044 | 0.8585 | 0.969 | 0.852 | 0.754 |
| 120 | 72.0 s | 328 | 0.8705 ± 0.1044 | 0.8592 | 0.970 | 0.853 | 0.755 |

## How to read

Each row uses N cal windows per gesture × 12 gestures = total cal duration shown. 60s of cued cal data (our main eval) = 60 windows/gesture (at 50 ms stride). If the curve saturates before 60 windows, the protocol could shorten the cal session. If it doesn't, we're already near the floor and longer cal would help.