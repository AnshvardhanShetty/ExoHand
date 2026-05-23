# Cal-data-size ablation

Patient-only HGB (no GrabMyo) on 12 PhysioMio patients × 6 cal sizes.
Cal windows per gesture sweep: [12, 24, 36, 60, 90, 120].

## Headline curve

| cal/gest | Total cal | n sessions | Mean acc | F1 macro | F1 rest | F1 close | F1 open |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 7.2 s | 91 | 0.8365 ± 0.0974 | 0.8204 | 0.949 | 0.817 | 0.695 |
| 24 | 14.4 s | 91 | 0.8432 ± 0.1019 | 0.8227 | 0.982 | 0.822 | 0.665 |
| 36 | 21.6 s | 91 | 0.8680 ± 0.0977 | 0.8529 | 0.986 | 0.846 | 0.727 |
| 60 | 36.0 s | 91 | 0.8694 ± 0.0988 | 0.8544 | 0.987 | 0.848 | 0.728 |
| 90 | 54.0 s | 91 | 0.8687 ± 0.0990 | 0.8538 | 0.987 | 0.847 | 0.728 |
| 120 | 72.0 s | 91 | 0.8686 ± 0.0999 | 0.8538 | 0.988 | 0.847 | 0.726 |

## How to read

Each row uses N cal windows per gesture × 12 gestures = total cal duration shown. 60s of cued cal data (our main eval) = 60 windows/gesture (at 50 ms stride). If the curve saturates before 60 windows, the protocol could shorten the cal session. If it doesn't, we're already near the floor and longer cal would help.