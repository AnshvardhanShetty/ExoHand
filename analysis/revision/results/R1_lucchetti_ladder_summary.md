# R1 — Lucchetti replication of the training-source ladder

n = 10 stroke patients. Rows 1, 3, 3b replicated. Row 2 (cross-arm same-patient) not applicable.

## Results

| Row | Training source | Mean acc | Median acc |
|---|---|---:|---:|
| 1 | Own impaired-arm 22s cal | 0.7949 | 0.8326 |
| 3 | LOPO (9 other stroke patients, full pool) | 0.6283 | 0.6257 |
| 3b | LOPO volume-matched to per-session cal size | 0.6543 | 0.6998 |
| 4 | Zero-shot GrabMyo (43 healthy) | 0.1943 | 0.1848 |

## Comparison to PhysioMio (n=48)

| Row | PhysioMio | Lucchetti | Ordering matches? |
|---|---:|---:|---:|
| 1 own cal | 0.88 | 0.7949 if avail | — |
| 3 LOPO full | 0.63 | 0.6283 | — |
| 3b LOPO VM | (running) | 0.6543 | — |
| 4 zero-shot | 0.35 | 0.19427299646552615 | — |

## Decision (pre-registered)

- If the qualitative ordering (per-session cal >> VM-LOPO >> zero-shot) holds on
  Lucchetti, replication is claimed. Independent-cohort confirmation.
- If ordering differs, replication is conditional and stated explicitly.