# Experiment A — three arms across cal sizes (PhysioMio)

For each cal_per_gesture, we fit three arms on the same balanced test:
  1. LDA on 16 Hudgins features
  2. HGB on 370 features — calibration-only
  3. HGB on 370 features — GrabMyo + calibration

## Patient-mean accuracy

| cal_per_gesture | LDA-Hudgins | HGB cal-only | HGB GrabMyo+cal |
|---:|---:|---:|---:|
| 3 | 0.685 | 0.333 | 0.717 |
| 6 | 0.754 | 0.770 | 0.778 |
| 12 | 0.809 | 0.836 | 0.827 |
| 24 | 0.820 | 0.871 | 0.875 |
| 36 | 0.834 | 0.877 | 0.881 |

## Reading the table

- At small cal (3), if LDA and PO both collapse toward 0.33 while GM+cal
  stays high, GrabMyo extends the viable operating range — the
  'protocol design assumes a backbone' story is defensible.
- At large cal (36), all three converge — GrabMyo doesn't add accuracy
  at the paper's operating point. That's a known truth we have to live
  with; the story is not about the operating point but about the range.