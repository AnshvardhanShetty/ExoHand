# PhysioMio error analysis

n = 48 patients, 329 sessions, 165,538 per-window predictions (Stage 1 classifier output).

## Overall confusion matrix (row % normalised)

| GT \ Pred | rest | close | open |
|---|---:|---:|---:|
| rest | 98.2% | 1.2% | 0.6% |
| close | 0.5% | 92.0% | 7.6% |
| open | 1.0% | 26.2% | 72.8% |

Diagonal = per-class recall. Off-diagonal patterns:
- **open → close** confusion: 26.2%
- **close → open** confusion: 7.6%
- **rest → close** confusion: 1.2%

## Per-arm recall

| Class | Healthy arm | Impaired arm | Δ (impaired − healthy) |
|---|---:|---:|---:|
| rest | 99.6% | 97.6% | -2.0% |
| close | 93.6% | 91.4% | -2.2% |
| open | 82.3% | 69.3% | -13.0% |

## Error rate vs time from ground-truth transition

Per-window error rate at offsets {-250 ms .. +2500 ms} from each GT class change:

| Offset (ms) | Error rate | Description |
|---:|---:|---|
| -250 (pre-transition, old class) | 9.1% | Last window of old class |
| +0 (transition) | 6.5% | First window of new class |
| +500 (reaction-buffer end) | 14.0% | End of typical reaction budget |
| +2500 (deep maintenance) | 5.2% | 2.5 s into new class |

If error rate is highest at +0 and decays through maintenance, the model is *latency-limited* (detects transitions slightly late but holds correctly). If error rate is flat across the window, the model is *class-confusion-limited* (it just gets the wrong class persistently).

## Where the errors are

- **Most-confused pair**: open → close (26.2%)
- **Impaired-arm penalty on rest recall**: -2.0%
- **Impaired-arm penalty on close recall**: -2.2%
- **Impaired-arm penalty on open recall**: -13.0%