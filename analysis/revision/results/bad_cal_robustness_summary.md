# Experiment B1 — bad-cal robustness

For each of 48 patients, session_01 impaired. Corrupt cal, evaluate on clean balanced test.

Arms:
- **PO** = HGB with per-session cal only (no GrabMyo)
- **GM** = HGB with GrabMyo + per-session cal (paper's method)
- **Δ** = GM − PO. Positive means GrabMyo helps under this corruption.

## Mode: noise

|   level |   po_mean |   gm_mean |   delta_mean |   n |
|--------:|----------:|----------:|-------------:|----:|
|     0   |    0.88   |    0.8851 |       0.0051 |  47 |
|     0.5 |    0.8954 |    0.8383 |      -0.0571 |  47 |
|     1   |    0.8776 |    0.73   |      -0.1477 |  47 |
|     2   |    0.806  |    0.5796 |      -0.2264 |  47 |
|     0   |    0.7949 |    0.8462 |       0.0513 |   1 |
|     0.5 |    0.9231 |    0.812  |      -0.1111 |   1 |
|     1   |    0.8034 |    0.6667 |      -0.1368 |   1 |
|     2   |    0.6923 |    0.5897 |      -0.1026 |   1 |

## Mode: drop

| level   |   po_mean |   gm_mean |   delta_mean |   n |
|:--------|----------:|----------:|-------------:|----:|
| drop_0  |    0.5575 |    0.5889 |       0.0313 |  48 |
| drop_1  |    0.6489 |    0.6743 |       0.0255 |  48 |
| drop_2  |    0.6538 |    0.7087 |       0.0548 |  48 |
| full    |    0.8782 |    0.8843 |       0.0061 |  48 |

## Reading

If Δ is near zero across all corruption levels, GrabMyo doesn't buy
robustness — the ablation story dies. If Δ grows with corruption level
(GM degrades more gracefully than PO), we have a defensible clinical
claim: 'GrabMyo is a safety net for cal-quality failures in deployment.'