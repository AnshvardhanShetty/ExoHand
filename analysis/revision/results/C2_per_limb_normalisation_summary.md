# C2 — per-limb normalisation cross-arm test

n = 48 patients. Three normalization variants for cross-arm PO.

## Results

| Variant | Description | Mean acc | Gap from own-cal (0.875) |
|---|---|---:|---:|
| **V1 baseline** | scaler on healthy-arm cal only | 0.5486 | +0.3264 |
| **V2 per-limb z** | separate scalers per limb | 0.5290 | +0.3460 |
| **V3 amplitude-eq** | rescale healthy to impaired-mean | 0.3999 | +0.4751 |

## Decision (pre-registered)

- If V2 or V3 closes the gap to < 20 pp (i.e., mean acc > 0.675): story is
  'scale/SNR mismatch', not 'pathology'. Report both numbers regardless.
- If both stay near V1 baseline: pathology story holds.