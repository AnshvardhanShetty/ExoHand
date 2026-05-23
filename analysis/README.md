# analysis/

All paper-bound experiments and reproductions. Each subdirectory is one analysis.

## Conventions

- **Seed**: `from analysis.seed import SEED, seed_everything` — canonical seed is 42.
- **Cache**: `analysis/.cache/` holds expensive intermediates (engineered feature CSVs, model artifacts). Gitignored.
- **Outputs**: each analysis writes its CSVs/figures to its own subdirectory.
- **Reproducibility**: every script takes `--seed` and `--force-rebuild-cache` flags where applicable.

## Contents

| Directory | Purpose |
|---|---|
| `reproduce_headline/` | LOSO reproduction of the README's 95.9% / 85.6% claims with bootstrap CIs. |
| `emgbench/` (planned) | Stream 1 — hybrid integration against EMGBench's 6 datasets. |
| `lucchetti/` (planned) | Stream 2 — stroke validation against Lucchetti et al. 2025. |

## Running

```bash
# Headline reproduction smoke test (one participant, ~10-15 min)
python analysis/reproduce_headline/loso_eval.py --participants participant1

# Full 43-fold reproduction (queue overnight)
python analysis/reproduce_headline/loso_eval.py
```
