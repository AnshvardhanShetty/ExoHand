"""
Frozen split generator — one canonical (cal_idx, test_idx) per patient, saved once,
consumed by every leakage-free re-run. Prevents the v2 MLP-big anomaly (split
comparability failure).

Outputs:
  analysis/revision/frozen_splits.parquet
    columns: patient, session, cal_idx (list), test_idx (list, only for impaired_01)
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from analysis.seed import SEED, seed_everything
from analysis.physiomio.per_session_eval import PHYSIOMIO_PKL, TEST_PER_CLASS
from analysis.revision.recompute_A_lda_at_cal_sizes import split_at

CAL_SIZE = 36
OUT = PROJECT_ROOT / "analysis" / "revision" / "frozen_splits.parquet"


def main():
    seed_everything(SEED)
    df = pd.read_pickle(PHYSIOMIO_PKL).reset_index(drop=True)

    rows = []
    for patient in sorted(df.participant.unique(), key=lambda s: int(s.replace("patient", ""))):
        # Impaired arm — both cal and test
        s_imp = df[(df.participant == patient) & (df.session == "impaired_01")]
        if len(s_imp) > 0:
            try:
                rng = np.random.RandomState(SEED)
                test_idx, cal_idx = split_at(s_imp, CAL_SIZE, TEST_PER_CLASS, rng)
                if len(test_idx) >= 15 and len(cal_idx) >= 6:
                    rows.append({
                        "patient": patient,
                        "session": "impaired_01",
                        "cal_idx": list(int(i) for i in cal_idx),
                        "test_idx": list(int(i) for i in test_idx),
                    })
            except Exception as e:
                print(f"  {patient} impaired_01 split failed: {e}")

        # Healthy arm — only cal used in Tier 1 (test doesn't get scored on)
        s_hlth = df[(df.participant == patient) & (df.session == "healthy_01")]
        if len(s_hlth) > 0:
            try:
                rng = np.random.RandomState(SEED + 1)
                _, cal_idx = split_at(s_hlth, CAL_SIZE, TEST_PER_CLASS, rng)
                if len(cal_idx) >= 6:
                    rows.append({
                        "patient": patient,
                        "session": "healthy_01",
                        "cal_idx": list(int(i) for i in cal_idx),
                        "test_idx": [],
                    })
            except Exception as e:
                print(f"  {patient} healthy_01 split failed: {e}")

    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    n_imp = out[out.session == "impaired_01"].patient.nunique()
    n_hlth = out[out.session == "healthy_01"].patient.nunique()
    print(f"Wrote {OUT}")
    print(f"  {len(out)} split rows: {n_imp} impaired_01, {n_hlth} healthy_01")


if __name__ == "__main__":
    main()
