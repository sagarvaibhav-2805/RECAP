"""
Standalone helper: given a sample-comparison CSV (the same format written
by run_experiment.py in this folder -- BLEU_1..BLEU_6 and
chrf++_1..chrf++_6 columns, one pair per sampling config), compute the
column-wise mean BLEU and mean chrF++ for each response across all rows
and print it in the same "mean_scores" shape used in sample_agreement.json:
    {
      "BLEU": {"sample_1": ..., "sample_2": ..., ...},
      "chrf++": {"sample_1": ..., "sample_2": ..., ...}
    }

Column detection is pattern-based (BLEU_<n> / chrf++_<n>), so this also
works on any CSV following the same naming convention regardless of how
many response columns it has (not hardcoded to exactly 6).

Usage:
    python compute_mean_scores.py /path/to/hi2tgt_sample_comparison.csv
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd


def main():
    if len(sys.argv) != 2:
        print("Usage: python compute_mean_scores.py <path_to_csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    bleu_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"BLEU_\d+", c)],
        key=lambda c: int(c.split("_")[1]),
    )
    chrf_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"chrf\+\+_\d+", c)],
        key=lambda c: int(c.split("_")[1]),
    )

    if not bleu_cols or not chrf_cols:
        print(f"[ERROR] No BLEU_N / chrf++_N columns found in {csv_path}.\n"
              f"        Got columns: {list(df.columns)}")
        sys.exit(1)

    mean_bleu = {
        c.replace("BLEU_", "sample_"): round(df[c].astype(float).mean(), 4)
        for c in bleu_cols
    }
    mean_chrf = {
        c.replace("chrf++_", "sample_"): round(df[c].astype(float).mean(), 4)
        for c in chrf_cols
    }
    mean_scores = {"BLEU": mean_bleu, "chrf++": mean_chrf}

    print(f"[Data] {csv_path}  ({len(df)} rows, {len(bleu_cols)} responses)\n")
    print(json.dumps(mean_scores, indent=2))


if __name__ == "__main__":
    main()
