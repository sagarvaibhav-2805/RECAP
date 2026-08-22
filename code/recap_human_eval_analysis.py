"""
Stage 11 -- once a sampled human-eval CSV (recap_sample_for_human_eval.py
output) comes back with the `human_preferred` column filled in
("chosen" | "rejected" | "tie"), computes Table 10's stats: agreement rate
with the automatic label, Spearman correlation, and chosen-win-rate.

Run:
    python recap_human_eval_analysis.py --labeled_csv recap_human_eval/sample_recap_dpo_500.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

import config as cfg


def analyze(labeled_csv: str) -> dict:
    df = pd.read_csv(labeled_csv)
    labeled = df[df["human_preferred"].isin(["chosen", "rejected", "tie"])].copy()
    if labeled.empty:
        raise ValueError(f"No filled-in human_preferred labels found in {labeled_csv}")

    # Agreement: does the human's pick match RECAP's automatic "chosen" pick?
    # ("tie" counts as disagreement -- the automatic label made a call, a tie
    # means the human didn't confirm it.)
    agreement = (labeled["human_preferred"] == "chosen").mean()

    # Spearman rho: automatic reward_margin (signed, positive = model agrees
    # "chosen" is better) vs a +1/0/-1 human signal for the same direction.
    # Spearman = Pearson correlation of ranks -- computed manually so this
    # doesn't pull in scipy as a dependency (pandas' own method="spearman"
    # calls into scipy internally).
    human_signal = labeled["human_preferred"].map({"chosen": 1, "tie": 0, "rejected": -1})
    rho = labeled["reward_margin"].rank().corr(human_signal.rank())

    chosen_win_rate = (labeled["human_preferred"] == "chosen").mean()

    result = {
        "n_pairs": len(labeled),
        "agreement": agreement,
        "spearman_rho": rho,
        "chosen_win_rate": chosen_win_rate,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_csv", required=True)
    args = parser.parse_args()
    result = analyze(args.labeled_csv)

    out_path = cfg.REPORT_ROOT / "tables" / "table10.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(out_path, index=False)
    print(f"[Done] {result} -> {out_path}")


if __name__ == "__main__":
    main()
