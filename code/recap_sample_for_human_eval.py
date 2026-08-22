"""
Stage 11 -- samples a balanced N-pair subset (default 500, paper Table 10)
across all six directions from one experiment's pairs_balanced.csv, for human
annotation. Re-run with a different --experiment to get the per-condition
rows Table 10 needs ("quality only", "+repetition+length", "+calibration",
"Full RECAP", ...).

Writes a CSV with an empty `human_preferred` column (values: "chosen",
"rejected", or "tie") for an annotator to fill in -- recap_human_eval_analysis.py
reads it back once labels exist.

Run:
    python recap_sample_for_human_eval.py --experiment recap_dpo --n 500
"""

from __future__ import annotations

import argparse

import pandas as pd

import config as cfg


def sample_for_human_eval(experiment: str, n: int, seed: int = cfg.SEED) -> None:
    frames = []
    n_directions = len(cfg.LANGUAGES) * len(cfg.DIRECTIONS)
    per_direction = max(1, n // n_directions)

    for lang in cfg.LANGUAGES:
        for direction in cfg.DIRECTIONS:
            path = cfg.pairs_experiment_dir(lang, direction, experiment) / "pairs_balanced.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            take = min(per_direction, len(df))
            sample = df.sample(n=take, random_state=seed)
            sample = sample.copy()
            sample["lang"] = lang
            sample["direction"] = direction
            frames.append(sample)

    if not frames:
        print(f"[Skip] {experiment}: no pairs_balanced.csv found for any direction")
        return

    combined = pd.concat(frames, ignore_index=True)
    if len(combined) > n:
        combined = combined.sample(n=n, random_state=seed).reset_index(drop=True)

    combined = combined[[
        "lang", "direction", "source_id", "source", "gold_truth",
        "chosen", "rejected", "chosen_model", "rejected_model", "reward_margin",
    ]].copy()
    combined["human_preferred"] = ""  # annotator fills in: "chosen" | "rejected" | "tie"

    out_dir = cfg.HUMAN_EVAL_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sample_{experiment}_{len(combined)}.csv"
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[Done] {experiment}: sampled {len(combined)} pairs -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=list(cfg.EXPERIMENTS.keys()), required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    args = parser.parse_args()
    sample_for_human_eval(args.experiment, args.n, args.seed)


if __name__ == "__main__":
    main()
