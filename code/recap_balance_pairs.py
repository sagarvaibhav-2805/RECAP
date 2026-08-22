"""
Stage 5 -- balances pairs_all.csv across generator-pair-type AND reward-margin
bins, so DPO training doesn't over-learn from one pair type or only
easy/high-margin examples. Also reports per-model preferred/rejected
participation (C_m+, C_m-) as a pathology-collapse check.

Run one combination:
    python recap_balance_pairs.py --lang Bhili --direction hi2tgt --experiment recap_dpo
Run all configured combinations x all reward-preset experiments:
    python recap_balance_pairs.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as cfg


def process_one(lang: str, direction: str, experiment: str) -> None:
    exp = cfg.EXPERIMENTS[experiment]
    if exp.reward_preset is None:
        print(f"[Skip] {experiment}: not a DPO/reward-preset experiment")
        return
    reward_config = cfg.REWARD_PRESETS[exp.reward_preset]

    exp_dir = cfg.pairs_experiment_dir(lang, direction, experiment)
    in_path = exp_dir / "pairs_all.csv"
    out_path = exp_dir / "pairs_balanced.csv"
    if not in_path.exists():
        print(f"[Skip] {lang}/{direction}/{experiment}: {in_path} not found -- run recap_mine_pairs.py first")
        return
    if out_path.exists():
        print(f"[Skip] {lang}/{direction}/{experiment}: {out_path} already exists (delete to redo)")
        return

    pairs_df = pd.read_csv(in_path)
    if not reward_config.use_balanced_sampling:
        pairs_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[Done] {lang}/{direction}/{experiment}: balancing disabled for this preset, "
              f"copied {len(pairs_df)} pairs as-is -> {out_path}")
        return

    if pairs_df.empty:
        pairs_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[Done] {lang}/{direction}/{experiment}: no pairs to balance -> {out_path}")
        return

    type_counts = pairs_df["pair_type"].value_counts()
    cap_per_type = cfg.BALANCE_CAP_PER_TYPE or int(type_counts.min())

    n_bins = cfg.BALANCE_MARGIN_BINS
    rng = np.random.default_rng(cfg.SPLIT_SEED)

    kept_frames = []
    for pair_type, group in pairs_df.groupby("pair_type"):
        group = group.copy()
        try:
            group["_margin_bin"] = pd.qcut(group["reward_margin"], q=min(n_bins, group["reward_margin"].nunique()), duplicates="drop")
        except ValueError:
            group["_margin_bin"] = 0  # too few unique margins to bin -- keep as one bin

        n_target = min(cap_per_type, len(group))
        bins = group["_margin_bin"].unique()
        per_bin_target = max(1, n_target // max(len(bins), 1))

        bin_frames = []
        for b in bins:
            bin_group = group[group["_margin_bin"] == b]
            n_take = min(per_bin_target, len(bin_group))
            idx = rng.choice(bin_group.index.to_numpy(), size=n_take, replace=False)
            bin_frames.append(bin_group.loc[idx])
        kept = pd.concat(bin_frames) if bin_frames else group.iloc[0:0]

        # top-up if binning under-filled the per-type target (uneven bin sizes)
        if len(kept) < n_target:
            remaining = group.drop(index=kept.index)
            n_more = min(n_target - len(kept), len(remaining))
            if n_more > 0:
                idx = rng.choice(remaining.index.to_numpy(), size=n_more, replace=False)
                kept = pd.concat([kept, remaining.loc[idx]])

        kept_frames.append(kept.drop(columns=["_margin_bin"]))

    balanced_df = pd.concat(kept_frames, ignore_index=True) if kept_frames else pairs_df.iloc[0:0]
    balanced_df = balanced_df.sample(frac=1.0, random_state=cfg.SPLIT_SEED).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    n_pair = len(balanced_df)
    participation = []
    for model in cfg.MODEL_NAMES:
        c_plus = (balanced_df["chosen_model"] == model).sum() / max(n_pair, 1)
        c_minus = (balanced_df["rejected_model"] == model).sum() / max(n_pair, 1)
        participation.append(f"{model}: chosen={c_plus:.1%} rejected={c_minus:.1%}")

    print(f"[Done] {lang}/{direction}/{experiment}: {len(pairs_df)} -> {n_pair} balanced pairs -> {out_path}")
    print("    " + " | ".join(participation))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=cfg.LANGUAGES, default=None)
    parser.add_argument("--direction", choices=cfg.DIRECTIONS, default=None)
    parser.add_argument("--experiment", choices=list(cfg.EXPERIMENTS.keys()), default=None)
    args = parser.parse_args()
    if bool(args.lang) != bool(args.direction):
        parser.error("--lang and --direction must be given together")

    lang_direction_jobs = [(args.lang, args.direction)] if args.lang else [
        (lang, direction) for lang in cfg.LANGUAGES for direction in cfg.DIRECTIONS
    ]
    experiments = [args.experiment] if args.experiment else [
        name for name, exp in cfg.EXPERIMENTS.items() if exp.reward_preset is not None
    ]

    combos = [(lang, direction, experiment) for lang, direction in lang_direction_jobs for experiment in experiments]
    for lang, direction, experiment in tqdm(combos, desc="Balancing pairs (all combos)", disable=len(combos) < 2):
        process_one(lang, direction, experiment)


if __name__ == "__main__":
    main()
