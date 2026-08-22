"""
Stage 4 -- builds DPO preference pairs from the train split, using
PreferenceBuilder + the cached Stage-3 raw metrics (no COMET recompute).
One reward-preset ("experiment") at a time, since different presets produce
different, non-interchangeable pair files.

Run one combination:
    python recap_mine_pairs.py --lang Bhili --direction hi2tgt --experiment recap_dpo
Run all configured combinations x all experiments that need a DPO reward preset:
    python recap_mine_pairs.py
"""

from __future__ import annotations

import argparse

import pandas as pd

import config as cfg
import recap_checks
from recap_preference import PreferenceBuilder
from recap_reward import RewardEngine


def _lang_prefix(columns: list[str]) -> str:
    for c in columns:
        if c.endswith("_nllb") and not c.startswith(("BLEU", "chrf", "COMET")):
            return c.split("_")[0]
    raise ValueError("Could not infer language prefix from columns")


def process_one(lang: str, direction: str, experiment: str) -> None:
    exp = cfg.EXPERIMENTS[experiment]
    if exp.reward_preset is None:
        print(f"[Skip] {experiment}: not a DPO/reward-preset experiment")
        return
    reward_config = cfg.REWARD_PRESETS[exp.reward_preset]

    out_dir = cfg.pairs_experiment_dir(lang, direction, experiment)
    out_path = out_dir / "pairs_all.csv"
    if out_path.exists():
        print(f"[Skip] {lang}/{direction}/{experiment}: {out_path} already exists (delete to redo)")
        return

    train_path = cfg.split_dir(lang, direction) / "train.csv"
    manifest_path = cfg.split_dir(lang, direction) / "manifest.csv"
    rewards_path = cfg.rewards_path(lang, direction)
    calib_path = cfg.calib_path(lang, direction)
    if not (train_path.exists() and manifest_path.exists() and rewards_path.exists() and calib_path.exists()):
        print(f"[Skip] {lang}/{direction}: missing split/rewards/calibration -- run Stages 1-3 first")
        return

    calib_check = recap_checks.check_calibration_stats_finite(lang, direction)
    if not calib_check.ok:
        raise recap_checks.CheckFailure(f"{lang}/{direction} calibration_finite: {calib_check.message}")

    train_df = pd.read_csv(train_path)
    rewards_df = pd.read_csv(rewards_path)
    rewards_df = rewards_df[rewards_df["split"] == "train"].reset_index(drop=True)

    prefix = _lang_prefix(train_df.columns.tolist())
    model_pred_cols = {m: f"{prefix}_{m}" for m in cfg.MODEL_NAMES}

    engine = RewardEngine.load(calib_path, reward_config=reward_config)
    builder = PreferenceBuilder(engine, margin_delta=cfg.MARGIN_DELTA)
    pairs = builder.build_pairs(train_df, model_pred_cols, rewards_df)

    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        manifest_df = pd.read_csv(manifest_path)
        leakage_check = recap_checks.check_split_leakage(manifest_df, pairs_df)
        if not leakage_check.ok:
            raise recap_checks.CheckFailure(f"{lang}/{direction}/{experiment} split_leakage: {leakage_check.message}")
        pairs_check = recap_checks.check_pairs_valid(pairs_df)
        if not pairs_check.ok:
            raise recap_checks.CheckFailure(f"{lang}/{direction}/{experiment} pairs_valid: {pairs_check.message}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_df["split"] = "train"
    pairs_df["random_seed"] = cfg.SPLIT_SEED
    pairs_df["config_hash"] = engine.reward_config.name  # cheap stand-in; full hash saved in run_manifest
    pairs_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[Done] {lang}/{direction}/{experiment}: {len(pairs_df)} pairs -> {out_path}")


def main() -> None:
    recap_checks.run_static_checks()

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

    for lang, direction in lang_direction_jobs:
        for experiment in experiments:
            process_one(lang, direction, experiment)


if __name__ == "__main__":
    main()
