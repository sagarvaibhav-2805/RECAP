"""
Stage 2 -- fits RewardEngine on the train split only, saves stats.json
(RewardEngine.serialize() output). One stats.json per (lang, direction),
shared by every reward-preset -- calibration stats don't depend on reward
weights, only on the raw metric distributions.

Run one combination:
    python recap_calibrate.py --lang Bhili --direction hi2tgt
Run all configured combinations sequentially:
    python recap_calibrate.py
"""

from __future__ import annotations

import argparse

import pandas as pd

import config as cfg
import recap_checks
from recap_reward import RewardEngine


def _melt_to_long(train_df: pd.DataFrame) -> pd.DataFrame:
    """Wide (one row per source, per-model prediction/BLEU/chrF++/COMET
    columns) -> long (one row per source x model)."""
    lang_prefix = [c.split("_")[0] for c in train_df.columns if c.endswith("_nllb") and not c.startswith(("BLEU", "chrf", "COMET"))][0]
    rows = []
    for model in cfg.MODEL_NAMES:
        pred_col = f"{lang_prefix}_{model}"
        bleu_col = f"BLEU_{model}"
        chrf_col = f"chrf++_{model}"
        comet_col = f"COMET_{model}"
        sub = train_df[["source", "gold_truth", pred_col, bleu_col, chrf_col, comet_col]].copy()
        sub.columns = ["source", "gold_truth", "prediction", "bleu", "chrf", "comet"]
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def process_one(lang: str, direction: str) -> None:
    train_path = cfg.split_dir(lang, direction) / "train.csv"
    if not train_path.exists():
        print(f"[Skip] {lang}/{direction}: {train_path} not found -- run recap_split.py first")
        return

    out_path = cfg.calib_path(lang, direction)
    if out_path.exists():
        print(f"[Skip] {lang}/{direction}: {out_path} already exists (delete to redo)")
        return

    train_df = pd.read_csv(train_path)
    long_df = _melt_to_long(train_df)

    # Any preset works here -- fit() ignores weights entirely, it only
    # computes descriptive stats of the raw metrics.
    engine = RewardEngine(cfg.REWARD_PRESETS["recap_dpo"])
    engine.fit(long_df)
    engine.serialize(out_path)

    check = recap_checks.check_calibration_stats_finite(lang, direction)
    if not check.ok:
        out_path.unlink()  # don't leave a broken stats.json for downstream stages to trust
        raise recap_checks.CheckFailure(f"{lang}/{direction} calibration_finite: {check.message}")

    print(f"[Done] {lang}/{direction}: calibrated on {len(long_df)} candidate rows -> {out_path}")
    for q, stat in engine.stats.items():
        print(f"    {q}: mu={stat['mu']:.4f} sigma={stat['sigma']:.4f}")


def main() -> None:
    recap_checks.run_static_checks()

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=cfg.LANGUAGES, default=None)
    parser.add_argument("--direction", choices=cfg.DIRECTIONS, default=None)
    args = parser.parse_args()
    if bool(args.lang) != bool(args.direction):
        parser.error("--lang and --direction must be given together")

    jobs = [(args.lang, args.direction)] if args.lang else [
        (lang, direction) for lang in cfg.LANGUAGES for direction in cfg.DIRECTIONS
    ]
    for lang, direction in jobs:
        process_one(lang, direction)


if __name__ == "__main__":
    main()
