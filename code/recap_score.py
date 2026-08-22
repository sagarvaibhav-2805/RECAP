"""
Stage 3 -- computes RAW per-candidate metrics (bleu, chrf, comet, rep, len,
valid) for every row (train+val+test) x every model, using the stored
BLEU/chrF++/COMET already in maha_data_2 -- no COMET model reload needed here.
Saved in long format so Stage 4/5 can cheaply re-score under any reward-preset
without ever re-touching the expensive metrics.

Run one combination:
    python recap_score.py --lang Bhili --direction hi2tgt
Run all configured combinations sequentially:
    python recap_score.py
"""

from __future__ import annotations

import argparse

import pandas as pd

import config as cfg
from recap_reward import RewardEngine


def _lang_prefix(columns: list[str]) -> str:
    for c in columns:
        if c.endswith("_nllb") and not c.startswith(("BLEU", "chrf", "COMET")):
            return c.split("_")[0]
    raise ValueError("Could not infer language prefix from columns")


def process_one(lang: str, direction: str) -> None:
    out_path = cfg.rewards_path(lang, direction)
    if out_path.exists():
        print(f"[Skip] {lang}/{direction}: {out_path} already exists (delete to redo)")
        return

    frames = []
    for split_name in ("train", "val", "test"):
        split_path = cfg.split_dir(lang, direction) / f"{split_name}.csv"
        if not split_path.exists():
            print(f"[Skip] {lang}/{direction}: {split_path} not found -- run recap_split.py first")
            return
        df = pd.read_csv(split_path)
        df["_split"] = split_name
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    prefix = _lang_prefix(all_df.columns.tolist())
    engine = RewardEngine(cfg.REWARD_PRESETS["recap_dpo"])  # only used for compute_raw_metrics, config-agnostic

    rows = []
    for model in cfg.MODEL_NAMES:
        pred_col = f"{prefix}_{model}"
        bleu_col, chrf_col, comet_col = f"BLEU_{model}", f"chrf++_{model}", f"COMET_{model}"
        precomputed = [
            {"bleu": b, "chrf": c, "comet": m}
            for b, c, m in zip(all_df[bleu_col], all_df[chrf_col], all_df[comet_col])
        ]
        raw = engine.compute_raw_metrics(
            all_df["source"].tolist(), all_df[pred_col].tolist(), all_df["gold_truth"].tolist(),
            precomputed=precomputed,
        )
        for i, r in enumerate(raw):
            rows.append({
                "source_id": all_df["source_id"].iloc[i],
                "split": all_df["_split"].iloc[i],
                "model": model,
                **r,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    n_invalid = sum(1 for r in rows if not r["valid"])
    print(f"[Done] {lang}/{direction}: {len(rows)} (source x model) rows -> {out_path} ({n_invalid} flagged invalid)")


def main() -> None:
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
