"""
Stage 10 -- single reusable inference entrypoint. Given a source sentence (or
a CSV of them) and a (lang, direction), resolves the winning checkpoint from
config.best_checkpoint_path() (written by recap_evaluate.py) and translates.

recap_evaluate.py (Stage 9) imports translate() from here directly, so
"what we evaluate" and "what we deploy" are always the same code path.

CLI:
    python recap_infer.py --lang Bhili --direction hi2tgt --source "..."
    python recap_infer.py --lang Bhili --direction hi2tgt --input_csv sources.csv --output_csv out.csv
    python recap_infer.py --lang Bhili --direction hi2tgt --source "..." --checkpoint /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import config as cfg
import recap_utils

_MODEL_CACHE: dict[str, tuple] = {}


def resolve_checkpoint(lang: str, direction: str, checkpoint: str | None = None, experiment: str | None = None) -> str:
    if checkpoint:
        return checkpoint
    if experiment:
        exp = cfg.EXPERIMENTS[experiment]
        stage_root = {"dpo": cfg.DPO_ROOT, "grpo": cfg.GRPO_ROOT, "ppo": cfg.PPO_ROOT}.get(exp.trainer)
        if stage_root is None:  # sft
            return str(cfg.sft_checkpoint_path(lang, direction))
        return str(cfg.checkpoint_dir(stage_root, lang, direction, experiment, cfg.SEED))

    best_path = cfg.best_checkpoint_path(lang, direction)
    if not best_path.exists():
        raise FileNotFoundError(
            f"No best-checkpoint recorded for {lang}/{direction} at {best_path}. "
            f"Run recap_evaluate.py first, or pass --checkpoint/--experiment explicitly."
        )
    with open(best_path) as f:
        return json.load(f)["checkpoint_path"]


def _load_model(checkpoint_path: str):
    if checkpoint_path in _MODEL_CACHE:
        return _MODEL_CACHE[checkpoint_path]
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    _MODEL_CACHE[checkpoint_path] = (model, tokenizer)
    return model, tokenizer


def translate(
    sources: list[str],
    lang: str,
    direction: str,
    checkpoint: str | None = None,
    experiment: str | None = None,
) -> list[str]:
    """The single translation entrypoint reused by CLI use, recap_evaluate.py,
    and (via recap_utils.generate_batch directly) training-time validation."""
    checkpoint_path = resolve_checkpoint(lang, direction, checkpoint=checkpoint, experiment=experiment)
    model, tokenizer = _load_model(checkpoint_path)
    label = experiment or Path(checkpoint_path).name
    return recap_utils.generate_batch(model, tokenizer, sources, cfg.INFERENCE_CONFIG, desc=f"Translating {lang}/{direction} [{label}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=cfg.LANGUAGES, required=True)
    parser.add_argument("--direction", choices=cfg.DIRECTIONS, required=True)
    parser.add_argument("--source", default=None, help="A single source sentence")
    parser.add_argument("--input_csv", default=None, help="CSV with a 'source' column")
    parser.add_argument("--output_csv", default=None, help="Where to write translations (with --input_csv)")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path override")
    parser.add_argument("--experiment", choices=list(cfg.EXPERIMENTS.keys()), default=None)
    args = parser.parse_args()

    if bool(args.source) == bool(args.input_csv):
        parser.error("Pass exactly one of --source or --input_csv")

    if args.source:
        result = translate([args.source], args.lang, args.direction, checkpoint=args.checkpoint, experiment=args.experiment)
        print(result[0])
    else:
        df = pd.read_csv(args.input_csv)
        translations = translate(df["source"].tolist(), args.lang, args.direction, checkpoint=args.checkpoint, experiment=args.experiment)
        df["translation"] = translations
        if args.output_csv:
            df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
            print(f"[Done] Wrote {len(df)} translations -> {args.output_csv}")
        else:
            print(df[["source", "translation"]].to_string(index=False))


if __name__ == "__main__":
    main()
