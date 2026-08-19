"""
Reorder columns in the maha_data_2/<lang>/<direction>.csv files so each
model's metrics are grouped together: prediction, BLEU, chrF++, COMET --
instead of all predictions/BLEU/chrF++ first and all COMET scores tacked on
at the end.

Before: source,gold_truth,<Lang>_nllb,BLEU_nllb,chrf++_nllb,<Lang>_mt5,BLEU_mt5,chrf++_mt5,...,COMET_nllb,COMET_mt5,COMET_qwen,COMET_llama
After:  source,gold_truth,<Lang>_nllb,BLEU_nllb,chrf++_nllb,COMET_nllb,<Lang>_mt5,BLEU_mt5,chrf++_mt5,COMET_mt5,...

Overwrites each CSV in place (same file, just reordered columns -- no data
is changed or dropped).

Run from the GRPO_RESEARCH root:
    python reorder_maha_data_2.py
"""

from pathlib import Path

import pandas as pd

DATA_ROOT = Path("./maha_data_2")

LANGUAGES = ["Bhili", "Gondi", "Mundari"]
DIRECTIONS = ["hi2tgt", "tgt2hi"]
MODEL_NAMES = ["nllb", "mt5", "qwen", "llama"]


def _find_pred_col(columns, model):
    candidates = [c for c in columns if c.endswith(f"_{model}")
                  and c not in (f"BLEU_{model}", f"chrf++_{model}", f"COMET_{model}")]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one prediction column for model "
                          f"'{model}', found {candidates} in {columns}")
    return candidates[0]


def reorder_one(lang, direction):
    path = DATA_ROOT / lang.lower() / f"{direction}.csv"
    if not path.exists():
        print(f"[Skip] {lang}/{direction}: {path} not found")
        return

    df = pd.read_csv(path)
    columns = list(df.columns)

    new_order = ["source", "gold_truth"]
    for model in MODEL_NAMES:
        pred_col = _find_pred_col(columns, model)
        new_order += [pred_col, f"BLEU_{model}", f"chrf++_{model}", f"COMET_{model}"]

    missing = set(new_order) - set(columns)
    extra = set(columns) - set(new_order)
    if missing or extra:
        print(f"[ERROR] {path}: column mismatch, refusing to touch it.\n"
              f"         missing: {missing}\n"
              f"         extra:   {extra}")
        return

    df = df[new_order]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Done] {path}")


def main():
    for lang in LANGUAGES:
        for direction in DIRECTIONS:
            reorder_one(lang, direction)


if __name__ == "__main__":
    main()
