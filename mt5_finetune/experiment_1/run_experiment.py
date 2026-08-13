"""
Experiment 1: mT5 beam-width ablation on Bhili <-> Hindi.

Randomly samples N_SAMPLES rows from the held-out Bhili test split
(datasets/bhilli/test.csv -- the actual train/val/test split, not the
separate inference_data CSV used by infer.py) and translates each one with
num_beams = 1, 2, 3, 4, 5 in turn (every other generation setting held
fixed at MAX_LENGTH=128, matching mt5_finetune/*/infer.py), for both
directions:
    hi2tgt : Hindi -> Bhili   (checkpoint: ../Bhili/mt5-bhili-hi2tgt-finetuned)
    tgt2hi : Bhili -> Hindi   (checkpoint: ../Bhili/mt5-bhili-tgt2hi-finetuned)

Output: one CSV per direction, 7 columns:
    source, gold_truth, beam_1, beam_2, beam_3, beam_4, beam_5

Run from this directory (mt5_finetune/experiment_1/) -- checkpoint paths
are relative to here, pointing at the sibling ../Bhili/ folder:
    python run_experiment.py
"""

import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MAX_LENGTH = 128   # same as mt5_finetune/*/infer.py -- only num_beams varies here
SEED       = 42
N_SAMPLES  = 10
BEAM_RANGE = range(1, 6)  # 1..5

TEST_CSV = "/home/scai/msr/aiy257590/flash/GRPO_RESEARCH/datasets/bhilli/test.csv"

DIRECTIONS = {
    "hi2tgt": {
        "checkpoint": "../Bhili/mt5-bhili-hi2tgt-finetuned",
        "src_col": "Hindi", "tgt_col": "Bhili",
        "src_name": "Hindi", "tgt_name": "Bhili",
    },
    "tgt2hi": {
        "checkpoint": "../Bhili/mt5-bhili-tgt2hi-finetuned",
        "src_col": "Bhili", "tgt_col": "Hindi",
        "src_name": "Bhili", "tgt_name": "Hindi",
    },
}


def load_samples():
    df = pd.read_csv(TEST_CSV)
    needed = {"English", "Hindi", "Bhili"}
    assert needed.issubset(df.columns), \
        f"Expected columns {needed} in {TEST_CSV}, got {list(df.columns)}"
    df = df[["English", "Hindi", "Bhili"]].dropna()
    for col in ["English", "Hindi", "Bhili"]:
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["English"] != "") & (df["Hindi"] != "") & (df["Bhili"] != "")].reset_index(drop=True)
    return df.sample(n=N_SAMPLES, random_state=SEED).reset_index(drop=True)


def run_direction(direction, cfg, sample, device):
    print(f"\n========== {direction} ==========")
    checkpoint = cfg["checkpoint"]
    src_col, tgt_col = cfg["src_col"], cfg["tgt_col"]
    src_name, tgt_name = cfg["src_name"], cfg["tgt_name"]
    prefix = f"translate {src_name} to {tgt_name}: "

    print(f"[Model] Loading tokenizer & model from {checkpoint} ...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model.to(device)
    model.eval()

    sources = sample[src_col].tolist()
    golds = sample[tgt_col].tolist()

    beam_preds = {}
    with torch.no_grad():
        for beam in BEAM_RANGE:
            print(f"  [Generate] num_beams={beam} ...")
            preds = []
            for text in sources:
                enc = tokenizer(
                    prefix + text, return_tensors="pt",
                    truncation=True, max_length=MAX_LENGTH,
                ).to(device)
                out = model.generate(**enc, max_length=MAX_LENGTH, num_beams=beam)
                preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
            beam_preds[beam] = preds

    result = pd.DataFrame({"source": sources, "gold_truth": golds})
    for beam in BEAM_RANGE:
        result[f"beam_{beam}"] = beam_preds[beam]

    out_path = Path(f"./{direction}_beam_comparison.csv")
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[Save] {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    sample = load_samples()
    print(f"[Data] Sampled {len(sample)} rows (seed={SEED}) from {TEST_CSV}")
    for direction, cfg in DIRECTIONS.items():
        run_direction(direction, cfg, sample, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
