"""
Experiment 1: NLLB beam-width ablation on Bhili <-> Hindi.

Same experiment as mt5_finetune/experiment_1/run_experiment.py, same random
sample (SEED=42 over the same test.csv) so the two models' outputs are
directly comparable row-for-row -- only the generation mechanics differ:
NLLB uses forced_bos_token_id + NLLB language codes instead of mT5's
"translate X to Y:" text prefix.

Randomly samples N_SAMPLES rows from the held-out Bhili test split
(datasets/bhilli/test.csv -- the actual train/val/test split, not the
separate inference_data CSV used by infer.py) and translates each one with
num_beams = 1, 2, 3, 4, 5 in turn (every other generation setting held
fixed at MAX_LENGTH=128, matching NLLB-finetune/*/infer.py), for both
directions:
    hi2tgt : Hindi -> Bhili   (checkpoint: ../Bhili/nllb-bhili-hi2tgt-finetuned)
    tgt2hi : Bhili -> Hindi   (checkpoint: ../Bhili/nllb-bhili-tgt2hi-finetuned)

Per-direction output CSV, 17 columns:
    source, gold_truth,
    beam_1, BLEU_1, chrf++_1, ... beam_5, BLEU_5, chrf++_5
BLEU/chrF++ are per-sentence scores (sacrebleu.sentence_bleu/sentence_chrf)
of that row's beam-N translation against gold_truth, not corpus-level.

At this scale (N_SAMPLES x 5 beams x 2 directions generate() calls) this is
a real, potentially long-running job, so progress is checkpointed every
SAVE_EVERY_ROWS rows -- if the job dies partway through, rerunning picks up
from the last flushed row instead of restarting.

After a direction's CSV is complete, beam_agreement.json is built (or
updated) with two things per direction:
  - "patterns": beam_1..beam_5 TEXT compared directly (identical text
    implies identical score) -- for every distinct partition of the 5
    beams into "agreed identically" groups, how many rows fall into that
    exact pattern, e.g.:
        "[1]|[2,3,4,5]"     -> beam 1 alone, beams 2-5 all agreed
        "[1,2]|[3,4,5]"     -> beams 1-2 agreed, beams 3-5 agreed
        "[1]|[2]|[3]|[4,5]" -> beams 1,2,3 each different, 4-5 agreed
  - "mean_scores": column-wise mean BLEU and mean chrF++ for each beam
    width (5 BLEU averages + 5 chrF++ averages), i.e. mean(BLEU_1) across
    all rows, mean(BLEU_2) across all rows, etc.

Run from this directory (NLLB-finetune/experiment_1/) -- checkpoint paths
are relative to here, pointing at the sibling ../Bhili/ folder:
    python run_experiment.py
"""

import json
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import pandas as pd
import sacrebleu
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MAX_LENGTH      = 128   # same as NLLB-finetune/*/infer.py -- only num_beams varies here
SEED            = 42
N_SAMPLES       = 5000
BEAM_RANGE      = list(range(1, 6))  # 1..5
SAVE_EVERY_ROWS = 500   # checkpoint interval -- see module docstring

TEST_CSV = "/home/scai/msr/aiy257590/flash/GRPO_RESEARCH/datasets/bhilli/test.csv"

DIRECTIONS = {
    "hi2tgt": {
        "checkpoint": "../Bhili/nllb-bhili-hi2tgt-finetuned",
        "src_col": "Hindi", "tgt_col": "Bhili",
        "src_lang": "hin_Deva", "tgt_lang": "bhb_Deva",
    },
    "tgt2hi": {
        "checkpoint": "../Bhili/nllb-bhili-tgt2hi-finetuned",
        "src_col": "Bhili", "tgt_col": "Hindi",
        "src_lang": "bhb_Deva", "tgt_lang": "hin_Deva",
    },
}

COLUMNS = ["source", "gold_truth"]
for _b in BEAM_RANGE:
    COLUMNS += [f"beam_{_b}", f"BLEU_{_b}", f"chrf++_{_b}"]


def load_samples():
    df = pd.read_csv(TEST_CSV)
    needed = {"English", "Hindi", "Bhili"}
    assert needed.issubset(df.columns), \
        f"Expected columns {needed} in {TEST_CSV}, got {list(df.columns)}"
    df = df[["English", "Hindi", "Bhili"]].dropna()
    for col in ["English", "Hindi", "Bhili"]:
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["English"] != "") & (df["Hindi"] != "") & (df["Bhili"] != "")].reset_index(drop=True)
    n = min(N_SAMPLES, len(df))
    return df.sample(n=n, random_state=SEED).reset_index(drop=True)


def _flush(rows, out_path):
    if not rows:
        return
    chunk_df = pd.DataFrame(rows, columns=COLUMNS)
    file_exists = out_path.exists()
    chunk_df.to_csv(out_path, mode="a", header=not file_exists, index=False, encoding="utf-8-sig")


def run_direction(direction, cfg, sample, device):
    print(f"\n========== {direction} ==========")
    total_rows = len(sample)
    out_path = Path(f"./{direction}_beam_comparison.csv")

    # ---- Resume check ----
    n_done = 0
    if out_path.exists():
        existing_cols = list(pd.read_csv(out_path, nrows=0).columns)
        if existing_cols != COLUMNS:
            print(f"[ERROR] {out_path} exists with a different schema than expected.\n"
                  f"         existing: {existing_cols}\n"
                  f"         expected: {COLUMNS}\n"
                  f"         Refusing to touch it (likely leftover from an earlier version) "
                  f"-- delete this file manually and rerun.")
            return
        n_done = len(pd.read_csv(out_path))
        print(f"[Resume] Found existing {out_path} with {n_done}/{total_rows} rows already done.")
    if n_done >= total_rows:
        print(f"[Resume] {direction} already fully done -- skipping.")
        return
    if n_done > 0:
        print(f"[Resume] Continuing from row {n_done}/{total_rows}.")

    checkpoint = cfg["checkpoint"]
    src_col, tgt_col = cfg["src_col"], cfg["tgt_col"]
    src_lang, tgt_lang = cfg["src_lang"], cfg["tgt_lang"]

    print(f"[Model] Loading tokenizer & model from {checkpoint} ...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model.to(device)
    model.eval()

    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if forced_bos_id == tokenizer.unk_token_id:
        raise ValueError(
            f"[{direction}] tgt_lang '{tgt_lang}' has no registered token in this "
            "checkpoint's tokenizer -- wrong checkpoint_path for this direction?"
        )
    tokenizer.src_lang = src_lang

    sources = sample[src_col].tolist()
    golds = sample[tgt_col].tolist()

    buf_rows = []
    with torch.no_grad():
        for i in range(n_done, total_rows):
            src_text, gold_text = sources[i], golds[i]
            row = [src_text, gold_text]
            for beam in BEAM_RANGE:
                enc = tokenizer(
                    src_text, return_tensors="pt",
                    truncation=True, max_length=MAX_LENGTH,
                ).to(device)
                out = model.generate(
                    **enc,
                    forced_bos_token_id=forced_bos_id,
                    max_length=MAX_LENGTH,
                    num_beams=beam,
                )
                pred = tokenizer.decode(out[0], skip_special_tokens=True)
                bleu = f"{sacrebleu.sentence_bleu(pred, [gold_text]).score:.4f}"
                chrfpp = f"{sacrebleu.sentence_chrf(pred, [gold_text], word_order=2).score:.4f}"
                row += [pred, bleu, chrfpp]
            buf_rows.append(row)

            if len(buf_rows) >= SAVE_EVERY_ROWS:
                _flush(buf_rows, out_path)
                print(f"  [Save] {direction}: {i + 1}/{total_rows} rows done")
                buf_rows = []

    _flush(buf_rows, out_path)
    print(f"[Save] {out_path}  ({total_rows} rows)")


def _group_beams_by_text(row):
    """row: pandas Series/dict with beam_1..beam_5 columns. Returns a list
    of groups (each a list of beam ids) that produced identical text."""
    groups = {}
    for beam in BEAM_RANGE:
        text = row[f"beam_{beam}"]
        groups.setdefault(text, []).append(beam)
    return sorted(groups.values(), key=lambda g: min(g))


def _pattern_key(groups):
    return "|".join("[" + ",".join(str(b) for b in sorted(g)) + "]" for g in groups)


def build_agreement_json():
    result = {}
    for direction in DIRECTIONS:
        out_path = Path(f"./{direction}_beam_comparison.csv")
        if not out_path.exists():
            continue
        df = pd.read_csv(out_path)
        total = len(df)

        pattern_counts = {}
        for _, row in df.iterrows():
            groups = _group_beams_by_text(row)
            key = _pattern_key(groups)
            pattern_counts[key] = pattern_counts.get(key, 0) + 1
        patterns = {
            k: {"count": v, "pct": round(100.0 * v / total, 2) if total else 0.0}
            for k, v in sorted(pattern_counts.items(), key=lambda kv: -kv[1])
        }

        mean_bleu = {f"beam_{b}": round(df[f"BLEU_{b}"].astype(float).mean(), 4) for b in BEAM_RANGE}
        mean_chrf = {f"beam_{b}": round(df[f"chrf++_{b}"].astype(float).mean(), 4) for b in BEAM_RANGE}

        result[direction] = {
            "total_rows": total,
            "patterns": patterns,
            "mean_scores": {"BLEU": mean_bleu, "chrf++": mean_chrf},
        }

    json_path = Path("./beam_agreement.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[Save] {json_path}")
    for direction, info in result.items():
        print(f"\n{direction} (total_rows={info['total_rows']}):")
        print(f"  mean BLEU   : {info['mean_scores']['BLEU']}")
        print(f"  mean chrF++ : {info['mean_scores']['chrf++']}")
        for pattern, stats in info["patterns"].items():
            print(f"  {pattern:30s} count={stats['count']:5d}  pct={stats['pct']:.2f}%")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    sample = load_samples()
    print(f"[Data] Sampled {len(sample)} rows (seed={SEED}) from {TEST_CSV}")
    for direction, cfg in DIRECTIONS.items():
        run_direction(direction, cfg, sample, device)
    build_agreement_json()
    print("\nDone.")


if __name__ == "__main__":
    main()
