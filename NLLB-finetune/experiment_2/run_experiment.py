"""
Experiment 2: NLLB sampling-diversity ablation on Bhili <-> Hindi.

Same experiment as mt5_finetune/experiment_2/run_experiment.py, same random
sample (SEED=42 over the same test.csv) so the two models are directly
comparable row-for-row -- only the generation mechanics differ: NLLB uses
forced_bos_token_id + NLLB language codes instead of mT5's
"translate X to Y:" text prefix.

6 outputs per sentence from 6 different sampling configurations, all with
num_beams=2 fixed (do_sample=True + num_beams=2 is HF's "beam sample"
decoding -- multinomial sampling within a 2-beam search):

    id  label                temperature  top_p
    1   temp_0.7             0.7          1.0   (mild diversity)
    2   temp_1.5             1.5          1.0   (strong diversity)
    3   topp_0.7             1.0          0.7   (mild-moderate diversity)
    4   temp_1.15_topp_0.9   1.15         0.9   (mild combined -- replaced a
                                                  plain topp_0.9 config that
                                                  scored almost identically to
                                                  temp_0.7, i.e. wasn't adding
                                                  distinct diversity)
    5   topp_0.99            1.0          0.99  (near full-distribution)
    6   temp_1.3_topp_0.85   1.3          0.85  (combined -- reshapes AND
                                                 truncates the distribution,
                                                 likely most different of all six)

Each config uses its own fixed seed (SEED + config id) set immediately
before that generate() call, so every individual output is reproducible
run-to-run despite do_sample=True.

Per-direction output CSV, 20 columns:
    source, gold_truth,
    sample_1, BLEU_1, chrf++_1, ... sample_6, BLEU_6, chrf++_6
BLEU/chrF++ are per-sentence scores (sacrebleu.sentence_bleu/sentence_chrf)
of that row's sample-N translation against gold_truth, not corpus-level.

Generation is batched across rows (BATCH_SIZE rows per generate() call, one
call per config) rather than one row at a time -- rows within a batch share
a config's temperature/top_p, so they can be generated together with
padding, which keeps the GPU actually busy instead of bottlenecked on
Python-loop/kernel-launch overhead between single-sequence calls. A full
row is only appended to the write buffer once all 6 configs have finished
for its batch, so if the job dies mid-batch that batch's work is lost (up
to BATCH_SIZE rows), not just the last row -- a worthwhile tradeoff for the
large GPU-utilization/wall-clock improvement. Checkpointed to disk every
SAVE_EVERY_ROWS rows (same resumability pattern as experiment_1) -- if the
job dies, rerunning picks up from the last flushed row instead of
restarting from scratch.

After a direction's CSV is complete, sample_agreement.json is built (or
updated) with, per direction:
  - "patterns": sample_1..sample_6 TEXT compared directly (identical text
    implies identical score) -- every distinct partition of the 6 configs
    into "agreed identically" groups, with counts, e.g. "[1]|[2,3,4,5,6]".
  - "mean_scores": column-wise mean BLEU and mean chrF++ for each config
    (6 BLEU averages + 6 chrF++ averages).
Plus a top-level "configs" legend mapping each numeric id to its actual
temperature/top_p/num_beams.

Run from this directory (NLLB-finetune/experiment_2/) -- checkpoint paths
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

MAX_LENGTH      = 128   # same as NLLB-finetune/*/infer.py
SEED            = 42
N_SAMPLES       = 5000
NUM_BEAMS       = 2     # fixed for all 6 configs, per instruction
BATCH_SIZE      = 32    # rows generated together per config, for GPU utilization
SAVE_EVERY_ROWS = 500

TEST_CSV = "/home/scai/msr/aiy257590/flash/GRPO_RESEARCH/datasets/bhilli/test.csv"

CONFIGS = [
    {"id": 1, "label": "temp_0.7",           "temperature": 0.7, "top_p": 1.0},
    {"id": 2, "label": "temp_1.5",           "temperature": 1.5, "top_p": 1.0},
    {"id": 3, "label": "topp_0.7",           "temperature": 1.0, "top_p": 0.7},
    {"id": 4, "label": "temp_1.15_topp_0.9",  "temperature": 1.15, "top_p": 0.9},
    {"id": 5, "label": "topp_0.99",          "temperature": 1.0, "top_p": 0.99},
    {"id": 6, "label": "temp_1.3_topp_0.85", "temperature": 1.3, "top_p": 0.85},
]
CONFIG_IDS = [c["id"] for c in CONFIGS]

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
for _c in CONFIGS:
    COLUMNS += [f"sample_{_c['id']}", f"BLEU_{_c['id']}", f"chrf++_{_c['id']}"]


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
    out_path = Path(f"./{direction}_sample_comparison.csv")

    # ---- Resume check ----
    n_done = 0
    if out_path.exists():
        existing_cols = list(pd.read_csv(out_path, nrows=0).columns)
        if existing_cols != COLUMNS:
            print(f"[ERROR] {out_path} exists with a different schema than expected.\n"
                  f"         existing: {existing_cols}\n"
                  f"         expected: {COLUMNS}\n"
                  f"         Refusing to touch it -- delete this file manually and rerun.")
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
        for batch_start in range(n_done, total_rows, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_rows)
            batch_src = sources[batch_start:batch_end]
            batch_gold = golds[batch_start:batch_end]

            # rows[j] accumulates [source, gold, sample_1, BLEU_1, chrf++_1, ...]
            # for the j-th sentence in this batch, filled in one config at a time.
            rows = [[s, g] for s, g in zip(batch_src, batch_gold)]

            for sc in CONFIGS:
                torch.manual_seed(SEED + sc["id"])
                enc = tokenizer(
                    batch_src, return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_LENGTH,
                ).to(device)
                out = model.generate(
                    **enc,
                    forced_bos_token_id=forced_bos_id,
                    max_length=MAX_LENGTH,
                    num_beams=NUM_BEAMS,
                    do_sample=True,
                    temperature=sc["temperature"],
                    top_p=sc["top_p"],
                )
                preds = tokenizer.batch_decode(out, skip_special_tokens=True)
                for j, pred in enumerate(preds):
                    gold_text = batch_gold[j]
                    bleu = f"{sacrebleu.sentence_bleu(pred, [gold_text]).score:.4f}"
                    chrfpp = f"{sacrebleu.sentence_chrf(pred, [gold_text], word_order=2).score:.4f}"
                    rows[j] += [pred, bleu, chrfpp]

            buf_rows.extend(rows)
            print(f"  [Progress] {direction}: {batch_end}/{total_rows} rows done")

            if len(buf_rows) >= SAVE_EVERY_ROWS:
                _flush(buf_rows, out_path)
                print(f"  [Save] {direction}: {batch_end}/{total_rows} rows done")
                buf_rows = []

    _flush(buf_rows, out_path)
    print(f"[Save] {out_path}  ({total_rows} rows)")


def _group_samples_by_text(row):
    groups = {}
    for cid in CONFIG_IDS:
        text = row[f"sample_{cid}"]
        groups.setdefault(text, []).append(cid)
    return sorted(groups.values(), key=lambda g: min(g))


def _pattern_key(groups):
    return "|".join("[" + ",".join(str(c) for c in sorted(g)) + "]" for g in groups)


def build_agreement_json():
    result = {
        "configs": {
            str(c["id"]): {
                "label": c["label"], "temperature": c["temperature"],
                "top_p": c["top_p"], "num_beams": NUM_BEAMS,
            } for c in CONFIGS
        },
        "directions": {},
    }
    for direction in DIRECTIONS:
        out_path = Path(f"./{direction}_sample_comparison.csv")
        if not out_path.exists():
            continue
        df = pd.read_csv(out_path)
        total = len(df)

        pattern_counts = {}
        for _, row in df.iterrows():
            groups = _group_samples_by_text(row)
            key = _pattern_key(groups)
            pattern_counts[key] = pattern_counts.get(key, 0) + 1
        patterns = {
            k: {"count": v, "pct": round(100.0 * v / total, 2) if total else 0.0}
            for k, v in sorted(pattern_counts.items(), key=lambda kv: -kv[1])
        }

        mean_bleu = {f"sample_{c}": round(df[f"BLEU_{c}"].astype(float).mean(), 4) for c in CONFIG_IDS}
        mean_chrf = {f"sample_{c}": round(df[f"chrf++_{c}"].astype(float).mean(), 4) for c in CONFIG_IDS}

        result["directions"][direction] = {
            "total_rows": total,
            "patterns": patterns,
            "mean_scores": {"BLEU": mean_bleu, "chrf++": mean_chrf},
        }

    json_path = Path("./sample_agreement.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[Save] {json_path}")
    for direction, info in result["directions"].items():
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
