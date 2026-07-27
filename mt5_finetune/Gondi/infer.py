"""
Run inference + scoring for finetuned mT5 checkpoints, across any number of
directions in one pass.

Each entry in infer_config.json is one inference job: a CSV to translate, the
finetuned checkpoint to load, and the natural-language source/target names
used to build the T5-style instruction prefix -- same convention as
mt5_finetune.py's training-time prefix: "translate {src_name} to {tgt_name}: ".
Checkpoints are self-contained (saved via trainer.save_model() +
tokenizer.save_pretrained() during training), so each job loads its own
tokenizer+model directly from checkpoint_path.

infer_config.json format (a list of jobs, run in order):
    [
      {
        "name": "gondi_en2tgt",
        "csv_path": "/path/to/test.csv",
        "checkpoint_path": "./mt5-gondi-en2tgt-finetuned",
        "src_col": "English",
        "tgt_col": "Gondi",
        "src_name": "English",
        "tgt_name": "Gondi"
      },
      ...
    ]

"name" is optional (defaults to the checkpoint folder name) and is only used
to label output files/rows. If "tgt_col" is present in the CSV, BLEU/chrF++
are computed against it as the reference; if not, the job runs pure
inference (predictions only, no scores).

Run:
    python infer.py
"""

import os
import json
import csv
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import pandas as pd
import torch

print("=" * 60)
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("=" * 60)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu

MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_BEAMS  = 5

SCORES_CSV = Path("./infer_scores.csv")


def append_scores(job_name, n_rows, bleu, chrfpp):
    header = ["job", "rows", "bleu", "chrf++"]
    row = [job_name, n_rows,
           f"{bleu:.4f}" if bleu is not None else "",
           f"{chrfpp:.4f}" if chrfpp is not None else ""]
    is_new = not SCORES_CSV.exists()
    with open(SCORES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(header)
        w.writerow(row)
    print(f"[Scores] Appended '{job_name}' to {SCORES_CSV}")


def run_job(job, device):
    csv_path        = job["csv_path"]
    checkpoint_path = job["checkpoint_path"]
    src_col         = job["src_col"]
    tgt_col         = job.get("tgt_col")
    src_name        = job["src_name"]
    tgt_name        = job["tgt_name"]
    job_name        = job.get("name", Path(checkpoint_path).name)

    prefix = f"translate {src_name} to {tgt_name}: "

    print(f"\n========== {job_name} ==========")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  csv:        {csv_path}")
    print(f"  {src_col} -> {tgt_col or '?'}  |  prefix: '{prefix}'")

    df = pd.read_csv(csv_path)
    assert src_col in df.columns, \
        f"'{src_col}' not found in {csv_path}, got {list(df.columns)}"
    df[src_col] = df[src_col].astype(str).str.strip()
    df = df[df[src_col] != ""].reset_index(drop=True)

    has_ref = tgt_col is not None and tgt_col in df.columns
    if has_ref:
        df[tgt_col] = df[tgt_col].astype(str).str.strip()
    print(f"[Data] rows to translate: {len(df)}  (reference available: {has_ref})")

    print("[Model] Loading tokenizer & model from checkpoint ...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    src_texts = df[src_col].tolist()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(src_texts), BATCH_SIZE):
            chunk = [prefix + str(t) for t in src_texts[i:i + BATCH_SIZE]]
            enc = tokenizer(
                chunk, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_LENGTH,
            ).to(device)
            out = model.generate(
                **enc,
                max_length=MAX_LENGTH,
                num_beams=NUM_BEAMS,
            )
            all_preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
            if (i // BATCH_SIZE) % 20 == 0:
                print(f"  decoded {i + len(chunk)}/{len(src_texts)}")

    out_cols = {f"source_{src_col.lower()}": src_texts, "prediction": all_preds}
    if has_ref:
        out_cols[f"reference_{tgt_col.lower()}"] = df[tgt_col].tolist()

    preds_path = f"infer_predictions_{job_name}.csv"
    pd.DataFrame(out_cols).to_csv(preds_path, index=False, encoding="utf-8-sig")
    print(f"[Save] Predictions: {preds_path}")

    bleu = chrfpp = None
    if has_ref:
        refs = df[tgt_col].tolist()
        bleu   = sacrebleu.corpus_bleu(all_preds, [refs]).score
        chrfpp = sacrebleu.corpus_chrf(all_preds, [refs], word_order=2).score
        print(f"BLEU   : {bleu:.4f}")
        print(f"chrF++ : {chrfpp:.4f}")
    else:
        print("[Score] No reference column found -- predictions-only run, no BLEU/chrF++.")

    append_scores(job_name, len(df), bleu, chrfpp)


def main():
    config_path = Path(__file__).resolve().parent / "infer_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    for job in jobs:
        run_job(job, device)

    print(f"\nAll jobs done. Scores: {SCORES_CSV}")


if __name__ == "__main__":
    main()
