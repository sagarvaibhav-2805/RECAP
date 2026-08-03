"""
Run inference + scoring for finetuned Llama-3.1-8B checkpoints, across any
number of directions in one pass.

Llama is a decoder-only causal LM finetuned on Llama-3's manual chat format
(see llama_finetune.py -- the base tokenizer has no chat_template, so the
prompt is built by hand with the same special tokens used at training time)
-- inference here reproduces that exact prompt format so generation matches
what the model was trained to continue from. Each checkpoint is
self-contained (saved via trainer.save_model() + tokenizer.save_pretrained()
during training -- DeepSpeed ZeRO-2 consolidates the sharded weights back
into a normal HF checkpoint on save, so loading it here needs nothing
DeepSpeed-specific, just plain AutoModelForCausalLM.from_pretrained()), so
each job loads its own tokenizer+model directly from checkpoint_path.

infer_config.json format (a list of jobs, run in order):
    [
      {
        "name": "gondi_en2tgt",
        "csv_path": "/path/to/test.csv",
        "checkpoint_path": "./en2tgt/llama-gondi-en2tgt-finetuned",
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

checkpoint_path points into the per-direction training folders created
alongside this script (./en2tgt/, ./hi2tgt/, ./tgt2en/, ./tgt2hi/), since
each direction's training now runs as its own separate job/output location.

If more than one GPU is visible, jobs run in parallel -- one job per GPU,
round-robin if there are more jobs than GPUs (e.g. 4 direction-jobs on a
4-GPU node all run at once instead of sequentially). A single Llama-3.1-8B
checkpoint fits comfortably on one A100-80GB for inference (no DeepSpeed
sharding needed -- that's only required for full fine-tuning's optimizer
states). With a single GPU (or CPU), jobs run sequentially, same as before.

Run:
    python infer.py
"""

import os
import json
import csv
import multiprocessing as mp
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
from transformers import AutoTokenizer, AutoModelForCausalLM
import sacrebleu

MAX_LENGTH      = 256   # prompt token budget -- same as training (llama_finetune.py)
MAX_NEW_TOKENS  = 128   # generation cap for the response only
BATCH_SIZE      = 16
NUM_BEAMS       = 4
SAVE_EVERY_ROWS = 5000  # flush predictions to disk this often, for resumability
SYSTEM_PROMPT   = "You are a helpful assistant."

# Llama-3 special tokens for the chat format (present in the base tokenizer's
# vocab even though no chat_template is configured for it) -- must match
# llama_finetune.py exactly, since the model was trained on this format.
BOS_TOKEN = "<|begin_of_text|>"
EOT_TOKEN = "<|eot_id|>"

SCORES_CSV = Path("./infer_scores.csv")


def _turn(role, content):
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n{str(content).strip()}{EOT_TOKEN}"


def build_prompt_text(src_name, tgt_name, src_text):
    # Identical to llama_finetune.py's build_prompt_text() -- system + user
    # turns, plus the assistant header the model should continue from.
    system = _turn("system", SYSTEM_PROMPT)
    user = _turn("user", f"Translate the following text from {src_name} to {tgt_name}:\n{src_text}")
    return BOS_TOKEN + system + user + "<|start_header_id|>assistant<|end_header_id|>\n\n"


def _flush_predictions(preds_path, src_col_name, src_list, pred_list, ref_col_name, ref_list):
    """Append a chunk of rows to preds_path, writing the header only if the
    file doesn't exist yet. Called periodically during generation (not just
    once at the end) so progress survives a crash/timeout -- combined with
    the resume check in run_job(), an interrupted job picks back up here
    instead of starting over."""
    out = {src_col_name: src_list, "prediction": pred_list}
    if ref_list is not None:
        out[ref_col_name] = ref_list
    chunk_df = pd.DataFrame(out)
    file_exists = os.path.exists(preds_path)
    chunk_df.to_csv(preds_path, mode="a", header=not file_exists,
                     index=False, encoding="utf-8-sig")


def append_scores(job_name, n_rows, bleu, chrfpp, lock=None):
    header = ["job", "rows", "bleu", "chrf++"]
    row = [job_name, n_rows,
           f"{bleu:.4f}" if bleu is not None else "",
           f"{chrfpp:.4f}" if chrfpp is not None else ""]

    def _write():
        is_new = not SCORES_CSV.exists()
        with open(SCORES_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(header)
            w.writerow(row)

    # Guard the shared CSV with a lock when multiple processes may write to
    # it concurrently (parallel multi-GPU jobs) -- without it, interleaved
    # writes from separate processes can corrupt/garble rows.
    if lock is not None:
        with lock:
            _write()
    else:
        _write()
    print(f"[Scores] Appended '{job_name}' to {SCORES_CSV}")


def run_job(job, device, lock=None):
    csv_path        = job["csv_path"]
    checkpoint_path = job["checkpoint_path"]
    src_col         = job["src_col"]
    tgt_col         = job.get("tgt_col")
    src_name        = job["src_name"]
    tgt_name        = job["tgt_name"]
    job_name        = job.get("name", Path(checkpoint_path).name)

    print(f"\n========== {job_name} ==========")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  csv:        {csv_path}")
    print(f"  {src_col} -> {tgt_col or '?'}")

    df = pd.read_csv(csv_path)
    assert src_col in df.columns, \
        f"'{src_col}' not found in {csv_path}, got {list(df.columns)}"
    df[src_col] = df[src_col].astype(str).str.strip()
    df = df[df[src_col] != ""].reset_index(drop=True)

    has_ref = tgt_col is not None and tgt_col in df.columns
    if has_ref:
        df[tgt_col] = df[tgt_col].astype(str).str.strip()
    total_rows = len(df)
    print(f"[Data] rows to translate: {total_rows}  (reference available: {has_ref})")

    src_out_col = f"source_{src_col.lower()}"
    ref_out_col = f"reference_{tgt_col.lower()}" if has_ref else None
    preds_path = f"infer_predictions_{job_name}.csv"

    # ---- Resume check: how much of this job is already done? ----
    # Assumes csv_path/src_col haven't changed since the last run -- row
    # order is deterministic (same filter, no shuffling), so "n_done" rows
    # already in preds_path map 1:1 onto the first n_done rows of df.
    n_done = 0
    if os.path.exists(preds_path):
        n_done = len(pd.read_csv(preds_path))
        print(f"[Resume] Found existing {preds_path} with {n_done}/{total_rows} rows already done.")

    if n_done >= total_rows:
        print(f"[Resume] {job_name} already fully done -- skipping generation.")
    else:
        if n_done > 0:
            print(f"[Resume] Continuing from row {n_done}/{total_rows}.")

        print("[Model] Loading tokenizer & model from checkpoint ...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # required for correct batched causal-LM generation

        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
        model.to(device)
        model.eval()

        eot_id = tokenizer.convert_tokens_to_ids(EOT_TOKEN)
        eos_ids = [tokenizer.eos_token_id, eot_id]

        remaining = df.iloc[n_done:].reset_index(drop=True)
        src_texts = remaining[src_col].tolist()
        ref_texts = remaining[tgt_col].tolist() if has_ref else None

        buf_src, buf_pred, buf_ref = [], [], ([] if has_ref else None)

        with torch.no_grad():
            for i in range(0, len(src_texts), BATCH_SIZE):
                raw_chunk = src_texts[i:i + BATCH_SIZE]
                prompts = [build_prompt_text(src_name, tgt_name, str(t)) for t in raw_chunk]
                enc = tokenizer(
                    prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_LENGTH,
                    add_special_tokens=False,
                ).to(device)

                out = model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=NUM_BEAMS,
                    eos_token_id=eos_ids,
                    pad_token_id=tokenizer.pad_token_id,
                )

                gen_only = out[:, enc["input_ids"].shape[1]:]
                preds = [p.strip() for p in tokenizer.batch_decode(gen_only, skip_special_tokens=True)]

                buf_src.extend(raw_chunk)
                buf_pred.extend(preds)
                if has_ref:
                    buf_ref.extend(ref_texts[i:i + len(raw_chunk)])

                if (i // BATCH_SIZE) % 20 == 0:
                    print(f"  decoded {n_done + i + len(raw_chunk)}/{total_rows}")

                if len(buf_src) >= SAVE_EVERY_ROWS:
                    _flush_predictions(preds_path, src_out_col, buf_src, buf_pred, ref_out_col, buf_ref)
                    buf_src, buf_pred = [], []
                    buf_ref = [] if has_ref else None

        if buf_src:
            _flush_predictions(preds_path, src_out_col, buf_src, buf_pred, ref_out_col, buf_ref)

        print(f"[Save] Predictions saved incrementally (every {SAVE_EVERY_ROWS} rows) to {preds_path}")

    # ---- Score against the FULL predictions file (resumed + newly generated) ----
    full_preds = pd.read_csv(preds_path)
    bleu = chrfpp = None
    if has_ref and ref_out_col in full_preds.columns:
        preds_list = full_preds["prediction"].astype(str).tolist()
        refs_list  = full_preds[ref_out_col].astype(str).tolist()
        bleu   = sacrebleu.corpus_bleu(preds_list, [refs_list]).score
        chrfpp = sacrebleu.corpus_chrf(preds_list, [refs_list], word_order=2).score
        print(f"BLEU   : {bleu:.4f}")
        print(f"chrF++ : {chrfpp:.4f}")
    else:
        print("[Score] No reference column found -- predictions-only run, no BLEU/chrF++.")

    append_scores(job_name, len(full_preds), bleu, chrfpp, lock=lock)


def _worker(gpu_id, job, lock):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    run_job(job, device, lock=lock)


def main():
    config_path = Path(__file__).resolve().parent / "infer_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    n_gpus = torch.cuda.device_count()
    print(f"[Device] {n_gpus} GPU(s) visible")

    if n_gpus <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for job in jobs:
            run_job(job, device)
    else:
        # One process per job, pinned to a GPU round-robin -- all jobs run
        # concurrently instead of sequentially. "spawn" is required (not the
        # Linux default "fork") since each worker needs its own clean CUDA
        # context; forking a process that already touched CUDA is unsafe.
        ctx = mp.get_context("spawn")
        lock = ctx.Lock()
        procs = []
        for i, job in enumerate(jobs):
            gpu_id = i % n_gpus
            p = ctx.Process(target=_worker, args=(gpu_id, job, lock))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    print(f"\nAll jobs done. Scores: {SCORES_CSV}")


if __name__ == "__main__":
    main()
