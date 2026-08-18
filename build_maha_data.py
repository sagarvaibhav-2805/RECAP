"""
Add a COMET score for each model's prediction to the existing Maha_data
comparison CSVs -- no re-running inference, no re-merging from the raw
per-model infer_predictions_*.csv files. Just reads the already-built
Maha_data/<lang>/<direction>.csv (source, gold_truth, and each model's
prediction/BLEU/chrF++ already aligned in one row), computes COMET(source,
prediction, gold_truth) for each model's prediction column, and writes the
result with one new COMET_<model> column per model to maha_data_2/.

Restricted to hi2tgt and tgt2hi only (see DIRECTIONS below).

Input:  Maha_data/<lang>/<direction>.csv       (14 columns, no COMET)
Output: maha_data_2/<lang>/<direction>.csv     (18 columns, adds COMET_<model>)

IMPORTANT -- COMET is a neural metric, unlike BLEU/chrF++:
  - Requires the `unbabel-comet` package (`pip install unbabel-comet`).
  - Loads its own model (COMET_MODEL_NAME) and benefits heavily from a GPU
    -- run this on a GPU allocation, not a plain CPU/login session.
  - First run downloads the COMET checkpoint from Hugging Face Hub (needs
    internet access) -- if compute nodes are offline, fetch it once from a
    node with internet access first so it's cached locally.

Progress is flushed to disk every CHUNK_SIZE rows, and resumes by counting
rows already in the output file -- if the job is killed partway through,
rerunning picks up from the last flushed row instead of starting over.

There are 3 languages x 2 directions = 6 total (lang, direction) jobs. If
more than one GPU is visible, jobs run in parallel -- one job per GPU,
round-robin if there are more jobs than GPUs (e.g. on 4 GPUs, 2 jobs run
twice in sequence while the other 2 GPUs each only run once). Each worker
process loads its own COMET model instance pinned to its assigned GPU and
writes to its own separate output file, so no cross-process coordination is
needed. With a single GPU (or CPU), jobs run sequentially, same as before.

Run from the GRPO_RESEARCH root (same level as the Maha_data/ folder), on a
GPU allocation:
    python build_maha_data.py
"""

import math
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd
import torch

INPUT_ROOT = Path("./Maha_data")
OUTPUT_ROOT = Path("./maha_data_2")

LANGUAGES = ["Bhili", "Gondi", "Mundari"]
DIRECTIONS = ["hi2tgt", "tgt2hi"]
MODEL_NAMES = ["nllb", "mt5", "qwen", "llama"]

CHUNK_SIZE = 2000   # rows per COMET batch + disk flush

COMET_MODEL_NAME = "Unbabel/wmt22-comet-da"
COMET_BATCH_SIZE = 64
COMET_GPUS = 1 if torch.cuda.is_available() else 0

_comet_model = None


def get_comet_model():
    global _comet_model
    if _comet_model is None:
        from comet import download_model, load_from_checkpoint
        print(f"[COMET] Downloading/loading {COMET_MODEL_NAME} ...")
        model_path = download_model(COMET_MODEL_NAME)
        _comet_model = load_from_checkpoint(model_path)
        print(f"[COMET] Model loaded. gpus={COMET_GPUS}")
    return _comet_model


def compute_comet_scores(comet_model, sources, hyps, refs):
    """Same-length lists of strings, "" in refs where no reference is
    available. Returns a list of scores (float, NaN where ref was missing)
    aligned 1:1 with the input order."""
    idx_map = []
    data = []
    for i, (s, h, r) in enumerate(zip(sources, hyps, refs)):
        if r:
            idx_map.append(i)
            data.append({"src": s, "mt": h, "ref": r})
    scores = [float("nan")] * len(sources)
    if data:
        output = comet_model.predict(data, batch_size=COMET_BATCH_SIZE, gpus=COMET_GPUS)
        for idx, score in zip(idx_map, output.scores):
            scores[idx] = score
    return scores


def _find_pred_col(columns, model):
    """The prediction column for a model is whatever ends in "_<model>" and
    isn't the BLEU_<model>/chrf++_<model> score column, e.g. "Bhili_nllb"."""
    candidates = [c for c in columns if c.endswith(f"_{model}")
                  and c not in (f"BLEU_{model}", f"chrf++_{model}")]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one prediction column for model "
                          f"'{model}', found {candidates} in {columns}")
    return candidates[0]


def process_one(lang, direction):
    in_path = INPUT_ROOT / lang.lower() / f"{direction}.csv"
    if not in_path.exists():
        print(f"[Skip] {lang}/{direction}: {in_path} not found")
        return

    src_df = pd.read_csv(in_path)
    total_rows = len(src_df)
    pred_cols = {model: _find_pred_col(list(src_df.columns), model) for model in MODEL_NAMES}

    columns = list(src_df.columns) + [f"COMET_{model}" for model in MODEL_NAMES]

    out_dir = OUTPUT_ROOT / lang.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{direction}.csv"

    # ---- Resume check ----
    n_done = 0
    if out_path.exists():
        existing_cols = list(pd.read_csv(out_path, nrows=0).columns)
        if existing_cols != columns:
            print(f"[ERROR] {out_path} exists with a different schema than expected.\n"
                  f"         existing: {existing_cols}\n"
                  f"         expected: {columns}\n"
                  f"         Refusing to touch it -- delete this file manually and rerun.")
            return
        n_done = len(pd.read_csv(out_path))
        print(f"[Resume] Found existing {out_path} with {n_done}/{total_rows} rows already done.")

    if n_done >= total_rows:
        print(f"[Resume] {lang}/{direction} already fully done -- skipping.")
        return
    if n_done > 0:
        print(f"[Resume] Continuing from row {n_done}/{total_rows}.")

    comet_model = get_comet_model()

    for chunk_start in range(n_done, total_rows, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_rows)
        chunk = src_df.iloc[chunk_start:chunk_end].reset_index(drop=True)

        sources = chunk["source"].astype(str).tolist()
        golds = chunk["gold_truth"].astype(str).tolist()

        comet_cols = {}
        for model in MODEL_NAMES:
            hyps = chunk[pred_cols[model]].astype(str).tolist()
            scores = compute_comet_scores(comet_model, sources, hyps, golds)
            comet_cols[f"COMET_{model}"] = [
                "" if math.isnan(s) else f"{s:.4f}" for s in scores
            ]

        out_chunk = chunk.copy()
        for col, values in comet_cols.items():
            out_chunk[col] = values

        file_exists = out_path.exists()
        out_chunk.to_csv(out_path, mode="a", header=not file_exists, index=False, encoding="utf-8-sig")
        print(f"  [Save] {lang}/{direction}: {chunk_end}/{total_rows} rows done")

    print(f"[Save] {out_path}  ({total_rows} rows)")


def _worker(gpu_id, lang, direction):
    # Restrict this process's CUDA visibility to just its assigned GPU,
    # BEFORE any CUDA call happens. torch.cuda.set_device() alone is not
    # enough here: COMET's .predict() goes through its own PyTorch Lightning
    # Trainer, which does its own GPU auto-selection internally and ignores
    # whatever "current device" was set beforehand -- it just grabs GPU
    # index 0 of whatever's visible. That caused multiple workers to pile
    # onto the same physical GPU and OOM. Setting CUDA_VISIBLE_DEVICES makes
    # that GPU the only one this process can see, so there's no way for
    # Lightning to pick the wrong one.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU {gpu_id}] Starting {lang}/{direction}")
    process_one(lang, direction)


def main():
    jobs = [(lang, direction) for lang in LANGUAGES for direction in DIRECTIONS]
    n_gpus = torch.cuda.device_count()
    print(f"[Device] {n_gpus} GPU(s) visible, {len(jobs)} jobs total")

    if n_gpus <= 1:
        for lang, direction in jobs:
            process_one(lang, direction)
    else:
        # One process per job, pinned to a GPU round-robin -- jobs run
        # concurrently instead of sequentially. "spawn" is required (not the
        # Linux default "fork") since each worker needs its own clean CUDA
        # context; forking a process that already touched CUDA is unsafe.
        ctx = mp.get_context("spawn")
        procs = []
        for i, (lang, direction) in enumerate(jobs):
            gpu_id = i % n_gpus
            p = ctx.Process(target=_worker, args=(gpu_id, lang, direction))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    print(f"\nDone. Output under {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
