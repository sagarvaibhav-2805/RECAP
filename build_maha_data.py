"""
Merge per-model inference outputs (NLLB, mT5, Qwen, Llama) into combined
per-direction comparison CSVs, including gold-truth references and
per-sentence BLEU/chrF++/COMET scores for each model's prediction.

Restricted to hi2tgt and tgt2hi only (see DIRECTIONS below) -- en2tgt and
tgt2en are intentionally skipped.

Each model's infer.py writes infer_predictions_<lang>_<direction>.csv with a
"source_<col>" column, a "prediction" column, and (if a reference was
available) a "reference_<col>" column -- same convention across all four
model folders since their infer.py scripts share this format.

This script reads all four models' outputs for each (language, direction)
pair, merges them by row position, and computes per-sentence BLEU/chrF++
(sacrebleu) and COMET (Unbabel/wmt22-comet-da, reference-based) for each
model's prediction against the shared gold-truth reference. Output is 18
columns:
    source, gold_truth,
    <TargetLang>_nllb,  BLEU_nllb,  chrf++_nllb,  COMET_nllb,
    <TargetLang>_mt5,   BLEU_mt5,   chrf++_mt5,   COMET_mt5,
    <TargetLang>_qwen,  BLEU_qwen,  chrf++_qwen,  COMET_qwen,
    <TargetLang>_llama, BLEU_llama, chrf++_llama, COMET_llama

IMPORTANT -- unlike BLEU/chrF++, COMET is a neural metric:
  - Requires the `unbabel-comet` package (`pip install unbabel-comet`).
  - Loads its own model (COMET_MODEL_NAME) and benefits heavily from a GPU
    -- run this on a GPU allocation, not a plain CPU/login session.
  - The first time it runs, it downloads the COMET checkpoint from Hugging
    Face Hub, which needs internet access. If Pragya's compute nodes are
    offline (as we've hit before with other Hub downloads), download it
    once on a node with internet access first so it's cached locally
    before running this on a compute node.

Output layout (replaces the previous 14-column version at the same paths):
    maha_data_2/
      bhili/{hi2tgt,tgt2hi}.csv
      gondi/{hi2tgt,tgt2hi}.csv
      mundari/{hi2tgt,tgt2hi}.csv

COMET is much more expensive per row than BLEU/chrF++ (a full neural
forward pass per example, batched via the comet library's own batching),
so progress is flushed to disk every CHUNK_SIZE rows -- smaller than the
old BLEU/chrF++-only SAVE_EVERY_ROWS=5000, since losing a partially-scored
chunk on a crash is more costly now. If the job is killed partway through,
rerunning picks up from the last flushed row instead of starting over.

IMPORTANT: if a file from the OLD 14-column (no COMET) version of this
script still exists at maha_data_2/<lang>/<direction>.csv, delete it before
running this version -- the resume logic checks the existing file's header
against the new 18-column schema and will refuse to touch a file whose
header doesn't match, rather than risk appending misaligned columns.

Run from the GRPO_RESEARCH root (same level as NLLB-finetune/, mt5_finetune/,
qwen_finetune/, llama_finetune/), on a GPU allocation:
    python build_maha_data.py
"""

import math
from pathlib import Path

import pandas as pd
import sacrebleu
import torch

MODELS = {
    "nllb":  "NLLB-finetune",
    "mt5":   "mt5_finetune",
    "qwen":  "qwen_finetune",
    "llama": "llama_finetune",
}

LANGUAGES = ["Bhili", "Gondi", "Mundari"]
DIRECTIONS = ["hi2tgt", "tgt2hi"]   # restricted per instruction -- was all 4

OUTPUT_ROOT = Path("./maha_data_2")
CHUNK_SIZE = 1000   # rows per COMET batch + disk flush (see module docstring)

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
    """sources/hyps/refs: same-length lists of strings, "" in refs where no
    reference is available. Returns a list of scores (float, NaN where ref
    was missing) aligned 1:1 with the input order."""
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


def _direction_langs(lang, direction):
    """Return (source_lang_name, target_lang_name) for a direction."""
    if direction == "en2tgt":
        return "English", lang
    if direction == "hi2tgt":
        return "Hindi", lang
    if direction == "tgt2en":
        return lang, "English"
    if direction == "tgt2hi":
        return lang, "Hindi"
    raise ValueError(direction)


def _load_model_outputs(lang, direction):
    """Load source/reference/prediction columns from all 4 models for one
    (language, direction) pair, truncated to the common min row count.
    Returns (source_series, gold_series_or_None, {model: pred_series}), or
    None if no usable model outputs exist at all."""
    job_name = f"{lang.lower()}_{direction}"
    fname = f"infer_predictions_{job_name}.csv"

    all_sources = {}
    all_refs = {}
    preds = {}
    missing = []

    for model, base_dir in MODELS.items():
        path = Path(base_dir) / lang / fname
        if not path.exists():
            missing.append(model)
            continue
        df = pd.read_csv(path)
        src_cols = [c for c in df.columns if c.startswith("source_")]
        ref_cols = [c for c in df.columns if c.startswith("reference_")]
        if not src_cols or "prediction" not in df.columns:
            print(f"[WARN] {path} missing expected columns ({list(df.columns)}), skipping")
            continue
        all_sources[model] = df[src_cols[0]].reset_index(drop=True)
        if ref_cols:
            all_refs[model] = df[ref_cols[0]].reset_index(drop=True)
        preds[model] = df["prediction"].reset_index(drop=True)

    if not preds:
        print(f"[Skip] {lang}/{direction}: no usable model outputs found")
        return None
    if missing:
        print(f"[WARN] {lang}/{direction}: missing output file from {missing}")
    if not all_refs:
        print(f"[WARN] {lang}/{direction}: no model has a reference column -- gold_truth will be blank")

    lengths = {m: len(s) for m, s in all_sources.items()}
    min_len = min(lengths.values())
    if len(set(lengths.values())) > 1:
        print(f"[WARN] {lang}/{direction}: row count mismatch {lengths} -- truncating all to {min_len} rows")

    # Sanity check: the source column should be identical text across all
    # models (same source CSV was used for every model's infer_config.json).
    # A mismatch here means the merge-by-position below is misaligned.
    ref_model, ref_src = next(iter(all_sources.items()))
    ref_src_trunc = ref_src.iloc[:min_len].reset_index(drop=True)
    for model, src_series in all_sources.items():
        if model == ref_model:
            continue
        src_trunc = src_series.iloc[:min_len].reset_index(drop=True)
        n_diff = (src_trunc.astype(str) != ref_src_trunc.astype(str)).sum()
        if n_diff:
            print(f"[WARN] {lang}/{direction}: source text mismatch between "
                  f"'{ref_model}' and '{model}' on {n_diff}/{min_len} rows -- "
                  f"these CSVs may not be row-aligned, double check before trusting the merge.")

    gold = None
    if all_refs:
        gold_model, gold_series = next(iter(all_refs.items()))
        gold = gold_series.iloc[:min_len].reset_index(drop=True)
        for model, ref_series in all_refs.items():
            if model == gold_model:
                continue
            ref_trunc = ref_series.iloc[:min_len].reset_index(drop=True)
            n_diff = (ref_trunc.astype(str) != gold.astype(str)).sum()
            if n_diff:
                print(f"[WARN] {lang}/{direction}: reference text mismatch between "
                      f"'{gold_model}' and '{model}' on {n_diff}/{min_len} rows.")

    preds_trunc = {m: s.iloc[:min_len].reset_index(drop=True) for m, s in preds.items()}
    return ref_src_trunc, gold, preds_trunc


def _flush(rows, columns, out_path):
    if not rows:
        return
    chunk_df = pd.DataFrame(rows, columns=columns)
    file_exists = out_path.exists()
    chunk_df.to_csv(out_path, mode="a", header=not file_exists, index=False, encoding="utf-8-sig")


def merge_one(lang, direction):
    _, tgt_lang = _direction_langs(lang, direction)
    loaded = _load_model_outputs(lang, direction)
    if loaded is None:
        return
    source, gold, preds = loaded
    total_rows = len(source)
    models_present = list(preds.keys())

    columns = ["source", "gold_truth"]
    for model in models_present:
        columns += [f"{tgt_lang}_{model}", f"BLEU_{model}", f"chrf++_{model}", f"COMET_{model}"]

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
                  f"         Refusing to touch it (likely leftover from the old 14-column, "
                  f"no-COMET version) -- delete this file manually and rerun.")
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
        chunk_source = source.iloc[chunk_start:chunk_end].astype(str).tolist()
        if gold is not None:
            chunk_gold = gold.iloc[chunk_start:chunk_end].astype(str).tolist()
        else:
            chunk_gold = [""] * (chunk_end - chunk_start)

        chunk_preds = {
            model: preds[model].iloc[chunk_start:chunk_end].astype(str).tolist()
            for model in models_present
        }
        chunk_comet = {
            model: compute_comet_scores(comet_model, chunk_source, chunk_preds[model], chunk_gold)
            for model in models_present
        }

        buf_rows = []
        for j in range(chunk_end - chunk_start):
            gold_text = chunk_gold[j]
            row = [chunk_source[j], gold_text]
            for model in models_present:
                pred_text = chunk_preds[model][j]
                if gold_text:
                    bleu = sacrebleu.sentence_bleu(pred_text, [gold_text]).score
                    chrfpp = sacrebleu.sentence_chrf(pred_text, [gold_text], word_order=2).score
                else:
                    bleu = float("nan")
                    chrfpp = float("nan")
                comet_score = chunk_comet[model][j]
                comet_str = "" if math.isnan(comet_score) else f"{comet_score:.4f}"
                row += [pred_text, f"{bleu:.4f}", f"{chrfpp:.4f}", comet_str]
            buf_rows.append(row)

        _flush(buf_rows, columns, out_path)
        print(f"  [Save] {lang}/{direction}: {chunk_end}/{total_rows} rows done")

    print(f"[Save] {out_path}  ({total_rows} rows, models: {models_present})")


def main():
    for lang in LANGUAGES:
        for direction in DIRECTIONS:
            merge_one(lang, direction)
    print(f"\nDone. Output under {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
