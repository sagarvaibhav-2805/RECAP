"""
Merge per-model inference outputs (NLLB, mT5, Qwen, Llama) into combined
per-direction comparison CSVs.

Each model's infer.py writes infer_predictions_<lang>_<direction>.csv with a
"source_<col>" column, a "prediction" column, and (if a reference was
available) a "reference_<col>" column -- same convention across all four
model folders since their infer.py scripts share this format.

This script reads all four models' outputs for each (language, direction)
pair and merges them by row position into one combined CSV: one source
column (named after the source language) plus one prediction column per
model (named "<TargetLang>_<model>"), e.g. for gondi/en2tgt.csv:
    English, Gondi_nllb, Gondi_mt5, Gondi_qwen, Gondi_llama

Output layout:
    Maha_data/
      bhili/{en2tgt,hi2tgt,tgt2en,tgt2hi}.csv
      gondi/...
      mundari/...

Run from the GRPO_RESEARCH root (same level as NLLB-finetune/, mt5_finetune/,
qwen_finetune/, llama_finetune/):
    python build_maha_data.py
"""

from pathlib import Path
import pandas as pd

MODELS = {
    "nllb":  "NLLB-finetune",
    "mt5":   "mt5_finetune",
    "qwen":  "qwen_finetune",
    "llama": "llama_finetune",
}

LANGUAGES = ["Bhili", "Gondi", "Mundari"]
DIRECTIONS = ["en2tgt", "hi2tgt", "tgt2en", "tgt2hi"]

OUTPUT_ROOT = Path("./Maha_data")


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


def merge_one(lang, direction):
    job_name = f"{lang.lower()}_{direction}"
    src_lang, tgt_lang = _direction_langs(lang, direction)
    fname = f"infer_predictions_{job_name}.csv"

    all_sources = {}   # model -> source Series (for cross-checking alignment)
    preds = {}         # model -> prediction Series
    missing = []

    for model, base_dir in MODELS.items():
        path = Path(base_dir) / lang / fname
        if not path.exists():
            missing.append(model)
            continue
        df = pd.read_csv(path)
        src_cols = [c for c in df.columns if c.startswith("source_")]
        if not src_cols or "prediction" not in df.columns:
            print(f"[WARN] {path} missing expected columns ({list(df.columns)}), skipping")
            continue
        all_sources[model] = df[src_cols[0]].reset_index(drop=True)
        preds[model] = df["prediction"].reset_index(drop=True)

    if not preds:
        print(f"[Skip] {lang}/{direction}: no usable model outputs found")
        return
    if missing:
        print(f"[WARN] {lang}/{direction}: missing output file from {missing}")

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

    merged = pd.DataFrame({src_lang: ref_src_trunc})
    for model, series in preds.items():
        merged[f"{tgt_lang}_{model}"] = series.iloc[:min_len].reset_index(drop=True)

    out_dir = OUTPUT_ROOT / lang.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{direction}.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[Save] {out_path}  ({len(merged)} rows, models: {list(preds.keys())})")


def main():
    for lang in LANGUAGES:
        for direction in DIRECTIONS:
            merge_one(lang, direction)
    print(f"\nDone. Output under {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
