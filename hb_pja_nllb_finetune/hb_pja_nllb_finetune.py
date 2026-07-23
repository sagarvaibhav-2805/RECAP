"""
Fine-tune facebook/nllb-200-distilled-600M for English/Hindi <-> {tribal language}
Supports 7 languages: bhili, gondi, kokborok, santhali, kui, mundari, garo

Language + direction are read from config.json (same directory as this script):
    {"language": "gondi", "direction": "en2tgt"}

Run 4 directions per language (28 total runs). Same train/val/test splits per
language across all 4 of its directions (built once, cached on disk).

Example config.json:
    {"language": "gondi", "direction": "en2tgt"}
    {"language": "santhali", "direction": "tgt2hi"}
"""

import os
import json
import csv
from pathlib import Path

# --- pin GPU and silence wandb / tokenizer warnings BEFORE importing torch/transformers ---
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
import pandas as pd
import torch
import torch

print("=" * 60)
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("=" * 60)
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import sacrebleu

# =============================================================
# CONFIG
# =============================================================
MODEL_NAME     = "/home/scai/msr/aiy257590/flash/GRPO_RESEARCH/Models/nllb-200-distilled-600M"
MAX_LENGTH     = 128
SEED           = 42

ENG_CODE = "eng_Latn"
HIN_CODE = "hin_Deva"

# Per-language config:
#   - csv_file:    path to the 20k parallel CSV with columns English, Hindi, <column>
#   - column:      the column name in that CSV for the tribal language
#   - lang_code:   the NLLB-style language token we register for it (script-specific)
#
# *** UPDATE THE csv_file PATHS BELOW IF YOURS ARE DIFFERENT ***
LANGUAGES = {
    "bhili":    {"csv_file": "/home/scai/msr/aiy257590/flash/180k_script_mill_gyi/dataset/Bhi_Hin_Eng.csv",    "column": "Bhili",    "lang_code": "bhb_Deva"},
    #"gondi":    {"csv_file": "/home/siddhant/Nishant/EMNLP/data2/eng_hin_gondi_20k_finetuning.csv",    "column": "Gondi",    "lang_code": "gon_Deva"},
    #"kokborok": {"csv_file": "/home/siddhant/Nishant/EMNLP/data2/Eng_Hin_Kok_20k.csv",                 "column": "Kokborok", "lang_code": "kok_Latn"},
    #"santhali": {"csv_file": "/home/siddhant/Nishant/EMNLP/data2/Eng_Hin_Santali_20k_finetuning.csv", "column": "Santhali", "lang_code": "sat_Olck"},
    #"kui":      {"csv_file": "/home/siddhant/Nishant/EMNLP/data2/eng_hin_kui_20k_finetuning.csv",      "column": "Kui",      "lang_code": "kxu_Orya"},
    #"mundari":  {"csv_file": "/home/siddhant/Nishant/EMNLP/data2/eng_hin_Mundari_20k_finetuning.csv",  "column": "Mundari",  "lang_code": "unr_Deva"},
    # "garo":     {"csv_file": "/home/siddhant/Nishant/EMNLP/data2/eng_hin_garo_20k_finetuning.csv",     "column": "Garo",     "lang_code": "grt_Latn"},
    # "santhali": {"csv_file": "/home/siddhant/Nishant/EMNLP/data2/Eng_Hin_Santali_train_20k.csv",       "column": "Santali",  "lang_code": "sat_Olck"},
    # "mundari":  {"csv_file": "/home/siddhant/Nishant/EMNLP/data2/Eng_Hin_Mundari_train_20k.csv",       "column": "Mundari",  "lang_code": "unr_Deva"},
    # "kokborok_latn": {"csv_file": "/home/siddhant/Nishant/EMNLP/data2/eng_hin_kok_20k_processed.csv", "column": "Kokborok_Latn", "lang_code": "kok_Latn"},
    # "kokborok_deva": {"csv_file": "/home/siddhant/Nishant/EMNLP/data2/eng_hin_kok_20k_processed.csv", "column": "Kokborok_Deva", "lang_code": "kok_Deva"},
    # "kui_latn": {"csv_file": "/home/siddhant/Nishant/EMNLP/data/Eng_Hin_Kui_train_20k_processed.csv", "column": "Kui (Latin)",      "lang_code": "kxu_Latn"},
    # "kui_deva": {"csv_file": "/home/siddhant/Nishant/EMNLP/data/Eng_Hin_Kui_train_20k_processed.csv", "column": "Kui (Devanagari)", "lang_code": "kxu_Deva"},
}

# Master scores file -- one shared CSV across ALL languages/directions
SCORES_CSV = Path("./all_languages_scores.csv")

torch.manual_seed(SEED)
np.random.seed(SEED)


# =============================================================
# 1. LOAD + CLEAN DATA (train/val/test paths given explicitly)
# =============================================================
def _load_and_clean(csv_file: str, tgt_column: str) -> pd.DataFrame:
    print(f"[Data] Loading {csv_file} ...")
    df_raw = pd.read_csv(csv_file)

    needed = {"English", "Hindi", tgt_column}
    assert needed.issubset(df_raw.columns), \
        f"Expected columns {needed} in {csv_file}, got {list(df_raw.columns)}"

    df = df_raw[["English", "Hindi", tgt_column]].dropna()
    for col in ["English", "Hindi", tgt_column]:
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["English"] != "") & (df["Hindi"] != "") & (df[tgt_column] != "")].reset_index(drop=True)
    print(f"[Data] Usable parallel rows (all 3 langs present): {len(df)}")
    return df


def load_splits(train_csv: str, val_csv: str, test_csv: str, tgt_column: str):
    train_df = _load_and_clean(train_csv, tgt_column)
    val_df   = _load_and_clean(val_csv,   tgt_column)
    test_df  = _load_and_clean(test_csv,  tgt_column)
    print(f"[Data] Loaded splits | "
          f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    return train_df, val_df, test_df


# =============================================================
# 2. TOKENIZER + MODEL  (register new lang token for tribal lang)
# =============================================================
def load_tokenizer_and_model(src_lang_code: str, tribal_code: str):
    print("[Model] Loading tokenizer & model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=src_lang_code)

    existing_specials = list(tokenizer.additional_special_tokens)
    if tribal_code not in existing_specials:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": existing_specials + [tribal_code]}
        )

    new_lang_id = tokenizer.convert_tokens_to_ids(tribal_code)
    if hasattr(tokenizer, "lang_code_to_id"):
        tokenizer.lang_code_to_id[tribal_code] = new_lang_id
    if hasattr(tokenizer, "id_to_lang_code"):
        tokenizer.id_to_lang_code[new_lang_id] = tribal_code
    if hasattr(tokenizer, "fairseq_tokens_to_ids"):
        tokenizer.fairseq_tokens_to_ids[tribal_code] = new_lang_id
    if hasattr(tokenizer, "fairseq_ids_to_tokens"):
        tokenizer.fairseq_ids_to_tokens[new_lang_id] = tribal_code

    print(f"[Tokenizer] {tribal_code} token id = {new_lang_id}, vocab size = {len(tokenizer)}")

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    print("Inside load_tokenizer_and_model()")
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Model] Using device: {device}  "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    model.to(device)
    return tokenizer, model, device


# =============================================================
# 3. PREPROCESSING
# =============================================================
def make_preprocess_fn(tokenizer, src_col, tgt_col, src_lang_code, tgt_lang_code):
    def preprocess_function(examples):
        inputs  = examples[src_col]
        targets = examples[tgt_col]
        tokenizer.src_lang = src_lang_code
        tokenizer.tgt_lang = tgt_lang_code

        model_inputs = tokenizer(
            inputs,
            text_target=targets,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )
        return model_inputs
    return preprocess_function


# =============================================================
# 4. METRICS  (BLEU + chrF + chrF++)
# =============================================================
def make_compute_metrics(tokenizer):
    def _postprocess(preds, labels):
        return [p.strip() for p in preds], [l.strip() for l in labels]

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        labels      = np.where(labels      != -100, labels,      tokenizer.pad_token_id)
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)

        decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels,      skip_special_tokens=True)
        decoded_preds, decoded_labels = _postprocess(decoded_preds, decoded_labels)

        bleu   = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
        chrf   = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels]).score
        chrfpp = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2).score
        # "chrf" key (no ++) is used for metric_for_best_model per your training args
        return {"bleu": bleu, "chrf": chrf, "chrf++": chrfpp}
    return compute_metrics


# =============================================================
# 5. APPEND TO MASTER SCORES CSV
# =============================================================
def append_master_scores(language, direction, val_bleu, val_chrf, test_bleu, test_chrf,
                         manual_bleu, manual_chrf):
    header = ["language", "direction",
              "val_bleu", "val_chrf++", "test_bleu", "test_chrf++",
              "test_bleu_beam5", "test_chrf++_beam5"]
    row = [language, direction,
           f"{val_bleu:.4f}",   f"{val_chrf:.4f}",
           f"{test_bleu:.4f}",  f"{test_chrf:.4f}",
           f"{manual_bleu:.4f}", f"{manual_chrf:.4f}"]

    is_new = not SCORES_CSV.exists()
    with open(SCORES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(header)
        w.writerow(row)
    print(f"[Scores] Appended row '{language}/{direction}' to {SCORES_CSV}")


# =============================================================
# 6. MAIN
# =============================================================
def run_direction(language, direction, train_csv, val_csv, test_csv):
    assert language in LANGUAGES, \
        f"Unknown language '{language}' in config.json, choices: {list(LANGUAGES.keys())}"
    assert direction in ["en2tgt", "hi2tgt", "tgt2en", "tgt2hi"], \
        f"Unknown direction '{direction}' in config.json, choices: en2tgt, hi2tgt, tgt2en, tgt2hi"

    cfg       = LANGUAGES[language]
    tgt_column  = cfg["column"]
    tribal_code = cfg["lang_code"]

    # Resolve direction -> (src_col, tgt_col, src_code, tgt_code)
    if direction == "en2tgt":
        src_col, tgt_col, src_code, tgt_code = "English", tgt_column, ENG_CODE, tribal_code
    elif direction == "hi2tgt":
        src_col, tgt_col, src_code, tgt_code = "Hindi", tgt_column, HIN_CODE, tribal_code
    elif direction == "tgt2en":
        src_col, tgt_col, src_code, tgt_code = tgt_column, "English", tribal_code, ENG_CODE
    elif direction == "tgt2hi":
        src_col, tgt_col, src_code, tgt_code = tgt_column, "Hindi", tribal_code, HIN_CODE

    output_dir = f"./nllb-{language}-{direction}-finetuned"
    print(f"\n========== Language: {language} | Direction: {direction} ==========")
    print(f"  {src_col} [{src_code}]  ->  {tgt_col} [{tgt_code}]")
    print(f"[Output] {output_dir}")

    # ---- Data ----
    train_df, val_df, test_df = load_splits(train_csv, val_csv, test_csv, tgt_column)
    train_set = Dataset.from_pandas(train_df, preserve_index=False)
    val_set   = Dataset.from_pandas(val_df,   preserve_index=False)
    test_set  = Dataset.from_pandas(test_df,  preserve_index=False)
    dataset_dict = DatasetDict({"train": train_set, "validation": val_set, "test": test_set})
    print(f"[Data] train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    # ---- Tokenizer + model ----
    tokenizer, model, device = load_tokenizer_and_model(src_code, tribal_code)

    # ---- Tokenize ----
    pre_fn = make_preprocess_fn(tokenizer, src_col, tgt_col, src_code, tgt_code)
    candidate_cols = ["English", "Hindi", tgt_column, "Unique_ID"]
    cols_to_remove = [c for c in candidate_cols if c in train_set.column_names]

    print("[Tokenize] Mapping splits ...")
    tokenized_train = dataset_dict["train"].map(
        pre_fn, batched=True, remove_columns=cols_to_remove, desc="train")
    tokenized_val = dataset_dict["validation"].map(
        pre_fn, batched=True, remove_columns=cols_to_remove, desc="val")
    tokenized_test = dataset_dict["test"].map(
        pre_fn, batched=True, remove_columns=cols_to_remove, desc="test")

    # ---- Trainer ----
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_code)
    model.config.forced_bos_token_id = forced_bos_id

    # Use bf16 if GPU supports it; otherwise no mixed precision
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"[Train] bf16={use_bf16}")

    # === CUSTOM TRAINING ARGS (kept as you specified) ===
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        lr_scheduler_type="inverse_sqrt",
        warmup_steps=150,
        eval_strategy="steps",
        eval_steps=900,
        save_strategy="steps",
        save_steps=900,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        generation_max_length=128,
        bf16=use_bf16,
        fp16=False,
        max_grad_norm=1.0,
        logging_dir=f"{output_dir}/logs",
        logging_steps=1000,
        dataloader_num_workers=2,
        report_to="none",
        predict_with_generate=True,
        eval_accumulation_steps=4,
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    print(f"[Train] epochs=5  steps/epoch~{len(tokenized_train)//(16*2)}")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[Save] Best model written to {output_dir}")

    # ---- Final eval ----
    print("\n=== Validation set (Trainer.evaluate, beam search) ===")
    val_results = trainer.evaluate(metric_key_prefix="val")
    val_bleu = val_results.get("val_bleu",   float("nan"))
    val_chrf = val_results.get("val_chrf++", float("nan"))
    print(f"VAL  BLEU   = {val_bleu:.4f}")
    print(f"VAL  chrF++ = {val_chrf:.4f}")

    print("\n=== Test set (Trainer.evaluate, beam search) ===")
    test_results = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")
    test_bleu = test_results.get("test_bleu",   float("nan"))
    test_chrf = test_results.get("test_chrf++", float("nan"))
    print(f"TEST BLEU   = {test_bleu:.4f}")
    print(f"TEST chrF++ = {test_chrf:.4f}")

    # ---- Manual sentence-level inference (beam=5) for inspection ----
    print("\n=== Manual inference on test set (beam=5) ===")
    model.eval()
    test_records = list(dataset_dict["test"])
    src_texts = [r[src_col] for r in test_records]
    ref_texts = [r[tgt_col] for r in test_records]

    all_preds = []
    batch = 16
    tokenizer.src_lang = src_code
    with torch.no_grad():
        for i in range(0, len(src_texts), batch):
            chunk = src_texts[i:i + batch]
            enc = tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
            ).to(device)
            out = model.generate(
                **enc,
                forced_bos_token_id=forced_bos_id,
                max_length=MAX_LENGTH,
                num_beams=5,
            )
            all_preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
            if (i // batch) % 20 == 0:
                print(f"  decoded {i + len(chunk)}/{len(src_texts)}")

    preds_path = f"test_predictions_{language}_{direction}.csv"
    pd.DataFrame({
        f"source_{src_col.lower()}":     src_texts,
        f"reference_{tgt_col.lower()}":  ref_texts,
        f"prediction_{tgt_col.lower()}": all_preds,
    }).to_csv(preds_path, index=False, encoding="utf-8-sig")
    print(f"[Save] Predictions: {preds_path}")

    final_bleu   = sacrebleu.corpus_bleu(all_preds, [ref_texts]).score
    final_chrfpp = sacrebleu.corpus_chrf(all_preds, [ref_texts], word_order=2).score

    print(f"\n=== FINAL TEST SCORES [{language}/{direction}] (manual beam=5) ===")
    print(f"BLEU   : {final_bleu:.4f}")
    print(f"chrF++ : {final_chrfpp:.4f}")

    with open(f"final_scores_{language}_{direction}.txt", "w", encoding="utf-8") as f:
        f.write(f"Language: {language}  Direction: {direction}  ({src_col} -> {tgt_col})\n")
        f.write(f"VAL  BLEU   : {val_bleu:.4f}\n")
        f.write(f"VAL  chrF++ : {val_chrf:.4f}\n")
        f.write(f"TEST BLEU   : {test_bleu:.4f}\n")
        f.write(f"TEST chrF++ : {test_chrf:.4f}\n")
        f.write(f"TEST BLEU (beam=5 manual)   : {final_bleu:.4f}\n")
        f.write(f"TEST chrF++ (beam=5 manual) : {final_chrfpp:.4f}\n")

    append_master_scores(language, direction, val_bleu, val_chrf, test_bleu, test_chrf,
                         final_bleu, final_chrfpp)
    print(f"\nDone with {language}/{direction}. Master scores: {SCORES_CSV}")


def main():
    config_path = Path(__file__).resolve().parent / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        run_cfg = json.load(f)

    language   = run_cfg["language"]
    directions = run_cfg["direction"]
    train_csv  = run_cfg["train_csv"]
    val_csv    = run_cfg["val_csv"]
    test_csv   = run_cfg["test_csv"]

    # "direction" can be a single string or a list of the four directions
    if isinstance(directions, str):
        directions = [directions]

    for direction in directions:
        run_direction(language, direction, train_csv, val_csv, test_csv)


if __name__ == "__main__":
    main()
