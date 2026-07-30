"""
Full fine-tune Qwen2.5-0.5B (base) for English/Hindi <-> {tribal language}
translation via supervised instruction fine-tuning (SFT) on Qwen's ChatML
template.

Qwen2.5 is a decoder-only causal LM, not an encoder-decoder seq2seq model like
NLLB/mT5 -- there is no forced_bos_token_id or task-prefix trick here. Each
training example is instead formatted as a chat turn:

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    Translate the following text from English to Gondi:
    <source text><|im_end|>
    <|im_start|>assistant
    <target text><|im_end|>

and the loss is computed ONLY on the assistant's response tokens -- the
system/user (prompt) tokens are masked out with label id -100. This is the
standard SFT recipe for instruction-following LLMs (the same masking TRL's
SFTTrainer/DataCollatorForCompletionOnlyLM does internally).

Checkpoint selection is by eval_loss (perplexity), not a generation metric --
plain Trainer (unlike Seq2SeqTrainer) doesn't generate during periodic eval,
and running beam search every eval step during LLM SFT is not standard
practice. Real BLEU/chrF++ are computed once at the end via actual
generation on val + test, same as the manual beam-5 pass in the mT5/NLLB
scripts.

Language, direction(s), data paths, and epochs are read from config.json
(same directory as this script):
    {
      "language": "gondi",
      "direction": ["en2tgt", "hi2tgt", "tgt2en", "tgt2hi"],
      "epochs": 3,
      "train_csv": "...", "val_csv": "...", "test_csv": "..."
    }
"""

import os
import json
import csv
from pathlib import Path

# --- silence wandb / tokenizer warnings BEFORE importing torch/transformers ---
os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
import pandas as pd
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
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
import sacrebleu

# =============================================================
# CONFIG
# =============================================================
MODEL_NAME    = "/home/scai/msr/aiy257590/flash/GRPO_RESEARCH/Models/Qwen2.5-0.5B"
MAX_LENGTH    = 256   # full prompt+response token budget (chat template adds overhead)
MAX_NEW_TOKENS = 128  # generation cap for the response only
SEED          = 42
SYSTEM_PROMPT = "You are a helpful assistant."

ENG_NAME = "English"
HIN_NAME = "Hindi"

# Per-language config:
#   - column: the column name in the CSV for the tribal language (also used
#             as its natural-language name in the translation instruction)
LANGUAGES = {
    "bhili":   {"column": "Bhili"},
    "gondi":   {"column": "Gondi"},
    "mundari": {"column": "Mundari"},
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
# 2. TOKENIZER + MODEL
# =============================================================
def load_tokenizer_and_model():
    print("[Model] Loading tokenizer & model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

    print("Inside load_tokenizer_and_model()")
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Model] Using device: {device}  "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    model.to(device)
    return tokenizer, model, device


# =============================================================
# 3. PROMPT FORMATTING + PREPROCESSING (response-only loss masking)
# =============================================================
def build_messages(src_name, tgt_name, src_text, tgt_text=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Translate the following text from {src_name} to {tgt_name}:\n{src_text}"},
    ]
    if tgt_text is not None:
        messages.append({"role": "assistant", "content": str(tgt_text)})
    return messages


def make_preprocess_fn(tokenizer, src_col, tgt_col, src_name, tgt_name):
    def preprocess_function(examples):
        all_input_ids, all_labels, all_attention_mask = [], [], []

        for src_text, tgt_text in zip(examples[src_col], examples[tgt_col]):
            prompt_messages = build_messages(src_name, tgt_name, src_text)
            full_messages    = build_messages(src_name, tgt_name, src_text, tgt_text)

            # add_generation_prompt=True renders exactly the text the
            # assistant's turn continues from, so prompt_text is guaranteed
            # to be a string (and therefore token) prefix of full_text.
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False)

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids   = tokenizer(full_text,   add_special_tokens=False)["input_ids"]

            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

            # Truncate from the left (drop oldest tokens first) if the whole
            # example overflows -- keeps the response intact.
            if len(full_ids) > MAX_LENGTH:
                overflow = len(full_ids) - MAX_LENGTH
                full_ids = full_ids[overflow:]
                labels   = labels[overflow:]

            all_input_ids.append(full_ids)
            all_labels.append(labels)
            all_attention_mask.append([1] * len(full_ids))

        return {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_mask,
        }
    return preprocess_function


class CausalLMPaddingCollator:
    """Pads pre-tokenized input_ids/labels/attention_mask to the batch max
    length. Labels are already response-only-masked (-100 on prompt tokens)
    by make_preprocess_fn above -- this collator only pads, it does not
    recompute labels the way DataCollatorForLanguageModeling would."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# =============================================================
# 4. GENERATION-BASED SCORING  (BLEU + chrF++ via real model.generate())
# =============================================================
def generate_and_score(records, src_col, tgt_col, src_name, tgt_name,
                        model, tokenizer, device, num_beams=4, batch_size=32):
    model.eval()
    tokenizer.padding_side = "left"  # required for correct batched causal-LM generation

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tokenizer.eos_token_id, im_end_id]

    src_texts = [r[src_col] for r in records]
    ref_texts = [str(r[tgt_col]) for r in records]

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(src_texts), batch_size):
            chunk = src_texts[i:i + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    build_messages(src_name, tgt_name, t),
                    tokenize=False, add_generation_prompt=True)
                for t in chunk
            ]
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_LENGTH,
                add_special_tokens=False,
            ).to(device)

            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=num_beams,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
            )

            gen_only = out[:, enc["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
            all_preds.extend(p.strip() for p in decoded)

            if (i // batch_size) % 20 == 0:
                print(f"  decoded {i + len(chunk)}/{len(src_texts)}")

    tokenizer.padding_side = "right"  # restore training-time default

    bleu   = sacrebleu.corpus_bleu(all_preds, [ref_texts]).score
    chrfpp = sacrebleu.corpus_chrf(all_preds, [ref_texts], word_order=2).score
    return all_preds, ref_texts, bleu, chrfpp


# =============================================================
# 5. APPEND TO MASTER SCORES CSV
# =============================================================
def append_master_scores(language, direction, val_loss, val_bleu, val_chrf,
                          test_bleu, test_chrf):
    header = ["language", "direction", "val_loss",
              "val_bleu_beam5", "val_chrf++_beam5",
              "test_bleu_beam5", "test_chrf++_beam5"]
    row = [language, direction, f"{val_loss:.4f}",
           f"{val_bleu:.4f}", f"{val_chrf:.4f}",
           f"{test_bleu:.4f}", f"{test_chrf:.4f}"]

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
def _has_final_model(output_dir):
    """A direction counts as fully trained if trainer.save_model()'s output
    sits directly in output_dir (config.json + a weights file at the top
    level -- NOT just inside a checkpoint-*/ subfolder, which may be an
    incomplete/stale intermediate save)."""
    if not os.path.isdir(output_dir):
        return False
    has_config = os.path.exists(os.path.join(output_dir, "config.json"))
    has_weights = (
        os.path.exists(os.path.join(output_dir, "model.safetensors"))
        or os.path.exists(os.path.join(output_dir, "pytorch_model.bin"))
        or os.path.exists(os.path.join(output_dir, "model.safetensors.index.json"))
    )
    return has_config and has_weights


def run_direction(language, direction, train_csv, val_csv, test_csv, epochs):
    assert language in LANGUAGES, \
        f"Unknown language '{language}' in config.json, choices: {list(LANGUAGES.keys())}"
    assert direction in ["en2tgt", "hi2tgt", "tgt2en", "tgt2hi"], \
        f"Unknown direction '{direction}' in config.json, choices: en2tgt, hi2tgt, tgt2en, tgt2hi"

    cfg = LANGUAGES[language]
    tgt_column = cfg["column"]

    # Resolve direction -> (src_col, tgt_col, src_name, tgt_name)
    if direction == "en2tgt":
        src_col, tgt_col, src_name, tgt_name = "English", tgt_column, ENG_NAME, tgt_column
    elif direction == "hi2tgt":
        src_col, tgt_col, src_name, tgt_name = "Hindi", tgt_column, HIN_NAME, tgt_column
    elif direction == "tgt2en":
        src_col, tgt_col, src_name, tgt_name = tgt_column, "English", tgt_column, ENG_NAME
    elif direction == "tgt2hi":
        src_col, tgt_col, src_name, tgt_name = tgt_column, "Hindi", tgt_column, HIN_NAME

    output_dir = f"./qwen-{language}-{direction}-finetuned"
    final_scores_path = f"final_scores_{language}_{direction}.txt"
    print(f"\n========== Language: {language} | Direction: {direction} ==========")
    print(f"  {src_col} -> {tgt_col}")
    print(f"[Output] {output_dir}")

    # If this direction already ran to full completion (training + eval +
    # scoring) in a previous invocation, there's nothing left to do -- skip
    # it entirely rather than reloading data/model just to redo work whose
    # results are already on disk.
    if os.path.exists(final_scores_path):
        print(f"[Skip] {language}/{direction} already fully completed "
              f"(found {final_scores_path}) -- nothing to do.")
        return

    # ---- Data ----
    train_df, val_df, test_df = load_splits(train_csv, val_csv, test_csv, tgt_column)
    train_set = Dataset.from_pandas(train_df, preserve_index=False)
    val_set   = Dataset.from_pandas(val_df,   preserve_index=False)
    test_set  = Dataset.from_pandas(test_df,  preserve_index=False)
    dataset_dict = DatasetDict({"train": train_set, "validation": val_set, "test": test_set})
    print(f"[Data] train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    # Training already finished and saved a final model here, but scoring
    # didn't finish (no final_scores file) -- load the ALREADY-TRAINED model
    # straight from output_dir instead of retraining. This is deliberately
    # NOT the same as resume_from_checkpoint: every rank loads the same
    # complete, already-saved weights via plain from_pretrained (the same
    # mechanism used to load the base model in the first place), so there's
    # no risk of the cross-rank shape mismatch that resuming from a stale/
    # incomplete checkpoint-*/ folder caused.
    already_trained = _has_final_model(output_dir)

    if already_trained:
        print(f"[Resume] Found a completed final model in {output_dir} -- "
              "skipping training, only re-running evaluation/scoring.")
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        tokenizer, model, device = load_tokenizer_and_model()

    # ---- Tokenize (response-only loss masking) ----
    pre_fn = make_preprocess_fn(tokenizer, src_col, tgt_col, src_name, tgt_name)
    candidate_cols = ["English", "Hindi", tgt_column, "Unique_ID"]
    cols_to_remove = [c for c in candidate_cols if c in train_set.column_names]

    print("[Tokenize] Mapping validation split ...")
    tokenized_val = dataset_dict["validation"].map(
        pre_fn, batched=True, remove_columns=cols_to_remove, desc="val")

    if already_trained:
        tokenized_train = None
    else:
        print("[Tokenize] Mapping train split ...")
        tokenized_train = dataset_dict["train"].map(
            pre_fn, batched=True, remove_columns=cols_to_remove, desc="train")

    # ---- Trainer ----
    data_collator = CausalLMPaddingCollator(pad_token_id=tokenizer.pad_token_id)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"[Train] bf16={use_bf16}")

    # === TRAINING ARGS (standard full-parameter LLM SFT recipe: AdamW,
    #     small LR, short warmup, few epochs -- unlike the MT encoder-decoder
    #     models, LLM SFT typically uses far fewer epochs since the model
    #     already has strong pretrained priors and overfits fast) ===
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        eval_strategy="steps",
        eval_steps=900,
        save_strategy="steps",
        save_steps=900,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=False,
        max_grad_norm=1.0,
        logging_dir=f"{output_dir}/logs",
        logging_steps=1000,
        dataloader_num_workers=2,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if already_trained:
        print(f"[Skip] Training skipped -- using existing model from {output_dir}")
    else:
        # NOTE: deliberately NOT resuming from any leftover checkpoint-*/
        # folder here. trainer.train(resume_from_checkpoint=...) reliably
        # crashed on this cluster with a DDP _verify_param_shape_across_processes
        # overflow (RuntimeError: value cannot be converted to type int
        # without overflow) -- reproduced with two different checkpoints
        # from two different directions, so it's not one bad file, it's the
        # resume mechanism itself on this environment. Always training fresh
        # is slower on a re-run but reliable; the "already_trained" check
        # above still avoids wasting a full retrain on directions that
        # already finished.
        print("[Train] Training from scratch (checkpoint-resume disabled -- "
              "see comment above).")
        print(f"[Train] epochs={epochs}  steps/epoch~{len(tokenized_train)//(8*4)}")
        trainer.train()

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[Save] Best model written to {output_dir}")

    # ---- Final eval loss (for the master scores row) ----
    # trainer.evaluate() is a collective/synchronized op under DDP -- every
    # rank must call it (skipping it on non-zero ranks would hang the others
    # waiting on a collective that never gets their contribution).
    val_metrics = trainer.evaluate()
    val_loss = val_metrics.get("eval_loss", float("nan"))

    # Everything below is plain model.generate() + file I/O, NOT a
    # Trainer-managed collective op. Under DDP (torchrun --nproc_per_node>1)
    # every rank holds an identical trained model, so running this on all
    # ranks would redundantly repeat the same generation 4x AND have every
    # rank write to the same output files at once (corruption/duplicate
    # rows). Restrict it to the main process only.
    if trainer.is_world_process_zero():
        print(f"\nVAL  loss = {val_loss:.4f}")

        # ---- Real generation-based BLEU/chrF++ on val + test (beam=5) ----
        print("\n=== Generating on validation set (beam=5) ===")
        _, _, val_bleu, val_chrf = generate_and_score(
            list(dataset_dict["validation"]), src_col, tgt_col, src_name, tgt_name,
            model, tokenizer, device)
        print(f"VAL  BLEU   = {val_bleu:.4f}")
        print(f"VAL  chrF++ = {val_chrf:.4f}")

        print("\n=== Generating on test set (beam=5) ===")
        test_preds, test_refs, test_bleu, test_chrf = generate_and_score(
            list(dataset_dict["test"]), src_col, tgt_col, src_name, tgt_name,
            model, tokenizer, device)
        print(f"TEST BLEU   = {test_bleu:.4f}")
        print(f"TEST chrF++ = {test_chrf:.4f}")

        test_src_texts = [r[src_col] for r in dataset_dict["test"]]
        preds_path = f"test_predictions_{language}_{direction}.csv"
        pd.DataFrame({
            f"source_{src_col.lower()}":     test_src_texts,
            f"reference_{tgt_col.lower()}":  test_refs,
            f"prediction_{tgt_col.lower()}": test_preds,
        }).to_csv(preds_path, index=False, encoding="utf-8-sig")
        print(f"[Save] Predictions: {preds_path}")

        print(f"\n=== FINAL TEST SCORES [{language}/{direction}] (beam=5) ===")
        print(f"BLEU   : {test_bleu:.4f}")
        print(f"chrF++ : {test_chrf:.4f}")

        with open(f"final_scores_{language}_{direction}.txt", "w", encoding="utf-8") as f:
            f.write(f"Language: {language}  Direction: {direction}  ({src_col} -> {tgt_col})\n")
            f.write(f"VAL  loss   : {val_loss:.4f}\n")
            f.write(f"VAL  BLEU   : {val_bleu:.4f}\n")
            f.write(f"VAL  chrF++ : {val_chrf:.4f}\n")
            f.write(f"TEST BLEU   : {test_bleu:.4f}\n")
            f.write(f"TEST chrF++ : {test_chrf:.4f}\n")

        append_master_scores(language, direction, val_loss, val_bleu, val_chrf, test_bleu, test_chrf)
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
    epochs     = run_cfg["epochs"]

    # "direction" can be a single string or a list of the four directions
    if isinstance(directions, str):
        directions = [directions]

    for direction in directions:
        run_direction(language, direction, train_csv, val_csv, test_csv, epochs)


if __name__ == "__main__":
    main()
