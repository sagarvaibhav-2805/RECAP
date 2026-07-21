import os

# Force offline mode to prevent network-related hangs during distributed loading
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import csv
import gc
import json
import logging
import os
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import sacrebleu
import torch
import torch.distributed as dist
from peft import PeftModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)


CONFIG_PATH = "configs/eval_config.json"


def setup_ddp():
    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )

    use_ddp = world_size > 1

    if not use_ddp:
        return 0, 1, 0, False

    rank = int(
        os.environ["RANK"]
    )

    local_rank = int(
        os.environ["LOCAL_RANK"]
    )

    backend = (
        "nccl"
        if torch.cuda.is_available()
        else "gloo"
    )

    torch.cuda.set_device(
        local_rank
    )

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    return (
        rank,
        world_size,
        local_rank,
        True,
    )


def cleanup_ddp(use_ddp):
    if (
        use_ddp
        and dist.is_initialized()
    ):
        dist.destroy_process_group()


def gather_object(
    obj,
    world_size,
):
    if world_size == 1:
        return obj

    gathered = [
        None
        for _ in range(world_size)
    ]

    dist.all_gather_object(
        gathered,
        obj,
    )

    merged = []

    for item in gathered:
        merged.extend(item)

    return merged


def init_logger(
    log_path,
    name,
):
    logger = logging.getLogger(name)

    logger.handlers = []

    logger.setLevel(
        logging.INFO
    )

    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_path,
        mode="a",
        encoding="utf-8",
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        file_handler
    )

    return logger


class _NullLogger:
    """Drop-in logger for non-main DDP ranks to avoid NoneType crashes."""

    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass


def _safe_logger(logger):
    """Return a no-op logger when logger is None (non-rank-0 processes)."""
    return logger if logger is not None else _NullLogger()


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


class TranslationEvalDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        src_col: str,
        tgt_col: str,
        src_lang: str,
        tgt_lang: str,
        max_source_length: int,
        max_target_length: int,
    ):
        self.df = dataframe.reset_index(
            drop=True
        )

        self.tokenizer = tokenizer

        self.src_col = src_col
        self.tgt_col = tgt_col

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.max_source_length = (
            max_source_length
        )

        self.max_target_length = (
            max_target_length
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(
        self,
        idx,
    ):
        row = self.df.iloc[idx]

        src_text = str(
            row[self.src_col]
        ).strip()

        tgt_text = str(
            row[self.tgt_col]
        ).strip()

        self.tokenizer.src_lang = (
            self.src_lang
        )

        # Use text_target= instead of the deprecated as_target_tokenizer() API.
        model_inputs = self.tokenizer(
            text=src_text,
            text_target=tgt_text,
            truncation=True,
            max_length=self.max_source_length,
            padding=False,
            return_attention_mask=True,
        )

        # Truncate labels independently to max_target_length.
        model_inputs["labels"] = model_inputs["labels"][: self.max_target_length]

        model_inputs["src_text"] = src_text

        model_inputs["tgt_text"] = tgt_text

        return model_inputs


class EvalCollator:
    def __init__(
        self,
        tokenizer,
        model,
    ):
        self.tokenizer = tokenizer

        self.collator = (
            DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
        )

    def __call__(
        self,
        batch,
    ):
        src_texts = [
            x["src_text"]
            for x in batch
        ]

        tgt_texts = [
            x["tgt_text"]
            for x in batch
        ]

        processed = []

        for item in batch:
            processed.append(
                {
                    "input_ids": item[
                        "input_ids"
                    ],
                    "attention_mask": item[
                        "attention_mask"
                    ],
                    "labels": item[
                        "labels"
                    ],
                }
            )

        out = self.collator(
            processed
        )

        out["src_texts"] = src_texts

        out["tgt_texts"] = tgt_texts

        return out


def build_eval_dataset(
    cfg: Dict,
    tokenizer,
):
    eval_df = pd.read_csv(
        cfg["test_csv"]
    ).dropna()

    if (
        cfg.get(
            "max_test_rows_debug"
        )
        is not None
    ):
        eval_df = eval_df.iloc[
            : cfg[
                "max_test_rows_debug"
            ]
        ]

    dataset = TranslationEvalDataset(
        dataframe=eval_df,
        tokenizer=tokenizer,
        src_col=cfg["src_col"],
        tgt_col=cfg["tgt_col"],
        src_lang=cfg["src_lang"],
        tgt_lang=cfg["tgt_lang"],
        max_source_length=cfg.get(
            "max_source_length",
            256,
        ),
        max_target_length=cfg.get(
            "max_target_length",
            256,
        ),
    )

    return dataset


def build_eval_dataloader(
    dataset,
    tokenizer,
    model,
    cfg,
    rank,
    world_size,
    use_ddp,
):
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if use_ddp
        else None
    )

    collator = EvalCollator(
        tokenizer=tokenizer,
        model=model,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get(
            "eval_batch_size",
            4,
        ),
        sampler=sampler,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.get(
            "num_workers",
            4,
        ),
        pin_memory=True,
        persistent_workers=cfg.get(
            "persistent_workers",
            True,
        ),
        prefetch_factor=cfg.get(
            "prefetch_factor",
            2,
        ),
    )

    return dataloader


def load_model_and_tokenizer(
    cfg: Dict,
    logger,
):
    base_output_dir = os.path.join("directions", cfg["direction_name"])
    checkpoint_dir = cfg.get(
        "checkpoint_dir",
        os.path.join(base_output_dir, "best-checkpoint")
    )
    
    # Resolve rank for logging (passed via logger context or inferred)
    rank = int(os.environ.get("RANK", "0"))
    
    print(f"[Rank {rank}] Resolving checkpoint path for {cfg['direction_name']}...")

    # Resolve "best-checkpoint" or "latest-checkpoint" aliases if used.
    # train.py writes the actual dynamic directory path to text files.
    if checkpoint_dir.endswith("best-checkpoint"):
        best_path_file = os.path.join(os.path.dirname(checkpoint_dir), "_best_checkpoint_path.txt")
        if os.path.isfile(best_path_file):
            with open(best_path_file, "r") as f:
                checkpoint_dir = f.read().strip()
    elif checkpoint_dir.endswith("latest-checkpoint"):
        latest_path_file = os.path.join(os.path.dirname(checkpoint_dir), "_latest_checkpoint_path.txt")
        if os.path.isfile(latest_path_file):
            with open(latest_path_file, "r") as f:
                checkpoint_dir = f.read().strip()

    model_name = cfg[
        "model_name"
    ]

    offline_dir = cfg.get(
        "offline_model_dir",
        None,
    )

    use_qlora = cfg.get(
        "use_qlora",
        False,
    )

    attn_impl = (
        "flash_attention_2"
        if cfg.get(
            "use_flash_attention",
            False,
        )
        else "eager"
    )

    quantization_config = None

    if use_qlora:
        from transformers import (
            BitsAndBytesConfig,
        )

        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        )

    log = _safe_logger(logger)

    # ── Tokenizer loading (checkpoint → offline → online) ─────────────────
    tokenizer_loaded = False

    try:
        print(f"[Rank {rank}] Loading tokenizer from checkpoint: {checkpoint_dir}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_dir,
                local_files_only=True,
                use_fast=False,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_dir,
                local_files_only=True,
                use_fast=True,
            )
        tokenizer_loaded = True
        log.info("[TOKENIZER] Loaded tokenizer from checkpoint")
        print(f"[Rank {rank}] Tokenizer loaded from checkpoint.")

    except Exception:
        log.info("[TOKENIZER] Checkpoint tokenizer not found, trying fallback")

    if not tokenizer_loaded:
        try:
            if offline_dir is None or not os.path.isdir(offline_dir):
                raise RuntimeError("Offline tokenizer unavailable")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    offline_dir,
                    local_files_only=True,
                    use_fast=False,
                )
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(
                    offline_dir,
                    local_files_only=True,
                    use_fast=True,
                )
            log.info("[TOKENIZER] Loaded offline tokenizer")

        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            log.info("[TOKENIZER] Loaded online tokenizer")

    # ── Model loading — branch on LoRA vs full fine-tune ──────────────────
    # LoRA checkpoints only store adapter weights, NOT full model weights.
    # Attempting AutoModelForSeq2SeqLM.from_pretrained on a LoRA checkpoint
    # directory is unreliable.  We detect the intent via config flags and
    # choose the correct loading path upfront.
    use_lora = cfg.get("use_lora", False) or cfg.get("use_qlora", False)

    if use_lora:
        # Step 1: load base model (offline preferred, then online).
        log.info("[MODEL] LoRA/QLoRA mode — loading base model first")
        try:
            if offline_dir is None or not os.path.isdir(offline_dir):
                raise RuntimeError("Offline model unavailable")
            config = AutoConfig.from_pretrained(offline_dir)
            config.tie_word_embeddings = False
            model = AutoModelForSeq2SeqLM.from_pretrained(
                offline_dir,
                config=config,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
                attn_implementation=attn_impl,
            )
            log.info("[MODEL] Loaded offline base model")

        except Exception:
            log.info(f"[MODEL] Falling back to online base model: {model_name}")
            config = AutoConfig.from_pretrained(model_name)
            config.tie_word_embeddings = False
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
                attn_implementation=attn_impl,
            )

        # Step 2: overlay the adapter weights from the checkpoint directory.
        log.info(f"[LORA] Applying PEFT adapters from {checkpoint_dir}")
        model = PeftModel.from_pretrained(
            model,
            checkpoint_dir,
        )

    else:
        # Full fine-tune: checkpoint contains full model weights.
        log.info(f"[MODEL] Full fine-tune mode — loading checkpoint: {checkpoint_dir}")
        print(f"[Rank {rank}] Loading full model from checkpoint: {checkpoint_dir}")
        try:
            config = AutoConfig.from_pretrained(checkpoint_dir, local_files_only=True)
            config.tie_word_embeddings = False
            model = AutoModelForSeq2SeqLM.from_pretrained(
                checkpoint_dir,
                config=config,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
                attn_implementation=attn_impl,
            )
            log.info("[MODEL] Loaded checkpoint successfully")
            print(f"[Rank {rank}] Model loaded from checkpoint successfully.")

        except Exception:
            log.info("[MODEL] Checkpoint load failed, falling back to base model")
            try:
                if offline_dir is None or not os.path.isdir(offline_dir):
                    raise RuntimeError("Offline model unavailable")
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    offline_dir,
                    local_files_only=True,
                    torch_dtype=torch.bfloat16,
                    quantization_config=quantization_config,
                    attn_implementation=attn_impl,
                )
                log.info("[MODEL] Loaded offline base model as fallback")

            except Exception:
                log.info(f"[MODEL] Falling back to online base model: {model_name}")
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    quantization_config=quantization_config,
                    attn_implementation=attn_impl,
                )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    return tokenizer, model


def _apply_tokenizer_extension(
    tokenizer,
    model,
    cfg: Dict,
    logger,
):
    """
    Re-apply SentencePiece tokenizer extension for eval/infer.

    During training, the extended tokenizer is saved inside the checkpoint
    directory and is loaded automatically.  This function is a safety net:
    if the checkpoint tokenizer already contains all extension tokens,
    add_tokens() / add_special_tokens() will be no-ops (idempotent).

    Modes (per side, read from cfg):
      \"nllb_default\" → skip extension entirely
      \"custom_spm\"   → load SPM from tokenizer_extension_dir, add tokens
      \"auto\"         → same as custom_spm for eval (no corpus available
                        to compute UNK ratio, so we apply unconditionally)
    """
    import sentencepiece as spm  # local import — only needed when extending

    log = _safe_logger(logger)
    ext_dir = cfg.get(
        "tokenizer_extension_dir",
        os.path.join("tok_extensions", cfg["direction_name"])
    )

    for side in ("src", "tgt"):
        mode = cfg.get(f"{side}_tokenizer", "nllb_default")

        if mode == "nllb_default":
            log.info(
                f"[TOKENIZER] {side.upper()} mode=nllb_default — "
                "no extension applied"
            )
            continue

        # custom_spm or auto — load saved SPM and extend vocabulary.
        if ext_dir is None:
            log.warning(
                f"[TOKENIZER] {side.upper()} mode={mode} but "
                "'tokenizer_extension_dir' not set in config — skipping"
            )
            continue

        # SPM model file is saved as <direction_name>_<side>.model by train.py
        spm_path = os.path.join(
            ext_dir,
            f"{cfg['direction_name']}_{side}.model",
        )
        if not os.path.isfile(spm_path):
            log.warning(
                f"[TOKENIZER] SPM model not found at {spm_path} — "
                "skipping (train first to generate the SPM model)"
            )
            continue

        processor = spm.SentencePieceProcessor()
        processor.load(spm_path)

        existing_vocab = tokenizer.get_vocab()
        new_regular_tokens = []
        new_special_tokens = []

        for idx in range(processor.get_piece_size()):
            piece = processor.id_to_piece(idx)
            if piece not in existing_vocab:
                new_regular_tokens.append(piece)

        for lang_tag in (cfg.get("src_lang", ""), cfg.get("tgt_lang", "")):
            if lang_tag and lang_tag not in existing_vocab:
                new_special_tokens.append(lang_tag)

        if new_special_tokens:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": new_special_tokens}
            )
            log.info(
                f"[TOKENIZER] [{side.upper()}] Registered language tags: "
                f"{new_special_tokens}"
            )

        if new_regular_tokens:
            tokenizer.add_tokens(new_regular_tokens)
            model.resize_token_embeddings(len(tokenizer))
            log.info(
                f"[TOKENIZER] [{side.upper()}] Added {len(new_regular_tokens)} "
                "subword tokens and resized embeddings"
            )
        else:
            log.info(
                f"[TOKENIZER] [{side.upper()}] All SPM tokens already in vocab — "
                "checkpoint tokenizer was pre-extended (no-op)"
            )

    return tokenizer, model


def prepare_model_for_eval(
    model,
    device,
    use_ddp,
    local_rank,
    cfg,
):
    if not cfg.get(
        "use_qlora",
        False,
    ):
        model.to(device)

    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
        )

    model.eval()

    return model


@torch.no_grad()
def run_distributed_evaluation(
    model,
    tokenizer,
    dataloader,
    device,
    cfg,
    rank,
    world_size,
    logger,
):
    predictions = []

    references = []

    sources = []

    sentence_bleus = []

    sentence_chrfs = []

    log = _safe_logger(logger)

    # Safe forced_bos lookup — identical to train.py fallback chain.
    tgt_lang = cfg["tgt_lang"]
    lang_map = getattr(tokenizer, "lang_code_to_id", {})
    if tgt_lang in lang_map:
        forced_bos_token_id = lang_map[tgt_lang]
    elif tgt_lang in tokenizer.get_vocab():
        forced_bos_token_id = tokenizer.get_vocab()[tgt_lang]
    else:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        if forced_bos_token_id == tokenizer.unk_token_id:
            raw_model = model.module if isinstance(model, DDP) else model
            fallback_id = raw_model.config.decoder_start_token_id
            if fallback_id == tokenizer.eos_token_id:
                raise ValueError(
                    f"[EVAL] tgt_lang '{tgt_lang}' has no registered token in the "
                    "tokenizer, and decoder_start_token_id == eos_token_id "
                    f"({fallback_id}). Forcing this as the first generated token "
                    "makes every hypothesis empty, silently zeroing BLEU/CHRF++. "
                    "This checkpoint's tokenizer never learned this language tag — "
                    "retrain with the fixed train.py (ensure_lang_tag_registered)."
                )
            forced_bos_token_id = fallback_id

    log.info(f"[EVAL] forced_bos_token_id = {forced_bos_token_id} for lang '{tgt_lang}'")

    progress = tqdm(
        dataloader,
        disable=rank != 0,
        desc=f"{cfg['direction_name']} Evaluation",
    )

    for batch in progress:
        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        # labels are not used during inference — src/tgt raw strings drive metrics.
        src_texts = batch[
            "src_texts"
        ]

        tgt_texts = batch[
            "tgt_texts"
        ]

        amp_context = (
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )
            if torch.cuda.is_available()
            else nullcontext()
        )

        with amp_context:
            generated = (
                model.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    forced_bos_token_id=forced_bos_token_id,
                    num_beams=cfg.get(
                        "beam_size",
                        5,
                    ),
                    max_new_tokens=cfg.get(
                        "generation_max_length",
                        128,
                    ),
                    max_length=None,
                    repetition_penalty=cfg.get(
                        "repetition_penalty",
                        1.1,
                    ),
                    length_penalty=cfg.get(
                        "length_penalty",
                        1.0,
                    ),
                    no_repeat_ngram_size=cfg.get(
                        "no_repeat_ngram_size",
                        3,
                    ),
                    use_cache=True,
                )
                if isinstance(model, DDP)
                else model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    forced_bos_token_id=forced_bos_token_id,
                    num_beams=cfg.get(
                        "beam_size",
                        5,
                    ),
                    max_new_tokens=cfg.get(
                        "generation_max_length",
                        128,
                    ),
                    max_length=None,
                    repetition_penalty=cfg.get(
                        "repetition_penalty",
                        1.1,
                    ),
                    length_penalty=cfg.get(
                        "length_penalty",
                        1.0,
                    ),
                    no_repeat_ngram_size=cfg.get(
                        "no_repeat_ngram_size",
                        3,
                    ),
                    use_cache=True,
                )
            )

        decoded_preds = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        clean_preds = [
            p.strip()
            for p in decoded_preds
        ]

        clean_refs = [
            r.strip()
            for r in tgt_texts
        ]

        predictions.extend(
            clean_preds
        )

        references.extend(
            clean_refs
        )

        sources.extend(
            src_texts
        )

        for pred, ref in zip(
            clean_preds,
            clean_refs,
        ):
            sent_bleu = (
                sacrebleu.sentence_bleu(
                    pred,
                    [ref],
                ).score
            )

            sent_chrf = (
                sacrebleu.sentence_chrf(
                    pred,
                    [ref],
                    word_order=2,
                ).score
            )

            sentence_bleus.append(
                sent_bleu
            )

            sentence_chrfs.append(
                sent_chrf
            )

    predictions = gather_object(
        predictions,
        world_size,
    )

    references = gather_object(
        references,
        world_size,
    )

    sources = gather_object(
        sources,
        world_size,
    )

    sentence_bleus = gather_object(
        sentence_bleus,
        world_size,
    )

    sentence_chrfs = gather_object(
        sentence_chrfs,
        world_size,
    )

    if rank != 0:
        return None

    corpus_bleu = (
        sacrebleu.corpus_bleu(
            predictions,
            [references],
        ).score
    )

    corpus_chrf = (
        sacrebleu.corpus_chrf(
            predictions,
            [references],
            word_order=2,
        ).score
    )

    mean_sentence_bleu = (
        sum(sentence_bleus)
        / max(
            1,
            len(sentence_bleus),
        )
    )

    mean_sentence_chrf = (
        sum(sentence_chrfs)
        / max(
            1,
            len(sentence_chrfs),
        )
    )

    metrics = {
        "Corpus_BLEU": corpus_bleu,
        "Corpus_CHRF++": corpus_chrf,
        "Mean_Sentence_BLEU": mean_sentence_bleu,
        "Mean_Sentence_CHRF++": mean_sentence_chrf,
        "Num_Samples": len(
            predictions
        ),
    }

    log.info(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print(
        "\n========== Evaluation Metrics =========="
    )

    for k, v in metrics.items():
        print(f"{k}: {v}")

    return {
        "metrics": metrics,
        "predictions": predictions,
        "references": references,
        "sources": sources,
        "sentence_bleus": sentence_bleus,
        "sentence_chrfs": sentence_chrfs,
    }


def dump_predictions_csv(
    results,
    cfg,
    output_dir,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    csv_path = os.path.join(
        output_dir,
        "predictions.csv",
    )

    with open(
        csv_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "Source",
                "Reference",
                "Prediction",
                "Sentence_BLEU",
                "Sentence_CHRF++",
            ]
        )

        for (
            src,
            ref,
            pred,
            s_bleu,
            s_chrf,
        ) in zip(
            results["sources"],
            results["references"],
            results["predictions"],
            results["sentence_bleus"],
            results["sentence_chrfs"],
        ):
            writer.writerow(
                [
                    src,
                    ref,
                    pred,
                    s_bleu,
                    s_chrf,
                ]
            )

    metrics_path = os.path.join(
        output_dir,
        "metrics.json",
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results["metrics"],
            f,
            indent=2,
            ensure_ascii=False,
        )

    return csv_path, metrics_path


def evaluate_direction(
    rank,
    world_size,
    local_rank,
    use_ddp,
    cfg,
):
    direction = cfg[
        "direction_name"
    ]

    is_main = rank == 0

    # ── Directory Hierarchy ───────────────────────────────────────────
    base_output_dir = os.path.join("directions", cfg["direction_name"])
    output_dir = os.path.join(base_output_dir, "eval")
    
    if is_main:
        os.makedirs(output_dir, exist_ok=True)

    logger = (
        init_logger(
            os.path.join(
                output_dir,
                "eval.log",
            ),
            f"eval_{direction}",
        )
        if is_main
        else None
    )

    device = torch.device(
        f"cuda:{local_rank}"
        if torch.cuda.is_available()
        else "cpu"
    )

    if is_main:
        print(
            "\n"
            + "=" * 20
            + f" Evaluating {direction} "
            + "=" * 20
        )

    print(f"[Rank {rank}] Loading model and tokenizer for {direction}...")

    tokenizer, model = load_model_and_tokenizer(cfg, logger)

    # Re-apply tokenizer extension if cfg requests it.
    # If the checkpoint already contains the extended vocab (normal case),
    # this is a no-op.  Handles the fallback case where base model is loaded.
    tokenizer, model = _apply_tokenizer_extension(
        tokenizer=tokenizer,
        model=model,
        cfg=cfg,
        logger=logger,
    )

    model = prepare_model_for_eval(
        model=model,
        device=device,
        use_ddp=use_ddp,
        local_rank=local_rank,
        cfg=cfg,
    )

    dataset = build_eval_dataset(
        cfg=cfg,
        tokenizer=tokenizer,
    )

    dataloader = (
        build_eval_dataloader(
            dataset=dataset,
            tokenizer=tokenizer,
            model=model,
            cfg=cfg,
            rank=rank,
            world_size=world_size,
            use_ddp=use_ddp,
        )
    )

    if use_ddp:
        dist.barrier()

    results = (
        run_distributed_evaluation(
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            device=device,
            cfg=cfg,
            rank=rank,
            world_size=world_size,
            logger=logger,
        )
    )

    if is_main:
        dump_predictions_csv(
            results=results,
            cfg=cfg,
            output_dir=output_dir,
        )

        print(
            "\n[INFO] Evaluation artifacts saved"
        )

        logger.info(
            "[INFO] Evaluation completed successfully"
        )

    if use_ddp:
        dist.barrier()

    gc.collect()

    torch.cuda.empty_cache()


def main():
    rank, world_size, local_rank, use_ddp = (
        setup_ddp()
    )

    try:
        config = load_json(
            CONFIG_PATH
        )

        directions = config[
            "directions"
        ]

        for direction_cfg in directions:
            if use_ddp:
                dist.barrier()

            evaluate_direction(
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                use_ddp=use_ddp,
                cfg=direction_cfg,
            )

            if use_ddp:
                dist.barrier()

    except KeyboardInterrupt:
        if rank == 0:
            print(
                "\n[INFO] Evaluation interrupted by user"
            )

    except Exception as e:
        print(
            f"[FATAL ERROR] {str(e)}"
        )

        if (
            use_ddp
            and dist.is_initialized()
        ):
            try:
                dist.abort()
            except Exception:
                pass

        raise

    finally:
        cleanup_ddp(
            use_ddp
        )


if __name__ == "__main__":
    main()