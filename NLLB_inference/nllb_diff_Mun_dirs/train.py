import os

# Force offline mode to prevent network-related hangs during distributed loading
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import gc
import json
import logging
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import sacrebleu
import sentencepiece as spm
import torch
import torch.distributed as dist
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.amp import GradScaler
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
    get_cosine_schedule_with_warmup,
)

CONFIG_PATH = "configs/train_config.json"

GLOBAL_SEED = 42


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def seed_everything(seed: int):
    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    use_ddp = world_size > 1

    if not use_ddp:
        return 0, 1, 0, False

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(minutes=120),
        device_id=local_rank if backend == "nccl" else None,
    )

    return rank, world_size, local_rank, True


def cleanup_ddp(use_ddp: bool):
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


def reduce_sum_scalar(value, device, world_size):
    if world_size <= 1:
        return float(value)

    tensor = torch.tensor(
        float(value),
        device=device,
    )

    dist.all_reduce(
        tensor,
        op=dist.ReduceOp.SUM,
    )

    return tensor.item()


def gather_object(obj, world_size):
    if world_size == 1:
        return obj

    gathered = [None for _ in range(world_size)]

    dist.all_gather_object(
        gathered,
        obj,
    )

    merged = []

    for item in gathered:
        merged.extend(item)

    return merged


def init_logger(log_path: str, name: str):
    logger = logging.getLogger(name)

    logger.handlers = []

    logger.setLevel(logging.INFO)

    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_path,
        mode="a",
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


class _NullLogger:
    """Drop-in logger for non-main DDP ranks to avoid NoneType crashes."""

    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass


def _safe_logger(logger):
    """Return a no-op logger when logger is None (non-rank-0 processes)."""
    return logger if logger is not None else _NullLogger()


def safe_exp(x: float):
    return math.exp(min(x, 20))


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_academic_plots(csv_path: str, save_dir: str, direction: str):
    """Generates ACL/EMNLP style plots for training metrics."""
    df = pd.read_csv(csv_path)
    epochs = df["epoch"].values

    # Use a clean style
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "grid.alpha": 0.3
    })

    metrics_to_plot = [
        ("Loss", ["train_loss", "val_loss"], "Loss"),
        ("BLEU", ["train_bleu", "val_bleu"], "BLEU Score"),
        ("CHRF", ["train_chrf", "val_chrf"], "CHRF++ Score"),
        ("Learning Rate", ["lr"], "Learning Rate")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    colors = ["#1f77b4", "#d62728"]  # Academic Blue & Red

    for i, (title, cols, ylabel) in enumerate(metrics_to_plot):
        ax = axes[i]
        for idx, col in enumerate(cols):
            label = col.replace("_", " ").title()
            ax.plot(epochs, df[col], label=label, marker='o', markersize=4, 
                    color=colors[idx] if len(cols) > 1 else colors[0], linewidth=1.5)
        
        ax.set_title(f"{direction}: {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_metrics_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def maybe_train_sentencepiece(
    texts: List[str],
    save_dir: str,
    vocab_size: int,
    model_type: str,
    lang_name: str,
    logger,
    force_train: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)

    model_prefix = os.path.join(
        save_dir,
        f"spm_{lang_name}",
    )
    model_path = model_prefix + ".model"

    # Check if the model already exists to avoid redundant training,
    # unless a forced re-train is requested.
    if os.path.isfile(model_path) and not force_train:
        logger.info(
            f"[TOKENIZER] Existing SentencePiece model found for {lang_name} at {model_path} — skipping training"
        )
        return model_path

    corpus_path = os.path.join(
        save_dir,
        f"{lang_name}_corpus.txt",
    )

    with open(corpus_path, "w", encoding="utf-8") as f:
        for text in texts:
            text = str(text).replace("\n", " ").strip()

            if len(text) > 0:
                f.write(text + "\n")

    logger.info(
        f"[TOKENIZER] Training SentencePiece tokenizer for {lang_name}"
    )

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        split_digits=True,
        byte_fallback=True,
        normalization_rule_name="identity",
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        hard_vocab_limit=False,
    )

    return model_path


def compute_unknown_ratio(
    tokenizer,
    texts: List[str],
    sample_size: int = 1000,
):
    sample_size = min(sample_size, len(texts))

    texts = texts[:sample_size]

    total_tokens = 0
    unk_tokens = 0

    for text in texts:
        toks = tokenizer.tokenize(str(text))

        total_tokens += len(toks)

        unk_tokens += sum(
            1
            for t in toks
            if t == tokenizer.unk_token
        )

    if total_tokens == 0:
        return 0.0

    return unk_tokens / total_tokens


def ensure_lang_tag_registered(
    tokenizer,
    model,
    lang_tag: str,
    logger,
    is_main: bool = True,
    use_ddp: bool = False,
    init_from_lang: str = "hin_Deva",
):
    """
    Guarantee `lang_tag` resolves to a real token id before it is used as a
    src_lang/tgt_lang for tokenization or as forced_bos_token_id at generation
    time.

    NLLB's tokenizer only recognizes its native ~200 language codes. A tribal
    code like "bhili_Deva" that isn't one of them silently tokenizes to <unk>
    unless explicitly registered — regardless of the "nllb_default" /
    "auto" / "custom_spm" tokenizer-extension mode, since that logic only
    registers language tags as a side effect of deciding to extend the
    subword vocabulary (see extend_tokenizer_if_needed above). Under
    "nllb_default" mode the function returns immediately and the tag never
    gets registered at all.

    When this happens for tgt_lang, forced_bos_token_id resolution (see
    compute_bleu_chrf below) falls back to decoder_start_token_id, which for
    NLLB equals eos_token_id — forcing every generated sequence to be empty
    and BLEU/CHRF++ to be exactly 0.0 for the entire run, even while the
    underlying model trains normally (loss/PPL improve as expected).

    Safe to call unconditionally for every direction / every lang tag: it
    no-ops immediately if the tag is already a native NLLB code or was
    already registered.
    """
    log = _safe_logger(logger)

    lang_map = getattr(tokenizer, "lang_code_to_id", {})
    if lang_tag in lang_map or lang_tag in tokenizer.get_vocab():
        return tokenizer, model

    if is_main:
        log.info(
            f"[TOKENIZER] '{lang_tag}' is not a native NLLB language code — "
            "registering it as a new special token before training."
        )

    tokenizer.add_special_tokens({"additional_special_tokens": [lang_tag]})
    model.resize_token_embeddings(len(tokenizer))

    new_id = tokenizer.convert_tokens_to_ids(lang_tag)
    ref_id = tokenizer.convert_tokens_to_ids(init_from_lang)

    if ref_id is not None and ref_id != tokenizer.unk_token_id:
        # tie_word_embeddings is set False for this model (see
        # load_model_and_tokenizer), so input and output embeddings are
        # separate tensors — warm-start both explicitly.
        embedding_matrices = {
            id(m): m
            for m in (model.get_input_embeddings(), model.get_output_embeddings())
            if m is not None
        }
        with torch.no_grad():
            for emb in embedding_matrices.values():
                emb.weight[new_id] = emb.weight[ref_id].clone()
        if is_main:
            log.info(
                f"[TOKENIZER] Warm-started '{lang_tag}' embedding (id={new_id}) "
                f"from '{init_from_lang}' (id={ref_id})"
            )
    elif is_main:
        log.warning(
            f"[TOKENIZER] Reference language '{init_from_lang}' not found — "
            f"'{lang_tag}' (id={new_id}) left with a randomly initialized embedding."
        )

    if use_ddp:
        dist.barrier()

    return tokenizer, model


def extend_tokenizer_if_needed(
    tokenizer,
    model,
    texts: List[str],
    cfg: Dict,
    logger,
    side: str = "tgt",
    is_main: bool = True,
    use_ddp: bool = False,
    force_train: bool = False,
):
    """
    Conditionally extend the NLLB tokenizer with a custom SentencePiece model.

    Three modes — set via cfg["src_tokenizer"] or cfg["tgt_tokenizer"]:

      "nllb_default"
          Never extend.  Use NLLB's built-in vocabulary as-is.
          Unseen scripts will produce <unk> tokens at inference time.
          Use for: English, Hindi, Marathi, Gujarati, Odia and any
          language NLLB-200 already covers well.

      "custom_spm"
          Always train a SentencePiece model on the given corpus and
          extend NLLB's vocabulary with new subword pieces.  The UNK
          ratio check is skipped — extension runs unconditionally.
          Use for: tribal/unseen languages (Bhili, Mundari, Gondi …)
          where NLLB has little or no coverage.

      "auto"
          Compute the UNK ratio on a sample of the corpus first.
          Extend only if the ratio exceeds tokenizer_extension_threshold.
          Use when: you are unsure whether NLLB covers the language.

    Args:
        side: "src" or "tgt" — selects the correct mode key from cfg.
    """
    mode_key = f"{side}_tokenizer"
    mode = cfg.get(mode_key, "nllb_default")
    log = _safe_logger(logger)

    if mode == "nllb_default":
        log.info(
            f"[TOKENIZER] {side.upper()} tokenizer mode=nllb_default — "
            "using NLLB vocabulary as-is (no extension)"
        )
        return tokenizer, model

    if mode == "auto":
        unk_ratio = compute_unknown_ratio(tokenizer, texts)
        threshold = cfg.get("tokenizer_extension_threshold", 0.05)
        log.info(
            f"[TOKENIZER] {side.upper()} mode=auto | "
            f"UNK ratio = {unk_ratio:.6f} | threshold = {threshold:.6f}"
        )
        if unk_ratio < threshold:
            log.info(
                f"[TOKENIZER] {side.upper()} UNK ratio below threshold — "
                "skipping extension (NLLB default sufficient)"
            )
            return tokenizer, model
        log.info(
            f"[TOKENIZER] {side.upper()} UNK ratio above threshold — "
            "proceeding with SentencePiece extension"
        )

    elif mode == "custom_spm":
        log.info(
            f"[TOKENIZER] {side.upper()} mode=custom_spm — "
            "forcing SentencePiece extension unconditionally"
        )

    else:
        log.warning(
            f"[TOKENIZER] Unknown mode '{mode}' for {side}_tokenizer. "
            "Valid values: 'nllb_default', 'custom_spm', 'auto'. "
            "Defaulting to nllb_default (no extension)."
        )
        return tokenizer, model

    tok_ext_dir = cfg.get(
        "tokenizer_extension_dir",
        os.path.join("tok_extensions", cfg["direction_name"])
    )

    # ── Train SentencePiece and extend vocabulary ──────────────────────────
    if is_main:
        sp_model = maybe_train_sentencepiece(
            texts=texts,
            save_dir=tok_ext_dir,
            vocab_size=cfg.get("spm_vocab_size", 4096),
            model_type=cfg.get("spm_model_type", "unigram"),
            lang_name=f"{cfg['direction_name']}_{side}",
            logger=log,
            force_train=force_train,
        )
    else:
        sp_model = os.path.join(
            tok_ext_dir,
            f"spm_{cfg['direction_name']}_{side}.model"
        )

    if use_ddp:
        dist.barrier()

    sp = spm.SentencePieceProcessor()
    sp.load(sp_model)

    existing_vocab = tokenizer.get_vocab()
    new_regular_tokens: List[str] = []
    new_special_tokens: List[str] = []

    for idx in range(sp.get_piece_size()):
        piece = sp.id_to_piece(idx)
        if piece not in existing_vocab:
            new_regular_tokens.append(piece)

    # Register src/tgt language tags as additional_special_tokens so
    # NLLB's forced_bos lookup works even for tribal/unseen lang codes.
    for lang_tag in (cfg.get("src_lang", ""), cfg.get("tgt_lang", "")):
        if lang_tag and lang_tag not in existing_vocab:
            new_special_tokens.append(lang_tag)

    if new_special_tokens:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": new_special_tokens}
        )
        log.info(
            f"[TOKENIZER] Registered {len(new_special_tokens)} "
            f"language tag(s) as special tokens: {new_special_tokens}"
        )

    log.info(
        f"[TOKENIZER] [{side.upper()}] Adding {len(new_regular_tokens)} "
        "new subword tokens"
    )

    old_vocab_size = len(tokenizer)
    tokenizer.add_tokens(new_regular_tokens)
    model.resize_token_embeddings(len(tokenizer))

    log.info(
        f"[TOKENIZER] Resized embeddings "
        f"{old_vocab_size} -> {len(tokenizer)}"
    )

    return tokenizer, model


def setup_lora(
    model,
    cfg: Dict,
):
    if (
        not cfg.get("use_lora", False)
        and not cfg.get("use_qlora", False)
    ):
        return model

    if cfg.get("use_qlora", False):
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.1),
        bias="none",
        target_modules=cfg.get(
            "lora_target_modules",
            # NLLB-200 attention projection names: q/k/v + out_proj (not o_proj)
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
            ],
        ),
    )

    model = get_peft_model(
        model,
        peft_config,
    )

    model.print_trainable_parameters()

    return model


def load_model_and_tokenizer(
    cfg: Dict,
    logger,
):
    # Resolve rank for logging
    rank = int(os.environ.get("RANK", "0"))
    
    model_name = cfg["model_name"]

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
        if cfg.get("use_flash_attention", False)
        else "eager"
    )

    quantization_config = None

    if use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    local_ok = (
        offline_dir is not None
        and os.path.isdir(offline_dir)
    )

    log = _safe_logger(logger)

    try:
        if not local_ok or offline_dir is None or not os.path.isdir(offline_dir):
            raise FileNotFoundError(f"Offline directory not found or unavailable: {offline_dir}")

        log.info(f"[MODEL] Loading offline weights from: {offline_dir}")
        print(f"[Rank {rank}] Loading offline weights from: {offline_dir}")

        # Try slow tokenizer first, fall back to fast if needed
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                offline_dir,
                local_files_only=True,
                use_fast=False,
            )
        except Exception:
            log.info("[TOKENIZER] Slow tokenizer failed, trying fast tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                offline_dir,
                local_files_only=True,
                use_fast=True,
            )
        print(f"[Rank {rank}] Tokenizer loaded.")

        config = AutoConfig.from_pretrained(offline_dir, local_files_only=True)
        config.tie_word_embeddings = False

        model = AutoModelForSeq2SeqLM.from_pretrained(
            offline_dir,
            config=config,
            local_files_only=True,
            torch_dtype=torch.bfloat16 if cfg.get("use_bf16") else torch.float32,
            quantization_config=quantization_config,
            attn_implementation=attn_impl,
        )
        log.info(f"[MODEL] Successfully loaded offline model from {offline_dir}")
        print(f"[Rank {rank}] Model weights loaded successfully.")

    except Exception as e:
        log.warning(f"[MODEL] Offline load failed! Reason: {str(e)}")
        log.info(f"[MODEL] Falling back to online download: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
        )

        config = AutoConfig.from_pretrained(model_name)
        config.tie_word_embeddings = False

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16 if cfg.get("use_bf16") else torch.float32,
            quantization_config=quantization_config,
            attn_implementation=attn_impl,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False

    model.gradient_checkpointing_enable()

    return tokenizer, model


class TranslationDataset(Dataset):
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
        self.df = dataframe.reset_index(drop=True)

        self.tokenizer = tokenizer

        self.src_col = src_col
        self.tgt_col = tgt_col

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        src_text = str(row[self.src_col]).strip()
        tgt_text = str(row[self.tgt_col]).strip()

        # -----------------------
        # Source tokenization
        # -----------------------
        self.tokenizer.src_lang = self.src_lang

        inputs = self.tokenizer(
            src_text,
            truncation=True,
            max_length=self.max_source_length,
            padding=False,
            return_attention_mask=True,
        )

        # -----------------------
        # Target tokenization
        # -----------------------
        with self.tokenizer.as_target_tokenizer():

            self.tokenizer.tgt_lang = self.tgt_lang

            pieces = self.tokenizer.tokenize(tgt_text)
            ids = self.tokenizer.convert_tokens_to_ids(pieces)

            # truncate
            ids = ids[: self.max_target_length - 1]

            # append EOS manually
            ids.append(self.tokenizer.eos_token_id)

            labels = {
                "input_ids": ids
            }

            inputs["labels"] = labels["input_ids"]

        return inputs

    # def __getitem__(self, idx):
    #     row = self.df.iloc[idx]

    #     src_text = str(
    #         row[self.src_col]
    #     ).strip()

    #     tgt_text = str(
    #         row[self.tgt_col]
    #     ).strip()

    #     self.tokenizer.src_lang = self.src_lang

    #     # Use text_target= instead of the deprecated as_target_tokenizer() API.

    #     print("\n====================")
    #     print("Index:", idx)
    #     print("SRC:", repr(src_text))
    #     print("TGT:", repr(tgt_text))
    #     print("src_lang:", self.src_lang)
    #     print("tgt_lang:", self.tgt_lang)
    #     print("====================")

    #     # model_inputs = self.tokenizer(
    #     #     text=src_text,
    #     #     text_target=tgt_text,
    #     #     truncation=True,
    #     #     max_length=self.max_source_length,
    #     #     padding=False,
    #     #     return_attention_mask=True,
    #     # )
    #     # print(model_inputs)
    #     # # Truncate labels independently to max_target_length.
    #     # model_inputs["labels"] = model_inputs["labels"][: self.max_target_length]

    #     # return model_inputs

    #     self.tokenizer.src_lang = self.src_lang

    #     inputs = self.tokenizer(
    #         src_text,
    #         truncation=True,
    #         max_length=self.max_source_length,
    #         padding=False,
    #         return_attention_mask=True,
    #     )

    #     with self.tokenizer.as_target_tokenizer():

    #         pieces = self.tokenizer.tokenize(tgt_text)
    #         print("TOKENS:", pieces)

    #         ids = self.tokenizer.convert_tokens_to_ids(pieces)
    #         print("IDS:", ids)

    #         labels = self.tokenizer.prepare_for_model(
    #             ids,
    #             padding=False,
    #             truncation=True,
    #             max_length=self.max_target_length,
    #         )
    #         # labels = self.tokenizer(
    #         #     tgt_text,
    #         #     truncation=True,
    #         #     max_length=self.max_target_length,
    #         #     padding=False,
    #         # )

    #     inputs["labels"] = labels["input_ids"]

    #     return inputs


class DynamicCollator:
    def __init__(
        self,
        tokenizer,
        model,
    ):
        self.collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    def __call__(self, batch):
        out = self.collator(batch)

        return Batch(
            input_ids=out["input_ids"],
            attention_mask=out["attention_mask"],
            labels=out["labels"],
        )


def build_datasets(
    cfg: Dict,
    tokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
):

    train_dataset = TranslationDataset(
        dataframe=train_df,
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

    val_dataset = TranslationDataset(
        dataframe=val_df,
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

    return (
        train_dataset,
        val_dataset,
    )


def build_dataloaders(
    train_dataset,
    val_dataset,
    tokenizer,
    model,
    cfg: Dict,
    rank: int,
    world_size: int,
    use_ddp: bool,
):
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        if use_ddp
        else None
    )

    val_sampler = (
        DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if use_ddp
        else None
    )

    collator = DynamicCollator(
        tokenizer=tokenizer,
        model=model,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        sampler=train_sampler,
        shuffle=train_sampler is None,
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

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.get(
            "eval_batch_size",
            cfg["batch_size"],
        ),
        sampler=val_sampler,
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

    return (
        train_loader,
        val_loader,
        train_sampler,
        val_sampler,
    )

@torch.no_grad()
def run_validation_loss(
    model,
    dataloader,
    device,
    world_size,
):
    model.eval()

    total_loss = 0.0
    total_tokens = 0.0

    for batch in dataloader:
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = batch.labels.to(device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        loss = outputs.loss

        toks = attention_mask.sum().item()

        total_loss += loss.item() * toks
        total_tokens += toks

    val_loss = reduce_sum_scalar(
        total_loss,
        device,
        world_size,
    ) / max(
        1.0,
        reduce_sum_scalar(
            total_tokens,
            device,
            world_size,
        ),
    )

    return val_loss


@torch.no_grad()
def compute_bleu_chrf(
    model,
    tokenizer,
    dataloader,
    device,
    cfg,
    world_size,
    max_samples: int = 0,
):
    """Compute corpus BLEU and CHRF++ via beam-search generation.

    Args:
        max_samples: If > 0, stop decoding after this many sentences.
                     Useful for train-set BLEU approximation.
    """
    model.eval()

    predictions = []
    references = []

    # Safe forced_bos lookup: handles tribal/extended language tags that
    # were added via add_special_tokens() and therefore exist in the
    # tokenizer vocab even if not in the original lang_code_to_id map.
    tgt_lang = cfg["tgt_lang"]
    lang_map = getattr(tokenizer, "lang_code_to_id", {})
    if tgt_lang in lang_map:
        forced_bos_token_id = lang_map[tgt_lang]
    elif tgt_lang in tokenizer.get_vocab():
        forced_bos_token_id = tokenizer.get_vocab()[tgt_lang]
    else:
        # Fallback: convert language tag to token id via tokenizer
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        if forced_bos_token_id == tokenizer.unk_token_id:
            # Last resort: use decoder_start_token_id from model config
            raw_model = model.module if isinstance(model, DDP) else model
            fallback_id = raw_model.config.decoder_start_token_id
            if fallback_id == tokenizer.eos_token_id:
                raise ValueError(
                    f"[BLEU/CHRF] tgt_lang '{tgt_lang}' has no registered token in "
                    "the tokenizer, and decoder_start_token_id == eos_token_id "
                    f"({fallback_id}). Forcing this as the first generated token "
                    "makes every hypothesis empty, silently zeroing BLEU/CHRF++ for "
                    "the whole run. Call ensure_lang_tag_registered() for this "
                    "language before training/evaluating."
                )
            forced_bos_token_id = fallback_id

    seen = 0

    raw_model = model.module if isinstance(model, DDP) else model

    # Beam search memory scales with batch_size * num_beams. Chunking the
    # generate() call independently of the DataLoader's batch size (which is
    # sized for training/loss throughput, not for beam decoding) keeps
    # generation's peak memory bounded regardless of beam_size — avoids OOM
    # when beam_size > 1 is combined with a large train/eval batch_size,
    # right after a training epoch that has already left the CUDA allocator
    # holding a lot of cached (but reusable) memory.
    gen_batch_size = cfg.get("generation_batch_size", cfg.get("eval_batch_size", 8))

    for batch in dataloader:
        if max_samples > 0 and seen >= max_samples:
            break

        input_ids_full = batch.input_ids.to(device)
        attention_mask_full = batch.attention_mask.to(device)

        labels = batch.labels.clone()

        bsz = input_ids_full.size(0)
        generated_chunks = []

        for start in range(0, bsz, gen_batch_size):
            end = min(start + gen_batch_size, bsz)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                gen = raw_model.generate(
                    input_ids=input_ids_full[start:end],
                    attention_mask=attention_mask_full[start:end],
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
                    repetition_penalty=1.1,
                    length_penalty=1.0,
                )

            generated_chunks.append(gen)

        # generate() pads each chunk to its own longest sequence, which can
        # differ across chunks — pad to a common length before concatenating.
        max_len = max(g.size(1) for g in generated_chunks)
        pad_id = tokenizer.pad_token_id

        padded_chunks = []
        for g in generated_chunks:
            if g.size(1) < max_len:
                pad = g.new_full((g.size(0), max_len - g.size(1)), pad_id)
                g = torch.cat([g, pad], dim=1)
            padded_chunks.append(g)

        generated = torch.cat(padded_chunks, dim=0)

        preds = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        labels[labels == -100] = tokenizer.pad_token_id

        refs = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        predictions.extend(preds)
        references.extend(refs)
        seen += len(preds)

    predictions = gather_object(
        predictions,
        world_size,
    )

    references = gather_object(
        references,
        world_size,
    )

    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references],
    ).score

    chrf = sacrebleu.corpus_chrf(
        predictions,
        [references],
        word_order=2,
    ).score

    return bleu, chrf


def save_checkpoint(
    save_dir,
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    best_val,
    wait,
    cfg,
):
    os.makedirs(save_dir, exist_ok=True)

    real_model = (
        model.module
        if isinstance(model, DDP)
        else model
    )

    real_model.save_pretrained(save_dir)

    tokenizer.save_pretrained(save_dir)

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val": best_val,
            "wait": wait,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "config": cfg,
        },
        os.path.join(
            save_dir,
            "trainer_state.pt",
        ),
    )


def train_direction(
    rank,
    world_size,
    local_rank,
    use_ddp,
    cfg,
):
    direction = cfg["direction_name"]

    is_main = rank == 0

    device = torch.device(
        f"cuda:{local_rank}"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ── Directory Hierarchy ───────────────────────────────────────────
    # Base: directions/<direction_name>
    # Train artifacts: directions/<direction_name>/train/
    base_output_dir = os.path.join("directions", cfg["direction_name"])
    train_dir = os.path.join(base_output_dir, "train")
    
    if is_main:
        os.makedirs(train_dir, exist_ok=True)

    logger = (
        init_logger(
            os.path.join(
                train_dir,
                "train.log",
            ),
            f"train_{direction}",
        )
        if is_main
        else None
    )

    tokenizer, model = load_model_and_tokenizer(
        cfg,
        logger,
    )

    # ── Register any non-native language tags (e.g. bhili_Deva) ────────────
    # Must happen before datasets are built (they set tokenizer.src_lang /
    # tokenizer.tgt_lang) and before extend_tokenizer_if_needed (which only
    # registers tags as a side effect of subword-vocab extension, and never
    # runs at all under "nllb_default" mode).
    lang_tag_init_from = cfg.get("lang_tag_warm_start_from", "hin_Deva")
    for lang_tag in (cfg["src_lang"], cfg["tgt_lang"]):
        tokenizer, model = ensure_lang_tag_registered(
            tokenizer=tokenizer, model=model, lang_tag=lang_tag,
            logger=logger, is_main=is_main, use_ddp=use_ddp,
            init_from_lang=lang_tag_init_from,
        )

    # ── Dynamic Train/Val Disjoint Splitting ──────────────────────────────────
    if cfg["train_csv"] == cfg["val_csv"]:
        full_df = pd.read_csv(cfg["train_csv"]).dropna()
        # Randomly shuffle deterministically across all DDP ranks
        full_df = full_df.sample(frac=1.0, random_state=GLOBAL_SEED).reset_index(drop=True)
        
        # Logic: If train==val, we split the single file.
        val_split = cfg.get("val_split", 0.1)
        val_size = int(len(full_df) * val_split)
        
        # Override with debug limit if specified
        if cfg.get("max_val_rows_debug") is not None:
            val_size = min(val_size, cfg["max_val_rows_debug"])
            
        val_df = full_df.iloc[:val_size]
        train_df = full_df.iloc[val_size:]
        
        if cfg.get("max_train_rows_debug") is not None:
            train_df = train_df.iloc[:cfg["max_train_rows_debug"]]
    else:
        train_df = pd.read_csv(cfg["train_csv"]).dropna()
        val_df = pd.read_csv(cfg["val_csv"]).dropna()
        
        if cfg.get("max_train_rows_debug") is not None:
            train_df = train_df.iloc[: cfg["max_train_rows_debug"]]
        if cfg.get("max_val_rows_debug") is not None:
            val_df = val_df.iloc[: cfg["max_val_rows_debug"]]

    # ── Tokenizer extension (per-side, independent modes) ─────────────────
    # src_tokenizer / tgt_tokenizer each accept:
    #   "nllb_default" → never extend (NLLB vocab used as-is)
    #   "custom_spm"   → always train SentencePiece and extend
    #   "auto"         → extend only if UNK ratio > tokenizer_extension_threshold
    src_texts = list(train_df[cfg["src_col"]].astype(str).values)
    tgt_texts = list(train_df[cfg["tgt_col"]].astype(str).values)

    tokenizer, model = extend_tokenizer_if_needed(
        tokenizer=tokenizer,
        model=model,
        texts=src_texts,
        cfg=cfg,
        logger=logger,
        side="src",
        is_main=is_main,
        use_ddp=use_ddp,
        force_train=not cfg.get("resume_from_checkpoint", True),
    )

    tokenizer, model = extend_tokenizer_if_needed(
        tokenizer=tokenizer,
        model=model,
        texts=tgt_texts,
        cfg=cfg,
        logger=logger,
        side="tgt",
        is_main=is_main,
        use_ddp=use_ddp,
        force_train=not cfg.get("resume_from_checkpoint", True),
    )

    model = setup_lora(
        model,
        cfg,
    )

    if not cfg.get(
        "use_qlora",
        False,
    ):
        model.to(device)

    train_dataset, val_dataset = build_datasets(
        cfg,
        tokenizer,
        train_df,
        val_df,
    )

    (
        train_loader,
        val_loader,
        train_sampler,
        val_sampler,
    ) = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        model=model,
        cfg=cfg,
        rank=rank,
        world_size=world_size,
        use_ddp=use_ddp,
    )

    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            static_graph=True,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=cfg.get(
            "weight_decay",
            0.01,
        ),
    )

    total_steps = (
        len(train_loader)
        * cfg["epochs"]
        // cfg.get(
            "gradient_accumulation_steps",
            1,
        )
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.get(
            "warmup_steps",
            500,
        ),
        num_training_steps=total_steps,
    )

    scaler = GradScaler(
        "cuda",
        enabled=cfg.get(
            "use_fp16",
            False,
        )
    )

    best_val = float("inf")
    wait = 0
    global_step = 0
    start_epoch = 1

    patience = cfg.get(
        "patience",
        3,
    )

    # ── Resume from checkpoint ───────────────────────────────────────────
    # We look in base_output_dir for checkpoints.
    resume_ckpt = None
    if cfg.get("resume_from_checkpoint", True):
        resume_ckpt = find_latest_checkpoint(base_output_dir)

    if resume_ckpt is not None and is_main:
        print(f"[RESUME] Resuming from checkpoint: {resume_ckpt}")

    if use_ddp:
        # Broadcast resume decision from rank-0 so all ranks agree.
        _resume_flag = torch.tensor(
            [1 if resume_ckpt is not None else 0],
            device=device,
        )
        dist.broadcast(_resume_flag, src=0)
        if not bool(_resume_flag.item()):
            resume_ckpt = None

    if resume_ckpt is not None:
        start_epoch, global_step, best_val, wait = restore_checkpoint(
            checkpoint_dir=resume_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        if is_main:
            last_completed_epoch = start_epoch - 1
            print(
                f"[RESUME] Successfully restored state from Epoch {last_completed_epoch}."
            )
            print(
                f"[RESUME] Training will now continue starting from Epoch {start_epoch}."
            )
            print(
                f"[RESUME] Current Best Val Loss: {best_val:.6f} | Global Step: {global_step}"
            )

    for epoch in range(
        start_epoch,
        cfg["epochs"] + 1,
    ):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        model.train()

        epoch_loss_sum = 0.0
        epoch_tokens = 0.0

        progress = tqdm(
            train_loader,
            disable=not is_main,
            desc=f"{direction} Epoch {epoch}/{cfg['epochs']}",
        )

        best_checkpoint_dir = None

        optimizer.zero_grad(
            set_to_none=True
        )

        for step, batch in enumerate(progress):
            input_ids = batch.input_ids.to(device)

            attention_mask = (
                batch.attention_mask.to(device)
            )

            labels = batch.labels.to(device)

            amp_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=(
                        torch.bfloat16
                        if cfg.get(
                            "use_bf16",
                            True,
                        )
                        else torch.float16
                    ),
                )
                if torch.cuda.is_available()
                else nullcontext()
            )

            with amp_context:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss

                loss = (
                    loss
                    / cfg.get(
                        "gradient_accumulation_steps",
                        1,
                    )
                )

            if not torch.isfinite(loss):
                if is_main:
                    print(
                        f"[WARN] Non-finite loss detected: {loss.item()}"
                    )

                optimizer.zero_grad(
                    set_to_none=True
                )

                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (
                (step + 1)
                % cfg.get(
                    "gradient_accumulation_steps",
                    1,
                )
                == 0
            ):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.get(
                        "max_grad_norm",
                        1.0,
                    ),
                )

                if scaler.is_enabled():
                    scaler.step(optimizer)

                    scaler.update()

                else:
                    optimizer.step()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                global_step += 1

            toks = attention_mask.sum().item()

            # Re-scale loss back to the un-divided value for metric tracking.
            accum = cfg.get("gradient_accumulation_steps", 1)
            epoch_loss_sum += loss.item() * accum * toks

            epoch_tokens += toks

            fractional_epoch = (
                epoch
                + (
                    step
                    / max(
                        1,
                        len(train_loader),
                    )
                )
            )

            if is_main:
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    epoch=f"{fractional_epoch:.1f}",
                )

        train_loss = reduce_sum_scalar(
            epoch_loss_sum,
            device,
            world_size,
        ) / max(
            1.0,
            reduce_sum_scalar(
                epoch_tokens,
                device,
                world_size,
            ),
        )

        # Release the training loop's cached (but unused) CUDA allocator
        # blocks before generation. Beam search needs differently-shaped
        # tensors than training's backward/optimizer step left behind, and
        # without this the allocator can OOM on a fresh large allocation
        # even though aggregate "reserved" memory would have been enough.
        gc.collect()
        torch.cuda.empty_cache()

        val_loss = run_validation_loss(
            model=model,
            dataloader=val_loader,
            device=device,
            world_size=world_size,
        )

        # Compute separate train and val BLEU/CHRF++ metrics.
        # Train BLEU uses a capped sample set to keep cost manageable.
        train_bleu_max = cfg.get("max_train_bleu_samples", 256)
        train_bleu, train_chrf = compute_bleu_chrf(
            model=model,
            tokenizer=tokenizer,
            dataloader=train_loader,
            device=device,
            cfg=cfg,
            world_size=world_size,
            max_samples=train_bleu_max,
        )

        val_bleu, val_chrf = compute_bleu_chrf(
            model=model,
            tokenizer=tokenizer,
            dataloader=val_loader,
            device=device,
            cfg=cfg,
            world_size=world_size,
        )

        # Restore training mode after eval functions set model.eval().
        model.train()

        train_ppl = safe_exp(
            train_loss
        )

        val_ppl = safe_exp(
            val_loss
        )

        stop_training = False

        if is_main:
            active_model_path = cfg.get("offline_model_dir", cfg["model_name"])
            if active_model_path is None:
                active_model_path = cfg["model_name"]
            model_tag = Path(active_model_path).name

            if cfg.get(
                "use_lora",
                False,
            ):
                model_tag += "-LoRA"

            if cfg.get(
                "use_qlora",
                False,
            ):
                model_tag += "-QLoRA"

            msg = (
                f"[{model_tag}] Epoch {epoch} | "
                f"Train {train_loss:.4f} | "
                f"Val {val_loss:.4f} | "
                f"Train PPL {train_ppl:.4f} | "
                f"Val PPL {val_ppl:.4f} | "
                f"Train BLEU {train_bleu:.4f} | "
                f"Val BLEU {val_bleu:.4f} | "
                f"Train CHRF++ {train_chrf:.4f} | "
                f"Val CHRF++ {val_chrf:.4f} | "
                f"LR {scheduler.get_last_lr()[0]:.2e}"
            )

            print(msg)

            logger.info(msg)

            # ── CSV Metric Tracking ──────────────────────────────────────────
            metrics_csv = os.path.join(train_dir, "train_metrics.csv")
            new_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_ppl": train_ppl,
                "val_ppl": val_ppl,
                "train_bleu": train_bleu,
                "val_bleu": val_bleu,
                "train_chrf": train_chrf,
                "val_chrf": val_chrf,
                "lr": scheduler.get_last_lr()[0],
            }
            
            if not os.path.exists(metrics_csv):
                pd.DataFrame([new_row]).to_csv(metrics_csv, index=False)
            else:
                pd.DataFrame([new_row]).to_csv(metrics_csv, mode="a", header=False, index=False)

            # ── Academic Plotting ────────────────────────────────────────────
            try:
                generate_academic_plots(metrics_csv, train_dir, direction)
            except Exception as pe:
                logger.warning(f"[PLOT] Could not generate plots: {str(pe)}")

            if val_loss < best_val:
                best_val = val_loss

                wait = 0

                checkpoint_dir = os.path.join(
                    base_output_dir,
                    f"checkpoint-epoch-{epoch}-bleu-{val_bleu:.2f}",
                )

                prev_best_path = os.path.join(
                    base_output_dir,
                    "_best_checkpoint_path.txt",
                )

                if os.path.isfile(prev_best_path):
                    with open(prev_best_path, "r") as _f:
                        prev_best = _f.read().strip()
                    if os.path.isdir(prev_best) and prev_best != checkpoint_dir:
                        import shutil
                        shutil.rmtree(prev_best, ignore_errors=True)
                        logger.info(
                            f"[CHECKPOINT] Removed old best → {prev_best}"
                        )

                save_checkpoint(
                    save_dir=checkpoint_dir,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    best_val=best_val,
                    wait=wait,
                    cfg=cfg,
                )

                with open(prev_best_path, "w") as _f:
                    _f.write(checkpoint_dir)

                logger.info(
                    f"[CHECKPOINT] New best saved → {checkpoint_dir}"
                )

                best_checkpoint_dir = checkpoint_dir

            else:
                wait += 1


            # ── Always save the latest checkpoint (for resume safety) ──────
            latest_checkpoint_dir = os.path.join(
                base_output_dir,
                f"checkpoint-latest-epoch-{epoch}",
            )

            prev_latest_path = os.path.join(
                base_output_dir,
                "_latest_checkpoint_path.txt",
            )

            if os.path.isfile(prev_latest_path):
                with open(prev_latest_path, "r") as _f:
                    prev_latest = _f.read().strip()
                if (
                    os.path.isdir(prev_latest)
                    and prev_latest != best_checkpoint_dir
                ):
                    import shutil
                    shutil.rmtree(prev_latest, ignore_errors=True)
                    logger.info(
                        f"[CHECKPOINT] Removed old latest → {prev_latest}"
                    )

            save_checkpoint(
                save_dir=latest_checkpoint_dir,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_val=best_val,
                wait=wait,
                cfg=cfg,
            )

            with open(prev_latest_path, "w") as _f:
                _f.write(latest_checkpoint_dir)

            logger.info(
                f"[CHECKPOINT] Latest saved → {latest_checkpoint_dir}"
            )

            stop_training = (
                wait >= patience
            )

        if use_ddp:
            stop_tensor = torch.tensor(
                [
                    1
                    if stop_training
                    else 0
                ],
                device=device,
            )

            dist.broadcast(
                stop_tensor,
                src=0,
            )

            stop_training = bool(
                stop_tensor.item()
            )

        if stop_training:
            if is_main:
                print(
                    "[EARLY STOPPING] Triggered"
                )

            break

        if use_ddp:
            dist.barrier()

        gc.collect()

        torch.cuda.empty_cache()

def restore_checkpoint(
    checkpoint_dir,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
):
    trainer_state_path = os.path.join(
        checkpoint_dir,
        "trainer_state.pt",
    )

    if not os.path.isfile(
        trainer_state_path
    ):
        raise FileNotFoundError(
            trainer_state_path
        )

    state = torch.load(
        trainer_state_path,
        map_location=device,
        weights_only=False,  # RNG states require full pickle
    )

    optimizer.load_state_dict(
        state["optimizer"]
    )

    scheduler.load_state_dict(
        state["scheduler"]
    )

    if (
        scaler is not None
        and state["scaler"] is not None
    ):
        scaler.load_state_dict(
            state["scaler"]
        )

    torch.set_rng_state(
        state["rng_state"].cpu()
    )

    if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
        cpu_cuda_states = [s.cpu() if isinstance(s, torch.Tensor) else s for s in state["cuda_rng_state"]]
        torch.cuda.set_rng_state_all(
            cpu_cuda_states
        )

    start_epoch = state["epoch"] + 1

    global_step = state.get(
        "global_step",
        0,
    )

    best_val = state.get(
        "best_val",
        float("inf"),
    )

    wait = state.get(
        "wait",
        0,
    )

    return (
        start_epoch,
        global_step,
        best_val,
        wait,
    )


def find_latest_checkpoint(
    output_dir,
):
    if not os.path.isdir(output_dir):
        return None

    checkpoints = []

    for item in os.listdir(output_dir):
        full_path = os.path.join(
            output_dir,
            item,
        )

        if (
            os.path.isdir(full_path)
            and item.startswith(
                "checkpoint-epoch-"
            )
        ):
            checkpoints.append(full_path)

    if len(checkpoints) == 0:
        return None

    checkpoints = sorted(
        checkpoints,
        key=lambda x: os.path.getmtime(x),
    )

    return checkpoints[-1]


def main():
    seed_everything(
        GLOBAL_SEED
    )

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
            direction_name = direction_cfg[
                "direction_name"
            ]

            if rank == 0:
                print(
                    "\n"
                    + "=" * 20
                    + f" Training: {direction_name} "
                    + "=" * 20
                )
    
            print(f"[Rank {rank}] Starting training process for {direction_name}...")

            if use_ddp:
                dist.barrier()

            train_direction(
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
                "\n[INFO] Training interrupted by user"
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
        cleanup_ddp(use_ddp)


if __name__ == "__main__":
    main()