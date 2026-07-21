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
from typing import Dict, List, Optional

import pandas as pd
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

CONFIG_PATH = "configs/infer_config.json"


# ─── DDP helpers ──────────────────────────────────────────────────────────────

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


def gather_strings(strings: List[str], world_size: int) -> List[str]:
    """All-gather string lists across DDP ranks into a flat merged list."""
    if world_size == 1:
        return strings
    gathered = [None] * world_size
    dist.all_gather_object(gathered, strings)
    merged: List[str] = []
    for part in gathered:
        merged.extend(part)
    return merged


# ─── Logging ──────────────────────────────────────────────────────────────────

def init_logger(log_path: str, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(fh)
    return logger


class _NullLogger:
    """Drop-in no-op logger for non-main DDP ranks."""
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass


def _safe_logger(logger):
    return logger if logger is not None else _NullLogger()


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Model & tokenizer loading ────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: Dict, logger) -> tuple:
    """
    Load tokenizer and model.
    Priority: checkpoint_dir → offline_model_dir → HuggingFace online.
    Correctly handles full fine-tune vs LoRA/QLoRA checkpoints.
    """
    log = _safe_logger(logger)
    
    # Resolve rank for logging
    rank = int(os.environ.get("RANK", "0"))
    print(f"[Rank {rank}] Resolving checkpoint path for {cfg['direction_name']}...")
    base_output_dir = os.path.join("directions", cfg["direction_name"])
    checkpoint_dir = cfg.get(
        "checkpoint_dir",
        os.path.join(base_output_dir, "best-checkpoint")
    )

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

    model_name = cfg["model_name"]
    offline_dir = cfg.get("offline_model_dir", None)
    use_qlora = cfg.get("use_qlora", False)
    attn_impl = (
        "flash_attention_2"
        if cfg.get("use_flash_attention", False)
        else "eager"
    )

    quantization_config = None
    if use_qlora:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # ── Tokenizer: checkpoint → offline → online ───────────────────────────
    tokenizer_loaded = False
    try:
        print(f"[Rank {rank}] Loading tokenizer from checkpoint: {checkpoint_dir}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_dir, local_files_only=True, use_fast=False
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_dir, local_files_only=True, use_fast=True
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
                    offline_dir, local_files_only=True, use_fast=False
                )
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(
                    offline_dir, local_files_only=True, use_fast=True
                )
            log.info("[TOKENIZER] Loaded offline tokenizer")
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            log.info("[TOKENIZER] Loaded online tokenizer")

    # ── Model: LoRA path vs full fine-tune path ────────────────────────────
    use_lora = cfg.get("use_lora", False) or use_qlora

    if use_lora:
        # LoRA checkpoints contain only adapter weights — load base first.
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
        log.info(f"[LORA] Applying PEFT adapters from {checkpoint_dir}")
        model = PeftModel.from_pretrained(model, checkpoint_dir)

    else:
        # Full fine-tune: checkpoint contains complete model weights.
        log.info(
            f"[MODEL] Full fine-tune mode — loading checkpoint: {checkpoint_dir}"
        )
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
                log.info(
                    f"[MODEL] Falling back to online base model: {model_name}"
                )
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
    Re-apply SentencePiece tokenizer extension for infer.

    The extended tokenizer is normally baked into the checkpoint and loaded
    automatically.  This is a safety net: add_tokens() is idempotent, so
    calling it when tokens already exist is a no-op.

    Modes (per side, read from cfg):
      "nllb_default" → skip extension entirely
      "custom_spm"   → load SPM from tokenizer_extension_dir, add tokens
      "auto"         → same as custom_spm for infer (no corpus to check UNK)
    """
    import sentencepiece as spm  # local import

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

        if ext_dir is None:
            log.warning(
                f"[TOKENIZER] {side.upper()} mode={mode} but "
                "'tokenizer_extension_dir' not set — skipping"
            )
            continue

        spm_path = os.path.join(
            ext_dir,
            f"{cfg['direction_name']}_{side}.model",
        )
        if not os.path.isfile(spm_path):
            log.warning(
                f"[TOKENIZER] SPM model not found at {spm_path} — "
                "skipping (run train.py first to generate it)"
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


def prepare_model_for_infer(
    model,
    device: torch.device,
    use_ddp: bool,
    local_rank: int,
    cfg: Dict,
):
    """Move model to device and optionally wrap in DDP."""

    if not cfg.get("use_qlora", False):
        model.to(device)

    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
        )

    model.eval()
    return model


# ─── Dataset & collator ───────────────────────────────────────────────────────

class InferDataset(Dataset):
    """
    Holds flat source-text strings from the infer CSV.
    No reference column — we only need the source side.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        src_lang: str,
        max_source_length: int,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.max_source_length = max_source_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        src_text = str(self.texts[idx]).strip()
        self.tokenizer.src_lang = self.src_lang

        enc = self.tokenizer(
            text=src_text,
            truncation=True,
            max_length=self.max_source_length,
            padding=False,
            return_attention_mask=True,
        )
        # Store raw text to pass through the collator without tokenisation.
        enc["src_text"] = src_text
        return enc


class InferCollator:
    """
    Pads input_ids/attention_mask with DataCollatorForSeq2Seq.
    Passes raw src_text strings through as a plain list (no padding).
    """

    def __init__(self, tokenizer, model):
        self._pad = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
            label_pad_token_id=-100,
        )

    def __call__(self, batch: List[Dict]) -> Dict:
        src_texts = [x["src_text"] for x in batch]
        tensor_items = [
            {
                "input_ids": x["input_ids"],
                "attention_mask": x["attention_mask"],
            }
            for x in batch
        ]
        out = self._pad(tensor_items)
        out["src_texts"] = src_texts
        return out


def build_infer_dataloader(
    texts: List[str],
    tokenizer,
    model,
    cfg: Dict,
    rank: int,
    world_size: int,
    use_ddp: bool,
) -> DataLoader:
    dataset = InferDataset(
        texts=texts,
        tokenizer=tokenizer,
        src_lang=cfg["src_lang"],
        max_source_length=cfg.get("max_source_length", 256),
    )

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

    return DataLoader(
        dataset,
        batch_size=cfg.get("infer_batch_size", 8),
        sampler=sampler,
        shuffle=False,
        collate_fn=InferCollator(tokenizer=tokenizer, model=model),
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
        persistent_workers=cfg.get("persistent_workers", True),
        prefetch_factor=cfg.get("prefetch_factor", 2),
    )


# ─── Generation helpers ───────────────────────────────────────────────────────

def _resolve_forced_bos(tokenizer, cfg: Dict, model) -> int:
    """
    Safe 3-tier forced_bos_token_id resolution for NLLB-200.
    Handles standard NLLB codes AND tribal/extended codes registered
    via add_special_tokens() during tokenizer extension.
    """
    tgt_lang = cfg["tgt_lang"]
    lang_map = getattr(tokenizer, "lang_code_to_id", {})

    if tgt_lang in lang_map:
        return lang_map[tgt_lang]

    vocab = tokenizer.get_vocab()
    if tgt_lang in vocab:
        return vocab[tgt_lang]

    token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if token_id != tokenizer.unk_token_id:
        return token_id

    # Last resort: model's decoder_start_token_id
    raw = model.module if isinstance(model, DDP) else model
    fallback_id = raw.config.decoder_start_token_id
    if fallback_id == tokenizer.eos_token_id:
        raise ValueError(
            f"[INFER] tgt_lang '{tgt_lang}' has no registered token in the "
            "tokenizer, and decoder_start_token_id == eos_token_id "
            f"({fallback_id}). Forcing this as the first generated token makes "
            "every output empty. This checkpoint's tokenizer never learned this "
            "language tag — retrain with the fixed train.py "
            "(ensure_lang_tag_registered)."
        )
    return fallback_id


def _build_generate_kwargs(cfg: Dict, forced_bos_token_id: int) -> Dict:
    """Assemble generation kwargs with clean defaults."""
    kwargs: Dict = {
        "forced_bos_token_id": forced_bos_token_id,
        "max_new_tokens": cfg.get("generation_max_length", 128),
        "max_length": None,
        "use_cache": True,
    }

    use_sampling = cfg.get("use_sampling", False)
    if use_sampling:
        kwargs["do_sample"] = True
        kwargs["temperature"] = cfg.get("temperature", 1.0)
        top_k = cfg.get("top_k", 0)
        top_p = cfg.get("top_p", 1.0)
        if top_k > 0:
            kwargs["top_k"] = top_k
        if top_p < 1.0:
            kwargs["top_p"] = top_p
    else:
        kwargs["num_beams"] = cfg.get("beam_size", 5)
        kwargs["length_penalty"] = cfg.get("length_penalty", 1.0)

    no_repeat = cfg.get("no_repeat_ngram_size", 3)
    if no_repeat > 0:
        kwargs["no_repeat_ngram_size"] = no_repeat

    rep_penalty = cfg.get("repetition_penalty", 1.1)
    if rep_penalty != 1.0:
        kwargs["repetition_penalty"] = rep_penalty

    return kwargs


# ─── Core inference loop ──────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model,
    tokenizer,
    dataloader: DataLoader,
    device: torch.device,
    cfg: Dict,
    rank: int,
    world_size: int,
    logger,
) -> List[Dict[str, str]]:
    """
    Translate all batches in the dataloader.
    Gathers predictions across DDP ranks.
    Returns list of {"src": ..., "pred": ...} dicts on rank-0 only.
    Non-rank-0 returns an empty list.
    """
    log = _safe_logger(logger)

    forced_bos_token_id = _resolve_forced_bos(tokenizer, cfg, model)
    log.info(
        f"[INFER] forced_bos_token_id = {forced_bos_token_id} "
        f"for tgt_lang '{cfg['tgt_lang']}'"
    )

    gen_kwargs = _build_generate_kwargs(cfg, forced_bos_token_id)
    raw_model = model.module if isinstance(model, DDP) else model

    sources: List[str] = []
    predictions: List[str] = []

    progress = tqdm(
        dataloader,
        disable=(rank != 0),
        desc=f"{cfg['direction_name']} Inference",
        unit="batch",
        dynamic_ncols=True,
    )

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else nullcontext()
    )

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        src_texts: List[str] = batch["src_texts"]

        with amp_ctx:
            generated = raw_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        clean_preds = [p.strip() for p in decoded]

        sources.extend(src_texts)
        predictions.extend(clean_preds)

        if rank == 0:
            progress.set_postfix(
                translated=len(predictions),
                preview=clean_preds[0][:50] if clean_preds else "",
            )

    # Gather across DDP ranks so rank-0 has the full result set.
    sources = gather_strings(sources, world_size)
    predictions = gather_strings(predictions, world_size)

    if rank != 0:
        return []

    return [{"src": s, "pred": p} for s, p in zip(sources, predictions)]


# ─── Output writer ────────────────────────────────────────────────────────────

def write_output_csv(
    results: List[Dict[str, str]],
    output_path: str,
    tgt_col_display: str,
):
    """
    Write inference results to CSV.
    Columns: "Source" | <tgt_col_display>  (e.g. "Source" | "Bhili")
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Source", tgt_col_display])
        for row in results:
            writer.writerow([row["src"], row["pred"]])


# ─── Direction orchestrator ───────────────────────────────────────────────────

def infer_direction(
    rank: int,
    world_size: int,
    local_rank: int,
    use_ddp: bool,
    cfg: Dict,
):
    """
    Run full inference for one translation direction.
    Reads from cfg["infer_csv"] / cfg["src_col"].
    Writes to cfg["output_dir"]/predictions.csv with columns
    ["Source", cfg["tgt_col_display"]].
    """
    direction = cfg["direction_name"]
    is_main = rank == 0
    # ── Directory Hierarchy ───────────────────────────────────────────
    base_output_dir = os.path.join("directions", cfg["direction_name"])
    output_dir = os.path.join(base_output_dir, "infer")
    
    if is_main:
        os.makedirs(output_dir, exist_ok=True)

    logger = (
        init_logger(
            os.path.join(output_dir, "infer.log"),
            f"infer_{direction}",
        )
        if is_main
        else None
    )
    log = _safe_logger(logger)

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )

    if is_main:
        print(
            "\n"
            + "=" * 20
            + f" Inference: {direction} "
            + "=" * 20
        )
    
    print(f"[Rank {rank}] Starting inference for {direction}...")

    # ── Load CSV ───────────────────────────────────────────────────────────
    infer_csv = cfg["infer_csv"]
    src_col = cfg["src_col"]
    tgt_col_display = cfg.get("tgt_col_display", "Prediction")

    df = pd.read_csv(infer_csv)
    df = df.dropna(subset=[src_col])

    max_rows: Optional[int] = cfg.get("max_infer_rows_debug", None)
    if max_rows is not None:
        df = df.iloc[:max_rows]

    texts = df[src_col].astype(str).tolist()

    if is_main:
        print(
            f"[INFER] {len(texts)} sentences | "
            f"{src_col} → {tgt_col_display}"
        )
        log.info(
            f"[INFER] {len(texts)} sentences | "
            f"{src_col} → {tgt_col_display}"
        )

    # ── Model ──────────────────────────────────────────────────────────────
    tokenizer, model = load_model_and_tokenizer(cfg, logger)

    # Re-apply tokenizer extension (idempotent safety-net).
    tokenizer, model = _apply_tokenizer_extension(
        tokenizer=tokenizer,
        model=model,
        cfg=cfg,
        logger=logger,
    )

    model = prepare_model_for_infer(
        model=model,
        device=device,
        use_ddp=use_ddp,
        local_rank=local_rank,
        cfg=cfg,
    )

    # ── DataLoader ─────────────────────────────────────────────────────────
    dataloader = build_infer_dataloader(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        cfg=cfg,
        rank=rank,
        world_size=world_size,
        use_ddp=use_ddp,
    )

    if use_ddp:
        dist.barrier()

    # ── Inference ──────────────────────────────────────────────────────────
    results = run_inference(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        device=device,
        cfg=cfg,
        rank=rank,
        world_size=world_size,
        logger=logger,
    )

    # ── Save (rank-0 only) ─────────────────────────────────────────────────
    if is_main:
        output_path = os.path.join(output_dir, "predictions.csv")
        write_output_csv(
            results=results,
            output_path=output_path,
            tgt_col_display=tgt_col_display,
        )
        print(
            f"[INFER] Saved {len(results)} predictions → {output_path}"
        )
        log.info(
            f"[INFER] Saved {len(results)} predictions to {output_path}"
        )

    if use_ddp:
        dist.barrier()

    gc.collect()
    torch.cuda.empty_cache()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    rank, world_size, local_rank, use_ddp = setup_ddp()

    try:
        config = load_json(CONFIG_PATH)
        directions: List[Dict] = config["directions"]

        for direction_cfg in directions:
            if use_ddp:
                dist.barrier()

            infer_direction(
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
            print("\n[INFO] Inference interrupted by user")

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        if use_ddp and dist.is_initialized():
            try:
                dist.abort()
            except Exception:
                pass
        raise

    finally:
        cleanup_ddp(use_ddp)


if __name__ == "__main__":
    main()
