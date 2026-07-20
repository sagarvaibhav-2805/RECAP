import os
import re

def patch_eval():
    with open('eval.py', 'r') as f:
        content = f.read()
    
    # Replace call sequence
    old_call = '''    print(f"[Rank {rank}] Loading model and tokenizer for {direction}...")

    tokenizer, model = load_model_and_tokenizer(cfg, logger)

    tokenizer, model = _apply_tokenizer_extension(
        tokenizer=tokenizer,
        model=model,
        cfg=cfg,
        logger=logger,
    )

    model = prepare_model_for_eval('''
    
    new_call = '''    print(f"[Rank {rank}] Loading model and tokenizer for {direction}...")

    tokenizer = load_tokenizer(cfg, logger)

    tokenizer = _apply_tokenizer_extension(
        tokenizer=tokenizer,
        cfg=cfg,
        logger=logger,
    )
    
    model = load_model(cfg, tokenizer, logger)

    model = prepare_model_for_eval('''
    content = content.replace(old_call, new_call)
    
    # Replace truncation
    old_trunc = '''    sentence_chrfs = gather_object(
        sentence_chrfs,
        world_size,
    )

    if rank != 0:
        return None'''
        
    new_trunc = '''    sentence_chrfs = gather_object(
        sentence_chrfs,
        world_size,
    )

    total_samples = len(dataloader.dataset)
    if rank == 0:
        predictions = predictions[:total_samples]
        references = references[:total_samples]
        sources = sources[:total_samples]
        sentence_bleus = sentence_bleus[:total_samples]
        sentence_chrfs = sentence_chrfs[:total_samples]
    else:
        return None'''
    content = content.replace(old_trunc, new_trunc)
    
    # Now replace the function definitions.
    # We will use regex to find load_model_and_tokenizer and _apply_tokenizer_extension and replace them.
    # From 'def load_model_and_tokenizer(' to '    return tokenizer, model\n'
    
    pattern = re.compile(r'def load_model_and_tokenizer\(.*?return tokenizer, model\n', re.DOTALL)
    
    new_funcs = '''def load_tokenizer(
    cfg: dict,
    logger,
):
    base_output_dir = os.path.join("directions", cfg["direction_name"])
    checkpoint_dir = cfg.get(
        "checkpoint_dir",
        os.path.join(base_output_dir, "best-checkpoint")
    )
    rank = int(os.environ.get("RANK", "0"))
    
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
    log = _safe_logger(logger)

    from transformers import AutoTokenizer
    tokenizer_loaded = False

    try:
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(
    cfg: dict,
    tokenizer,
    logger,
):
    import torch
    from transformers import AutoConfig, AutoModelForSeq2SeqLM
    from peft import PeftModel
    
    base_output_dir = os.path.join("directions", cfg["direction_name"])
    checkpoint_dir = cfg.get(
        "checkpoint_dir",
        os.path.join(base_output_dir, "best-checkpoint")
    )
    rank = int(os.environ.get("RANK", "0"))

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
    attn_impl = "flash_attention_2" if cfg.get("use_flash_attention", False) else "eager"
    quantization_config = None

    if use_qlora:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    log = _safe_logger(logger)
    use_lora = cfg.get("use_lora", False) or cfg.get("use_qlora", False)

    if use_lora:
        log.info("[MODEL] LoRA/QLoRA mode — loading base model first")
        try:
            if offline_dir is None or not os.path.isdir(offline_dir):
                raise RuntimeError("Offline model unavailable")
            config = AutoConfig.from_pretrained(offline_dir)
            config.tie_word_embeddings = True
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
            config.tie_word_embeddings = True
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
                attn_implementation=attn_impl,
            )
            
        model.resize_token_embeddings(len(tokenizer))
        
        log.info(f"[LORA] Applying PEFT adapters from {checkpoint_dir}")
        model = PeftModel.from_pretrained(model, checkpoint_dir)
    else:
        log.info(f"[MODEL] Full fine-tune mode — loading checkpoint: {checkpoint_dir}")
        try:
            config = AutoConfig.from_pretrained(checkpoint_dir, local_files_only=True)
            config.tie_word_embeddings = True
            model = AutoModelForSeq2SeqLM.from_pretrained(
                checkpoint_dir,
                config=config,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
                attn_implementation=attn_impl,
            )
            log.info("[MODEL] Loaded checkpoint successfully")
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
                
        model.resize_token_embeddings(len(tokenizer))

    model.eval()
    return model


def _apply_tokenizer_extension(
    tokenizer,
    cfg: dict,
    logger,
):
    import sentencepiece as spm

    log = _safe_logger(logger)
    ext_dir = cfg.get(
        "tokenizer_extension_dir",
        os.path.join("tok_extensions", cfg["direction_name"])
    )

    for side in ("src", "tgt"):
        mode = cfg.get(f"{side}_tokenizer", "mt5_default")

        if mode == "mt5_default":
            log.info(f"[TOKENIZER] {side.upper()} mode=mt5_default — no extension applied")
            continue

        if ext_dir is None:
            log.warning(f"[TOKENIZER] {side.upper()} mode={mode} but 'tokenizer_extension_dir' not set in config — skipping")
            continue

        spm_path = os.path.join(ext_dir, f"{cfg['direction_name']}_{side}.model")
        if not os.path.isfile(spm_path):
            log.warning(f"[TOKENIZER] SPM model not found at {spm_path} — skipping (train first to generate the SPM model)")
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
            tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
            log.info(f"[TOKENIZER] [{side.upper()}] Registered language tags: {new_special_tokens}")

        if new_regular_tokens:
            tokenizer.add_tokens(new_regular_tokens)
            log.info(f"[TOKENIZER] [{side.upper()}] Added {len(new_regular_tokens)} subword tokens")
        else:
            log.info(f"[TOKENIZER] [{side.upper()}] All SPM tokens already in vocab — checkpoint tokenizer was pre-extended (no-op)")

    return tokenizer
'''
    content = pattern.sub(new_funcs, content)
    
    with open('eval.py', 'w') as f:
        f.write(content)
        
    print("Patched eval.py")

def patch_infer():
    with open('infer.py', 'r') as f:
        content = f.read()
    
    # Replace call sequence
    old_call = '''    print(f"[Rank {rank}] Loading model and tokenizer for {direction}...")

    tokenizer, model = load_model_and_tokenizer(cfg, logger)

    tokenizer, model = _apply_tokenizer_extension(
        tokenizer=tokenizer,
        model=model,
        cfg=cfg,
        logger=logger,
    )

    model = prepare_model_for_infer('''
    
    new_call = '''    print(f"[Rank {rank}] Loading model and tokenizer for {direction}...")

    tokenizer = load_tokenizer(cfg, logger)

    tokenizer = _apply_tokenizer_extension(
        tokenizer=tokenizer,
        cfg=cfg,
        logger=logger,
    )
    
    model = load_model(cfg, tokenizer, logger)

    model = prepare_model_for_infer('''
    content = content.replace(old_call, new_call)
    
    # Replace truncation
    old_trunc = '''    sources = gather_strings(
        sources,
        world_size,
    )

    predictions = gather_strings(
        predictions,
        world_size,
    )

    if rank != 0:
        return None'''
        
    new_trunc = '''    sources = gather_strings(
        sources,
        world_size,
    )

    predictions = gather_strings(
        predictions,
        world_size,
    )

    total_samples = len(dataloader.dataset)
    if rank == 0:
        sources = sources[:total_samples]
        predictions = predictions[:total_samples]
    else:
        return None'''
    content = content.replace(old_trunc, new_trunc)
    
    # Replace the functions in infer.py exactly as in eval.py
    pattern = re.compile(r'def load_model_and_tokenizer\(.*?return tokenizer, model\n', re.DOTALL)
    
    new_funcs = '''def load_tokenizer(
    cfg: dict,
    logger,
):
    base_output_dir = os.path.join("directions", cfg["direction_name"])
    checkpoint_dir = cfg.get(
        "checkpoint_dir",
        os.path.join(base_output_dir, "best-checkpoint")
    )
    rank = int(os.environ.get("RANK", "0"))
    
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
    log = _safe_logger(logger)

    from transformers import AutoTokenizer
    tokenizer_loaded = False

    try:
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(
    cfg: dict,
    tokenizer,
    logger,
):
    import torch
    from transformers import AutoConfig, AutoModelForSeq2SeqLM
    from peft import PeftModel
    
    base_output_dir = os.path.join("directions", cfg["direction_name"])
    checkpoint_dir = cfg.get(
        "checkpoint_dir",
        os.path.join(base_output_dir, "best-checkpoint")
    )
    rank = int(os.environ.get("RANK", "0"))

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
    attn_impl = "flash_attention_2" if cfg.get("use_flash_attention", False) else "eager"
    quantization_config = None

    if use_qlora:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    log = _safe_logger(logger)
    use_lora = cfg.get("use_lora", False) or cfg.get("use_qlora", False)

    if use_lora:
        log.info("[MODEL] LoRA/QLoRA mode — loading base model first")
        try:
            if offline_dir is None or not os.path.isdir(offline_dir):
                raise RuntimeError("Offline model unavailable")
            config = AutoConfig.from_pretrained(offline_dir)
            config.tie_word_embeddings = True
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
            config.tie_word_embeddings = True
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
                attn_implementation=attn_impl,
            )
            
        model.resize_token_embeddings(len(tokenizer))
        
        log.info(f"[LORA] Applying PEFT adapters from {checkpoint_dir}")
        model = PeftModel.from_pretrained(model, checkpoint_dir)
    else:
        log.info(f"[MODEL] Full fine-tune mode — loading checkpoint: {checkpoint_dir}")
        try:
            config = AutoConfig.from_pretrained(checkpoint_dir, local_files_only=True)
            config.tie_word_embeddings = True
            model = AutoModelForSeq2SeqLM.from_pretrained(
                checkpoint_dir,
                config=config,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
                attn_implementation=attn_impl,
            )
            log.info("[MODEL] Loaded checkpoint successfully")
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
                
        model.resize_token_embeddings(len(tokenizer))

    model.eval()
    return model


def _apply_tokenizer_extension(
    tokenizer,
    cfg: dict,
    logger,
):
    import sentencepiece as spm

    log = _safe_logger(logger)
    ext_dir = cfg.get(
        "tokenizer_extension_dir",
        os.path.join("tok_extensions", cfg["direction_name"])
    )

    for side in ("src", "tgt"):
        mode = cfg.get(f"{side}_tokenizer", "mt5_default")

        if mode == "mt5_default":
            log.info(f"[TOKENIZER] {side.upper()} mode=mt5_default — no extension applied")
            continue

        if ext_dir is None:
            log.warning(f"[TOKENIZER] {side.upper()} mode={mode} but 'tokenizer_extension_dir' not set in config — skipping")
            continue

        spm_path = os.path.join(ext_dir, f"{cfg['direction_name']}_{side}.model")
        if not os.path.isfile(spm_path):
            log.warning(f"[TOKENIZER] SPM model not found at {spm_path} — skipping (train first to generate the SPM model)")
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
            tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
            log.info(f"[TOKENIZER] [{side.upper()}] Registered language tags: {new_special_tokens}")

        if new_regular_tokens:
            tokenizer.add_tokens(new_regular_tokens)
            log.info(f"[TOKENIZER] [{side.upper()}] Added {len(new_regular_tokens)} subword tokens")
        else:
            log.info(f"[TOKENIZER] [{side.upper()}] All SPM tokens already in vocab — checkpoint tokenizer was pre-extended (no-op)")

    return tokenizer
'''
    content = pattern.sub(new_funcs, content)
    
    with open('infer.py', 'w') as f:
        f.write(content)
        
    print("Patched infer.py")

if __name__ == '__main__':
    patch_eval()
    patch_infer()
