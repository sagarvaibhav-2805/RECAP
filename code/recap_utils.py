"""
Shared helpers reused by every training/eval script: seeding, DDP safety,
run-manifest saving. Kept in one place so DPO/GRPO/PPO scripts never duplicate
this logic (see IMPLEMENTATION_PLAN.md "Reproducibility" / "Deadlock-free DDP").
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Below this many rows, a tqdm bar is more noise than signal (matches
# recap_reward.py's TQDM_MIN_ITEMS) -- validation/eval calls on a full
# dataset are comfortably above this and show real progress.
TQDM_MIN_ITEMS = 200


def set_seed(seed: int, rank: int = 0) -> None:
    """Seeds Python/NumPy/Torch (CPU+CUDA) and transformers. rank offsets the
    seed deterministically so DDP workers shuffle data differently but
    reproducibly."""
    effective_seed = seed + rank
    random.seed(effective_seed)
    try:
        import numpy as np
        np.random.seed(effective_seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(effective_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(effective_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        import transformers
        transformers.set_seed(effective_seed)
    except ImportError:
        pass


def is_main_process() -> bool:
    """True if this is rank 0, or if we're not running under DDP at all.
    Use this to gate logging/checkpoint-saving/COMET-validation -- never
    forward/backward/optimizer-step."""
    try:
        from accelerate.state import PartialState
        return PartialState().is_main_process
    except Exception:
        return int(os.environ.get("RANK", "0")) == 0


def wait_for_everyone() -> None:
    """Barrier so fast processes don't race ahead of a slow one (e.g. during
    checkpoint save)."""
    try:
        from accelerate.state import PartialState
        PartialState().wait_for_everyone()
    except Exception:
        pass


def configure_ddp_env() -> None:
    """Set env vars that turn a silent NCCL hang into a crash, so a deadlock
    fails fast instead of burning HPC wall-time. Call once, before any CUDA
    call, at process start."""
    import config as cfg
    for key, value in cfg.DDP_ENV.items():
        os.environ.setdefault(key, value)


def make_accelerator():
    """Accelerator() with cfg.DDP_TIMEOUT_SECONDS actually applied to the
    process-group init -- plain Accelerator() ignores DDP_TIMEOUT_SECONDS
    entirely and falls back to torch's default (30 min), which defeats the
    "deadlock fails fast" goal for any step slower than that. Use this
    instead of calling Accelerator() directly in any script that trains
    under DDP."""
    from datetime import timedelta

    import config as cfg
    from accelerate import Accelerator, InitProcessGroupKwargs

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=cfg.DDP_TIMEOUT_SECONDS))
    return Accelerator(kwargs_handlers=[kwargs])


def _git_commit_hash() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _library_versions() -> dict[str, str]:
    versions = {}
    for name in ["torch", "transformers", "trl", "accelerate", "pandas"]:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


def _gpu_info() -> dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "count": torch.cuda.device_count(),
                "name": torch.cuda.get_device_name(0),
            }
    except ImportError:
        pass
    return {"count": 0, "name": None}


def config_hash(resolved_config: dict[str, Any]) -> str:
    """Stable hash of a resolved (JSON-serializable) config dict, used to
    detect drift between what a pair-file/checkpoint claims and what actually
    produced it."""
    blob = json.dumps(resolved_config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def generate_batch(
    model, tokenizer, sources: list[str], inference_config, batch_size: int = 32, desc: str = "Generating",
) -> list[str]:
    """One shared seq2seq generation routine, used by recap_infer.py AND the
    validation callbacks inside recap_train_dpo.py / recap_train_grpo.py /
    recap_train_ppo.py -- so "what we evaluate" and "what we deploy" (and
    what we validate mid-training) never drift apart."""
    import torch

    device = next(model.parameters()).device
    outputs: list[str] = []
    was_training = model.training  # restore afterward -- callers may invoke this mid-training-loop
    model.eval()
    show_progress = len(sources) >= TQDM_MIN_ITEMS and is_main_process()
    try:
        with torch.no_grad():
            batch_starts = range(0, len(sources), batch_size)
            for i in tqdm(batch_starts, desc=desc, disable=not show_progress,
                          total=(len(sources) + batch_size - 1) // batch_size):
                batch = sources[i:i + batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
                gen_kwargs = {"max_new_tokens": inference_config.max_new_tokens}
                if inference_config.strategy == "beam":
                    gen_kwargs["num_beams"] = inference_config.num_beams
                else:
                    gen_kwargs["num_beams"] = 1
                    gen_kwargs["do_sample"] = False
                generated = model.generate(**inputs, **gen_kwargs)
                outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    finally:
        model.train(was_training)
    return outputs


def save_training_state_meta(path: Path, step: int, extra: dict | None = None) -> None:
    """Small resume-metadata JSON (loop step counter, best-so-far composite)
    for hand-rolled training loops (GRPO, classic-API PPO) that don't get HF
    Trainer's built-in resume_from_checkpoint. Pair with
    accelerator.save_state(path)/load_state(path) for the actual model +
    optimizer + RNG weights -- that's the DDP-safe way to checkpoint those,
    accelerate handles the unwrapping/gathering itself."""
    if not is_main_process():
        return
    path.mkdir(parents=True, exist_ok=True)
    state = {"step": step}
    if extra:
        state.update(extra)
    with open(path / "training_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def load_training_state_meta(path: Path) -> dict[str, Any] | None:
    """Returns the saved {'step': ..., ...} dict if a resumable checkpoint
    exists at path, else None -- callers use the None case to mean "start
    fresh from update 0"."""
    state_file = path / "training_state.json"
    if not state_file.exists():
        return None
    with open(state_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_run_manifest(path: Path, resolved_config: dict[str, Any], seed: int, extra: dict | None = None) -> None:
    """Writes run_manifest.json next to a checkpoint: everything needed to
    retrace a reported number back to an exact reproducible run."""
    if not is_main_process():
        return
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "python_version": sys.version,
        "library_versions": _library_versions(),
        "gpu_info": _gpu_info(),
        "seed": seed,
        "resolved_config": resolved_config,
        "config_hash": config_hash(resolved_config),
    }
    if extra:
        manifest.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
