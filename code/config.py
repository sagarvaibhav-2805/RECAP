"""
RECAP central config -- single source of truth for the whole pipeline.

No stage script hardcodes a language, direction, model name, path, or
hyperparameter. Everything comes from here. New language/direction/ablation =
one new entry in this file, never a code change elsewhere.

See ../IMPLEMENTATION_PLAN.md for the full design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Data / plug-and-play lists
# ---------------------------------------------------------------------------

LANGUAGES = ["Bhili", "Gondi", "Mundari"]
DIRECTIONS = ["hi2tgt", "tgt2hi"]
MODEL_NAMES = ["nllb", "mt5", "qwen", "llama"]

# ---------------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MAHA_DATA_2_ROOT = ROOT / "maha_data_2"
SPLITS_ROOT = ROOT / "recap_splits"
CALIB_ROOT = ROOT / "recap_calib"
REWARDS_ROOT = ROOT / "recap_rewards"
PAIRS_ROOT = ROOT / "recap_pairs"
DPO_ROOT = ROOT / "recap_dpo"
GRPO_ROOT = ROOT / "recap_grpo"
PPO_ROOT = ROOT / "recap_ppo"
EVAL_ROOT = ROOT / "recap_eval"
REPORT_ROOT = ROOT / "recap_report"
HUMAN_EVAL_ROOT = ROOT / "recap_human_eval"


def maha_data_2_csv(lang: str, direction: str) -> Path:
    return MAHA_DATA_2_ROOT / lang.lower() / f"{direction}.csv"


def split_dir(lang: str, direction: str) -> Path:
    return SPLITS_ROOT / lang.lower() / direction


def calib_path(lang: str, direction: str) -> Path:
    return CALIB_ROOT / lang.lower() / direction / "stats.json"


def rewards_path(lang: str, direction: str) -> Path:
    return REWARDS_ROOT / lang.lower() / direction / "rewards.csv"


def pairs_dir(lang: str, direction: str) -> Path:
    return PAIRS_ROOT / lang.lower() / direction


def pairs_experiment_dir(lang: str, direction: str, experiment: str) -> Path:
    """Each reward-preset/experiment gets its own pairs_all.csv /
    pairs_balanced.csv -- different presets are NOT interchangeable."""
    return PAIRS_ROOT / lang.lower() / direction / experiment


def checkpoint_dir(stage_root: Path, lang: str, direction: str, experiment: str, seed: int) -> Path:
    """stage_root is one of DPO_ROOT / GRPO_ROOT / PPO_ROOT."""
    return stage_root / lang.lower() / direction / experiment / f"seed_{seed}" / "checkpoint"


def run_manifest_path(stage_root: Path, lang: str, direction: str, experiment: str, seed: int) -> Path:
    return stage_root / lang.lower() / direction / experiment / f"seed_{seed}" / "run_manifest.json"


def eval_report_path(lang: str, direction: str, experiment: str, seed: int) -> Path:
    return EVAL_ROOT / lang.lower() / direction / experiment / f"seed_{seed}" / "report.json"


def sft_checkpoint_path(lang: str, direction: str) -> Path:
    """The existing, already-trained mT5 SFT checkpoint for one direction --
    RECAP never retrains this, only loads it as the DPO/PPO/GRPO starting
    point. Matches the naming convention already used on the Pragya server
    (mt5_finetune/<Lang>/mt5-<lang>-<direction>-finetuned, no direction
    subfolder -- unlike llama_finetune, which does use one)."""
    return ROOT / "mt5_finetune" / lang / f"mt5-{lang.lower()}-{direction}-finetuned"


def best_checkpoint_path(lang: str, direction: str) -> Path:
    """Written by recap_evaluate.py after the main matrix is scored. This is a
    runtime lookup file, not a hand-edited config value -- avoids requiring a
    manual config.py edit after every evaluation run."""
    return EVAL_ROOT / lang.lower() / direction / "best_checkpoint.json"


# ---------------------------------------------------------------------------
# 3. Split settings (Stage 1)
# ---------------------------------------------------------------------------

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
SPLIT_SEED = 13

# ---------------------------------------------------------------------------
# 4. Calibration settings (Stage 2)
# ---------------------------------------------------------------------------

CALIBRATION_QUANTITIES = ["bleu", "chrf", "comet", "rep", "len"]
CALIBRATION_EPSILON = 1e-6

REPETITION_NGRAM_SIZE = 4  # word n-grams, for P_rep
LENGTH_METRIC = "char"     # "char" or "token", for P_len / rho_len

# ---------------------------------------------------------------------------
# 5. Reward presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardConfig:
    """Everything RewardEngine + PreferenceBuilder need to build one flavor of
    preference data. See IMPLEMENTATION_PLAN.md Central-Config #5 (a)/(b)."""

    name: str
    use_calibration: bool          # z-score standardize BLEU/chrF++/COMET/rep/len?
    w_quality: float
    w_rep: float
    w_len: float
    use_margin_filter: bool        # apply delta threshold?
    use_dedup: bool
    use_balanced_sampling: bool


# --- (a) Preference-construction cumulative ablation (paper Section 8.3, Table 8) ---
# NOTE: none of these use margin filtering until the last two steps -- that is
# the point of this sequence (it isolates each reward COMPONENT one at a time).
PREFERENCE_ABLATIONS = {
    "ablation_quality_only": RewardConfig(
        "ablation_quality_only", use_calibration=True,
        w_quality=1.0, w_rep=0.0, w_len=0.0,
        use_margin_filter=False, use_dedup=True, use_balanced_sampling=False,
    ),
    "ablation_quality_plus_rep": RewardConfig(
        "ablation_quality_plus_rep", use_calibration=True,
        w_quality=0.75, w_rep=0.25, w_len=0.0,
        use_margin_filter=False, use_dedup=True, use_balanced_sampling=False,
    ),
    "ablation_quality_plus_len": RewardConfig(
        "ablation_quality_plus_len", use_calibration=True,
        w_quality=0.75, w_rep=0.0, w_len=0.25,
        use_margin_filter=False, use_dedup=True, use_balanced_sampling=False,
    ),
    "ablation_full_reward": RewardConfig(
        "ablation_full_reward", use_calibration=True,
        w_quality=0.60, w_rep=0.25, w_len=0.15,
        use_margin_filter=False, use_dedup=True, use_balanced_sampling=False,
    ),
    "ablation_full_reward_margin": RewardConfig(
        "ablation_full_reward_margin", use_calibration=True,
        w_quality=0.60, w_rep=0.25, w_len=0.15,
        use_margin_filter=True, use_dedup=True, use_balanced_sampling=False,
    ),
    "ablation_full_recap": RewardConfig(
        "ablation_full_recap", use_calibration=True,
        w_quality=0.60, w_rep=0.25, w_len=0.15,
        use_margin_filter=True, use_dedup=True, use_balanced_sampling=True,
    ),
}

# --- (b) Main experiment matrix reward variants (paper Section 11.9, Table 7) ---
# NOTE: dpo_quality_only here KEEPS margin filter + balancing (unlike the
# ablation_quality_only above, which does not) -- this is the paper's
# deliberate distinction between the two ablation studies. See
# IMPLEMENTATION_PLAN.md Central-Config #5.
MAIN_MATRIX_REWARDS = {
    "dpo_raw": RewardConfig(
        "dpo_raw", use_calibration=False,
        w_quality=1.0, w_rep=0.0, w_len=0.0,
        use_margin_filter=False, use_dedup=True, use_balanced_sampling=False,
    ),
    "dpo_quality_only": RewardConfig(
        "dpo_quality_only", use_calibration=True,
        w_quality=1.0, w_rep=0.0, w_len=0.0,
        use_margin_filter=True, use_dedup=True, use_balanced_sampling=True,
    ),
    "dpo_no_confidence": RewardConfig(
        "dpo_no_confidence", use_calibration=True,
        w_quality=0.60, w_rep=0.25, w_len=0.15,
        use_margin_filter=False, use_dedup=True, use_balanced_sampling=True,
    ),
    "recap_dpo": RewardConfig(
        "recap_dpo", use_calibration=True,
        w_quality=0.60, w_rep=0.25, w_len=0.15,
        use_margin_filter=True, use_dedup=True, use_balanced_sampling=True,
    ),
}

REWARD_PRESETS = {**PREFERENCE_ABLATIONS, **MAIN_MATRIX_REWARDS}

# ---------------------------------------------------------------------------
# 6. Pair-mining settings (Stage 4 & 5)
# ---------------------------------------------------------------------------

MARGIN_DELTA = 0.10          # confidence margin (validation-selected; placeholder)
BALANCE_CAP_PER_TYPE = None  # None = no cap; else max pairs per generator-pair-type
BALANCE_MARGIN_BINS = 5      # number of reward-margin bins for Stage 5 balancing
MAX_PAIRS_PER_SOURCE = 6     # natural cap: C(4,2) = 6 unordered pairs per source

# ---------------------------------------------------------------------------
# 7. DPO settings (Stage 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DPOSettings:
    beta: float = 0.1
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    max_length: int = 256
    max_prompt_length: int = 128
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    eval_steps: int = 500
    save_steps: int = 500
    use_confidence_weighting: bool = False


DPO_SETTINGS = DPOSettings()

# ---------------------------------------------------------------------------
# 8. GRPO settings (Stage 7 -- scratch implementation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GRPOSettings:
    group_size: int = 2               # G in {2, 3}
    learning_rate: float = 1e-6
    kl_coef: float = 0.05
    clip_epsilon: float = 0.2
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 128
    num_updates: int = 2000
    per_device_batch_size: int = 4     # sources per step (each yields group_size completions)
    eval_steps: int = 200
    save_steps: int = 200
    source_subset_size: int = 50_000   # fixed subset per direction (paper Section 8.6)


GRPO_SETTINGS = GRPOSettings()

# NOT YET WIRED -- deferred to a separate scope. These lists (and the 6 other
# targeted ablations from IMPLEMENTATION_PLAN.md Central-Config #10:
# candidate-diversity, calibration-method, confidence/delta-grid,
# pair-selection-strategy, weight-sensitivity) are not runnable via
# --experiment yet -- only the 8-condition main matrix and the 6-step
# preference-construction ablation are. Building this out means: per-preset
# GRPO_SETTINGS overrides (group_size, source_subset_size), a swept
# MARGIN_DELTA per experiment (currently one global cfg.MARGIN_DELTA), and
# corresponding entries in EXPERIMENTS/REWARD_PRESETS.
GRPO_DATA_SIZE_ABLATION = [10_000, 25_000, 50_000]
GRPO_GROUP_SIZE_ABLATION = [2, 3]

# ---------------------------------------------------------------------------
# 9. PPO settings (Stage 8 -- TRL-based)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PPOSettings:
    learning_rate: float = 1e-6
    init_kl_coef: float = 0.05
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    gamma: float = 1.0
    lam: float = 0.95
    batch_size: int = 32
    mini_batch_size: int = 4
    ppo_epochs: int = 4
    max_new_tokens: int = 128
    num_updates: int = 2000
    eval_steps: int = 200
    save_steps: int = 200


PPO_SETTINGS = PPOSettings()

# ---------------------------------------------------------------------------
# 10. Experiment registry -- main matrix + targeted ablations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Experiment:
    name: str
    trainer: str            # "sft" | "dpo" | "grpo" | "ppo"
    reward_preset: str | None = None   # key into REWARD_PRESETS, None for sft/ppo/pure-grpo
    init_from: str | None = None       # None = SFT checkpoint, else another experiment name
    grpo_group_size: int | None = None  # NOT YET CONSUMED -- see GRPO_GROUP_SIZE_ABLATION note above


# Main matrix -- 8 conditions, paper Section 11.9 minimum core matrix.
EXPERIMENTS: dict[str, Experiment] = {
    "sft": Experiment("sft", trainer="sft"),
    "dpo_raw": Experiment("dpo_raw", trainer="dpo", reward_preset="dpo_raw"),
    "dpo_quality_only": Experiment("dpo_quality_only", trainer="dpo", reward_preset="dpo_quality_only"),
    "dpo_no_confidence": Experiment("dpo_no_confidence", trainer="dpo", reward_preset="dpo_no_confidence"),
    "recap_dpo": Experiment("recap_dpo", trainer="dpo", reward_preset="recap_dpo"),
    "sft_ppo": Experiment("sft_ppo", trainer="ppo"),
    "sft_grpo": Experiment("sft_grpo", trainer="grpo", init_from=None),
    "recap_dpo_grpo": Experiment("recap_dpo_grpo", trainer="grpo", init_from="recap_dpo"),
}

# Preference-construction cumulative ablation -- each trains its own DPO run
# from the same SFT checkpoint, for Table 8.
for _name in PREFERENCE_ABLATIONS:
    EXPERIMENTS[_name] = Experiment(_name, trainer="dpo", reward_preset=_name)

MAIN_MATRIX_ORDER = [
    "sft", "dpo_raw", "dpo_quality_only", "dpo_no_confidence",
    "recap_dpo", "sft_ppo", "sft_grpo", "recap_dpo_grpo",
]
PREFERENCE_ABLATION_ORDER = list(PREFERENCE_ABLATIONS.keys())

# ---------------------------------------------------------------------------
# 11. Reproducibility settings
# ---------------------------------------------------------------------------

SEED = 13
SEEDS = [13, 42, 2026]

# ---------------------------------------------------------------------------
# 12. DDP / distributed settings
# ---------------------------------------------------------------------------

DDP_BACKEND = "nccl"
DDP_TIMEOUT_SECONDS = 1800
DDP_ENV = {"NCCL_ASYNC_ERROR_HANDLING": "1"}

# ---------------------------------------------------------------------------
# 13. Inference config -- shared by recap_infer.py and recap_evaluate.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceConfig:
    strategy: str = "greedy"   # "greedy" | "beam"
    num_beams: int = 1
    max_new_tokens: int = 128


INFERENCE_CONFIG = InferenceConfig()

# ---------------------------------------------------------------------------
# 14. Best-checkpoint registry -- see best_checkpoint_path() above.
# Not a hardcoded dict here: recap_evaluate.py writes one JSON file per
# direction after the main matrix is scored, and recap_infer.py reads it at
# runtime. Keeps config.py free of results that change every time eval reruns.
# ---------------------------------------------------------------------------
