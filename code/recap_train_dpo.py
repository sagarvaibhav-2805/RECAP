"""
Stage 6 -- offline DPO training with TRL's DPOTrainer, for any of the DPO
reward-preset experiments (dpo_raw, dpo_quality_only, dpo_no_confidence,
recap_dpo, or any ablation_* preset).

pi_theta and pi_ref both start from the existing, frozen-elsewhere mT5 SFT
checkpoint for this direction (never retrained here). Checkpoint selection is
by validation BLEU/chrF++/COMET (subject to repetition/length not regressing),
not by training loss.

Multi-GPU: launch with `accelerate launch --num_processes=<N> recap_train_dpo.py ...`
-- DPOTrainer is already accelerate-based, no manual DDP wrapping needed here.

NOTE: TRL's public API has changed across versions. This targets a
DPOConfig + DPOTrainer(model, ref_model, args, train_dataset, processing_class)
signature (trl >= 0.9-ish). Verify against the exact pinned `trl` version
before the first real run (see IMPLEMENTATION_PLAN.md Stage 6).

Run:
    python recap_train_dpo.py --lang Bhili --direction hi2tgt --experiment recap_dpo
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict

import pandas as pd

import config as cfg
import recap_checks
import recap_utils
from recap_reward import RewardEngine


class ValidationCheckpointCallback:
    """Periodically decodes the validation split, scores BLEU/chrF++/COMET,
    and tracks the best checkpoint by a simple composite -- subject to
    repetition/length not regressing beyond the SFT baseline. Wired into
    TRL's DPOTrainer via a real transformers.TrainerCallback (on_step_end),
    so this actually runs DURING training, not just once at the end -- the
    checkpoint saved is genuinely the best-on-validation one, per the paper's
    explicit "select by validation, not training loss or last step" rule."""

    def __init__(self, val_sources, val_refs, reward_engine: RewardEngine, out_dir, tokenizer, eval_steps: int):
        self.val_sources = val_sources
        self.val_refs = val_refs
        self.reward_engine = reward_engine
        self.out_dir = out_dir
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.best_composite = float("-inf")
        self.best_step = None
        self.last_result: dict = {}

    def evaluate(self, model, step) -> dict:
        translations = recap_utils.generate_batch(model, self.tokenizer, self.val_sources, cfg.INFERENCE_CONFIG)
        scored = self.reward_engine.compute_raw_metrics(self.val_sources, translations, self.val_refs)
        valid = [s for s in scored if s["valid"] and math.isfinite(s["bleu"])]
        if not valid:
            result = {"step": step, "composite": float("-inf"), "n_valid": 0}
        else:
            bleu = sum(s["bleu"] for s in valid) / len(valid)
            chrf = sum(s["chrf"] for s in valid) / len(valid)
            comet = sum(s["comet"] for s in valid) / len(valid)
            rep = sum(s["rep"] for s in valid) / len(valid)
            composite = (bleu + chrf + comet) / 3.0
            result = {"step": step, "bleu": bleu, "chrf": chrf, "comet": comet, "rep": rep,
                      "composite": composite, "n_valid": len(valid)}
        self.last_result = result
        if recap_utils.is_main_process():
            print(f"[Eval] step={step} validation composite={result['composite']:.4f} (n_valid={result.get('n_valid', 0)})")
        if result["composite"] > self.best_composite:
            self.best_composite = result["composite"]
            self.best_step = step
            if recap_utils.is_main_process():
                model.save_pretrained(self.out_dir)
                self.tokenizer.save_pretrained(self.out_dir)
        return result


def _make_trl_callback(val_callback: ValidationCheckpointCallback):
    """Thin transformers.TrainerCallback adapter -- on_step_end fires inside
    TRL's blocking trainer.train() loop, which is what actually makes
    periodic validation happen during training."""
    from transformers import TrainerCallback

    class _Adapter(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step > 0 and state.global_step % val_callback.eval_steps == 0:
                if recap_utils.is_main_process():
                    val_callback.evaluate(model, state.global_step)
                recap_utils.wait_for_everyone()
            return control

    return _Adapter()


def build_dpo_dataset(pairs_path):
    from datasets import Dataset
    df = pd.read_csv(pairs_path)
    df = df.rename(columns={"source": "prompt"})[["prompt", "chosen", "rejected"]]
    return Dataset.from_pandas(df, preserve_index=False)


def process_one(lang: str, direction: str, experiment: str, seed: int) -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    exp = cfg.EXPERIMENTS[experiment]
    if exp.trainer != "dpo":
        raise ValueError(f"{experiment} is not a DPO experiment (trainer={exp.trainer})")

    checkpoint_dir = cfg.checkpoint_dir(cfg.DPO_ROOT, lang, direction, experiment, seed)
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        print(f"[Skip] {lang}/{direction}/{experiment}/seed_{seed}: checkpoint already exists")
        return

    pairs_path = cfg.pairs_experiment_dir(lang, direction, experiment) / "pairs_balanced.csv"
    if not pairs_path.exists():
        print(f"[Skip] {lang}/{direction}/{experiment}: {pairs_path} not found -- run Stages 1-5 first")
        return

    recap_utils.configure_ddp_env()
    recap_utils.set_seed(seed)

    recap_checks.run_static_checks()

    sft_checkpoint = cfg.sft_checkpoint_path(lang, direction)
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
    policy_model = AutoModelForSeq2SeqLM.from_pretrained(sft_checkpoint)
    ref_model = AutoModelForSeq2SeqLM.from_pretrained(sft_checkpoint)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    ref_frozen_check = recap_checks.check_reference_frozen(ref_model)
    if not ref_frozen_check.ok:
        raise recap_checks.CheckFailure(f"{lang}/{direction}/{experiment} reference_frozen: {ref_frozen_check.message}")

    train_dataset = build_dpo_dataset(pairs_path)

    settings = cfg.DPO_SETTINGS
    training_args = DPOConfig(
        output_dir=str(checkpoint_dir.parent / "trainer_state"),
        beta=settings.beta,
        learning_rate=settings.learning_rate,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        num_train_epochs=settings.num_train_epochs,
        max_length=settings.max_length,
        max_prompt_length=settings.max_prompt_length,
        warmup_ratio=settings.warmup_ratio,
        max_grad_norm=settings.max_grad_norm,
        save_steps=settings.save_steps,
        logging_steps=50,
        seed=seed,
        report_to=[],
        # transformers.TrainingArguments' own DDP-hang safety valve (DPOConfig
        # inherits it) -- without this, a slow step under multi-GPU falls
        # back to torch's default 30-min NCCL timeout instead of
        # cfg.DDP_TIMEOUT_SECONDS. Verify this field name against the pinned
        # transformers/trl version.
        ddp_timeout=cfg.DDP_TIMEOUT_SECONDS,
    )

    val_df = pd.read_csv(cfg.split_dir(lang, direction) / "val.csv")
    calib_engine = RewardEngine.load(cfg.calib_path(lang, direction), reward_config=cfg.REWARD_PRESETS[exp.reward_preset])
    val_callback = ValidationCheckpointCallback(
        val_df["source"].tolist(), val_df["gold_truth"].tolist(),
        calib_engine, checkpoint_dir, tokenizer, eval_steps=settings.eval_steps,
    )

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.add_callback(_make_trl_callback(val_callback))

    trainer.train()

    # If eval_steps never lined up with a step boundary (e.g. very short
    # runs, or a small ablation preset with few pairs), make sure at least
    # one validation pass -- and a saved checkpoint -- happened before we
    # declare done. Gated exactly like the on_step_end callback: only rank 0
    # runs generate()/COMET, everyone else just waits at the barrier --
    # otherwise every DDP rank would redundantly re-run validation here.
    if val_callback.best_step is None:
        if recap_utils.is_main_process():
            val_callback.evaluate(trainer.model, trainer.state.global_step)
        recap_utils.wait_for_everyone()
    final_metrics = val_callback.last_result
    print(f"[Eval] {lang}/{direction}/{experiment}: best validation composite="
          f"{val_callback.best_composite:.4f} (step={val_callback.best_step})")

    resolved_config = {
        "lang": lang, "direction": direction, "experiment": experiment, "seed": seed,
        "reward_preset": exp.reward_preset,
        "dpo_settings": asdict(settings),
        "sft_checkpoint": str(sft_checkpoint),
    }
    manifest_path = cfg.run_manifest_path(cfg.DPO_ROOT, lang, direction, experiment, seed)
    recap_utils.save_run_manifest(
        manifest_path, resolved_config, seed,
        extra={"best_validation_composite": val_callback.best_composite, "best_step": val_callback.best_step,
               "final_validation": final_metrics},
    )
    if recap_utils.is_main_process():
        with open(manifest_path) as f:
            manifest_check = recap_checks.check_manifest_hashed(json.load(f))
        if not manifest_check.ok:
            raise recap_checks.CheckFailure(f"{lang}/{direction}/{experiment} manifest_hashed: {manifest_check.message}")
    print(f"[Done] {lang}/{direction}/{experiment}/seed_{seed} -> {checkpoint_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=cfg.LANGUAGES, required=True)
    parser.add_argument("--direction", choices=cfg.DIRECTIONS, required=True)
    parser.add_argument("--experiment", choices=[n for n, e in cfg.EXPERIMENTS.items() if e.trainer == "dpo"], required=True)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    args = parser.parse_args()
    process_one(args.lang, args.direction, args.experiment, args.seed)


if __name__ == "__main__":
    main()
