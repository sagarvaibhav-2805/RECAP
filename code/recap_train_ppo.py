"""
Stage 8 -- PPO baseline with TRL's PPOTrainer. Independent RL baseline: starts
from the SFT checkpoint (not from DPO), reuses the SAME fixed RewardEngine
reward as DPO/GRPO (no separately trained reward model), and does not use the
static four-model candidate pool as PPO actions -- every rollout is a fresh
sample from the current policy.

NOTE: targets the classic seq2seq-compatible TRL PPO API
(AutoModelForSeq2SeqLMWithValueHead + PPOTrainer(config, model, ref_model,
tokenizer) + ppo_trainer.step(queries, responses, scores)). TRL's PPOTrainer
API has changed across versions and the newer rewrite skews causal-LM-first;
verify this against the exact pinned `trl` version before the first real run
(see IMPLEMENTATION_PLAN.md Stage 8 / Section 11.5 reconnaissance step).

Run:
    python recap_train_ppo.py --lang Bhili --direction hi2tgt --experiment sft_ppo
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


def process_one(lang: str, direction: str, experiment: str, seed: int) -> None:
    from datetime import timedelta

    import torch
    from accelerate import InitProcessGroupKwargs
    from transformers import AutoTokenizer
    from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model

    exp = cfg.EXPERIMENTS[experiment]
    if exp.trainer != "ppo":
        raise ValueError(f"{experiment} is not a PPO experiment (trainer={exp.trainer})")

    checkpoint_dir = cfg.checkpoint_dir(cfg.PPO_ROOT, lang, direction, experiment, seed)
    # Completion is judged by run_manifest.json (written only at the very end
    # of a successful run), NOT by checkpoint_dir having files -- a resumed
    # run can have a partial best-checkpoint saved there already while
    # training is still genuinely in progress (see resume logic below).
    manifest_path = cfg.run_manifest_path(cfg.PPO_ROOT, lang, direction, experiment, seed)
    if manifest_path.exists():
        print(f"[Skip] {lang}/{direction}/{experiment}/seed_{seed}: already completed (manifest exists)")
        return

    train_path = cfg.split_dir(lang, direction) / "train.csv"
    calib_path = cfg.calib_path(lang, direction)
    if not (train_path.exists() and calib_path.exists()):
        print(f"[Skip] {lang}/{direction}: missing split/calibration -- run Stages 1-2 first")
        return

    recap_utils.configure_ddp_env()
    recap_utils.set_seed(seed)

    sft_checkpoint = str(cfg.sft_checkpoint_path(lang, direction))
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(sft_checkpoint)
    ref_model = create_reference_model(model)  # frozen copy, PPO's KL-control reference

    recap_checks.run_static_checks()
    ref_frozen_check = recap_checks.check_reference_frozen(ref_model)
    if not ref_frozen_check.ok:
        raise recap_checks.CheckFailure(f"{lang}/{direction}/{experiment} reference_frozen: {ref_frozen_check.message}")

    settings = cfg.PPO_SETTINGS
    ppo_config = PPOConfig(
        learning_rate=settings.learning_rate,
        batch_size=settings.batch_size,
        mini_batch_size=settings.mini_batch_size,
        ppo_epochs=settings.ppo_epochs,
        init_kl_coef=settings.init_kl_coef,
        cliprange=settings.cliprange,
        cliprange_value=settings.cliprange_value,
        vf_coef=settings.vf_coef,
        gamma=settings.gamma,
        lam=settings.lam,
        seed=seed,
        # Classic TRL PPOConfig forwards accelerator_kwargs straight to its
        # internal Accelerator(**accelerator_kwargs) -- without this, the
        # same "hang forever past torch's 30-min default" risk applies here
        # as everywhere else under DDP. Verify this field name against the
        # pinned trl version.
        accelerator_kwargs={
            "kwargs_handlers": [InitProcessGroupKwargs(timeout=timedelta(seconds=cfg.DDP_TIMEOUT_SECONDS))],
        },
    )
    ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(cfg.split_dir(lang, direction) / "val.csv")
    reward_engine = RewardEngine.load(calib_path, reward_config=cfg.REWARD_PRESETS["recap_dpo"])
    device = ppo_trainer.accelerator.device

    # Resume support -- same idiom as recap_train_grpo.py: accelerate-native
    # save_state()/load_state() for the model+optimizer+RNG (PPOTrainer
    # already manages its own Accelerator, so we reuse it rather than
    # constructing a second one), plus a small metadata JSON for the loop
    # step counter and best-so-far validation composite.
    latest_state_dir = checkpoint_dir.parent / "latest_state"
    resume_meta = recap_utils.load_training_state_meta(latest_state_dir)
    start_update = 0
    best_composite = float("-inf")
    if resume_meta is not None:
        ppo_trainer.accelerator.load_state(str(latest_state_dir / "accelerate"))
        start_update = resume_meta["step"] + 1
        best_composite = resume_meta.get("best_composite", float("-inf"))
        if recap_utils.is_main_process():
            print(f"[Resume] loaded PPO state from {latest_state_dir}, "
                  f"resuming at update={start_update} (best_composite={best_composite:.4f})")

    for update in range(start_update, settings.num_updates):
        # Rank-aware sampling -- see recap_train_grpo.py for why: without the
        # process_index offset, every DDP rank samples the identical batch.
        rank_seed = seed + update * 1000 + ppo_trainer.accelerator.process_index
        batch_df = train_df.sample(n=min(settings.batch_size, len(train_df)), random_state=rank_seed)
        sources = batch_df["source"].tolist()
        references = batch_df["gold_truth"].tolist()

        query_tensors = [
            tokenizer(s, return_tensors="pt", truncation=True).input_ids[0].to(device) for s in sources
        ]
        response_tensors = ppo_trainer.generate(
            query_tensors, max_new_tokens=settings.max_new_tokens, do_sample=True,
        )
        responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

        if update == 0:
            live_gen_check = recap_checks.check_live_generation(sources, responses)
            if not live_gen_check.ok:
                raise recap_checks.CheckFailure(live_gen_check.message)
            rollout_records = [
                {"source": s, "reference": r, "candidate": c, "direction": direction}
                for s, r, c in zip(sources, references, responses)
            ]
            rollout_check = recap_checks.check_rollout_alignment(rollout_records)
            if not rollout_check.ok:
                raise recap_checks.CheckFailure(rollout_check.message)

        scored = reward_engine.score_batch(sources, responses, references)

        # Invalid (empty/malformed/extreme-length) completions are dropped
        # from the PPO update entirely, not given a substitute reward -- see
        # recap_train_grpo.py for the reward-hacking rationale. TRL's
        # ppo_trainer.step() needs matched-length lists, so we filter all
        # three together before calling it.
        keep_idx = [i for i, s in enumerate(scored) if math.isfinite(s["reward"])]
        if not keep_idx:
            if update % 50 == 0 and recap_utils.is_main_process():
                print(f"[{lang}/{direction}/{experiment}] update={update}: no valid completions this step, skipping")
            continue

        kept_queries = [query_tensors[i] for i in keep_idx]
        kept_responses = [response_tensors[i] for i in keep_idx]
        kept_rewards = [torch.tensor(scored[i]["reward"], device=device) for i in keep_idx]

        stats = ppo_trainer.step(kept_queries, kept_responses, kept_rewards)

        if update % 50 == 0 and recap_utils.is_main_process():
            mean_reward = sum(float(r) for r in kept_rewards) / len(kept_rewards)
            n_invalid = len(scored) - len(keep_idx)
            print(f"[{lang}/{direction}/{experiment}] update={update} mean_reward={mean_reward:.4f} "
                  f"kl={stats.get('objective/kl', float('nan')):.4f} n_invalid={n_invalid}/{len(scored)}")

        if update % settings.eval_steps == 0 and update > 0 and recap_utils.is_main_process():
            val_translations = recap_utils.generate_batch(
                ppo_trainer.model, tokenizer, val_df["source"].tolist(), cfg.INFERENCE_CONFIG,
            )
            val_scored = reward_engine.compute_raw_metrics(
                val_df["source"].tolist(), val_translations, val_df["gold_truth"].tolist(),
            )
            valid = [s for s in val_scored if s["valid"] and math.isfinite(s["bleu"])]
            if valid:
                composite = sum((s["bleu"] + s["chrf"] + s["comet"]) / 3.0 for s in valid) / len(valid)
                print(f"[Eval] update={update} validation composite={composite:.4f}")
                if composite > best_composite:
                    best_composite = composite
                    ppo_trainer.model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

        if update % settings.save_steps == 0 and update > 0:
            # Periodic RESUME checkpoint -- separate from checkpoint_dir
            # (best-by-validation only). Called by every rank; accelerate
            # coordinates the actual save internally.
            ppo_trainer.accelerator.save_state(str(latest_state_dir / "accelerate"))
            recap_utils.save_training_state_meta(latest_state_dir, update, extra={"best_composite": best_composite})

    resolved_config = {
        "lang": lang, "direction": direction, "experiment": experiment, "seed": seed,
        "sft_checkpoint": sft_checkpoint, "ppo_settings": asdict(settings),
        "reward_preset": "recap_dpo",
    }
    recap_utils.save_run_manifest(
        manifest_path, resolved_config, seed, extra={"best_validation_composite": best_composite},
    )
    if recap_utils.is_main_process():
        with open(manifest_path) as f:
            manifest_check = recap_checks.check_manifest_hashed(json.load(f))
        if not manifest_check.ok:
            raise recap_checks.CheckFailure(f"{lang}/{direction}/{experiment} manifest_hashed: {manifest_check.message}")
        if latest_state_dir.exists():
            import shutil
            shutil.rmtree(latest_state_dir)
    print(f"[Done] {lang}/{direction}/{experiment}/seed_{seed} -> {checkpoint_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=cfg.LANGUAGES, required=True)
    parser.add_argument("--direction", choices=cfg.DIRECTIONS, required=True)
    parser.add_argument("--experiment", choices=[n for n, e in cfg.EXPERIMENTS.items() if e.trainer == "ppo"], required=True)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    args = parser.parse_args()
    process_one(args.lang, args.direction, args.experiment, args.seed)


if __name__ == "__main__":
    main()
