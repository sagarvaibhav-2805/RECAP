"""
Stage 7 -- GRPO, written from scratch (not TRL's GRPOTrainer -- see
IMPLEMENTATION_PLAN.md Stage 7 for why: TRL's GRPOTrainer targets causal LMs,
mT5 is encoder-decoder). Two conditions, both driven by --experiment:

  sft_grpo        -- policy AND frozen KL-reference both start from the SFT
                      checkpoint ("pure GRPO" ablation).
  recap_dpo_grpo  -- policy starts from the recap_dpo checkpoint, frozen
                      KL-reference is a frozen copy of that SAME DPO
                      checkpoint (not SFT). This is the headline result.

Design: single inner epoch per rollout batch, so the importance ratio is
exactly 1 at the point of the gradient step (generation and the scored
forward pass use the same weights, no optimizer.step() in between) -- this
reduces the clipped objective to a plain KL-regularized policy-gradient,
which is the correct, simplest form of GRPO. The ratio/clip machinery is
still present so this generalizes cleanly if a multi-epoch-per-batch variant
is ever needed.

Every rollout is FRESH generation from the current policy -- static
maha_data_2 candidate files are never read here (see recap_checks.py check 7).

Multi-GPU: wrap with accelerate.Accelerator(); the frozen reference model is
NEVER passed to accelerator.prepare() (see IMPLEMENTATION_PLAN.md
"Deadlock-free DDP").

Run:
    python recap_train_grpo.py --lang Bhili --direction hi2tgt --experiment recap_dpo_grpo
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


def _batch_logprobs(model, tokenizer, sources: list[str], targets: list[str], device):
    """Sum of per-token log-probs of `targets` under `model`, conditioned on
    `sources` (teacher-forced). Returns a 1D tensor, one value per example."""
    import torch
    import torch.nn.functional as F

    enc = tokenizer(sources, return_tensors="pt", padding=True, truncation=True).to(device)
    labels = tokenizer(text_target=targets, return_tensors="pt", padding=True, truncation=True).to(device)
    label_ids = labels["input_ids"]
    pad_id = tokenizer.pad_token_id
    masked_labels = label_ids.masked_fill(label_ids == pad_id, -100)

    outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=masked_labels)
    logits = outputs.logits  # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)
    token_mask = (label_ids != pad_id).float()
    gathered = torch.gather(log_probs, 2, label_ids.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (gathered * token_mask).sum(dim=1)


def process_one(lang: str, direction: str, experiment: str, seed: int) -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    exp = cfg.EXPERIMENTS[experiment]
    if exp.trainer != "grpo":
        raise ValueError(f"{experiment} is not a GRPO experiment (trainer={exp.trainer})")

    checkpoint_dir = cfg.checkpoint_dir(cfg.GRPO_ROOT, lang, direction, experiment, seed)
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        print(f"[Skip] {lang}/{direction}/{experiment}/seed_{seed}: checkpoint already exists")
        return

    train_path = cfg.split_dir(lang, direction) / "train.csv"
    calib_path = cfg.calib_path(lang, direction)
    if not (train_path.exists() and calib_path.exists()):
        print(f"[Skip] {lang}/{direction}: missing split/calibration -- run Stages 1-2 first")
        return

    recap_utils.configure_ddp_env()
    recap_utils.set_seed(seed)
    accelerator = recap_utils.make_accelerator()

    if exp.init_from is None:
        init_checkpoint = str(cfg.sft_checkpoint_path(lang, direction))
    else:
        init_exp = cfg.EXPERIMENTS[exp.init_from]
        init_checkpoint = str(cfg.checkpoint_dir(cfg.DPO_ROOT, lang, direction, exp.init_from, seed))

    tokenizer = AutoTokenizer.from_pretrained(init_checkpoint)
    policy = AutoModelForSeq2SeqLM.from_pretrained(init_checkpoint)
    ref_model = AutoModelForSeq2SeqLM.from_pretrained(init_checkpoint)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    recap_checks.run_static_checks()
    ref_frozen_check = recap_checks.check_reference_frozen(ref_model)
    if not ref_frozen_check.ok:
        raise recap_checks.CheckFailure(f"{lang}/{direction}/{experiment} reference_frozen: {ref_frozen_check.message}")

    settings = cfg.GRPO_SETTINGS
    optimizer = torch.optim.AdamW(policy.parameters(), lr=settings.learning_rate)
    policy, optimizer = accelerator.prepare(policy, optimizer)
    ref_model = ref_model.to(accelerator.device)  # frozen -- never accelerator.prepare()'d

    train_df = pd.read_csv(train_path)
    subset = train_df.sample(
        n=min(settings.source_subset_size, len(train_df)), random_state=seed,
    ).reset_index(drop=True)
    # Keyed by source_id (unique, guaranteed by Stage 1), not raw source text,
    # and built once outside the hot loop.
    refs_by_source_id = dict(zip(subset["source_id"], subset["gold_truth"]))
    sources_by_id = dict(zip(subset["source_id"], subset["source"]))

    reward_config = cfg.REWARD_PRESETS["recap_dpo"]  # full reward, matches SFT/DPO reward definition
    reward_engine = RewardEngine.load(calib_path, reward_config=reward_config)

    val_df = pd.read_csv(cfg.split_dir(lang, direction) / "val.csv")
    best_composite = float("-inf")

    for update in range(settings.num_updates):
        # Rank-aware sampling: offset by accelerator.process_index so DDP
        # ranks see DIFFERENT sources each step instead of all redoing the
        # same identical batch (which would waste every extra GPU).
        rank_seed = seed + update * 1000 + accelerator.process_index
        batch_ids = subset["source_id"].sample(
            n=min(settings.per_device_batch_size, len(subset)), random_state=rank_seed,
        ).tolist()
        batch_sources = [sources_by_id[i] for i in batch_ids]

        all_sources, all_completions, all_refs, group_ids = [], [], [], []
        with torch.no_grad():
            for g, src in enumerate(batch_sources):
                enc = tokenizer([src] * settings.group_size, return_tensors="pt", padding=True, truncation=True)
                enc = {k: v.to(accelerator.device) for k, v in enc.items()}
                unwrapped = accelerator.unwrap_model(policy)
                generated = unwrapped.generate(
                    **enc, max_new_tokens=settings.max_new_tokens, do_sample=True,
                    temperature=settings.temperature, top_p=settings.top_p,
                )
                texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
                all_sources.extend([src] * settings.group_size)
                all_completions.extend(texts)
                all_refs.extend([refs_by_source_id[batch_ids[g]]] * settings.group_size)
                group_ids.extend([g] * settings.group_size)

        if update == 0:
            live_gen_check = recap_checks.check_live_generation(all_sources, all_completions)
            if not live_gen_check.ok:
                raise recap_checks.CheckFailure(live_gen_check.message)
            rollout_records = [
                {"source": s, "reference": r, "candidate": c, "direction": direction}
                for s, r, c in zip(all_sources, all_refs, all_completions)
            ]
            rollout_check = recap_checks.check_rollout_alignment(rollout_records)
            if not rollout_check.ok:
                raise recap_checks.CheckFailure(rollout_check.message)

        scored = reward_engine.score_batch(all_sources, all_completions, all_refs)

        # Invalid (empty/malformed/extreme-length) completions are EXCLUDED
        # from the group's advantage computation and from the loss entirely
        # -- not given a substitute reward. A fabricated reward (e.g. 0.0)
        # for garbage output is a reward-hacking risk: calibrated rewards can
        # be negative for weak-but-real translations, so 0.0 can look better
        # than a genuine low-quality output and teach the policy to degenerate.
        valid_mask = [math.isfinite(s["reward"]) for s in scored]
        advantages = [0.0] * len(scored)
        n_groups = max(group_ids) + 1 if group_ids else 0
        for g in range(n_groups):
            idx = [i for i, gg in enumerate(group_ids) if gg == g and valid_mask[i]]
            if len(idx) < 2:
                continue  # not enough valid completions in this group to form a baseline -- zero advantage
            group_rewards = [scored[i]["reward"] for i in idx]
            mean_r = sum(group_rewards) / len(group_rewards)
            var_r = sum((r - mean_r) ** 2 for r in group_rewards) / len(group_rewards)
            if var_r < 1e-8:
                continue  # zero-variance group -> zero advantage (documented rule)
            std_r = math.sqrt(var_r + 1e-8)
            for i in idx:
                advantages[i] = (scored[i]["reward"] - mean_r) / std_r

        keep_idx = [i for i in range(len(scored)) if valid_mask[i] and advantages[i] != 0.0]
        if not keep_idx:
            if update % 50 == 0 and recap_utils.is_main_process():
                print(f"[{lang}/{direction}/{experiment}] update={update}: no valid non-zero-advantage "
                      f"completions this step, skipping update")
            continue

        kept_sources = [all_sources[i] for i in keep_idx]
        kept_completions = [all_completions[i] for i in keep_idx]
        kept_advantages = [advantages[i] for i in keep_idx]

        with torch.no_grad():
            logprob_ref = _batch_logprobs(ref_model, tokenizer, kept_sources, kept_completions, accelerator.device)

        # No optimizer.step() has happened since generation, so the policy
        # weights used to sample all_completions are identical to the current
        # weights -- logprob_old == logprob_new.detach() exactly. Using
        # logprob_new here (instead of a second redundant no-grad forward
        # pass) keeps the ratio/clip machinery correct and cheap; see module
        # docstring for how this generalizes to a multi-epoch-per-batch
        # variant later.
        logprob_new = _batch_logprobs(policy, tokenizer, kept_sources, kept_completions, accelerator.device)
        logprob_old = logprob_new.detach()
        advantages_t = torch.tensor(kept_advantages, device=accelerator.device, dtype=torch.float32)

        ratio = torch.exp(logprob_new - logprob_old)
        clipped_ratio = torch.clamp(ratio, 1 - settings.clip_epsilon, 1 + settings.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages_t, clipped_ratio * advantages_t).mean()
        kl = (logprob_new - logprob_ref).mean()
        loss = policy_loss + settings.kl_coef * kl

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if update % 50 == 0 and recap_utils.is_main_process():
            valid_rewards = [scored[i]["reward"] for i in range(len(scored)) if valid_mask[i]]
            mean_reward = sum(valid_rewards) / len(valid_rewards) if valid_rewards else float("nan")
            n_invalid = len(scored) - sum(valid_mask)
            print(f"[{lang}/{direction}/{experiment}] update={update} loss={loss.item():.4f} "
                  f"policy_loss={policy_loss.item():.4f} kl={kl.item():.4f} "
                  f"mean_reward={mean_reward:.4f} n_invalid={n_invalid}/{len(scored)}")

        if update % settings.eval_steps == 0 and update > 0:
            recap_utils.wait_for_everyone()
            if recap_utils.is_main_process():
                unwrapped = accelerator.unwrap_model(policy)
                val_translations = recap_utils.generate_batch(
                    unwrapped, tokenizer, val_df["source"].tolist(), cfg.INFERENCE_CONFIG,
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
                        unwrapped.save_pretrained(checkpoint_dir)
                        tokenizer.save_pretrained(checkpoint_dir)

    recap_utils.wait_for_everyone()
    resolved_config = {
        "lang": lang, "direction": direction, "experiment": experiment, "seed": seed,
        "init_checkpoint": init_checkpoint, "grpo_settings": asdict(settings),
        "reward_preset": "recap_dpo",
    }
    manifest_path = cfg.run_manifest_path(cfg.GRPO_ROOT, lang, direction, experiment, seed)
    recap_utils.save_run_manifest(
        manifest_path, resolved_config, seed, extra={"best_validation_composite": best_composite},
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
    parser.add_argument("--experiment", choices=[n for n, e in cfg.EXPERIMENTS.items() if e.trainer == "grpo"], required=True)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    args = parser.parse_args()
    process_one(args.lang, args.direction, args.experiment, args.seed)


if __name__ == "__main__":
    main()
