#!/usr/bin/env bash
# ============================================================================
# RECAP -- full experiment run, ALL IN ONE SEQUENTIAL SCRIPT.
#
# NOTE: for actually running this on HPC, use JOB_GUIDE.md instead -- it
# breaks this exact same set of commands into independent job groups (data
# prep / DPO / GRPO / PPO / eval / report) with the dependency rules between
# them made explicit, so you can submit them as separate parallel jobs
# rather than one long sequential run. This file is kept as a reference for
# "what exactly gets run" and as a fallback if you'd genuinely rather run
# everything in one sitting.
#
# Every stage, every (lang, direction), every currently-wired experiment:
# the 8-condition main matrix (Table 7) plus the 6-step
# preference-construction ablation (Table 8) = 14 experiments total, single
# seed (config.py's default cfg.SEED = 13).
#
# NOT included here (deferred, separate scope -- see config.py's "NOT YET
# WIRED" comments): the 8 targeted ablations (candidate-diversity,
# calibration-method, confidence/delta-grid, pair-selection-strategy,
# data-scale, GRPO-group-size, weight-sensitivity).
#
# Compute heads-up: Stage 6-8 below launch
#   3 languages x 2 directions x 10 DPO experiments = 60 DPO runs
#   3 languages x 2 directions x 2  GRPO experiments = 12 GRPO runs
#   3 languages x 2 directions x 1  PPO experiment   =  6 PPO runs
#   ---------------------------------------------------------------
#   78 training runs total, at this single seed.
# Each is independent and safe to launch as its own HPC job (matches the
# `qsub -I ... -lngpus=1` single-GPU pattern used elsewhere in this project)
# instead of running the loops below sequentially in one session.
#
# Run from inside RECAP/code/. Stops on first error.
# ============================================================================
set -euo pipefail

LANGS=("Bhili" "Gondi" "Mundari")
DIRECTIONS=("hi2tgt" "tgt2hi")
DPO_EXPERIMENTS=(
  "dpo_raw" "dpo_quality_only" "dpo_no_confidence" "recap_dpo"
  "ablation_quality_only" "ablation_quality_plus_rep" "ablation_quality_plus_len"
  "ablation_full_reward" "ablation_full_reward_margin" "ablation_full_recap"
)

# ----------------------------------------------------------------------------
# Stage 0-5: data pipeline. CPU-only (metrics come from maha_data_2's stored
# BLEU/chrF++/COMET columns, no COMET model reload needed), no GPU required.
# Each script loops ALL languages/directions/experiments internally when
# called with no --lang/--direction/--experiment, so one call each covers
# everything. Pre-flight checks (recap_checks.py) run automatically inside
# these.
# ----------------------------------------------------------------------------
echo "=== Stage 1: split ==="
python recap_split.py

echo "=== Stage 2: calibrate ==="
python recap_calibrate.py

echo "=== Stage 3: score ==="
python recap_score.py

echo "=== Stage 4: mine pairs (all 10 reward-preset experiments) ==="
python recap_mine_pairs.py

echo "=== Stage 5: balance pairs (all 10 reward-preset experiments) ==="
python recap_balance_pairs.py

# ----------------------------------------------------------------------------
# Stage 6: DPO training -- one job per (lang, direction, experiment). GPU
# required. This trains recap_dpo among the 10, which Stage 7's
# recap_dpo_grpo below depends on (same lang/direction/seed).
# ----------------------------------------------------------------------------
echo "=== Stage 6: DPO training (60 runs) ==="
for lang in "${LANGS[@]}"; do
  for direction in "${DIRECTIONS[@]}"; do
    for experiment in "${DPO_EXPERIMENTS[@]}"; do
      echo "--- DPO: $lang / $direction / $experiment ---"
      python recap_train_dpo.py --lang "$lang" --direction "$direction" --experiment "$experiment"
    done
  done
done

# ----------------------------------------------------------------------------
# Stage 7: GRPO -- "pure GRPO" (sft_grpo, init from SFT) and the headline
# RECAP-DPO+GRPO (recap_dpo_grpo, init from the recap_dpo checkpoint trained
# above). GPU required.
# ----------------------------------------------------------------------------
echo "=== Stage 7: GRPO training (12 runs) ==="
for lang in "${LANGS[@]}"; do
  for direction in "${DIRECTIONS[@]}"; do
    echo "--- GRPO: $lang / $direction / sft_grpo ---"
    python recap_train_grpo.py --lang "$lang" --direction "$direction" --experiment sft_grpo
    echo "--- GRPO: $lang / $direction / recap_dpo_grpo ---"
    python recap_train_grpo.py --lang "$lang" --direction "$direction" --experiment recap_dpo_grpo
  done
done

# ----------------------------------------------------------------------------
# Stage 8: PPO baseline -- from SFT directly. GPU required.
# ----------------------------------------------------------------------------
echo "=== Stage 8: PPO training (6 runs) ==="
for lang in "${LANGS[@]}"; do
  for direction in "${DIRECTIONS[@]}"; do
    echo "--- PPO: $lang / $direction / sft_ppo ---"
    python recap_train_ppo.py --lang "$lang" --direction "$direction" --experiment sft_ppo
  done
done

# ----------------------------------------------------------------------------
# Stage 9: evaluate every trained condition (+ untouched SFT baseline) on the
# held-out test split, compute deltas vs SFT with paired bootstrap CIs, and
# write each direction's best-by-VALIDATION checkpoint for deployment.
# ----------------------------------------------------------------------------
echo "=== Stage 9: evaluation (all experiments, all directions) ==="
python recap_evaluate.py

# ----------------------------------------------------------------------------
# Stage 11: reporting. Tables 3-9 and Figures 1-3 from the saved outputs
# above only -- no retraining/re-decoding. Figures need matplotlib:
#   pip install matplotlib
# ----------------------------------------------------------------------------
echo "=== Stage 11: report tables + plots ==="
python recap_report_tables.py
python recap_report_plots.py

# Table 10 (human validation) is a two-step, human-in-the-loop process --
# NOT fully automatic:
echo "=== Stage 11: Table 10 sampling (fill in human_preferred, then analyze) ==="
python recap_sample_for_human_eval.py --experiment recap_dpo --n 500
# After an annotator fills in the human_preferred column of the sampled CSV:
#   python recap_human_eval_analysis.py --labeled_csv ../recap_human_eval/sample_recap_dpo_500.csv

echo "=== Done. ==="

# ============================================================================
# Optional: multi-seed reruns for paper-grade mean +- std reporting
# (config.py's SEEDS = [13, 42, 2026]; paper Section 8.1/11.9 asks for this
# "whenever possible"). Repeat Stages 6-9 with --seed 42 and --seed 2026 --
# e.g. wrap the Stage 6-8 loops above in an outer `for seed in 13 42 2026`
# loop, adding `--seed "$seed"` to each training command, then rerun
# `python recap_evaluate.py --seed 42` / `--seed 2026` for Stage 9. Left out
# of the default run above because it triples the 78 training runs to 234.
# ============================================================================

# ============================================================================
# Example: standalone inference after everything above has run
#   python recap_infer.py --lang Bhili --direction hi2tgt --source "..."
# ============================================================================
