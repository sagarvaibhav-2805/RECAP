# RECAP Job Guide — running the experiment as independent HPC jobs

This is the practical companion to `run_all_experiments.sh` (which runs
everything sequentially, in one long session — useful as a reference for
"what exactly gets run," but not how you'd actually want to burn HPC time).
This guide breaks the same work into **independent job groups** you can
submit separately, with the dependency rules between them made explicit, so
you know exactly what can run in parallel and what has to wait.

Every command below is copy-pasteable from `RECAP/code/`.

**Before committing to any of this on real HPC time**, smoke-test the whole
pipeline on a tiny slice of data first -- see §14.

---

## 0. Dependency graph (the whole thing, at a glance)

```
Group A (data prep, x6, per direction)
   |
   +--> Group B (DPO training, x60, per direction x experiment)
   |        |
   |        +--> Group D (recap_dpo_grpo, x6) -- needs THIS direction's
   |                                              Group B "recap_dpo" run done
   |
   +--> Group C (sft_grpo, x6, per direction)
   |
   +--> Group E (sft_ppo, x6, per direction)
                |
                v
        Group F (evaluate, x1 or x6) -- needs everything above done
                |
                v
        Group G (report tables + plots, x1) -- needs Group F done
                |
                v
        Group H (human-eval sampling, x1; analysis, x1 -- human-in-the-loop)
```

Nothing in Groups B, C, D, E writes to any other group's files, so within a
group every job is safe to run at the same time as every other job in that
group (and in most cases, across groups too — see the table below).

---

## 1. Job groups — what's in each, and when it can start

| Group | What | # jobs | GPU? | Can start when... | Max parallel |
|---|---|---|---|---|---|
| **A** | Stages 1-5 (split/calibrate/score/mine/balance), one job per direction | 6 | No | immediately | 6 (all at once) |
| **B** | Stage 6 DPO training, one job per (lang, direction, experiment) | 60 | Yes | Group A done **for that direction** | up to 60 (GPU-limited) |
| **C** | Stage 7 GRPO, `sft_grpo` (from SFT), one job per direction | 6 | Yes | Group A done for that direction | up to 6, and can run alongside B/E |
| **D** | Stage 7 GRPO, `recap_dpo_grpo` (from DPO), one job per direction | 6 | Yes | that direction's **Group B `recap_dpo` job** finished | up to 6, once each direction's B/recap_dpo is done |
| **E** | Stage 8 PPO, `sft_ppo`, one job per direction | 6 | Yes | Group A done for that direction | up to 6, and can run alongside B/C |
| **F** | Stage 9 evaluate | 1 (or 6, per direction) | Yes (decoding) | all of B, C, D, E finished | 1, or 6 if split by direction |
| **G** | Stage 11 tables + plots | 1 | No | Group F done | 1 |
| **H** | Stage 11 human-eval sample + analysis | 2 (sequential, human step between) | No | Group G (or just Group F) done | 1 |

**Practical read:** after Group A finishes for a direction, you can fire off
that direction's 10 DPO jobs (Group B) + 1 `sft_grpo` job (Group C) + 1
`sft_ppo` job (Group E) all at once — 12 simultaneous GPU jobs per direction,
72 across all 6 directions, GPU-availability permitting. The scheduler queues
whatever doesn't fit immediately. Only Group D needs an explicit wait.

---

## 2. Group A — data prep (6 independent jobs, no GPU)

One job per direction. Cheap (CPU-only, metrics come from `maha_data_2`'s
stored columns, no COMET reload) — a few minutes each. Each job runs Stages
1-5 for that one direction, covering all 10 reward-preset experiments'
pair-mining within it.

```bash
# Job A1
python recap_split.py --lang Bhili --direction hi2tgt
python recap_calibrate.py --lang Bhili --direction hi2tgt
python recap_score.py --lang Bhili --direction hi2tgt
python recap_mine_pairs.py --lang Bhili --direction hi2tgt
python recap_balance_pairs.py --lang Bhili --direction hi2tgt

# Job A2
python recap_split.py --lang Bhili --direction tgt2hi
python recap_calibrate.py --lang Bhili --direction tgt2hi
python recap_score.py --lang Bhili --direction tgt2hi
python recap_mine_pairs.py --lang Bhili --direction tgt2hi
python recap_balance_pairs.py --lang Bhili --direction tgt2hi

# Job A3
python recap_split.py --lang Gondi --direction hi2tgt
python recap_calibrate.py --lang Gondi --direction hi2tgt
python recap_score.py --lang Gondi --direction hi2tgt
python recap_mine_pairs.py --lang Gondi --direction hi2tgt
python recap_balance_pairs.py --lang Gondi --direction hi2tgt

# Job A4
python recap_split.py --lang Gondi --direction tgt2hi
python recap_calibrate.py --lang Gondi --direction tgt2hi
python recap_score.py --lang Gondi --direction tgt2hi
python recap_mine_pairs.py --lang Gondi --direction tgt2hi
python recap_balance_pairs.py --lang Gondi --direction tgt2hi

# Job A5
python recap_split.py --lang Mundari --direction hi2tgt
python recap_calibrate.py --lang Mundari --direction hi2tgt
python recap_score.py --lang Mundari --direction hi2tgt
python recap_mine_pairs.py --lang Mundari --direction hi2tgt
python recap_balance_pairs.py --lang Mundari --direction hi2tgt

# Job A6
python recap_split.py --lang Mundari --direction tgt2hi
python recap_calibrate.py --lang Mundari --direction tgt2hi
python recap_score.py --lang Mundari --direction tgt2hi
python recap_mine_pairs.py --lang Mundari --direction tgt2hi
python recap_balance_pairs.py --lang Mundari --direction tgt2hi
```

Given how cheap these are, running all 6 as one sequential job is also
perfectly reasonable if you'd rather not manage 6 tiny job submissions.

---

## 3. Group B — DPO training (60 independent jobs, GPU)

One job = one `(lang, direction, experiment)`. All 60 are mutually
independent (each only touches its own `recap_dpo/<lang>/<direction>/<experiment>/`
folder). The 10 experiments per direction:
`dpo_raw, dpo_quality_only, dpo_no_confidence, recap_dpo, ablation_quality_only,
ablation_quality_plus_rep, ablation_quality_plus_len, ablation_full_reward,
ablation_full_reward_margin, ablation_full_recap`.

```bash
# --- Bhili / hi2tgt ---
python recap_train_dpo.py --lang Bhili --direction hi2tgt --experiment dpo_raw
python recap_train_dpo.py --lang Bhili --direction hi2tgt --experiment dpo_quality_only
python recap_train_dpo.py --lang Bhili --direction hi2tgt --experiment dpo_no_confidence
python recap_train_dpo.py --lang Bhili --direction hi2tgt --experiment recap_dpo
python recap_train_dpo.py --lang Bhili --direction hi2tgt --experiment ablation_quality_only
python recap_train_dpo.py --lang Bhili --direction hi2tgt --experiment ablation_quality_plus_rep
python recap_train_dpo.py --lang Bhili --direction hi2tgt --experiment ablation_quality_plus_len
python recap_train_dpo.py --lang Bhili --direction hi2tgt --experiment ablation_full_reward
python recap_train_dpo.py --lang Bhili --direction hi2tgt --experiment ablation_full_reward_margin
python recap_train_dpo.py --lang Bhili --direction hi2tgt --experiment ablation_full_recap

# --- Bhili / tgt2hi ---
python recap_train_dpo.py --lang Bhili --direction tgt2hi --experiment dpo_raw
python recap_train_dpo.py --lang Bhili --direction tgt2hi --experiment dpo_quality_only
python recap_train_dpo.py --lang Bhili --direction tgt2hi --experiment dpo_no_confidence
python recap_train_dpo.py --lang Bhili --direction tgt2hi --experiment recap_dpo
python recap_train_dpo.py --lang Bhili --direction tgt2hi --experiment ablation_quality_only
python recap_train_dpo.py --lang Bhili --direction tgt2hi --experiment ablation_quality_plus_rep
python recap_train_dpo.py --lang Bhili --direction tgt2hi --experiment ablation_quality_plus_len
python recap_train_dpo.py --lang Bhili --direction tgt2hi --experiment ablation_full_reward
python recap_train_dpo.py --lang Bhili --direction tgt2hi --experiment ablation_full_reward_margin
python recap_train_dpo.py --lang Bhili --direction tgt2hi --experiment ablation_full_recap

# --- Gondi / hi2tgt ---
python recap_train_dpo.py --lang Gondi --direction hi2tgt --experiment dpo_raw
python recap_train_dpo.py --lang Gondi --direction hi2tgt --experiment dpo_quality_only
python recap_train_dpo.py --lang Gondi --direction hi2tgt --experiment dpo_no_confidence
python recap_train_dpo.py --lang Gondi --direction hi2tgt --experiment recap_dpo
python recap_train_dpo.py --lang Gondi --direction hi2tgt --experiment ablation_quality_only
python recap_train_dpo.py --lang Gondi --direction hi2tgt --experiment ablation_quality_plus_rep
python recap_train_dpo.py --lang Gondi --direction hi2tgt --experiment ablation_quality_plus_len
python recap_train_dpo.py --lang Gondi --direction hi2tgt --experiment ablation_full_reward
python recap_train_dpo.py --lang Gondi --direction hi2tgt --experiment ablation_full_reward_margin
python recap_train_dpo.py --lang Gondi --direction hi2tgt --experiment ablation_full_recap

# --- Gondi / tgt2hi ---
python recap_train_dpo.py --lang Gondi --direction tgt2hi --experiment dpo_raw
python recap_train_dpo.py --lang Gondi --direction tgt2hi --experiment dpo_quality_only
python recap_train_dpo.py --lang Gondi --direction tgt2hi --experiment dpo_no_confidence
python recap_train_dpo.py --lang Gondi --direction tgt2hi --experiment recap_dpo
python recap_train_dpo.py --lang Gondi --direction tgt2hi --experiment ablation_quality_only
python recap_train_dpo.py --lang Gondi --direction tgt2hi --experiment ablation_quality_plus_rep
python recap_train_dpo.py --lang Gondi --direction tgt2hi --experiment ablation_quality_plus_len
python recap_train_dpo.py --lang Gondi --direction tgt2hi --experiment ablation_full_reward
python recap_train_dpo.py --lang Gondi --direction tgt2hi --experiment ablation_full_reward_margin
python recap_train_dpo.py --lang Gondi --direction tgt2hi --experiment ablation_full_recap

# --- Mundari / hi2tgt ---
python recap_train_dpo.py --lang Mundari --direction hi2tgt --experiment dpo_raw
python recap_train_dpo.py --lang Mundari --direction hi2tgt --experiment dpo_quality_only
python recap_train_dpo.py --lang Mundari --direction hi2tgt --experiment dpo_no_confidence
python recap_train_dpo.py --lang Mundari --direction hi2tgt --experiment recap_dpo
python recap_train_dpo.py --lang Mundari --direction hi2tgt --experiment ablation_quality_only
python recap_train_dpo.py --lang Mundari --direction hi2tgt --experiment ablation_quality_plus_rep
python recap_train_dpo.py --lang Mundari --direction hi2tgt --experiment ablation_quality_plus_len
python recap_train_dpo.py --lang Mundari --direction hi2tgt --experiment ablation_full_reward
python recap_train_dpo.py --lang Mundari --direction hi2tgt --experiment ablation_full_reward_margin
python recap_train_dpo.py --lang Mundari --direction hi2tgt --experiment ablation_full_recap

# --- Mundari / tgt2hi ---
python recap_train_dpo.py --lang Mundari --direction tgt2hi --experiment dpo_raw
python recap_train_dpo.py --lang Mundari --direction tgt2hi --experiment dpo_quality_only
python recap_train_dpo.py --lang Mundari --direction tgt2hi --experiment dpo_no_confidence
python recap_train_dpo.py --lang Mundari --direction tgt2hi --experiment recap_dpo
python recap_train_dpo.py --lang Mundari --direction tgt2hi --experiment ablation_quality_only
python recap_train_dpo.py --lang Mundari --direction tgt2hi --experiment ablation_quality_plus_rep
python recap_train_dpo.py --lang Mundari --direction tgt2hi --experiment ablation_quality_plus_len
python recap_train_dpo.py --lang Mundari --direction tgt2hi --experiment ablation_full_reward
python recap_train_dpo.py --lang Mundari --direction tgt2hi --experiment ablation_full_reward_margin
python recap_train_dpo.py --lang Mundari --direction tgt2hi --experiment ablation_full_recap
```

All 60 lines above, one line = one job. **Each line is its own single-GPU
job** (matches the `qsub -I ... -lngpus=1` pattern used elsewhere in this
project — see §6 for wrapping these as batch jobs).

Each is independently resumable (see §7) if a job dies mid-training, so it's
safe to just resubmit the exact same line.

---

## 4. Group C — pure GRPO, `sft_grpo` (6 independent jobs, GPU)

Only needs Group A (SFT checkpoint is pre-existing, never trained here).
Independent of Group B entirely.

```bash
python recap_train_grpo.py --lang Bhili   --direction hi2tgt --experiment sft_grpo
python recap_train_grpo.py --lang Bhili   --direction tgt2hi --experiment sft_grpo
python recap_train_grpo.py --lang Gondi   --direction hi2tgt --experiment sft_grpo
python recap_train_grpo.py --lang Gondi   --direction tgt2hi --experiment sft_grpo
python recap_train_grpo.py --lang Mundari --direction hi2tgt --experiment sft_grpo
python recap_train_grpo.py --lang Mundari --direction tgt2hi --experiment sft_grpo
```

---

## 5. Group D — RECAP-DPO+GRPO, `recap_dpo_grpo` (6 jobs, GPU, has a dependency)

Each job here **requires that same (lang, direction)'s Group B `recap_dpo` run
to have already finished** (it initializes from that checkpoint). Submit
these only after the corresponding `recap_dpo` line from §3 has completed —
or use a scheduler-level dependency (§6).

```bash
python recap_train_grpo.py --lang Bhili   --direction hi2tgt --experiment recap_dpo_grpo
python recap_train_grpo.py --lang Bhili   --direction tgt2hi --experiment recap_dpo_grpo
python recap_train_grpo.py --lang Gondi   --direction hi2tgt --experiment recap_dpo_grpo
python recap_train_grpo.py --lang Gondi   --direction tgt2hi --experiment recap_dpo_grpo
python recap_train_grpo.py --lang Mundari --direction hi2tgt --experiment recap_dpo_grpo
python recap_train_grpo.py --lang Mundari --direction tgt2hi --experiment recap_dpo_grpo
```

---

## 6. Group E — PPO, `sft_ppo` (6 independent jobs, GPU)

Only needs Group A. Independent of everything else.

```bash
python recap_train_ppo.py --lang Bhili   --direction hi2tgt --experiment sft_ppo
python recap_train_ppo.py --lang Bhili   --direction tgt2hi --experiment sft_ppo
python recap_train_ppo.py --lang Gondi   --direction hi2tgt --experiment sft_ppo
python recap_train_ppo.py --lang Gondi   --direction tgt2hi --experiment sft_ppo
python recap_train_ppo.py --lang Mundari --direction hi2tgt --experiment sft_ppo
python recap_train_ppo.py --lang Mundari --direction tgt2hi --experiment sft_ppo
```

---

## 7. Wrapping one line as an HPC job

Every line in Groups B/C/D/E is a single-GPU job. Using the interactive
pattern already established for this project:

```bash
qsub -I -P misn.mota2.spons -N recap_dpo_bhili_hi2tgt -lselect=1:ncpus=1:ngpus=1 -lwalltime=08:00:00
# once the session opens:
cd /path/to/RECAP/code
python recap_train_dpo.py --lang Bhili --direction hi2tgt --experiment recap_dpo
```

For non-interactive batch submission (recommended once you're launching 60+
of these), put the same two lines in a small `.pbs` script and `qsub` it
directly instead of `-I`:

```bash
#!/bin/bash
#PBS -P misn.mota2.spons
#PBS -N recap_dpo_bhili_hi2tgt
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=08:00:00

cd /path/to/RECAP/code
python recap_train_dpo.py --lang Bhili --direction hi2tgt --experiment recap_dpo
```

`qsub job.pbs` for each of the 60+12+6 lines above (swap in each job's exact
command). If your PBS setup supports job dependencies, chain Group D behind
its Group B counterpart instead of watching for completion manually:

```bash
qsub -W depend=afterok:<job_id_of_recap_dpo_run> recap_dpo_grpo_job.pbs
```

Walltime above (`08:00:00`) is a placeholder — size it to the dataset (a
50K-row direction's `recap_dpo` run will take longer than a small ablation
preset with a heavily-filtered pair set) and adjust per job.

---

## 8. Group F — evaluation (after everything trained)

```bash
python recap_evaluate.py
```

Runs everything (all 14 experiments x all 6 directions) in one process. Can
also be split into 6 per-direction jobs (`--lang X --direction Y`, no
`--experiment`) if you'd rather not wait for every direction's training to
finish before evaluating any of them.

---

## 9. Group G — reporting (after Group F)

```bash
python recap_report_tables.py
python recap_report_plots.py   # needs matplotlib: pip install matplotlib
```

---

## 10. Group H — Table 10 human validation (after Group G, human-in-the-loop)

```bash
python recap_sample_for_human_eval.py --experiment recap_dpo --n 500
# hand the sampled CSV to an annotator, fill in the human_preferred column, then:
python recap_human_eval_analysis.py --labeled_csv ../recap_human_eval/sample_recap_dpo_500.csv
```

---

## 11. Resume — what happens if a job dies mid-run

All three trainers are now resumable (walltime limits / preemption /
crashes on HPC are common enough that this matters):

- **Completion check**: every training script checks for
  `run_manifest.json` (written only at successful completion) before doing
  anything else. If it exists, the job just prints `[Skip] ... already
  completed` and exits — safe to blindly resubmit any line above.
- **DPO** (`recap_train_dpo.py`): uses HF Trainer's own
  `resume_from_checkpoint` against its periodic `trainer_state/checkpoint-*`
  saves. Also re-evaluates any already-saved best-validation checkpoint on
  resume so it doesn't lose track of the best score found before the crash.
- **GRPO / PPO** (hand-rolled loops): save a `latest_state/` snapshot every
  `save_steps` updates via `accelerate`'s own `save_state()`/`load_state()`
  (model + optimizer + RNG, DDP-safe) plus a small JSON with the loop step
  and best-so-far validation composite. On restart, if `latest_state/`
  exists, training picks up from that exact step instead of restarting from
  the SFT/DPO init checkpoint. This snapshot is deleted automatically once
  the run completes successfully (it's not needed after that, and can be
  sizeable).

So: **if a job dies, just resubmit the exact same command** -- no manual
cleanup needed.

---

## 12. Multi-seed (optional, not in the default job list above)

`config.py`'s `SEEDS = [13, 42, 2026]`. The paper asks for seed mean +- std
"whenever possible" for the learning conditions. To add this, repeat Groups
B/C/D/E with `--seed 42` and `--seed 2026` appended to each command (and
`recap_evaluate.py --seed 42` / `--seed 2026` for Group F) -- this triples
the 78 training jobs to 234. Left out of the default list above; add it once
the single-seed pass looks sane.

---

## 13. What's explicitly NOT in this guide (separate scope)

The 8 targeted ablations from `IMPLEMENTATION_PLAN.md` (candidate-diversity,
calibration-method, confidence/delta-grid, pair-selection-strategy,
data-scale, GRPO-group-size, weight-sensitivity) are not wired up as
runnable `--experiment` names yet -- see the "NOT YET WIRED" comments in
`config.py`. Only the 8-condition main matrix and 6-step
preference-construction ablation (14 experiments total, all covered above)
are currently runnable.

---

## 14. Smoke-testing on a small slice of data first

Don't launch 78 real GPU jobs against the full ~200K-row datasets without
first checking the pipeline actually runs end-to-end. `recap_split.py`
supports a `--n_samples N` flag that randomly samples N rows from
`maha_data_2` **before** splitting -- every downstream stage (calibrate,
score, mine_pairs, balance_pairs, train, evaluate) just reads whatever ends
up in `recap_splits/<lang>/<direction>/`, so shrinking the input at Stage 1
is enough to make the entire pipeline fast, with zero changes anywhere else.

```bash
# One direction, 200 rows instead of ~200K:
python recap_split.py --lang Bhili --direction hi2tgt --n_samples 200
python recap_calibrate.py --lang Bhili --direction hi2tgt
python recap_score.py --lang Bhili --direction hi2tgt
python recap_mine_pairs.py --lang Bhili --direction hi2tgt --experiment recap_dpo
python recap_balance_pairs.py --lang Bhili --direction hi2tgt --experiment recap_dpo
python recap_train_dpo.py --lang Bhili --direction hi2tgt --experiment recap_dpo
python recap_evaluate.py --lang Bhili --direction hi2tgt --experiment recap_dpo
```

Notes:

- `--n_samples` requires `--lang`/`--direction` (no bulk/loop-all mode) --
  this is deliberate, so you can't accidentally shrink every direction's
  real data at once.
- The 80/10/10 split still applies to the sampled rows, so `--n_samples 200`
  gives ~160/20/20 train/val/test -- small enough that DPO training finishes
  in minutes, not hours, and you can quickly check nothing crashes, the
  pre-flight checks pass, and the eval report looks sane.
- **Before a real run for the same `(lang, direction)`**, delete the test
  output first: `rm -rf recap_splits/<lang>/<direction>/` (and anything
  downstream that got built from it: `recap_calib`, `recap_rewards`,
  `recap_pairs`, `recap_dpo`/`recap_grpo`/`recap_ppo`, `recap_eval` for that
  direction) -- otherwise every later stage sees `manifest.csv` /
  `run_manifest.json` already there and skips as "already done," silently
  leaving you with the tiny test split instead of the real one.
- GRPO/PPO don't need a separate small-data flag for smoke-testing -- point
  `cfg.GRPO_SETTINGS.num_updates` / `cfg.PPO_SETTINGS.num_updates` down
  temporarily (e.g. 10) in `config.py` if you want a fast end-to-end check
  of those too, alongside the small `--n_samples` split.

---

## 15. Switching from a smoke test to the real full-data run

Once the small `--n_samples` test above looks sane, move to the real run for
that same `(lang, direction)`:

1. **Delete that direction's test output** -- everything built from the
   small split has to go, or every stage below will just see its manifest/
   run_manifest already there and skip as "already done," silently leaving
   you stuck on the tiny test data:

   ```bash
   rm -rf recap_splits/bhili/hi2tgt
   rm -rf recap_calib/bhili/hi2tgt
   rm -rf recap_rewards/bhili/hi2tgt
   rm -rf recap_pairs/bhili/hi2tgt
   rm -rf recap_dpo/bhili/hi2tgt
   rm -rf recap_grpo/bhili/hi2tgt
   rm -rf recap_ppo/bhili/hi2tgt
   rm -rf recap_eval/bhili/hi2tgt
   ```

   (swap in whichever `<lang>/<direction>` you tested with.)

2. **If you temporarily lowered `num_updates`** in `config.py` for a GRPO/PPO
   smoke test (§14's last bullet), set it back to the real value (2000 by
   default) before continuing -- otherwise the real run also stops after
   just a few updates.

3. **Rerun without `--n_samples`** -- this now processes the full dataset,
   and every stage after it (Groups B-E onward) runs exactly as documented
   in §2-§10 above:

   ```bash
   python recap_split.py --lang Bhili --direction hi2tgt
   python recap_calibrate.py --lang Bhili --direction hi2tgt
   python recap_score.py --lang Bhili --direction hi2tgt
   python recap_mine_pairs.py --lang Bhili --direction hi2tgt
   python recap_balance_pairs.py --lang Bhili --direction hi2tgt
   # ... then Groups B-E training commands as usual
   ```

Directions you never touched with `--n_samples` don't need any of this --
they were never shrunk, so their split is already full-size and ready to go
straight into Groups B-E.
