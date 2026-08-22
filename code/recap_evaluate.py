"""
Stage 9 -- test-set evaluation. Decodes the (never-touched-until-now) test
split via recap_infer.translate() (same code path used for deployment), scores
BLEU/chrF++/COMET as three SEPARATE metrics plus repetition/length-mismatch
reporting diagnostics, computes deltas from the SFT baseline, and reports
paired-bootstrap confidence intervals for key deltas.

Also selects, per direction, which main-matrix experiment gets deployed
(recap_infer's BEST_CHECKPOINT) -- chosen by each experiment's OWN best
validation-time composite (recorded in its run_manifest.json during
training), never by peeking at test-set numbers.

Run:
    python recap_evaluate.py --lang Bhili --direction hi2tgt --experiment recap_dpo
    python recap_evaluate.py --lang Bhili --direction hi2tgt   # all configured experiments for this direction
    python recap_evaluate.py                                   # everything
"""

from __future__ import annotations

import argparse
import json
import math
import random

import pandas as pd

import config as cfg
import recap_infer
from recap_reward import RewardEngine

N_BOOTSTRAP = 1000


def _rho_len(candidate: str, reference: str) -> float:
    return len(str(candidate)) / max(len(str(reference)), 1)


def _paired_bootstrap_ci(deltas: list[float], n_resamples: int = N_BOOTSTRAP, seed: int = 13) -> tuple[float, float]:
    if not deltas:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_resamples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    return (lo, hi)


def evaluate_one(lang: str, direction: str, experiment: str, seed: int) -> dict:
    out_path = cfg.eval_report_path(lang, direction, experiment, seed)
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)

    test_path = cfg.split_dir(lang, direction) / "test.csv"
    if not test_path.exists():
        print(f"[Skip] {lang}/{direction}: {test_path} not found -- run recap_split.py first")
        return {}

    test_df = pd.read_csv(test_path)
    source_ids = test_df["source_id"].tolist()
    sources, references = test_df["source"].tolist(), test_df["gold_truth"].tolist()

    translations = recap_infer.translate(sources, lang, direction, experiment=experiment)

    engine = RewardEngine(cfg.REWARD_PRESETS["recap_dpo"])  # config-agnostic for raw metric computation
    raw = engine.compute_raw_metrics(sources, translations, references)
    valid_mask = [r["valid"] and math.isfinite(r["bleu"]) for r in raw]
    valid_idx = [i for i, ok in enumerate(valid_mask) if ok]

    if not valid_idx:
        print(f"[Warn] {lang}/{direction}/{experiment}: no valid translations")
        report = {"lang": lang, "direction": direction, "experiment": experiment, "seed": seed, "n_valid": 0}
    else:
        import sacrebleu

        valid_hyps = [translations[i] for i in valid_idx]
        valid_refs = [references[i] for i in valid_idx]
        bleu_scores = [raw[i]["bleu"] for i in valid_idx]
        chrf_scores = [raw[i]["chrf"] for i in valid_idx]
        comet_scores = [raw[i]["comet"] for i in valid_idx]
        rho_rep_scores = [raw[i]["rep"] for i in valid_idx]  # same formula as training-time P_rep
        rho_len_scores = [_rho_len(translations[i], references[i]) for i in valid_idx]

        # True corpus-level BLEU/ChrF++ (n-gram statistics aggregated over the
        # whole test set, NOT the mean of sentence-level scores -- those are
        # well-known to differ due to the non-linear brevity penalty and
        # n-gram clipping; sacrebleu itself warns against conflating them).
        # COMET has no separate corpus formula -- segment-score mean IS the
        # standard "corpus COMET" convention, so sentence_avg==corpus there.
        corpus_bleu = sacrebleu.corpus_bleu(valid_hyps, [valid_refs]).score / 100.0
        corpus_chrf = sacrebleu.corpus_chrf(valid_hyps, [valid_refs]).score / 100.0

        report = {
            "lang": lang, "direction": direction, "experiment": experiment, "seed": seed,
            "n_total": len(test_df), "n_valid": len(valid_idx),
            "corpus": {
                "bleu": corpus_bleu,
                "chrf": corpus_chrf,
                "comet": sum(comet_scores) / len(comet_scores),
            },
            "sentence_avg": {
                "bleu": sum(bleu_scores) / len(bleu_scores),
                "chrf": sum(chrf_scores) / len(chrf_scores),
                "comet": sum(comet_scores) / len(comet_scores),
            },
            "diagnostics": {
                "rho_rep_mean": sum(rho_rep_scores) / len(rho_rep_scores),
                "rho_len_mean": sum(rho_len_scores) / len(rho_len_scores),
                "invalid_rate": 1 - len(valid_idx) / len(test_df),
            },
            # Keyed by source_id so compute_deltas() can align two conditions
            # by EXAMPLE, not by list position -- two conditions can flag
            # different rows invalid, which would silently misalign a
            # position-based zip().
            "per_sentence": {
                "source_id": [source_ids[i] for i in valid_idx],
                "bleu": bleu_scores, "chrf": chrf_scores, "comet": comet_scores,
            },
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Done] {lang}/{direction}/{experiment}: "
          f"BLEU={report.get('corpus', {}).get('bleu', float('nan')):.4f} "
          f"chrF++={report.get('corpus', {}).get('chrf', float('nan')):.4f} "
          f"COMET={report.get('corpus', {}).get('comet', float('nan')):.4f}")
    return report


def compute_deltas(lang: str, direction: str, experiment: str, seed: int) -> dict:
    """Delta from the SFT baseline, plus paired bootstrap CIs (Eq. 78-80).
    Paired means paired BY EXAMPLE: SFT and the condition can flag different
    rows invalid, so per-sentence scores are aligned by source_id (the
    intersection of both conditions' valid sets), never by raw list
    position."""
    sft_report = evaluate_one(lang, direction, "sft", seed)
    cond_report = evaluate_one(lang, direction, experiment, seed)
    if not sft_report.get("per_sentence") or not cond_report.get("per_sentence"):
        return {}

    sft_ids = sft_report["per_sentence"]["source_id"]
    cond_ids = cond_report["per_sentence"]["source_id"]
    sft_pos = {sid: i for i, sid in enumerate(sft_ids)}
    cond_pos = {sid: i for i, sid in enumerate(cond_ids)}
    common_ids = [sid for sid in sft_ids if sid in cond_pos]

    if not common_ids:
        return {}

    deltas = {"n_paired": len(common_ids)}
    for metric in ("bleu", "chrf", "comet"):
        sft_vals = [sft_report["per_sentence"][metric][sft_pos[sid]] for sid in common_ids]
        cond_vals = [cond_report["per_sentence"][metric][cond_pos[sid]] for sid in common_ids]
        per_sentence_delta = [c - s for c, s in zip(cond_vals, sft_vals)]
        point_delta = cond_report["corpus"][metric] - sft_report["corpus"][metric]
        ci_lo, ci_hi = _paired_bootstrap_ci(per_sentence_delta)
        deltas[f"delta_{metric}"] = point_delta
        deltas[f"delta_{metric}_ci"] = [ci_lo, ci_hi]
    return deltas


def select_best_checkpoint(lang: str, direction: str, seed: int = cfg.SEED) -> None:
    """Picks the deployed checkpoint by each experiment's OWN best
    validation-time composite (from run_manifest.json), never by test-set
    numbers -- test stays untouched for reporting only."""
    best_name, best_composite, best_path = None, float("-inf"), None
    for name in cfg.MAIN_MATRIX_ORDER:
        exp = cfg.EXPERIMENTS[name]
        if exp.trainer == "sft":
            candidate_path = cfg.sft_checkpoint_path(lang, direction)
            composite = float("-inf")  # SFT is the floor, never auto-selected over a real candidate
        else:
            stage_root = {"dpo": cfg.DPO_ROOT, "grpo": cfg.GRPO_ROOT, "ppo": cfg.PPO_ROOT}[exp.trainer]
            manifest_path = cfg.run_manifest_path(stage_root, lang, direction, name, seed)
            candidate_path = cfg.checkpoint_dir(stage_root, lang, direction, name, seed)
            if not manifest_path.exists():
                continue
            with open(manifest_path) as f:
                manifest = json.load(f)
            composite = manifest.get("best_validation_composite") or manifest.get("final_validation", {}).get("composite", float("-inf"))
        if composite > best_composite:
            best_composite, best_name, best_path = composite, name, candidate_path

    if best_name is None:
        print(f"[Skip] {lang}/{direction}: no trained checkpoints found yet for best-checkpoint selection")
        return

    out_path = cfg.best_checkpoint_path(lang, direction)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"experiment": best_name, "checkpoint_path": str(best_path), "validation_composite": best_composite}, f, indent=2)
    print(f"[Done] {lang}/{direction}: best checkpoint = {best_name} (val composite={best_composite:.4f}) -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=cfg.LANGUAGES, default=None)
    parser.add_argument("--direction", choices=cfg.DIRECTIONS, default=None)
    parser.add_argument("--experiment", choices=list(cfg.EXPERIMENTS.keys()), default=None)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    args = parser.parse_args()
    if bool(args.lang) != bool(args.direction):
        parser.error("--lang and --direction must be given together")

    lang_direction_jobs = [(args.lang, args.direction)] if args.lang else [
        (lang, direction) for lang in cfg.LANGUAGES for direction in cfg.DIRECTIONS
    ]
    experiments = [args.experiment] if args.experiment else list(cfg.EXPERIMENTS.keys())

    for lang, direction in lang_direction_jobs:
        for experiment in experiments:
            deltas = compute_deltas(lang, direction, experiment, args.seed)
            if deltas:
                print(f"    deltas vs SFT: "
                      f"dBLEU={deltas['delta_bleu']:.4f} dChrF++={deltas['delta_chrf']:.4f} "
                      f"dCOMET={deltas['delta_comet']:.4f}")
        select_best_checkpoint(lang, direction, args.seed)


if __name__ == "__main__":
    main()
