"""
PreferenceBuilder -- turns the static four-model candidate pool into fixed
DPO pairs, using a fitted RewardEngine. Only ever runs on training-split
sources. See IMPLEMENTATION_PLAN.md Stage 4 and paper Section 11.4.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

from tqdm import tqdm

import config as cfg
from recap_reward import RewardEngine


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


class PreferenceBuilder:
    def __init__(self, reward_engine: RewardEngine, margin_delta: float):
        self.reward_engine = reward_engine
        self.margin_delta = margin_delta

    def build_pairs(self, train_df, model_pred_cols: dict[str, str], raw_rewards_long) -> list[dict[str, Any]]:
        """train_df: one row per source (columns: source, gold_truth,
        source_id, plus one prediction column per model per model_pred_cols).
        raw_rewards_long: Stage 3's rewards.csv, long format -- one row per
        (source_id, model) with raw bleu/chrf/comet/rep/len/valid columns
        (already computed once, expensive metrics not recomputed here).
        Returns a flat list of pair records, one per retained (source, model
        pair)."""
        rc = self.reward_engine.reward_config
        all_pairs: list[dict[str, Any]] = []

        raw_by_source_model = {
            (r["source_id"], r["model"]): r for r in raw_rewards_long.to_dict("records")
        }

        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Mining pairs"):
            source = row["source"]
            reference = row["gold_truth"]
            source_id = row["source_id"]

            model_texts = {m: row[col] for m, col in model_pred_cols.items()}
            models = list(model_texts.keys())

            raw_rows = [raw_by_source_model[(source_id, m)] for m in models]
            scored = self.reward_engine.score_raw_rows(raw_rows)
            per_model = {m: s for m, s in zip(models, scored)}

            seen_norm_pairs: set[tuple[str, str]] = set()

            for model_a, model_b in itertools.combinations(models, 2):
                score_a, score_b = per_model[model_a], per_model[model_b]
                text_a, text_b = model_texts[model_a], model_texts[model_b]

                if not score_a["valid"] or not score_b["valid"]:
                    continue
                if math.isnan(score_a["reward"]) or math.isnan(score_b["reward"]):
                    continue

                norm_a, norm_b = _normalize_text(text_a), _normalize_text(text_b)
                if norm_a == norm_b:
                    continue
                dedup_key = tuple(sorted([norm_a, norm_b]))
                if dedup_key in seen_norm_pairs:
                    continue
                seen_norm_pairs.add(dedup_key)

                delta = score_a["reward"] - score_b["reward"]
                if delta == 0:
                    continue  # exact tie -- not a valid preference signal, regardless of margin-filter setting
                if rc.use_margin_filter and abs(delta) <= self.margin_delta:
                    continue

                if delta > 0:
                    chosen_model, rejected_model = model_a, model_b
                    chosen_text, rejected_text = text_a, text_b
                    chosen_score, rejected_score = score_a, score_b
                else:
                    chosen_model, rejected_model = model_b, model_a
                    chosen_text, rejected_text = text_b, text_a
                    chosen_score, rejected_score = score_b, score_a

                all_pairs.append({
                    "source_id": source_id,
                    "source": source,
                    "gold_truth": reference,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                    "chosen_model": chosen_model,
                    "rejected_model": rejected_model,
                    "pair_type": "-".join(sorted([model_a, model_b])),
                    "chosen_reward": chosen_score["reward"],
                    "rejected_reward": rejected_score["reward"],
                    "reward_margin": abs(delta),
                    "chosen_bleu": chosen_score["bleu"], "chosen_chrf": chosen_score["chrf"],
                    "chosen_comet": chosen_score["comet"], "chosen_rep": chosen_score["rep"],
                    "chosen_len": chosen_score["len"],
                    "rejected_bleu": rejected_score["bleu"], "rejected_chrf": rejected_score["chrf"],
                    "rejected_comet": rejected_score["comet"], "rejected_rep": rejected_score["rep"],
                    "rejected_len": rejected_score["len"],
                })

        # Natural per-source cap: at most 6 unordered pairs already enforced by
        # itertools.combinations over 4 models. If MAX_PAIRS_PER_SOURCE is set
        # lower, trim here (keep highest-margin pairs per source).
        if cfg.MAX_PAIRS_PER_SOURCE < 6:
            by_source: dict[Any, list[dict]] = {}
            for p in all_pairs:
                by_source.setdefault(p["source_id"], []).append(p)
            trimmed = []
            for pairs in by_source.values():
                pairs.sort(key=lambda p: p["reward_margin"], reverse=True)
                trimmed.extend(pairs[: cfg.MAX_PAIRS_PER_SOURCE])
            all_pairs = trimmed

        return all_pairs
