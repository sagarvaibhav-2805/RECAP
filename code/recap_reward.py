"""
RewardEngine -- the one place BLEU/chrF++/COMET/repetition/length get turned
into a RECAP reward. Stage 2 (calibration) calls fit(); Stage 3 (scoring),
PreferenceBuilder, GRPO, and PPO all call score_batch() on the same fitted
instance (or a reloaded serialize()d one). See IMPLEMENTATION_PLAN.md
"Section 11.3 C. RewardEngine" and Stage 2/3.

Two scoring paths:
  - precomputed metrics (bleu/chrf/comet already known, from maha_data_2 CSVs)
    -- used by Stage 2/3 on the static four-model candidate pool.
  - raw text only -- BLEU/chrF++ computed with sacrebleu, COMET computed with
    a lazily-loaded model -- used by GRPO/PPO to score fresh rollouts.
"""

from __future__ import annotations

# Silence noisy COMET/Lightning chatter before torch/comet are imported --
# same fix as build_maha_data.py, applies here for the same reason.
import logging
import warnings

warnings.filterwarnings("ignore")
for _logger_name in [
    "pytorch_lightning", "pytorch_lightning.utilities.rank_zero",
    "pytorch_lightning.accelerators.cuda",
    "lightning_fabric", "lightning_fabric.utilities.rank_zero",
    "lightning.pytorch", "lightning.pytorch.utilities.rank_zero",
]:
    logging.getLogger(_logger_name).setLevel(logging.ERROR)
try:
    import pytorch_lightning.utilities.rank_zero as _plrz
    _plrz.rank_zero_info = lambda *a, **k: None
except ImportError:
    pass

import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import config as cfg
import recap_utils

COMET_MODEL_NAME = "Unbabel/wmt22-comet-da"
COMET_BATCH_SIZE = 64
EXTREME_LENGTH_RATIO = 10.0        # candidate > 10x reference length -> invalid
EXTREME_LENGTH_CHARS_ABSOLUTE = 4000  # backstop cap, independent of reference length


class RewardEngine:
    def __init__(self, reward_config: cfg.RewardConfig):
        self.reward_config = reward_config
        self.stats: dict[str, dict[str, float]] = {}
        self._fitted = False
        self._comet_model = None

    # -- metric computation --------------------------------------------

    def _get_comet_model(self):
        if self._comet_model is None:
            from comet import download_model, load_from_checkpoint
            model_path = download_model(COMET_MODEL_NAME)
            self._comet_model = load_from_checkpoint(model_path)
        return self._comet_model

    @staticmethod
    def _repetition_penalty(text: str) -> float:
        words = text.split()
        n = cfg.REPETITION_NGRAM_SIZE
        if len(words) < n:
            return 0.0
        ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        seen: set[tuple[str, ...]] = set()
        repeated = 0
        for g in ngrams:
            if g in seen:
                repeated += 1
            else:
                seen.add(g)
        return repeated / max(len(ngrams), 1)

    @staticmethod
    def _length_penalty(candidate: str, reference: str) -> float:
        if cfg.LENGTH_METRIC == "char":
            len_c, len_r = len(candidate), len(reference)
        else:
            len_c, len_r = len(candidate.split()), len(reference.split())
        return abs(len_c / max(len_r, 1) - 1)

    @staticmethod
    def _is_valid(candidate: str, reference: str) -> bool:
        """Empty candidates are always invalid. 'Extreme length' is judged
        RELATIVE to the reference (a ratio cap), not an absolute character
        count -- a long correct translation of a long reference should not be
        flagged, and a short degenerate repetition-loop against a short
        reference should be. An absolute ceiling is kept too, as a backstop
        against a runaway generation loop when the reference itself is only
        a word or two."""
        if candidate is None:
            return False
        text = str(candidate).strip()
        if not text:
            return False
        if len(text) > EXTREME_LENGTH_CHARS_ABSOLUTE:
            return False
        ref_text = str(reference).strip() if reference is not None else ""
        if ref_text:
            if cfg.LENGTH_METRIC == "char":
                cand_len, ref_len = len(text), len(ref_text)
            else:
                cand_len, ref_len = len(text.split()), len(ref_text.split())
            if cand_len > EXTREME_LENGTH_RATIO * max(ref_len, 1):
                return False
        return True

    def compute_raw_metrics(
        self,
        sources: list[str],
        candidates: list[str],
        references: list[str],
        precomputed: list[dict[str, float | None]] | None = None,
    ) -> list[dict[str, Any]]:
        """Returns one dict per row: bleu/chrf (0-1), comet, rep, len (all
        raw, un-standardized), plus a 'valid' flag. Invalid candidates
        (empty/malformed/extreme-length) are excluded from the sacrebleu/COMET
        calls entirely -- not just flagged afterward -- so a garbage rollout
        completion (which WILL happen during early GRPO/PPO training) can
        never crash or skew a batched metric call."""
        n = len(sources)
        valid_flags = [self._is_valid(candidates[i], references[i]) for i in range(n)]
        valid_idx = [i for i in range(n) if valid_flags[i]]

        need_bleu_chrf = precomputed is None or any(p.get("bleu") is None for p in precomputed)
        need_comet = precomputed is None or any(p.get("comet") is None for p in precomputed)

        bleu_vals: dict[int, float] = {}
        chrf_vals: dict[int, float] = {}
        if need_bleu_chrf and valid_idx:
            import sacrebleu
            for i in valid_idx:
                if precomputed is not None and precomputed[i].get("bleu") is not None:
                    continue
                bleu_vals[i] = sacrebleu.sentence_bleu(candidates[i], [references[i]]).score / 100.0
                chrf_vals[i] = sacrebleu.sentence_chrf(candidates[i], [references[i]]).score / 100.0

        comet_vals: dict[int, float] = {}
        if need_comet and valid_idx:
            import torch
            comet_idx = [
                i for i in valid_idx
                if not (precomputed is not None and precomputed[i].get("comet") is not None)
            ]
            if comet_idx:
                model = self._get_comet_model()
                data = [{"src": sources[i], "mt": candidates[i], "ref": references[i]} for i in comet_idx]
                gpus = 1 if torch.cuda.is_available() else 0
                output = model.predict(data, batch_size=COMET_BATCH_SIZE, gpus=gpus, progress_bar=False, num_workers=0)
                comet_vals = dict(zip(comet_idx, output.scores))

        rows = []
        for i in range(n):
            candidate, reference = candidates[i], references[i]
            valid = valid_flags[i]

            if not valid:
                rows.append({
                    "bleu": float("nan"), "chrf": float("nan"), "comet": float("nan"),
                    "rep": float("nan"), "len": float("nan"), "valid": False,
                })
                continue

            if precomputed is not None and precomputed[i].get("bleu") is not None:
                raw_bleu = precomputed[i]["bleu"]
                bleu = raw_bleu / 100.0 if raw_bleu > 1.0 else raw_bleu
            else:
                bleu = bleu_vals[i]

            if precomputed is not None and precomputed[i].get("chrf") is not None:
                raw_chrf = precomputed[i]["chrf"]
                chrf = raw_chrf / 100.0 if raw_chrf > 1.0 else raw_chrf
            else:
                chrf = chrf_vals[i]

            if precomputed is not None and precomputed[i].get("comet") is not None:
                comet = precomputed[i]["comet"]
            else:
                comet = comet_vals[i]

            rep = self._repetition_penalty(candidate)
            length = self._length_penalty(candidate, reference)

            rows.append({
                "bleu": bleu, "chrf": chrf, "comet": comet,
                "rep": rep, "len": length, "valid": valid,
            })
        return rows

    # -- fit / score / serialize -----------------------------------------

    def fit(self, training_candidates) -> None:
        """training_candidates: a pandas DataFrame with columns
        source, gold_truth, prediction, bleu, chrf, comet (bleu/chrf on 0-100
        scale, one row per (source, model) pair). Fits mu/sigma per quantity
        from valid rows only."""
        precomputed = [
            {"bleu": b, "chrf": c, "comet": m}
            for b, c, m in zip(training_candidates["bleu"], training_candidates["chrf"], training_candidates["comet"])
        ]
        rows = self.compute_raw_metrics(
            training_candidates["source"].tolist(),
            training_candidates["prediction"].tolist(),
            training_candidates["gold_truth"].tolist(),
            precomputed=precomputed,
        )
        valid_rows = [
            r for r in rows
            if r["valid"] and all(math.isfinite(r[q]) for q in cfg.CALIBRATION_QUANTITIES)
        ]
        if not valid_rows:
            raise ValueError("RewardEngine.fit(): no valid training candidates to calibrate from")
        for q in cfg.CALIBRATION_QUANTITIES:
            vals = [r[q] for r in valid_rows]
            mu = statistics.fmean(vals)
            sigma = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            self.stats[q] = {"mu": mu, "sigma": sigma}
        self._fitted = True

    def score_raw_rows(self, raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Cheap path: raw_rows already have bleu/chrf/comet/rep/len/valid
        (e.g. loaded from rewards.csv, computed once in Stage 3). Applies
        standardization (using this engine's frozen fit() stats) and this
        engine's reward_config weights -- no COMET/sacrebleu call, so this is
        safe to call once per reward-preset ablation without re-touching the
        expensive metrics."""
        if not self._fitted:
            raise RuntimeError("RewardEngine.score_raw_rows() called before fit()/load()")
        results = []
        rc = self.reward_config
        for row in raw_rows:
            if not row["valid"] or not all(math.isfinite(row[q]) for q in cfg.CALIBRATION_QUANTITIES):
                results.append({**row, "reward": float("nan")})
                continue
            if rc.use_calibration:
                std = {
                    q: (row[q] - self.stats[q]["mu"]) / max(self.stats[q]["sigma"], cfg.CALIBRATION_EPSILON)
                    for q in cfg.CALIBRATION_QUANTITIES
                }
            else:
                std = {q: row[q] for q in cfg.CALIBRATION_QUANTITIES}
            quality = (std["bleu"] + std["chrf"] + std["comet"]) / 3.0
            reward = rc.w_quality * quality - rc.w_rep * std["rep"] - rc.w_len * std["len"]
            results.append({
                **row,
                "bleu_std": std["bleu"], "chrf_std": std["chrf"], "comet_std": std["comet"],
                "rep_std": std["rep"], "len_std": std["len"],
                "quality": quality, "reward": reward,
            })
        return results

    def score_batch(
        self,
        sources: list[str],
        candidates: list[str],
        references: list[str],
        precomputed: list[dict[str, float | None]] | None = None,
    ) -> list[dict[str, Any]]:
        """Full path: computes raw metrics (COMET/sacrebleu, expensive) then
        scores them. Used by GRPO/PPO on fresh rollouts, where no cached
        rewards.csv exists yet."""
        raw_rows = self.compute_raw_metrics(sources, candidates, references, precomputed=precomputed)
        return self.score_raw_rows(raw_rows)

    def serialize(self, path: Path) -> None:
        payload = {
            "reward_config": asdict(self.reward_config),
            "stats": self.stats,
            "calibration_quantities": cfg.CALIBRATION_QUANTITIES,
            "repetition_ngram_size": cfg.REPETITION_NGRAM_SIZE,
            "length_metric": cfg.LENGTH_METRIC,
            "comet_model_name": COMET_MODEL_NAME,
        }
        payload["config_hash"] = recap_utils.config_hash(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: Path, reward_config: "cfg.RewardConfig | None" = None) -> "RewardEngine":
        """fit() stats don't depend on reward_config (mu/sigma are just
        descriptive stats of the raw metrics), so one stats.json is shared by
        every reward-preset for a direction. Pass reward_config to attach the
        preset actually being scored; omit it to reuse whatever was
        serialized (informational only in that case)."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rc = reward_config if reward_config is not None else cfg.RewardConfig(**payload["reward_config"])
        engine = cls(rc)
        engine.stats = payload["stats"]
        engine._fitted = True
        return engine
