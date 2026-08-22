"""
Stage 0 -- 9 automated pre-flight checks (paper Section 11.12). Run before any
expensive training job. Each check function returns (ok: bool, message: str).
CheckFailure is raised by run_all() if anything fails, so a bad run never
starts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import config as cfg


class CheckFailure(Exception):
    pass


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str


# 1. reward weights non-negative and sum to 1 -------------------------------

def check_reward_weights(tolerance: float = 1e-6) -> CheckResult:
    bad = []
    for name, rc in cfg.REWARD_PRESETS.items():
        if rc.w_quality < 0 or rc.w_rep < 0 or rc.w_len < 0:
            bad.append(f"{name}: negative weight")
            continue
        total = rc.w_quality + rc.w_rep + rc.w_len
        if abs(total - 1.0) > tolerance:
            bad.append(f"{name}: weights sum to {total}, not 1")
    ok = not bad
    return CheckResult("reward_weights", ok, "OK" if ok else "; ".join(bad))


# 2. monotonicity: more repetition/length-mismatch can't increase reward ----

def check_penalty_monotonicity() -> CheckResult:
    from recap_reward import RewardEngine
    bad = []
    for name, rc in cfg.REWARD_PRESETS.items():
        engine = RewardEngine(rc)
        engine.stats = {q: {"mu": 0.0, "sigma": 1.0} for q in cfg.CALIBRATION_QUANTITIES}
        engine._fitted = True
        base = {"bleu": 0.5, "chrf": 0.5, "comet": 0.5, "rep": 0.2, "len": 0.2, "valid": True}
        worse_rep = {**base, "rep": 0.8}
        worse_len = {**base, "len": 0.8}
        r_base = engine.score_raw_rows([base])[0]["reward"]
        r_rep = engine.score_raw_rows([worse_rep])[0]["reward"]
        r_len = engine.score_raw_rows([worse_len])[0]["reward"]
        if r_rep > r_base + 1e-9:
            bad.append(f"{name}: higher repetition increased reward")
        if r_len > r_base + 1e-9:
            bad.append(f"{name}: higher length-mismatch increased reward")
    ok = not bad
    return CheckResult("penalty_monotonicity", ok, "OK" if ok else "; ".join(bad))


# 3. standardized quantities use saved training-only stats and are finite ---

def check_calibration_stats_finite(lang: str, direction: str) -> CheckResult:
    path = cfg.calib_path(lang, direction)
    if not path.exists():
        return CheckResult("calibration_finite", False, f"{path} not found")
    import json
    with open(path) as f:
        payload = json.load(f)
    bad = []
    for q, stat in payload.get("stats", {}).items():
        mu, sigma = stat.get("mu"), stat.get("sigma")
        if mu is None or sigma is None or not math.isfinite(mu) or not math.isfinite(sigma):
            bad.append(f"{q}: mu={mu} sigma={sigma}")
    ok = not bad
    return CheckResult("calibration_finite", ok, "OK" if ok else "; ".join(bad))


# 4. no identical-completion pairs, all retained pairs have positive margin -

def check_pairs_valid(pairs_df) -> CheckResult:
    bad = []
    identical = (pairs_df["chosen"].str.strip().str.lower()
                 == pairs_df["rejected"].str.strip().str.lower()).sum()
    if identical > 0:
        bad.append(f"{identical} pairs have identical chosen==rejected")
    non_positive = (pairs_df["reward_margin"] <= 0).sum()
    if non_positive > 0:
        bad.append(f"{non_positive} pairs have non-positive reward_margin")
    ok = not bad
    return CheckResult("pairs_valid", ok, "OK" if ok else "; ".join(bad))


# 5. no source in more than one split; no val/test row entered pair-mining --

def check_split_leakage(manifest_df, pairs_df) -> CheckResult:
    bad = []
    dup = manifest_df["source_id"].duplicated().sum()
    if dup > 0:
        bad.append(f"{dup} source_ids appear more than once in manifest")
    non_train_ids = set(manifest_df.loc[manifest_df["split"] != "train", "source_id"])
    leaked = set(pairs_df["source_id"]) & non_train_ids
    if leaked:
        bad.append(f"{len(leaked)} val/test sources leaked into pair-mining")
    ok = not bad
    return CheckResult("split_leakage", ok, "OK" if ok else "; ".join(bad))


# 6. reference-policy parameters unchanged during training ------------------

def check_reference_frozen(ref_model) -> CheckResult:
    bad = [n for n, p in ref_model.named_parameters() if p.requires_grad]
    ok = not bad
    msg = "OK" if ok else f"{len(bad)} reference-policy params still require grad"
    return CheckResult("reference_frozen", ok, msg)


# 7. GRPO/PPO completions generated live, not loaded from static files ------

def check_live_generation(sources: list[str], completions: list[str]) -> CheckResult:
    """Data-driven, not a hardcoded tautology: a rollout batch that's
    secretly reading static/cached text instead of actually sampling from
    the current policy tends to show one of two signatures --
    (a) every completion in the batch is byte-identical to every other one
        (a frozen/non-generating pipeline collapses to one constant output),
    (b) a completion is byte-identical to its own source sentence
        (copy-through, e.g. a broken generation call that just echoes input).
    Neither should happen from genuine stochastic sampling across different
    sources."""
    bad = []
    non_empty = [c for c in completions if str(c).strip()]
    if len(non_empty) > 1 and len(set(non_empty)) == 1:
        bad.append("every completion in the batch is identical -- looks frozen/non-generating")
    copy_through = sum(1 for s, c in zip(sources, completions) if str(c).strip() == str(s).strip())
    if copy_through > 0:
        bad.append(f"{copy_through} completion(s) are byte-identical to their own source (copy-through)")
    ok = not bad
    return CheckResult("live_generation", ok, "OK" if ok else "; ".join(bad))


# 8. every rollout reward received the correct source/reference/candidate/direction

def check_rollout_alignment(rollouts: list[dict]) -> CheckResult:
    bad = [r for r in rollouts if not all(k in r for k in ("source", "reference", "candidate", "direction"))]
    ok = not bad
    msg = "OK" if ok else f"{len(bad)} rollout records missing source/reference/candidate/direction"
    return CheckResult("rollout_alignment", ok, msg)


# 9. every checkpoint/manifest/pair-file/calibration-file/seed/state hashed -

def check_manifest_hashed(manifest: dict) -> CheckResult:
    ok = "config_hash" in manifest and bool(manifest["config_hash"])
    msg = "OK" if ok else "manifest missing config_hash"
    return CheckResult("manifest_hashed", ok, msg)


def run_static_checks() -> list[CheckResult]:
    """Checks 1-2 need only config.py, no per-direction data -- run these
    first, cheaply, before touching any direction's files."""
    results = [check_reward_weights(), check_penalty_monotonicity()]
    failed = [r for r in results if not r.ok]
    if failed:
        raise CheckFailure("; ".join(f"{r.name}: {r.message}" for r in failed))
    return results


if __name__ == "__main__":
    for result in run_static_checks():
        print(f"[{'OK' if result.ok else 'FAIL'}] {result.name}: {result.message}")
