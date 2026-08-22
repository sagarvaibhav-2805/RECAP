"""
Stage 1 -- splits maha_data_2/<lang>/<direction>.csv into train/val/test,
source-level, with a stable manifest for auditing.

Run one combination:
    python recap_split.py --lang Bhili --direction hi2tgt
Run all configured combinations sequentially:
    python recap_split.py
"""

from __future__ import annotations

import argparse
import hashlib

import pandas as pd

import config as cfg
import recap_checks

MODEL_NAMES = cfg.MODEL_NAMES


def _find_pred_col(columns: list[str], model: str) -> str:
    candidates = [
        c for c in columns
        if c.endswith(f"_{model}") and c not in (f"BLEU_{model}", f"chrf++_{model}", f"COMET_{model}")
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one prediction column for model '{model}', found {candidates}")
    return candidates[0]


def _checksum(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def _normalize(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def _source_id(lang: str, direction: str, dedup_key: str) -> str:
    raw = f"{lang}|{direction}|{dedup_key}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def process_one(lang: str, direction: str) -> None:
    in_path = cfg.maha_data_2_csv(lang, direction)
    if not in_path.exists():
        print(f"[Skip] {lang}/{direction}: {in_path} not found")
        return

    df = pd.read_csv(in_path)
    pred_cols = {m: _find_pred_col(list(df.columns), m) for m in MODEL_NAMES}

    out_dir = cfg.split_dir(lang, direction)
    manifest_path = out_dir / "manifest.csv"
    if manifest_path.exists():
        print(f"[Skip] {lang}/{direction}: already split (delete {manifest_path} to redo)")
        return

    # -- validation: reject-and-log, never silently drop -----------------
    rejects = []
    valid_rows = []
    for idx, row in df.iterrows():
        reasons = []
        if pd.isna(row.get("source")) or not str(row.get("source")).strip():
            reasons.append("missing source")
        if pd.isna(row.get("gold_truth")) or not str(row.get("gold_truth")).strip():
            reasons.append("missing gold_truth")
        for model, col in pred_cols.items():
            if col not in df.columns:
                reasons.append(f"missing candidate column {col}")
        if reasons:
            rejects.append({"row_index": idx, "reasons": "; ".join(reasons)})
        else:
            valid_rows.append(idx)

    if rejects:
        rejects_path = out_dir / "rejected_rows.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rejects).to_csv(rejects_path, index=False)
        print(f"[Warn] {lang}/{direction}: rejected {len(rejects)} rows, logged to {rejects_path}")

    df = df.loc[valid_rows].reset_index(drop=True)

    # -- duplicate-aware grouping: exact normalized-source duplicates stay
    #    together in one split. NOTE: this only catches EXACT normalized
    #    duplicates, not fuzzy near-duplicates (paper also asks for near-dup
    #    handling; a similarity-based clustering pass is a documented gap,
    #    not implemented here to avoid an unbounded-cost pairwise comparison
    #    over hundreds of thousands of rows).
    df["dedup_key"] = df["source"].map(_normalize) + "||" + df["gold_truth"].map(_normalize)
    df["source_id"] = df["dedup_key"].map(lambda k: _source_id(lang, direction, k))
    df["source_checksum"] = df["source"].map(_checksum)
    df["reference_checksum"] = df["gold_truth"].map(_checksum)

    groups = df.groupby("dedup_key")["source_id"].first().reset_index()
    groups = groups.sample(frac=1.0, random_state=cfg.SPLIT_SEED).reset_index(drop=True)

    n_total = len(df)
    n_train_target = int(cfg.SPLIT_RATIOS["train"] * n_total)
    n_val_target = int(cfg.SPLIT_RATIOS["val"] * n_total)

    group_sizes = df.groupby("dedup_key").size()
    split_assignment: dict[str, str] = {}
    running = {"train": 0, "val": 0, "test": 0}
    for _, g in groups.iterrows():
        key = g["dedup_key"]
        size = int(group_sizes[key])
        if running["train"] < n_train_target:
            split_assignment[key] = "train"
            running["train"] += size
        elif running["val"] < n_val_target:
            split_assignment[key] = "val"
            running["val"] += size
        else:
            split_assignment[key] = "test"
            running["test"] += size

    df["split"] = df["dedup_key"].map(split_assignment)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_cols = ["source_id", "source_checksum", "reference_checksum", "split"]
    manifest = df[manifest_cols].drop_duplicates(subset=["source_id"]).copy()
    manifest["direction"] = direction
    manifest["candidate_column_mapping"] = str(pred_cols)
    manifest["split_seed"] = cfg.SPLIT_SEED
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    data_cols = [c for c in df.columns if c not in ("dedup_key",)]
    for split_name in ("train", "val", "test"):
        split_df = df.loc[df["split"] == split_name, data_cols]
        split_df.to_csv(out_dir / f"{split_name}.csv", index=False, encoding="utf-8-sig")

    print(f"[Done] {lang}/{direction}: {len(df)} rows -> "
          f"train={running['train']} val={running['val']} test={running['test']} "
          f"({len(rejects)} rejected)")


def main() -> None:
    recap_checks.run_static_checks()  # raises CheckFailure before any expensive/IO work if config.py is broken

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=cfg.LANGUAGES, default=None)
    parser.add_argument("--direction", choices=cfg.DIRECTIONS, default=None)
    args = parser.parse_args()
    if bool(args.lang) != bool(args.direction):
        parser.error("--lang and --direction must be given together")

    jobs = [(args.lang, args.direction)] if args.lang else [
        (lang, direction) for lang in cfg.LANGUAGES for direction in cfg.DIRECTIONS
    ]
    for lang, direction in jobs:
        process_one(lang, direction)


if __name__ == "__main__":
    main()
