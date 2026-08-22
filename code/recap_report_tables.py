"""
Stage 11 -- builds Tables 3-9 (CSV) from already-saved outputs only (never
retrains/re-decodes). See IMPLEMENTATION_PLAN.md Stage 11 for the
paper-table-to-generator mapping. Tables 1, 2, 11 are manual/static (not
generated here).

Run:
    python recap_report_tables.py
"""

from __future__ import annotations

import json

import pandas as pd

import config as cfg


def _lang_prefix(columns: list[str]) -> str:
    for c in columns:
        if c.endswith("_nllb") and not c.startswith(("BLEU", "chrf", "COMET")):
            return c.split("_")[0]
    raise ValueError("Could not infer language prefix from columns")


def build_tables_3_5() -> dict[str, pd.DataFrame]:
    """Per-language reference-model BLEU/chrF++, split by direction/model,
    read directly from the stored maha_data_2 columns -- this reports the
    SFT candidate generators' own performance, not a RECAP experiment."""
    tables = {}
    for lang in cfg.LANGUAGES:
        rows = []
        for direction in cfg.DIRECTIONS:
            in_path = cfg.maha_data_2_csv(lang, direction)
            if not in_path.exists():
                continue
            df = pd.read_csv(in_path)
            prefix = _lang_prefix(df.columns.tolist())
            for model in cfg.MODEL_NAMES:
                rows.append({
                    "direction": direction, "model": model,
                    "bleu_mean": df[f"BLEU_{model}"].mean(),
                    "chrf_mean": df[f"chrf++_{model}"].mean(),
                })
        tables[lang] = pd.DataFrame(rows)
    return tables


def build_table_6() -> pd.DataFrame:
    """Macro-average validation BLEU/chrF++ per candidate model across all six
    directions -- used to justify the common mT5 backbone (paper Eq. 51-52).
    Uses the val split (Stage 1 output), not the raw maha_data_2 files, so the
    same held-out val rows are used everywhere else in the pipeline."""
    rows = []
    for model in cfg.MODEL_NAMES:
        per_direction = []
        for lang in cfg.LANGUAGES:
            for direction in cfg.DIRECTIONS:
                val_path = cfg.split_dir(lang, direction) / "val.csv"
                if not val_path.exists():
                    continue
                df = pd.read_csv(val_path)
                prefix = _lang_prefix(df.columns.tolist())
                score = (df[f"BLEU_{model}"].mean() / 100 + df[f"chrf++_{model}"].mean() / 100) / 2
                per_direction.append(score)
        macro = sum(per_direction) / len(per_direction) if per_direction else float("nan")
        rows.append({"model": model, "macro_avg_validation_score": macro})
    return pd.DataFrame(rows).sort_values("macro_avg_validation_score", ascending=False)


def _load_reports(experiment_names: list[str], seed: int) -> pd.DataFrame:
    rows = []
    for lang in cfg.LANGUAGES:
        for direction in cfg.DIRECTIONS:
            for experiment in experiment_names:
                path = cfg.eval_report_path(lang, direction, experiment, seed)
                if not path.exists():
                    continue
                with open(path) as f:
                    report = json.load(f)
                if "corpus" not in report:
                    continue
                rows.append({
                    "lang": lang, "direction": direction, "experiment": experiment,
                    "bleu": report["corpus"]["bleu"], "chrf": report["corpus"]["chrf"],
                    "comet": report["corpus"]["comet"],
                    "rho_rep": report["diagnostics"]["rho_rep_mean"],
                    "rho_len": report["diagnostics"]["rho_len_mean"],
                })
    return pd.DataFrame(rows)


def build_table_7(seed: int = cfg.SEED) -> pd.DataFrame:
    """Main results -- macro-average (equal weight per direction) across the
    8-condition main matrix."""
    df = _load_reports(cfg.MAIN_MATRIX_ORDER, seed)
    if df.empty:
        return df
    macro = df.groupby("experiment")[["bleu", "chrf", "comet", "rho_rep", "rho_len"]].mean().reset_index()
    macro["experiment"] = pd.Categorical(macro["experiment"], categories=cfg.MAIN_MATRIX_ORDER, ordered=True)
    return macro.sort_values("experiment")


def build_table_8(seed: int = cfg.SEED) -> pd.DataFrame:
    """Cumulative preference-construction ablation -- macro-average across
    the six directions, one row per ablation step."""
    df = _load_reports(cfg.PREFERENCE_ABLATION_ORDER, seed)
    if df.empty:
        return df
    macro = df.groupby("experiment")[["bleu", "chrf", "comet", "rho_rep", "rho_len"]].mean().reset_index()
    macro["experiment"] = pd.Categorical(macro["experiment"], categories=cfg.PREFERENCE_ABLATION_ORDER, ordered=True)
    return macro.sort_values("experiment")


def build_table_9(seed: int = cfg.SEED) -> pd.DataFrame:
    """Candidate-diversity / pair-strategy comparison. Requires the optional
    model-diversity ablation (self-sampling vs heterogeneous candidates,
    paper Section 8.5) which is not part of the core 19-file pipeline --
    returns whatever matching experiment reports already exist, empty
    otherwise (this table is a documented follow-up, not silently skipped)."""
    diversity_experiments = [n for n in cfg.EXPERIMENTS if "diversity" in n or "self_sample" in n]
    if not diversity_experiments:
        print("[Note] Table 9: no candidate-diversity experiments configured yet -- "
              "see IMPLEMENTATION_PLAN.md Stage 9 targeted ablations.")
        return pd.DataFrame()
    return _load_reports(diversity_experiments, seed)


def main() -> None:
    out_dir = cfg.REPORT_ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang, table in build_tables_3_5().items():
        table.to_csv(out_dir / f"table3_5_{lang.lower()}.csv", index=False)
        print(f"[Done] table3_5_{lang.lower()}.csv ({len(table)} rows)")

    build_table_6().to_csv(out_dir / "table6.csv", index=False)
    print("[Done] table6.csv")

    for name, builder in [("table7", build_table_7), ("table8", build_table_8), ("table9", build_table_9)]:
        df = builder()
        df.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"[Done] {name}.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
