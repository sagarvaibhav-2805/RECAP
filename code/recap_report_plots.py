"""
Stage 11 -- builds Figures 1-3 (PNG) from already-saved outputs only. Needs
`matplotlib` (not otherwise required by the pipeline -- install separately:
pip install matplotlib).

Run:
    python recap_report_plots.py
"""

from __future__ import annotations

import json

import pandas as pd

import config as cfg


def figure_1_margin_distribution(experiment: str = "recap_dpo") -> None:
    """Reward-margin distribution before (pairs_all.csv) vs after
    (pairs_balanced.csv) filtering, pooled across all directions."""
    import matplotlib.pyplot as plt

    before, after = [], []
    for lang in cfg.LANGUAGES:
        for direction in cfg.DIRECTIONS:
            exp_dir = cfg.pairs_experiment_dir(lang, direction, experiment)
            all_path, balanced_path = exp_dir / "pairs_all.csv", exp_dir / "pairs_balanced.csv"
            if all_path.exists():
                before.extend(pd.read_csv(all_path)["reward_margin"].tolist())
            if balanced_path.exists():
                after.extend(pd.read_csv(balanced_path)["reward_margin"].tolist())

    if not before:
        print("[Skip] Figure 1: no pairs_all.csv found yet")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(before, bins=40, alpha=0.5, label="before filtering (pairs_all)")
    if after:
        ax.hist(after, bins=40, alpha=0.5, label="after filtering (pairs_balanced)")
    ax.axvline(cfg.MARGIN_DELTA, color="red", linestyle="--", label=f"delta={cfg.MARGIN_DELTA}")
    ax.set_xlabel("Reward margin |R(y+) - R(y-)|")
    ax.set_ylabel("Pair count")
    ax.set_title(f"Reward-margin distribution ({experiment})")
    ax.legend()
    out_path = cfg.REPORT_ROOT / "plots" / "figure1.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] {out_path}")


def figure_2_model_participation(experiment: str = "recap_dpo") -> None:
    """Preferred (C_m+) / rejected (C_m-) participation bars per generator
    model, pooled across all directions -- Stage 5's balancing stats."""
    import matplotlib.pyplot as plt
    import numpy as np

    frames = []
    for lang in cfg.LANGUAGES:
        for direction in cfg.DIRECTIONS:
            path = cfg.pairs_experiment_dir(lang, direction, experiment) / "pairs_balanced.csv"
            if path.exists():
                frames.append(pd.read_csv(path))
    if not frames:
        print("[Skip] Figure 2: no pairs_balanced.csv found yet")
        return

    all_pairs = pd.concat(frames, ignore_index=True)
    n = len(all_pairs)
    c_plus = [(all_pairs["chosen_model"] == m).sum() / n for m in cfg.MODEL_NAMES]
    c_minus = [(all_pairs["rejected_model"] == m).sum() / n for m in cfg.MODEL_NAMES]

    x = np.arange(len(cfg.MODEL_NAMES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, c_plus, width, label="Preferred (C_m+)")
    ax.bar(x + width / 2, c_minus, width, label="Rejected (C_m-)")
    ax.set_xticks(x)
    ax.set_xticklabels(cfg.MODEL_NAMES)
    ax.set_ylabel("Fraction of retained pairs")
    ax.set_title(f"Model participation ({experiment})")
    ax.legend()
    out_path = cfg.REPORT_ROOT / "plots" / "figure2.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] {out_path}")


def figure_3_delta_heatmap(experiment: str = "recap_dpo", seed: int = cfg.SEED) -> None:
    """Heatmap of delta-BLEU/delta-ChrF++/delta-COMET vs SFT, one row per
    direction."""
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    for lang in cfg.LANGUAGES:
        for direction in cfg.DIRECTIONS:
            sft_path = cfg.eval_report_path(lang, direction, "sft", seed)
            cond_path = cfg.eval_report_path(lang, direction, experiment, seed)
            if not (sft_path.exists() and cond_path.exists()):
                continue
            with open(sft_path) as f:
                sft = json.load(f)
            with open(cond_path) as f:
                cond = json.load(f)
            if "corpus" not in sft or "corpus" not in cond:
                continue
            rows.append({
                "direction": f"{lang}/{direction}",
                "dBLEU": cond["corpus"]["bleu"] - sft["corpus"]["bleu"],
                "dChrF++": cond["corpus"]["chrf"] - sft["corpus"]["chrf"],
                "dCOMET": cond["corpus"]["comet"] - sft["corpus"]["comet"],
            })

    if not rows:
        print("[Skip] Figure 3: no matching sft/experiment report pairs found yet")
        return

    df = pd.DataFrame(rows).set_index("direction")

    # Anchor the colormap at zero with a symmetric range, rather than
    # imshow's default auto min/max -- deltas that are all roughly-equal and
    # small (e.g. three ~+0.05 improvements) can differ from each other by
    # nothing more than float noise (0.35-0.30 != 0.55-0.50 in float64), and
    # an auto-scaled colormap would stretch that machine-epsilon-level noise
    # across the full red-to-green range, making genuinely-similar positive
    # deltas look like one metric regressed. Centering at 0 makes "no color
    # contrast" mean "no real difference," which is the correct reading.
    from matplotlib.colors import TwoSlopeNorm

    abs_max = max(abs(df.values.min()), abs(df.values.max()), 1e-6)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    fig, ax = plt.subplots(figsize=(5, 0.6 * len(df) + 1))
    im = ax.imshow(df.values, cmap="RdYlGn", aspect="auto", norm=norm)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(j, i, f"{df.values[i, j]:+.3f}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    ax.set_title(f"Delta from SFT ({experiment})")
    out_path = cfg.REPORT_ROOT / "plots" / "figure3.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] {out_path}")


def main() -> None:
    figure_1_margin_distribution()
    figure_2_model_participation()
    figure_3_delta_heatmap()


if __name__ == "__main__":
    main()
