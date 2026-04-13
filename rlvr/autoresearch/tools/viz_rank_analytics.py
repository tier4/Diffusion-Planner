"""Visualize rank analytics: config win rates and reward component trends.

Usage:
    python -m rlvr.autoresearch.tools.viz_rank_analytics \
        --run_dir /path/to/experiment_dir \
        --output_dir /path/to/output_dir  # defaults to run_dir/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Category colors. Includes the experimental categories emitted by
# rlvr.rank_analytics.get_category() so they don't get dropped from plots.
_CAT_COLORS = {
    "det_pure": "#2196F3",
    "guided_det": "#FF9800",
    "guided_noisy": "#4CAF50",
    "random": "#9E9E9E",
    "noise_only_exp": "#9C27B0",
    "stretched_exp": "#E91E63",
    "lateral_exp": "#00BCD4",
    "decoupled_exp": "#795548",
    "collision_exp": "#F44336",
}

_CAT_ORDER = [
    "det_pure", "guided_det", "guided_noisy",
    "noise_only_exp", "stretched_exp", "lateral_exp", "decoupled_exp", "collision_exp",
    "random",
]


def load_summary(run_dir: Path) -> dict:
    path = run_dir / "rank_analytics_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"No summary found at {path}. Run training first.")
    with open(path) as f:
        return json.load(f)


def load_epoch_data(run_dir: Path) -> list[dict]:
    files = sorted(run_dir.glob("rank_analytics_epoch_*.json"))
    data = []
    for f in files:
        with open(f) as fh:
            data.append(json.load(fh))
    return data


def plot_category_stacked_area(summary: dict, output_dir: Path) -> None:
    """Stacked area: config category rates over epochs."""
    trends = summary["category_trends"]
    epochs = sorted(int(e) for e in trends.keys())
    if len(epochs) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    y_stacks = {cat: [] for cat in _CAT_ORDER}
    for ep in epochs:
        rates = trends[str(ep)]
        for cat in _CAT_ORDER:
            y_stacks[cat].append(rates.get(cat, 0) * 100)

    y_arrays = [np.array(y_stacks[cat]) for cat in _CAT_ORDER]
    ax.stackplot(
        epochs, *y_arrays,
        labels=[cat.replace("_", " ").title() for cat in _CAT_ORDER],
        colors=[_CAT_COLORS[cat] for cat in _CAT_ORDER],
        alpha=0.85,
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title("Guidance Category Win Rates Over Training", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(epochs[0], epochs[-1])
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "category_rates_over_epochs.png", dpi=150)
    plt.close(fig)
    print(f"  Saved category_rates_over_epochs.png")


def plot_config_heatmap(summary: dict, output_dir: Path) -> None:
    """Heatmap: individual config × epoch win rates."""
    config_trends = summary["config_trends"]
    epochs = sorted(int(e) for e in config_trends.keys())
    if not epochs:
        return

    # Collect the union of config labels across all epochs (preserving first-seen
    # order). Using only the first epoch silently drops slots that appear later
    # if the variant or label set ever changes mid-run.
    all_labels: list[str] = []
    seen: set[str] = set()
    for ep in epochs:
        for lbl in config_trends[str(ep)].keys():
            if lbl not in seen:
                seen.add(lbl)
                all_labels.append(lbl)

    # Build matrix [n_configs, n_epochs]
    matrix = np.zeros((len(all_labels), len(epochs)))
    for j, ep in enumerate(epochs):
        rates = config_trends[str(ep)]
        for i, lbl in enumerate(all_labels):
            matrix[i, j] = rates.get(lbl, 0) * 100

    fig, ax = plt.subplots(figsize=(max(10, len(epochs) * 0.8), max(6, len(all_labels) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels([str(e) for e in epochs], fontsize=9)
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=9)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Generation Config", fontsize=12)
    ax.set_title("Config Win Rate Heatmap (%)", fontsize=14)

    # Annotate cells with values
    for i in range(len(all_labels)):
        for j in range(len(epochs)):
            val = matrix[i, j]
            if val > 0:
                color = "white" if val > 20 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="Win Rate (%)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "config_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved config_heatmap.png")


def plot_dominant_components(epoch_data: list[dict], output_dir: Path) -> None:
    """Grouped bar chart: dominant reward component distribution per epoch."""
    if not epoch_data:
        return

    epochs = [d["epoch"] for d in epoch_data]
    all_comps = set()
    for d in epoch_data:
        all_comps.update(d["summary"]["dominant_components"].keys())
    all_comps = sorted(all_comps)

    # Build matrix [n_epochs, n_components]
    n_ep = len(epochs)
    n_comp = len(all_comps)
    matrix = np.zeros((n_ep, n_comp))
    for i, d in enumerate(epoch_data):
        total = sum(d["summary"]["dominant_components"].values()) or 1
        for j, comp in enumerate(all_comps):
            matrix[i, j] = d["summary"]["dominant_components"].get(comp, 0) / total * 100

    comp_colors = plt.cm.Set2(np.linspace(0, 1, n_comp))

    fig, ax = plt.subplots(figsize=(max(10, n_ep * 0.8), 5))
    x = np.arange(n_ep)
    bar_width = 0.8 / n_comp

    for j, comp in enumerate(all_comps):
        offset = (j - n_comp / 2 + 0.5) * bar_width
        ax.bar(x + offset, matrix[:, j], bar_width, label=comp, color=comp_colors[j])

    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in epochs], fontsize=9)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Frequency (%)", fontsize=12)
    ax.set_title("Dominant Reward Component Per Epoch", fontsize=14)
    ax.legend(fontsize=9, ncol=min(n_comp, 4))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "dominant_components.png", dpi=150)
    plt.close(fig)
    print(f"  Saved dominant_components.png")


def plot_scene_heatmap(summary: dict, output_dir: Path) -> None:
    """Per-scene heatmap: scene × epoch winner category."""
    scene_evo = summary.get("scene_evolution", {})
    if not scene_evo:
        return

    cat_to_int = {cat: i for i, cat in enumerate(_CAT_ORDER)}
    scenes = sorted(scene_evo.keys())
    # Derive epoch axis as the union of explicit epoch values across all scenes
    # (and category_trends, which is the source of truth). Indexing by list
    # position would silently drop scenes that skipped an epoch.
    epoch_set: set[int] = set()
    for ep_str in summary.get("category_trends", {}):
        try:
            epoch_set.add(int(ep_str))
        except (TypeError, ValueError):
            pass
    for entries in scene_evo.values():
        for entry in entries:
            ep = entry.get("epoch")
            if ep is not None:
                epoch_set.add(int(ep))
    epochs = sorted(epoch_set)
    if not epochs:
        return
    epoch_to_col = {ep: j for j, ep in enumerate(epochs)}

    matrix = np.full((len(scenes), len(epochs)), np.nan)
    unknown_idx = cat_to_int.get("random", len(_CAT_ORDER) - 1)
    for i, scene in enumerate(scenes):
        for entry in scene_evo[scene]:
            ep = entry.get("epoch")
            if ep is None or ep not in epoch_to_col:
                continue
            matrix[i, epoch_to_col[ep]] = cat_to_int.get(entry["category"], unknown_idx)

    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([_CAT_COLORS[c] for c in _CAT_ORDER])
    bounds = [-0.5] + [i + 0.5 for i in range(len(_CAT_ORDER))]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(max(10, len(epochs) * 0.8), max(8, len(scenes) * 0.25)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels([str(e) for e in epochs], fontsize=9)
    ax.set_yticks(range(len(scenes)))
    ax.set_yticklabels([s[:25] for s in scenes], fontsize=7)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Scene", fontsize=12)
    ax.set_title("Per-Scene Winner Category Over Training", fontsize=14)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_CAT_COLORS[cat], label=cat.replace("_", " ").title())
        for cat in _CAT_ORDER
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "scene_evolution_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved scene_evolution_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize rank analytics")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Output dir for plots (default: run_dir/plots)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading rank analytics from {args.run_dir}")
    summary = load_summary(args.run_dir)
    epoch_data = load_epoch_data(args.run_dir)

    print(f"Generating plots ({len(epoch_data)} epochs)...")
    plot_category_stacked_area(summary, output_dir)
    plot_config_heatmap(summary, output_dir)
    plot_dominant_components(epoch_data, output_dir)
    plot_scene_heatmap(summary, output_dir)
    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
