#!/usr/bin/env python3
"""Visualize a query scene and its top-K similar training scenes side by side.

Plots ego trajectory (past + future), neighbor agents, lane centerlines,
and route for each scene. Lets you visually confirm that similar scenes
actually look similar.

Usage:
    uv run python scripts/visualize_similar.py \
      --query data/or_scene_npz_v2/takanawa/frame.npz \
      --results data/demo_search_results_takanawa.json \
      --output data/demo_visualization.png \
      [--top_k 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_scene(npz_path: str) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def plot_scene(ax: plt.Axes, scene: dict, title: str) -> None:
    # Lanes (centerlines) — dim 33: first 3 are x, y, z per point
    lanes = scene["lanes"]  # (140, 20, 33)
    for i in range(lanes.shape[0]):
        lane_pts = lanes[i, :, :2]  # x, y
        if np.any(lane_pts != 0):
            ax.plot(lane_pts[:, 0], lane_pts[:, 1], color="#cccccc", linewidth=0.5, zorder=1)

    # Route lanes
    route = scene["route_lanes"]  # (25, 20, 33)
    for i in range(route.shape[0]):
        route_pts = route[i, :, :2]
        if np.any(route_pts != 0):
            ax.plot(
                route_pts[:, 0],
                route_pts[:, 1],
                color="#4CAF50",
                linewidth=1.5,
                alpha=0.6,
                zorder=2,
            )

    # Neighbor agents past trajectories
    neighbors = scene["neighbor_agents_past"]  # (320, 31, 11) — first 2 are x, y
    for i in range(neighbors.shape[0]):
        traj = neighbors[i, :, :2]
        if np.any(traj != 0):
            ax.plot(traj[:, 0], traj[:, 1], color="#FF9800", linewidth=0.8, alpha=0.5, zorder=3)
            ax.scatter(traj[-1, 0], traj[-1, 1], color="#FF9800", s=10, zorder=4)

    # Ego past (blue) and future (red)
    ego_past = scene["ego_agent_past"][:, :2]  # (31, 2)
    ego_future = scene["ego_agent_future"][:, :2]  # (80, 2)

    ax.plot(
        ego_past[:, 0], ego_past[:, 1], color="#2196F3", linewidth=2, zorder=5, label="ego past"
    )
    ax.plot(
        ego_future[:, 0],
        ego_future[:, 1],
        color="#F44336",
        linewidth=2,
        zorder=5,
        label="ego future",
    )
    ax.scatter(0, 0, color="#2196F3", s=60, marker="o", zorder=6, edgecolors="black")

    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.grid(True, alpha=0.2)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/demo_visualization.png"))
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)
    results = results[: args.top_k]

    n_cols = min(args.top_k + 1, 6)
    n_rows = (args.top_k + 1 + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, "__len__") else [row])]

    # Query scene
    query_scene = load_scene(str(args.query))
    plot_scene(axes_flat[0], query_scene, f"QUERY\n{args.query.name[:40]}")
    axes_flat[0].legend(fontsize=6, loc="upper right")

    # Matches
    for i, result in enumerate(results):
        ax = axes_flat[i + 1]
        match_path = result["path"]
        if Path(match_path).exists():
            match_scene = load_scene(match_path)
            plot_scene(
                ax,
                match_scene,
                f"Match #{result['rank']} (d={result['distance']:.3f})\n{Path(match_path).name[:40]}",
            )
        else:
            ax.text(
                0.5,
                0.5,
                f"File not found:\n{match_path}",
                ha="center",
                va="center",
                fontsize=7,
                transform=ax.transAxes,
            )
            ax.set_title(f"Match #{result['rank']} (MISSING)")

    # Hide unused axes
    for i in range(len(results) + 1, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("Similarity Search: Query vs Top-K Training Matches", fontsize=12, y=1.02)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
