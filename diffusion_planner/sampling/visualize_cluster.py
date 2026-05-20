# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualize ego_agent_future trajectories grouped by cluster.

Usage:
    python visualize_cluster.py \\
        --cluster_json /path/to/result.json \\
        --output       /path/to/figure.png \\
        [--max_samples 200] [--seed 42]

The cluster JSON is the output of cluster.py:
    {
        "cluster_id0": ["path/to/a.npz", ...],
        "cluster_id1": ["path/to/b.npz", ...],
        ...
    }

All clusters are plotted on a single axes, color-coded by cluster ID.
If --output is omitted the figure is shown interactively.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_cluster_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_trajectory(npz_path: str) -> np.ndarray:
    """Return ego_agent_future as (T, 2) array of (x, y) positions."""
    data = np.load(npz_path, allow_pickle=True)
    future = data["ego_agent_future"].astype(float)  # (T, 3): x, y, heading
    return future[:, :2]


# ── plotting ─────────────────────────────────────────────────────────────────

def _path_length(xy: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def visualize(cluster_json: str, output: str | None, max_samples: int, seed: int, bg_gray: float = 1.0) -> None:
    clusters = load_cluster_json(cluster_json)
    cluster_ids = sorted(clusters.keys(), key=lambda x: int(x.replace("cluster_id", "")))
    n_clusters = len(cluster_ids)

    cmap = plt.get_cmap("hsv")
    colors = [cmap(i / n_clusters) for i in range(n_clusters)]
    rng = random.Random(seed)

    # Collect all trajectories across all clusters, tagged with color and label
    all_entries = []  # (path_length, xy, color, label_or_None)
    label_used = [False] * n_clusters

    for idx, cluster_id in enumerate(cluster_ids):
        paths = clusters[cluster_id]
        color = colors[idx]
        cluster_label = f"{cluster_id} (n={len(paths)})"
        sampled = rng.sample(paths, min(max_samples, len(paths)))

        for npz_path in sampled:
            try:
                xy = load_trajectory(npz_path)
                all_entries.append((_path_length(xy), xy, color, idx, cluster_label))
            except Exception as e:
                print(f"  [warn] skipping {npz_path}: {e}")

    # Sort longest-first so shorter paths are drawn last (on top)
    all_entries.sort(key=lambda t: t[0], reverse=True)

    bg_color = (bg_gray, bg_gray, bg_gray)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    fig.suptitle(
        f"ego_agent_future trajectories by cluster  (max {max_samples} samples each)",
        fontsize=12,
        color="white" if bg_gray < 0.5 else "black",
    )

    for _, xy, color, idx, cluster_label in all_entries:
        label = cluster_label if not label_used[idx] else None
        ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=0.3, linewidth=1.0, label=label)
        label_used[idx] = True

    # Mark the ego origin
    ax.scatter([0], [0], color="black", s=40, zorder=5, label="ego origin")

    label_color = "white" if bg_gray < 0.5 else "black"
    ax.set_xlabel("x [m]", color=label_color)
    ax.set_ylabel("y [m]", color=label_color)
    ax.tick_params(colors=label_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(label_color)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.4, alpha=0.5, color=label_color)
    ax.legend(
        fontsize=7,
        loc="upper right",
        ncol=max(1, n_clusters // 20),
        markerscale=2,
        labelcolor=label_color,
        facecolor=bg_color,
        edgecolor=label_color,
    )

    fig.tight_layout()

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {out_path}")
    else:
        plt.show()


# ── CLI ──────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description="Visualize ego_agent_future trajectories grouped by cluster"
    )
    parser.add_argument(
        "--cluster_json",
        type=str,
        required=True,
        help="Path to the clustering result JSON produced by cluster.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save figure to this path (PNG/PDF/SVG). If omitted, display interactively.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Maximum number of trajectories to draw per cluster (default: 200)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bg_gray",
        type=float,
        default=1.0,
        help="Background grayscale level: 0.0=black, 1.0=white (default: 1.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    visualize(
        cluster_json=args.cluster_json,
        output=args.output,
        max_samples=args.max_samples,
        seed=args.seed,
        bg_gray=args.bg_gray,
    )
