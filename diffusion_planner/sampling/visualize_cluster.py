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

Each sub-plot shows the (x, y) paths from ego_agent_future for one cluster.
If --output is omitted the figure is shown interactively.
"""

import argparse
import json
import math
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


def _draw_cluster(
    ax: plt.Axes, paths: list, max_samples: int, color: str, rng: random.Random
) -> None:
    sampled = rng.sample(paths, min(max_samples, len(paths)))

    for npz_path in sampled:
        try:
            xy = load_trajectory(npz_path)
            ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=0.3, linewidth=0.8)
        except Exception as e:
            print(f"  [warn] skipping {npz_path}: {e}")

    # Mark the ego origin
    ax.scatter([0], [0], color="black", s=20, zorder=5)


def visualize(cluster_json: str, output: str | None, max_samples: int, seed: int) -> None:
    clusters = load_cluster_json(cluster_json)
    cluster_ids = sorted(clusters.keys(), key=lambda x: int(x.replace("cluster_id", "")))
    n_clusters = len(cluster_ids)

    # Grid layout: prefer roughly square arrangement
    n_cols = math.ceil(math.sqrt(n_clusters))
    n_rows = math.ceil(n_clusters / n_cols)

    cmap = plt.get_cmap("tab20")
    rng = random.Random(seed)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"ego_agent_future trajectories by cluster  (max {max_samples} samples each)",
        fontsize=12,
    )

    for idx, cluster_id in enumerate(cluster_ids):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        paths = clusters[cluster_id]
        color = cmap(idx % 20)

        _draw_cluster(ax, paths, max_samples, color, rng)

        ax.set_title(f"{cluster_id}  (n={len(paths)})", fontsize=9)
        ax.set_xlabel("x [m]", fontsize=7)
        ax.set_ylabel("y [m]", fontsize=7)
        ax.set_aspect("equal")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for idx in range(n_clusters, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

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
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    visualize(
        cluster_json=args.cluster_json,
        output=args.output,
        max_samples=args.max_samples,
        seed=args.seed,
    )
