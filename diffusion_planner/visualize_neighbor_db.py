"""Visualize a few random neighbor patterns from a neighbor-pattern DB.

Renders ``--num`` (default 10) randomly sampled patterns as a grid of subplots, each showing
one neighbor's past track (blue), future track (red), and its oriented bounding box at the
current pose. The ego origin (0, 0) is marked, since all patterns are stored in the
ego-centric frame of their source scene.

Example:
    python3 visualize_neighbor_db.py --db_path neighbor_db.npz --output_path sample.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from diffusion_planner.utils.neighbor_db import NeighborPatternDB  # noqa: E402

# Column layout of a neighbor past row (see loss.py / decoder.py).
_PAST_X, _PAST_Y, _PAST_COS, _PAST_SIN = 0, 1, 2, 3
_PAST_WIDTH, _PAST_LENGTH = 6, 7


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize random neighbor-DB patterns")
    parser.add_argument("--db_path", type=str, required=True, help="neighbor_db.npz path")
    parser.add_argument("--output_path", type=str, default="neighbor_db_sample.png")
    parser.add_argument("--num", type=int, default=10, help="number of patterns to render")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _nonzero_rows(track_xy: np.ndarray) -> np.ndarray:
    """Keep only rows whose (x, y) are not both exactly zero (== padding)."""
    return track_xy[np.any(track_xy != 0.0, axis=-1)]


def _bbox_corners(cx, cy, cos, sin, length, width) -> np.ndarray:
    """Return the 5 polygon points (4 corners + closing point) of an oriented box."""
    norm = np.hypot(cos, sin)
    if norm < 1e-6:
        cos, sin = 1.0, 0.0
    else:
        cos, sin = cos / norm, sin / norm
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw], [hl, hw]])
    rot = np.array([[cos, -sin], [sin, cos]])
    return local @ rot.T + np.array([cx, cy])


def visualize(db_path: str, output_path: str, num: int, seed: int) -> None:
    db = NeighborPatternDB(db_path)
    past = db.past.numpy()  # [M, 31, 11]
    future = db.future.numpy()  # [M, 80, 3]

    rng = np.random.default_rng(seed)
    num = min(num, db.num_patterns)
    indices = rng.choice(db.num_patterns, size=num, replace=False)

    cols = 5
    rows = int(np.ceil(num / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax_i, idx in enumerate(indices):
        ax = axes[ax_i]
        p = past[idx]
        f = future[idx]

        past_xy = _nonzero_rows(p[:, [_PAST_X, _PAST_Y]])
        future_xy = _nonzero_rows(f[:, :2])

        if past_xy.shape[0] > 0:
            ax.plot(past_xy[:, 0], past_xy[:, 1], "-o", color="tab:blue", ms=2, label="past")
        if future_xy.shape[0] > 0:
            ax.plot(future_xy[:, 0], future_xy[:, 1], "-o", color="tab:red", ms=2, label="future")

        cur = p[-1]
        corners = _bbox_corners(
            cur[_PAST_X], cur[_PAST_Y], cur[_PAST_COS], cur[_PAST_SIN],
            max(cur[_PAST_LENGTH], 0.1), max(cur[_PAST_WIDTH], 0.1),
        )
        ax.plot(corners[:, 0], corners[:, 1], "-", color="tab:green", lw=1.5)

        ax.plot(0, 0, "k*", ms=12, label="ego origin")
        dist = float(np.hypot(cur[_PAST_X], cur[_PAST_Y]))
        ax.set_title(f"#{idx}  dist={dist:.1f}m  L={cur[_PAST_LENGTH]:.1f} W={cur[_PAST_WIDTH]:.1f}",
                     fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        if ax_i == 0:
            ax.legend(fontsize=7, loc="best")

    for ax in axes[num:]:
        ax.axis("off")

    fig.suptitle(f"{num} random neighbor patterns from {db_path}  ({db.num_patterns} total)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    print(f"Saved {num} patterns to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    visualize(args.db_path, args.output_path, args.num, args.seed)
