#!/usr/bin/env python3
"""Visualize per-timestep penalty values along ego trajectory.

Usage:
    python util_scripts/visualize_penalties.py \
        --predictions_dir <path_to_predictions> \
        --valid_data_list <path_to_data_list.json> \
        [--save_dir <output_dir>] \
        [--sample_indices 0 1 2] \
        [--max_samples 10] \
        [--road_border_margin 1.0]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusion_planner.loss import (
    compute_ego_bbox_corners,
    neighbor_clearance_penalty,
    point_to_segment_distance,
    road_border_penalty,
    sample_ego_edge_points,
)
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.visualize_input import visualize_inputs
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize per-timestep penalty values along ego trajectory."
    )
    parser.add_argument("--predictions_dir", type=Path, required=True)
    parser.add_argument("--valid_data_list", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument(
        "--sample_indices",
        type=int,
        nargs="*",
        default=None,
        help="Specific sample indices to visualize",
    )
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--road_border_margin", type=float, default=0.25)
    parser.add_argument("--road_border_n_interp", type=int, default=0)
    return parser.parse_args()

def compute_min_edge_to_road_border_distance(
    ego_edge_points: torch.Tensor,
    line_strings_xy: torch.Tensor,
    road_border_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute minimum distance from ego edge points to road border segments.

    Args:
        ego_edge_points: [1, T, K, 2] sample points on ego bbox edges.
        line_strings_xy: [N, P, 2] line string xy coordinates.
        road_border_mask: [N] boolean.

    Returns:
        min_dist: [T] (inf where no road border is nearby).
    """
    T, K = ego_edge_points.shape[1], ego_edge_points.shape[2]

    if not road_border_mask.any():
        return torch.full((T,), float("inf"))

    # Segment endpoints from road border line strings
    seg_a = line_strings_xy[road_border_mask, :-1, :]  # [M, S, 2]
    seg_b = line_strings_xy[road_border_mask, 1:, :]  # [M, S, 2]
    M, S, _ = seg_a.shape
    seg_valid = (seg_a.abs().sum(-1) > 1e-6) & (seg_b.abs().sum(-1) > 1e-6)  # [M, S]

    # Flatten segments: [M*S, 2]
    seg_a_flat = seg_a.reshape(M * S, 2)
    seg_b_flat = seg_b.reshape(M * S, 2)
    seg_valid_flat = seg_valid.reshape(M * S)

    # ego_edge_points [1, T, K, 2] -> [T*K, 1, 2]
    p = ego_edge_points[0].reshape(T * K, 1, 2)
    a = seg_a_flat[None, :, :]  # [1, M*S, 2]
    b = seg_b_flat[None, :, :]  # [1, M*S, 2]

    dist = point_to_segment_distance(p, a, b)  # [T*K, M*S]
    dist = torch.where(seg_valid_flat[None, :], dist, torch.full_like(dist, float("inf")))

    # Min over segments, then over edge points per timestep
    min_per_point = dist.min(dim=-1).values  # [T*K]
    min_per_point = min_per_point.reshape(T, K)
    return min_per_point.min(dim=-1).values  # [T]


def process_sample(idx, valid_data_path, prediction_path, save_dir, road_border_margin, road_border_n_interp):
    valid_data_path = Path(valid_data_path)
    prediction_path = Path(prediction_path)

    valid_data = np.load(valid_data_path)
    output_dict = np.load(prediction_path)
    prediction = output_dict["prediction"]  # (P, T, 4)

    # Build input dict (raw, not normalized)
    inputs = {}
    for key, value in valid_data.items():
        if key in ("map_name", "token"):
            continue
        inputs[key] = torch.tensor(np.expand_dims(value, axis=0))

    inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
    inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])

    ego_shape = inputs["ego_shape"][0]  # [3]
    ego_traj = torch.tensor(prediction[0])  # [T, 4]
    T = ego_traj.shape[0]
    timesteps = np.arange(T) * 0.1  # seconds (10 Hz)

    # Compute ego bounding box corners [1, T, 4, 2]
    ego_bbox_corners = compute_ego_bbox_corners(ego_traj[None], ego_shape[None])

    # 1. Neighbor clearance penalty
    neighbors_future = inputs["neighbor_agents_future"]
    neighbor_future_mask = (
        torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
    )
    neighbors_future = heading_to_cos_sin(neighbors_future)
    neighbors_future[neighbor_future_mask] = 0.0
    neighbors_future_valid = ~neighbor_future_mask

    neighbor_pen = neighbor_clearance_penalty(
        ego_bbox_corners, neighbors_future, neighbors_future_valid, inputs
    )

    # 2. Road border penalty and distance
    ls = inputs["line_strings"][0]  # [N, P, D]
    D = ls.shape[-1]
    has_road_border = D >= 4
    if has_road_border:
        ls_xy = ls[..., :2]  # [N, P, 2]
        rb_mask = (ls[..., 3] > 0.5).any(dim=-1)  # [N]

        ego_edge_points = sample_ego_edge_points(ego_bbox_corners, n_interp=road_border_n_interp)
        rb_pen = road_border_penalty(
            ego_edge_points,
            ls_xy.unsqueeze(0),
            rb_mask.unsqueeze(0),
            margin=road_border_margin,
        )
        min_dist = compute_min_edge_to_road_border_distance(ego_edge_points, ls_xy, rb_mask)
    else:
        rb_pen = torch.zeros(1, T)
        min_dist = torch.full((T,), float("inf"))

    # Convert to numpy
    neighbor_pen_np = neighbor_pen[0].detach().numpy()
    rb_pen_np = rb_pen[0].detach().numpy()
    min_dist_np = min_dist.detach().numpy()

    # ===== Visualization =====
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

    # Top: Map view with trajectory colored by road_border penalty
    ax_map = fig.add_subplot(gs[0, :])

    vis_dict = {}
    for key, value in valid_data.items():
        if key in ("map_name", "token"):
            continue
        vis_dict[key] = torch.tensor(np.expand_dims(value, axis=0))
    vis_dict["ego_agent_past"] = heading_to_cos_sin(vis_dict["ego_agent_past"])
    vis_dict["goal_pose"] = heading_to_cos_sin(vis_dict["goal_pose"])

    visualize_inputs(vis_dict, save_path=None, ax=ax_map)

    # Color trajectory by road_border penalty
    vmax = max(float(rb_pen_np.max()), 0.01)
    scatter = ax_map.scatter(
        prediction[0, :, 0],
        prediction[0, :, 1],
        c=rb_pen_np,
        cmap="YlOrRd",
        s=20,
        zorder=10,
        vmin=0,
        vmax=vmax,
        edgecolors="none",
    )
    plt.colorbar(scatter, ax=ax_map, label="road_border penalty [m]", shrink=0.6)

    # Draw ego bbox at a few timesteps
    for t_idx in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
        corners = ego_bbox_corners[0, t_idx].numpy()
        polygon = plt.Polygon(
            corners, fill=False, edgecolor="gray", linewidth=0.5, alpha=0.5
        )
        ax_map.add_patch(polygon)

    ax_map.set_title(f"Sample {idx}: {valid_data_path.stem}")

    # Bottom-left: Time-series of penalties
    ax_pen = fig.add_subplot(gs[1, 0])
    ax_pen.plot(
        timesteps, neighbor_pen_np, label="neighbor", alpha=0.8, linewidth=1.5
    )
    ax_pen.plot(
        timesteps, rb_pen_np, label="road_border", alpha=0.8, linewidth=1.5
    )
    ax_pen.set_xlabel("Time [s]")
    ax_pen.set_ylabel("Penalty [m]")
    ax_pen.legend(loc="upper right", fontsize=8)
    ax_pen.set_title("Penalties per timestep")
    ax_pen.grid(True, alpha=0.3)
    ax_pen.set_xlim(0, timesteps[-1])

    # Bottom-right: Distance to nearest road border
    ax_dist = fig.add_subplot(gs[1, 1])
    finite_mask = np.isfinite(min_dist_np)
    if finite_mask.any():
        ax_dist.plot(
            timesteps[finite_mask],
            min_dist_np[finite_mask],
            label="min distance",
            color="tab:blue",
            linewidth=1.5,
        )
    ax_dist.axhline(
        y=road_border_margin,
        color="red",
        linestyle="--",
        label=f"margin ({road_border_margin:.2f}m)",
        linewidth=1,
    )
    ax_dist.set_xlabel("Time [s]")
    ax_dist.set_ylabel("Distance [m]")
    ax_dist.legend(loc="upper right", fontsize=8)
    ax_dist.set_title("Distance to nearest road border")
    ax_dist.grid(True, alpha=0.3)
    ax_dist.set_xlim(0, timesteps[-1])
    if finite_mask.any():
        y_max = min(float(min_dist_np[finite_mask].max()) * 1.2, 20.0)
        ax_dist.set_ylim(0, max(y_max, road_border_margin * 2))
    else:
        ax_dist.set_ylim(0, road_border_margin * 2)

    plt.tight_layout()
    save_path = save_dir / f"penalty_{idx:08d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    rb_min = float(min_dist_np[finite_mask].min()) if finite_mask.any() else float("inf")
    print(
        f"[{idx:4d}] neighbor={neighbor_pen_np.mean():.4f}, "
        f"rb={rb_pen_np.mean():.4f}, "
        f"rb_min_dist={rb_min:.2f}m"
    )


def main():
    args = parse_args()

    predictions_dir = args.predictions_dir
    save_dir = args.save_dir or predictions_dir.parent / "penalty_visualization"
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(args.valid_data_list) as f:
        valid_data_path_list = json.load(f)

    prediction_path_list = sorted(predictions_dir.glob("**/*.npz"))
    assert len(prediction_path_list) == len(valid_data_path_list), (
        f"Mismatch: {len(prediction_path_list)} predictions vs "
        f"{len(valid_data_path_list)} data"
    )

    if args.sample_indices is not None:
        indices = args.sample_indices
    else:
        total = len(valid_data_path_list)
        n = min(args.max_samples, total)
        indices = [int(round(i * (total - 1) / (n - 1))) for i in range(n)] if n > 1 else [0]

    for idx in tqdm(indices, desc="Visualizing penalties"):
        process_sample(
            idx,
            valid_data_path_list[idx],
            prediction_path_list[idx],
            save_dir,
            args.road_border_margin,
            args.road_border_n_interp,
        )

    print(f"Saved {len(indices)} images to {save_dir}")


if __name__ == "__main__":
    main()
