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

import numpy as np
import torch
from diffusion_planner.loss import (
    compute_ego_edge_points,
    compute_neighbor_collision_penalty,
    compute_road_border_penalty,
    point_to_segment_distance,
)
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.visualize_input import visualize_inputs
from tqdm import tqdm

from scenario_generation.canvas import Figure


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


def process_sample(
    idx, valid_data_path, prediction_path, save_dir, road_border_margin, road_border_n_interp
):
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

    # Compute ego edge points [1, T, K, 2]
    ego_edge_points = compute_ego_edge_points(
        ego_traj[None], ego_shape[None], n_interp=road_border_n_interp
    )

    # 1. Neighbor collision penalty
    neighbors_future = inputs["neighbor_agents_future"]
    neighbor_future_mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
    neighbors_future = heading_to_cos_sin(neighbors_future)
    neighbors_future[neighbor_future_mask] = 0.0
    neighbors_future_valid = ~neighbor_future_mask

    neighbor_pen = compute_neighbor_collision_penalty(
        ego_edge_points,
        neighbors_future,
        neighbors_future_valid,
        inputs["neighbor_agents_past"],
        margin=2.0,
    )

    # 2. Road border penalty and distance
    ls = inputs["line_strings"]  # [1, N, P, D]
    rb_pen = compute_road_border_penalty(
        ego_edge_points,
        ls,
        margin=road_border_margin,
    )  # [1, T]
    ls_xy = ls[0, ..., :2]  # [N, P, 2]
    rb_mask = (ls[0, ..., 3] > 0.5).any(dim=-1)  # [N]
    min_dist = compute_min_edge_to_road_border_distance(ego_edge_points, ls_xy, rb_mask)

    # Convert to numpy
    neighbor_pen_np = neighbor_pen[0].detach().numpy()
    rb_pen_np = rb_pen[0].detach().numpy()
    min_dist_np = min_dist.detach().numpy()

    # ===== Visualization =====
    fig = Figure((14, 10), dpi=150)

    # Top: Map view with trajectory colored by road_border penalty
    ax_map = fig.panel((0.06, 0.434, 0.91, 0.516))

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
    ax_map.markers(
        prediction[0, :, :2],
        "o",
        size=np.sqrt(20),
        values=rb_pen_np,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=vmax,
        z=10,
    )
    ax_map.colorbar("YlOrRd", 0.0, vmax, "road_border penalty [m]")

    # Draw ego edge points at a few timesteps
    for t_idx in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
        pts = ego_edge_points[0, t_idx].numpy()  # [K, 2]
        ax_map.markers(pts, "o", size=np.sqrt(5), color="#808080", alpha=0.5, z=5)

    ax_map.title(f"Sample {idx}: {valid_data_path.stem}")

    # Bottom-left: Time-series of penalties
    ax_pen = fig.panel((0.06, 0.06, 0.396, 0.258))
    ax_pen.lines([np.stack([timesteps, neighbor_pen_np], 1)], "#1f77b4", 1.5, 0.8, label="neighbor")
    ax_pen.lines([np.stack([timesteps, rb_pen_np], 1)], "#ff7f0e", 1.5, 0.8, label="road_border")
    ax_pen.xlabel("Time [s]")
    ax_pen.ylabel("Penalty [m]")
    ax_pen.legend("upper right")
    ax_pen.title("Penalties per timestep")
    ax_pen.grid(0.3)
    ax_pen.set_xlim(0, timesteps[-1])

    # Bottom-right: Distance to nearest road border
    ax_dist = fig.panel((0.574, 0.06, 0.396, 0.258))
    finite_mask = np.isfinite(min_dist_np)
    if finite_mask.any():
        ax_dist.lines(
            [np.stack([timesteps[finite_mask], min_dist_np[finite_mask]], 1)],
            "#1f77b4",
            1.5,
            label="min distance",
        )
    ax_dist.hline(
        road_border_margin,
        "#ff0000",
        1,
        dash="dash",
        label=f"margin ({road_border_margin:.2f}m)",
    )
    ax_dist.xlabel("Time [s]")
    ax_dist.ylabel("Distance [m]")
    ax_dist.legend("upper right")
    ax_dist.title("Distance to nearest road border")
    ax_dist.grid(0.3)
    ax_dist.set_xlim(0, timesteps[-1])
    if finite_mask.any():
        y_max = min(float(min_dist_np[finite_mask].max()) * 1.2, 20.0)
        ax_dist.set_ylim(0, max(y_max, road_border_margin * 2))
    else:
        ax_dist.set_ylim(0, road_border_margin * 2)

    save_path = save_dir / f"penalty_{idx:08d}.png"
    fig.save(save_path)

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
        f"Mismatch: {len(prediction_path_list)} predictions vs {len(valid_data_path_list)} data"
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
