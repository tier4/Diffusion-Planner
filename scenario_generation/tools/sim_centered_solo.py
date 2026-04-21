#!/usr/bin/env python3
"""Closed-loop sim with ego re-centered to the route lane and no neighbors.

Loads an NPZ, builds a SceneContext, then:
  - snaps the ego position to the nearest route-lane centerline point,
  - aligns ego heading with the lane's local direction,
  - zeros ego velocity/past history (new centered history),
  - removes all non-ego agents,
  - runs closed-loop simulation with the same model.

Purpose: check whether the model keeps the lane center when given an ideal
start (no traffic, dead-centered), isolating the policy's lane-keeping
behavior from any distraction.

Usage:
    python -m scenario_generation.tools.sim_centered_solo \
        --model_path /path/to/best_model.pth \
        --npz /path/to/scene.npz \
        --output_dir /path/to/out \
        --steps 80
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from scenario_generation.gt_route_extractor import assign_gt_goals_and_routes
from scenario_generation.npz_loader import from_npz
from scenario_generation.scene_context import AgentType, SceneContext
from scenario_generation.simulate import load_model, run_simulation


def recenter_ego_to_route(scene: SceneContext) -> tuple[np.ndarray, float, np.ndarray]:
    """Snap ego to the closest route-lane centerline point with aligned heading.

    Returns: (original_pos, original_heading, offset_vector) for logging.
    """
    ego = next(a for a in scene.agents if a.id == scene.ego_agent_id)
    if ego.route_lanes is None or ego.route_lanes.shape[0] == 0:
        raise ValueError("Ego has no route_lanes; cannot recenter.")

    # Route lanes: (S, P, 33). Centerline XY at cols 0:2, direction at 2:4.
    lanes = ego.route_lanes  # (S, P, 33)
    pts = lanes[..., :2].reshape(-1, 2)  # (S*P, 2)
    dirs = lanes[..., 2:4].reshape(-1, 2)
    valid = np.linalg.norm(pts, axis=-1) > 1e-3

    orig_pos = ego.current_position.copy()
    orig_heading = ego.current_heading

    dist = np.linalg.norm(pts - orig_pos[None], axis=-1)
    dist[~valid] = 1e9
    nearest_idx = int(np.argmin(dist))
    center_pt = pts[nearest_idx]
    lane_dir = dirs[nearest_idx]
    n = np.linalg.norm(lane_dir)
    if n < 1e-6:
        # Fallback: keep original heading
        new_heading = orig_heading
    else:
        lane_dir = lane_dir / n
        new_heading = float(np.arctan2(lane_dir[1], lane_dir[0]))

    # Rewrite past trajectory so the history is consistent with the new pose.
    # Easiest: replicate the new (x, y, heading) for all past timesteps,
    # so the model sees a stationary, centered vehicle at t=0.
    T_past = ego.past_trajectory.shape[0]
    new_past = np.zeros_like(ego.past_trajectory)
    new_past[:, 0] = center_pt[0]
    new_past[:, 1] = center_pt[1]
    new_past[:, 2] = new_heading
    ego.past_trajectory = new_past.astype(np.float32)
    if ego.past_velocities is not None:
        ego.past_velocities[:] = 0.0
    ego.acceleration[:] = 0.0
    ego.steering_angle = 0.0
    ego.yaw_rate = 0.0

    offset = center_pt - orig_pos
    return orig_pos, orig_heading, offset


def strip_neighbors(scene: SceneContext) -> int:
    """Remove all non-ego agents. Returns the number removed."""
    n_before = len(scene.agents)
    scene.agents = [a for a in scene.agents if a.id == scene.ego_agent_id]
    return n_before - len(scene.agents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", choices=["closed_loop", "semi_closed_loop"], default="closed_loop")
    parser.add_argument("--keep_neighbors", action="store_true",
                        help="Don't strip neighbors (for sanity comparison).")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"Loading model from {args.model_path}")
    model, model_args = load_model(args.model_path, device)

    print(f"Loading scene from {args.npz}")
    scene = from_npz(args.npz)
    # Ensure the ego has a route assigned (from GT)
    n_assigned = assign_gt_goals_and_routes(scene)
    print(f"Assigned GT routes/goals to {n_assigned} agents")

    orig_pos, orig_heading, offset = recenter_ego_to_route(scene)
    print(f"Ego recentered: original pos=({orig_pos[0]:.2f}, {orig_pos[1]:.2f}) "
          f"heading={np.degrees(orig_heading):.1f}° -> centered pos=("
          f"{scene.agents[0].current_position[0]:.2f}, {scene.agents[0].current_position[1]:.2f}) "
          f"new heading={np.degrees(scene.agents[0].current_heading):.1f}° "
          f"offset=({offset[0]:+.2f}, {offset[1]:+.2f}) dist={np.linalg.norm(offset):.2f}m")

    if not args.keep_neighbors:
        removed = strip_neighbors(scene)
        print(f"Removed {removed} neighbors (ego-only scene)")

    run_simulation(model, model_args, scene, args.steps, args.output_dir, device,
                   per_agent=False, mode=args.mode)


if __name__ == "__main__":
    main()
