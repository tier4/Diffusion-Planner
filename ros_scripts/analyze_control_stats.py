#!/usr/bin/env python3
"""Analyze acceleration/curvature statistics from a dataset.

Usage:
    python3 ros_scripts/analyze_control_stats.py <path_list.json>
"""

import json
import sys

import numpy as np
import torch
from diffusion_planner.utils.unicycle_accel_curvature import (
    UnicycleAccelCurvatureActionSpace,
    traj4d_to_action,
)
from tqdm import tqdm


def heading_to_cos_sin_np(traj_3d: np.ndarray) -> np.ndarray:
    """Convert (x, y, heading) to (x, y, cos, sin)."""
    out = np.zeros((*traj_3d.shape[:-1], 4), dtype=np.float32)
    out[..., :2] = traj_3d[..., :2]
    out[..., 2] = np.cos(traj_3d[..., 2])
    out[..., 3] = np.sin(traj_3d[..., 2])
    return out


def main():
    path_list_json = sys.argv[1]
    with open(path_list_json) as f:
        paths = json.load(f)
    print(f"Total files: {len(paths)}")

    action_space = UnicycleAccelCurvatureActionSpace(n_waypoints=80)

    all_accels = []
    all_curvatures = []
    all_v0s = []
    errors = 0

    for p in tqdm(paths, desc="Converting trajectories"):
        try:
            d = np.load(p)
            past_4d = heading_to_cos_sin_np(d["ego_agent_past"])  # (31, 4)
            future_4d = heading_to_cos_sin_np(d["ego_agent_future"])  # (80, 4)
            v0 = float(d["ego_current_state"][4])

            past_t = torch.from_numpy(past_4d).unsqueeze(0)
            future_t = torch.from_numpy(future_4d).unsqueeze(0)
            t0_states = {"v": torch.tensor([v0])}

            with torch.no_grad():
                actions = traj4d_to_action(action_space, past_t, future_t, t0_states=t0_states)

            actions_np = actions.squeeze(0).numpy()  # (80, 2)
            all_accels.append(actions_np[:, 0])
            all_curvatures.append(actions_np[:, 1])
            all_v0s.append(v0)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"Error on {p}: {e}")

    all_accels = np.concatenate(all_accels)
    all_curvatures = np.concatenate(all_curvatures)
    all_v0s = np.array(all_v0s)

    print(f"Processed: {len(paths) - errors}, Errors: {errors}")
    print(f"Total action samples: {len(all_accels)}")
    print()

    for name, data in [
        ("Acceleration (m/s^2)", all_accels),
        ("Curvature (1/m)", all_curvatures),
        ("Initial velocity v0 (m/s)", all_v0s),
    ]:
        print(f"=== {name} ===")
        print(f"  mean:   {np.mean(data):.6f}")
        print(f"  std:    {np.std(data):.6f}")
        print(f"  min:    {np.min(data):.6f}")
        print(f"  max:    {np.max(data):.6f}")
        print(f"  p1:     {np.percentile(data, 1):.6f}")
        print(f"  p5:     {np.percentile(data, 5):.6f}")
        print(f"  p25:    {np.percentile(data, 25):.6f}")
        print(f"  p50:    {np.percentile(data, 50):.6f}")
        print(f"  p75:    {np.percentile(data, 75):.6f}")
        print(f"  p95:    {np.percentile(data, 95):.6f}")
        print(f"  p99:    {np.percentile(data, 99):.6f}")
        print()

    print("=== Scale ratios ===")
    print(f"  accel_std / curvature_std = {np.std(all_accels) / np.std(all_curvatures):.2f}")


if __name__ == "__main__":
    main()
