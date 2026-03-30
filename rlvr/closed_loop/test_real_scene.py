"""Test state_update with a real NPZ scene.

Loads an actual scene, applies GT ego trajectory step-by-step,
verifies ego stays at origin and neighbors track correctly.

Run: python -m rlvr.closed_loop.test_real_scene
"""

from __future__ import annotations

import json
import math

import torch

from preference_optimization.utils import load_npz_data
from rlvr.closed_loop.state_update import (
    advance_neighbor_past,
    transform_positions_to_ego_frame,
    update_scene_state,
)

DEVICE = torch.device("cpu")
SSD = "/media/danielsanchez/2fb4af16-188c-4b7d-8ebb-4a7d0c90d207/auto_research"
VAL_SCENES = f"{SSD}/odaiba_grpo_experiments/val_v4_100.json"


def test_gt_rollout():
    """Apply GT ego trajectory step by step, verify state consistency."""
    with open(VAL_SCENES) as f:
        scenes = json.load(f)

    npz_path = scenes[0]
    print(f"Loading: {npz_path}")
    data = load_npz_data(npz_path, DEVICE)
    if "delay" not in data:
        data["delay"] = torch.zeros(1, dtype=torch.long, device=DEVICE)

    # GT ego future: [1, 80, 3] (x, y, heading) in original frame
    import numpy as np
    raw = np.load(npz_path)
    ego_future = torch.from_numpy(raw["ego_agent_future"]).to(DEVICE)  # [80, 3]
    nb_future = torch.from_numpy(raw["neighbor_agents_future"]).to(DEVICE)  # [32, 80, 3]

    print(f"  ego_future shape: {ego_future.shape}")
    print(f"  nb_future shape: {nb_future.shape}")
    print(f"  ego_current_state: {data['ego_current_state'][0, :6].tolist()}")

    # Convert GT future to [T, 4] format with cos/sin
    gt_traj = torch.zeros(80, 4, device=DEVICE)
    gt_traj[:, 0] = ego_future[:, 0]
    gt_traj[:, 1] = ego_future[:, 1]
    gt_traj[:, 2] = torch.cos(ego_future[:, 2])
    gt_traj[:, 3] = torch.sin(ego_future[:, 2])

    # Track absolute ego pose
    ego_abs_x, ego_abs_y, ego_abs_h = 0.0, 0.0, 0.0

    n_steps = 10  # test 10 steps (1 second)
    print(f"\nRolling out {n_steps} steps with GT trajectory...")

    for step_t in range(n_steps):
        # Advance neighbors using GT future
        if step_t < nb_future.shape[1]:
            nb_gt = nb_future[:, step_t, :]  # [32, 3] in original frame
            nb_curr_4d = transform_positions_to_ego_frame(
                nb_gt, ego_abs_x, ego_abs_y, ego_abs_h, DEVICE,
            )
            advance_neighbor_past(data, nb_curr_4d, dt=0.1)

        # Build relative trajectory from current frame
        # GT trajectory from original frame -> ego frame
        remaining_gt = ego_future[step_t:, :].clone()
        cos_h = math.cos(ego_abs_h)
        sin_h = math.sin(ego_abs_h)
        rel_traj = torch.zeros(remaining_gt.shape[0], 4, device=DEVICE)
        rel_x = remaining_gt[:, 0] - ego_abs_x
        rel_y = remaining_gt[:, 1] - ego_abs_y
        rel_traj[:, 0] = cos_h * rel_x + sin_h * rel_y
        rel_traj[:, 1] = -sin_h * rel_x + cos_h * rel_y
        rel_heading = remaining_gt[:, 2] - ego_abs_h
        rel_traj[:, 2] = torch.cos(rel_heading)
        rel_traj[:, 3] = torch.sin(rel_heading)

        # Update state
        data, (dx, dy, dh) = update_scene_state(data, rel_traj, step_idx=0, dt=0.1)

        # Update absolute pose
        ego_abs_x += dx * cos_h - dy * sin_h
        ego_abs_y += dx * sin_h + dy * cos_h
        ego_abs_h += dh

        # Verify ego at origin
        ego_x = data["ego_current_state"][0, 0].item()
        ego_y = data["ego_current_state"][0, 1].item()
        ego_cos = data["ego_current_state"][0, 2].item()
        ego_sin = data["ego_current_state"][0, 3].item()

        assert abs(ego_x) < 1e-4, f"Step {step_t}: ego x={ego_x}"
        assert abs(ego_y) < 1e-4, f"Step {step_t}: ego y={ego_y}"
        assert abs(ego_cos - 1.0) < 1e-4, f"Step {step_t}: ego cos={ego_cos}"
        assert abs(ego_sin) < 1e-4, f"Step {step_t}: ego sin={ego_sin}"

        vx = data["ego_current_state"][0, 4].item()
        vy = data["ego_current_state"][0, 5].item()
        speed = math.sqrt(vx**2 + vy**2)

        print(
            f"  Step {step_t}: abs=({ego_abs_x:.3f}, {ego_abs_y:.3f}, "
            f"{math.degrees(ego_abs_h):.1f}deg), "
            f"speed={speed:.2f} m/s, goal=({data['goal_pose'][0, 0]:.1f}, {data['goal_pose'][0, 1]:.1f})"
        )

    # Verify absolute position matches GT
    gt_x = ego_future[n_steps - 1, 0].item()
    gt_y = ego_future[n_steps - 1, 1].item()
    gt_h = ego_future[n_steps - 1, 2].item()

    print(f"\n  Final absolute: ({ego_abs_x:.4f}, {ego_abs_y:.4f}, {math.degrees(ego_abs_h):.2f}deg)")
    print(f"  GT position:    ({gt_x:.4f}, {gt_y:.4f}, {math.degrees(gt_h):.2f}deg)")

    pos_error = math.sqrt((ego_abs_x - gt_x) ** 2 + (ego_abs_y - gt_y) ** 2)
    heading_error = abs(ego_abs_h - gt_h)
    print(f"  Position error: {pos_error:.6f}m, Heading error: {math.degrees(heading_error):.4f}deg")

    assert pos_error < 0.01, f"Position error too large: {pos_error}m"
    assert heading_error < 0.01, f"Heading error too large: {heading_error}rad"

    print("\n  PASS: gt_rollout with real scene")


if __name__ == "__main__":
    print("Testing state_update with real NPZ scene...")
    test_gt_rollout()
    print("\nAll real scene tests passed!")
