"""Neighbor clearance and road-border distance for closed-loop steps."""

from __future__ import annotations

import numpy as np
import torch

from planner_metrics.config import RewardConfig
from planner_metrics.subscores import compute_road_border_penalty
from scenario_generation.reproducer_rollout import score_step


def score_safety_step(
    neighbors_live: np.ndarray,
    ego_shape: np.ndarray,
    np_dict: dict,
    device: str,
    *,
    margin_m: float = 0.3,
) -> dict:
    """Per-step neighbor clearance + road-border distance and violation flags."""
    clearance_m, collision, _n_valid = score_step(
        neighbors_live, ego_shape, ego_speed=0.0, device=device
    )

    rb_dist_m = float("inf")
    if "line_strings" in np_dict and "ego_shape" in np_dict:
        ego_shape_t = torch.tensor(
            np.asarray(np_dict["ego_shape"]).reshape(-1)[:3],
            dtype=torch.float32,
            device=device,
        )
        traj = torch.zeros(1, 1, 4, dtype=torch.float32, device=device)
        traj[0, 0, 2] = 1.0  # heading +x in ego frame
        data = {
            "line_strings": torch.tensor(
                np_dict["line_strings"][None], dtype=torch.float32, device=device
            )
        }
        _gate, _near, _wide, _steps, _cont, per_ts_min = compute_road_border_penalty(
            traj, ego_shape_t, data, config=RewardConfig()
        )
        rb_dist_m = float(per_ts_min[0, 0].item())

    neighbor_violation = clearance_m < margin_m
    rb_violation = rb_dist_m < margin_m
    return {
        "clearance_m": clearance_m,
        "collision": collision,
        "rb_dist_m": rb_dist_m,
        "neighbor_violation": neighbor_violation,
        "rb_violation": rb_violation,
        "safety_violation": neighbor_violation or rb_violation or collision,
    }
