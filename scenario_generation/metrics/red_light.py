"""Per-step red-light violation via ``planner_metrics`` red-light scorer."""

from __future__ import annotations

import numpy as np
import torch

from planner_metrics.config import RewardConfig
from planner_metrics.subscores import compute_red_light_score_batch
from scenario_generation.metrics.ego_traj import ego_traj_ego_frame

# Same moving gate as ``compute_red_light_score_batch`` (ego_speed > 0.5).
_RED_LIGHT_MIN_SPEED_MPS = 0.5


def score_red_light_step(
    np_dict: dict,
    *,
    device: str,
    ego_speed_mps: float,
    live_pose: np.ndarray,
    ego_hist: np.ndarray,
) -> dict:
    """Whether the live ego is violating a red light on ``route_lanes`` this step.

    Reuses ``compute_red_light_score_batch`` (bag-NPZ baked TL on ``route_lanes`` only).
    Trajectory comes from ``ego_traj_ego_frame(..., n_steps=2)`` (real ``ego_hist``),
    so the training scorer's finite-difference speed gate sees actual motion.
    """
    if float(ego_speed_mps) <= _RED_LIGHT_MIN_SPEED_MPS:
        return {"red_light_violation": False}
    if "route_lanes" not in np_dict:
        return {"red_light_violation": False}

    rl = np.asarray(np_dict["route_lanes"], dtype=np.float32)
    if rl.ndim not in (3, 4):
        return {"red_light_violation": False}
    rl_t = torch.tensor(rl, dtype=torch.float32, device=device)

    scores = compute_red_light_score_batch(
        ego_traj_ego_frame(live_pose, ego_hist, n_steps=2, device=device),
        {"route_lanes": rl_t},
        RewardConfig(),
    )
    return {"red_light_violation": bool(scores[0].item() < 0.0)}
