"""Per-step road-border (curb) clearance for closed-loop eval."""

from __future__ import annotations

import numpy as np
import torch

from planner_metrics.config import RewardConfig
from planner_metrics.subscores import compute_road_border_penalty


def score_road_border_step(np_dict: dict, *, device: str) -> dict:
    """Ego-to-road-border distance at the current step.

    Uses ``line_strings`` channel 3 as road border. Collision / miss masks are
    derived later in ``_finalize`` from the distance series. Current pose is the
    ego-frame origin; history is not needed.
    """
    rb_dist_m = float("inf")
    if "line_strings" in np_dict and "ego_shape" in np_dict:
        ego_shape_t = torch.tensor(
            np.asarray(np_dict["ego_shape"]).reshape(-1)[:3],
            dtype=torch.float32,
            device=device,
        )
        traj = torch.zeros(1, 1, 4, dtype=torch.float32, device=device)
        traj[..., 2] = 1.0  # origin, heading +x
        ls = np.asarray(np_dict["line_strings"])
        if ls.ndim == 4:
            ls = ls[0]
        data = {"line_strings": torch.tensor(ls, dtype=torch.float32, device=device)}
        _gate, _near, _wide, _steps, _cont, per_ts_min = compute_road_border_penalty(
            traj, ego_shape_t, data, config=RewardConfig()
        )
        rb_dist_m = float(per_ts_min[0, 0].item())

    return {"rb_dist_m": rb_dist_m}
