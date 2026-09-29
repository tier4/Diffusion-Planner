"""Per-step route-centerline distance for closed-loop eval."""

from __future__ import annotations

import numpy as np
import torch

from planner_metrics.centerline import compute_centerline_distance_batch


def score_centerline_step(np_dict: dict, *, device: str) -> dict:
    """Ego-to-route-centerline nearest-segment distance at the current step.

    Current pose is the ego-frame origin (``route_lanes``/``lanes`` are already
    re-centered onto the live ego each step, same as every other per-step scorer
    here). Falls back to ``lanes`` when ``route_lanes`` is absent, same
    precedence as the open-loop centerline metric.
    """
    centerline_dist_m = float("inf")
    lanes_key = "route_lanes" if "route_lanes" in np_dict else "lanes"
    if lanes_key in np_dict:
        lanes_t = torch.tensor(np.asarray(np_dict[lanes_key]), dtype=torch.float32, device=device)
        traj = torch.zeros(1, 1, 2, dtype=torch.float32, device=device)
        dist = compute_centerline_distance_batch(traj, {lanes_key: lanes_t})
        centerline_dist_m = float(dist[0, 0].item())
    return {"centerline_dist_m": centerline_dist_m}
