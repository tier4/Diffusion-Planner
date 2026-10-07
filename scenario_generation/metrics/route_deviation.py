"""Per-step route deviation (distance to the route centerline) for closed-loop eval."""

from __future__ import annotations

import numpy as np
import torch

from planner_metrics.centerline import compute_centerline_distance_batch, has_centerline_segments


def score_route_deviation_step(np_dict: dict, *, device: str) -> dict:
    """Ego-to-route-centerline nearest-segment distance at the current step.

    Current pose is the ego-frame origin (``route_lanes``/``lanes`` are already
    re-centered onto the live ego each step, same as every other per-step scorer
    here). Falls back to ``lanes`` when ``route_lanes`` is absent, same
    precedence as the open-loop centerline metric. A step whose lanes hold no usable segment
    (all zero padding, e.g. past the route end -- not a matter of how far the ego is) has no
    distance to report and scores ``inf``, like a step with no lanes at all; the mean skips
    non-finite steps and the summary reports their share as the unmeasured rate.
    """
    route_dist_m = float("inf")
    lanes_key = "route_lanes" if "route_lanes" in np_dict else "lanes"
    if lanes_key in np_dict:
        lanes_t = torch.tensor(np.asarray(np_dict[lanes_key]), dtype=torch.float32, device=device)
        if not has_centerline_segments(lanes_t.reshape(-1, *lanes_t.shape[-2:])):
            return {"route_dist_m": route_dist_m}
        traj = torch.zeros(1, 1, 2, dtype=torch.float32, device=device)
        dist = compute_centerline_distance_batch(traj, {lanes_key: lanes_t})
        route_dist_m = float(dist[0, 0].item())
    return {"route_dist_m": route_dist_m}
