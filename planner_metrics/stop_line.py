"""Stop-line metric: predicted ego must stop within the horizon without crossing the stop line.

Shared by the ``temporal_stop`` and ``traffic_light_stop`` scenario labels. The scene is
anchored around where the ego starts braking for a stop line (a stop sign, or a red signal
that stays red through the horizon), so the ego is usually still moving at the anchor. A
sample passes when, within ``horizon_seconds``, the predicted ego comes to a sustained stop
and its front never goes past the stop line.

The stop line is the nearest one ahead of the ego that crosses the current, connected chain
of route-lane centerlines (see ``planner_metrics/stop_overshoot.py``); positions are compared
as arclength ``s`` along that chain. Stop lines come from the native ``stop_lines`` polylines,
or from the NPZ ``line_strings`` rows whose stop-line flag is set. A scene with no stop line
on the route (or no route centerline at all) fails and records ``stop_line_found = 0``, which
separates a planner failure from a mis-curated list.
"""

from __future__ import annotations

import numpy as np
import torch

from planner_metrics.evaluation import MetricEvaluation
from planner_metrics.horizon import resolve_horizon_steps
from planner_metrics.stop_overshoot import (
    _current_route_segments,
    _project_along_route,
    _sample_lanes,
    _speed_mps,
    _valid_points,
)

_PREDICTION_TIMESTEP_SECONDS = 0.1
_STOP_SPEED_THRESHOLD_MPS = 0.5
_SUSTAINED_STOP_DURATION_S = 0.5
_LINE_STRING_STOP_LINE_FLAG = 2


def _sample_tensor(values: torch.Tensor, index: int, scene_ndim: int) -> np.ndarray:
    scene = values if values.ndim == scene_ndim else values[0 if values.shape[0] == 1 else index]
    return scene.detach().cpu().numpy().astype(float)


def _stop_line_polylines(data: dict[str, torch.Tensor], index: int) -> list[np.ndarray]:
    """Stop-line polylines ``(P, 2)`` of one sample, from ``stop_lines`` or ``line_strings``."""
    if "stop_lines" in data:
        lines = _sample_tensor(data["stop_lines"], index, scene_ndim=3)
        polylines = [_valid_points(line[:, :2]) for line in lines]
    elif "line_strings" in data:
        lines = _sample_tensor(data["line_strings"], index, scene_ndim=3)
        polylines = [
            _valid_points(line[:, :2])
            for line in lines
            if (line[:, _LINE_STRING_STOP_LINE_FLAG] > 0.5).any()
        ]
    else:
        raise ValueError("stop_line requires stop_lines (or line_strings) in data")
    return [line for line in polylines if len(line) > 1]


def _segment_intersections(a0, a1, b0, b1) -> np.ndarray:
    """Parameters ``t`` along segments ``a`` where they cross segments ``b``; NaN where not."""
    da = a1 - a0
    db = b1 - b0
    denom = da[:, None, 0] * db[None, :, 1] - da[:, None, 1] * db[None, :, 0]
    offset = b0[None] - a0[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (offset[..., 0] * db[None, :, 1] - offset[..., 1] * db[None, :, 0]) / denom
        u = (offset[..., 0] * da[:, None, 1] - offset[..., 1] * da[:, None, 0]) / denom
    hit = (np.abs(denom) > 1e-9) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return np.where(hit, t, np.nan)


def _stop_line_s(polylines: list[np.ndarray], segments: np.ndarray, ego_s: float) -> float:
    """Route arclength of the nearest stop line ahead of ``ego_s``; NaN when there is none."""
    start, end = segments[:, 0], segments[:, 1]
    length = np.linalg.norm(end - start, axis=1)
    keep = length > 1e-8
    start, end, length = start[keep], end[keep], length[keep]
    cumulative = np.concatenate(([0.0], np.cumsum(length[:-1])))
    best = np.inf
    for line in polylines:
        t = _segment_intersections(start, end, line[:-1], line[1:])
        crossing = cumulative[:, None] + t * length[:, None]
        ahead = crossing[np.isfinite(crossing) & (crossing > ego_s)]
        if ahead.size:
            best = min(best, float(ahead.min()))
    return best if np.isfinite(best) else np.nan


def _has_sustained_stop(speed: np.ndarray) -> bool:
    width = max(1, round(_SUSTAINED_STOP_DURATION_S / _PREDICTION_TIMESTEP_SECONDS))
    stopped = (speed <= _STOP_SPEED_THRESHOLD_MPS).astype(int)
    return bool((np.convolve(stopped, np.ones(width, dtype=int), mode="valid") == width).any())


@torch.no_grad()
def evaluate_stop_line_with_details(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    parameters: dict,
) -> MetricEvaluation:
    """Evaluate stopping before the stop line and return per-sample diagnostic details.

    A sample passes when, within ``horizon_seconds``, the prediction holds a sustained stop
    (speed at or below 0.5 m/s for 0.5 s) and the ego front never reaches the stop line. The
    front is ``ego_base_link_to_front`` ahead of the rear axle along the heading; data without
    it (NPZ) falls back to ``wheel_base / 2 + length / 2``, the front of the ego box of
    ``planner_metrics.geometry``, which assumes equal overhangs.
    """
    if ego_trajs.ndim != 3 or ego_trajs.shape[-1] < 4:
        raise ValueError(f"ego_trajs must have shape (N, T, D>=4), got {tuple(ego_trajs.shape)}")
    horizon_seconds = float(parameters["horizon_seconds"])
    steps = resolve_horizon_steps(
        horizon_seconds,
        ego_trajs.shape[1],
        label="stop_line",
        timestep_seconds=_PREDICTION_TIMESTEP_SECONDS,
    )
    lanes = data.get("route_lanes", data.get("lanes"))
    ego_shape = data.get("ego_shape")
    base_link_to_front = data.get("ego_base_link_to_front")
    if lanes is None or (ego_shape is None and base_link_to_front is None):
        raise ValueError(
            "stop_line requires route_lanes (or lanes) and ego_base_link_to_front "
            "(or ego_shape) in data"
        )

    batch_size = ego_trajs.shape[0]
    stop_line_found = np.zeros(batch_size, dtype=bool)
    stopped = np.zeros(batch_size, dtype=bool)
    crossed = np.zeros(batch_size, dtype=bool)
    stop_line_distance = np.full(batch_size, np.nan)
    front_margin = np.full(batch_size, np.nan)

    for index in range(batch_size):
        trajectory = ego_trajs[index, :steps].detach().cpu().numpy().astype(float)
        stopped[index] = _has_sustained_stop(_speed_mps(trajectory[:, :2]))
        try:
            segments = _current_route_segments(_sample_lanes(lanes, index))
        except ValueError:  # no usable route centerline, so no stop line on the route either
            continue
        if base_link_to_front is not None:
            front_offset = float(_sample_tensor(base_link_to_front, index, scene_ndim=0))
        else:
            wheel_base, length = _sample_tensor(ego_shape, index, scene_ndim=1)[:2]
            front_offset = 0.5 * wheel_base + 0.5 * length
        heading = trajectory[:, 2:4] / np.maximum(
            np.linalg.norm(trajectory[:, 2:4], axis=1, keepdims=True), 1e-6
        )
        front = trajectory[:, :2] + heading * front_offset
        ego_s = float(_project_along_route(np.zeros((1, 2)), segments)[0])
        line_s = _stop_line_s(_stop_line_polylines(data, index), segments, ego_s)
        if np.isnan(line_s):
            continue
        margin = line_s - _project_along_route(front, segments).max()
        stop_line_found[index] = True
        crossed[index] = margin < 0
        stop_line_distance[index] = line_s - ego_s
        front_margin[index] = margin

    passed = stop_line_found & stopped & ~crossed

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=ego_trajs.dtype, device=ego_trajs.device)

    return MetricEvaluation(
        scores={"success_rate_percent": tensor(passed) * 100.0},
        details={
            "stop_line": {
                "horizon_seconds": torch.full((batch_size,), horizon_seconds),
                "stop_line_found": tensor(stop_line_found),
                "stop_line_distance_m": tensor(stop_line_distance),
                "min_front_to_stop_line_m": tensor(front_margin),
                "crossed_stop_line": tensor(crossed),
                "predicted_sustained_stop": tensor(stopped),
            }
        },
    )


__all__ = ["evaluate_stop_line_with_details"]
