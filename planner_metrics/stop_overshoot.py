"""Stop-position overshoot metric: predicted stop point vs. GT stop point.

Shared by the ``obstacle_stop`` and ``traffic_light_stop`` scenario labels:
the scene ends with the ego stopped ahead of an obstacle or a red signal, and
the GT ego trajectory records where it actually stopped. The predicted ego is
expected to stop at or before that same point along the route, projected as
arclength ``s`` along the current, connected chain of route-lane centerlines.

Ported near-verbatim (numpy) from the reference open-loop analysis script:
the lane-chaining/arclength-projection geometry is inherently per-sample
(variable lane count/connectivity) and not easily vectorized across a batch
— the same reasoning behind the per-sample loop in
``planner_metrics/neighbor_clearance.py``.
"""

from __future__ import annotations

import numpy as np
import torch

from planner_metrics.evaluation import MetricEvaluation

_PREDICTION_TIMESTEP_SECONDS = 0.1
_STOP_SPEED_THRESHOLD_MPS = 0.5
_SUSTAINED_STOP_DURATION_S = 0.5
_ROUTE_CHAIN_GAP_M = 8.0


def _speed_mps(xy: np.ndarray) -> np.ndarray:
    result = np.zeros(len(xy), dtype=float)
    if len(xy) > 1:
        result[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) / _PREDICTION_TIMESTEP_SECONDS
        result[0] = result[1]
    return result


def _valid_points(values: np.ndarray) -> np.ndarray:
    pts = np.asarray(values, dtype=float)
    return pts[np.isfinite(pts).all(axis=1) & (np.abs(pts).sum(axis=1) > 1e-6)]


def _point_to_segments_min_dist(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    start, end = segments[:, 0], segments[:, 1]
    direction = end - start
    denom = np.einsum("ij,ij->i", direction, direction)
    keep = denom > 1e-8
    start, direction, denom = start[keep], direction[keep], denom[keep]
    values = []
    for point in points:
        t = np.clip(np.einsum("ij,ij->i", point - start, direction) / denom, 0.0, 1.0)
        closest = start + direction * t[:, None]
        values.append(float(np.linalg.norm(closest - point, axis=1).min()))
    return np.asarray(values)


def _current_route_segments(route_lanes: np.ndarray) -> np.ndarray:
    """Chain the connected route-lane centerlines starting from the current lane."""
    lines = [_valid_points(lane[:, :2]) for lane in route_lanes]
    usable = [(i, line) for i, line in enumerate(lines) if len(line) > 1]
    if not usable:
        raise ValueError("route_lanes contains no usable centerline")
    segments_all = [np.asarray(list(zip(line[:-1], line[1:]))) for _, line in usable]
    current_pos, (_, current) = min(
        enumerate(usable),
        key=lambda item: float(
            _point_to_segments_min_dist(np.zeros((1, 2)), segments_all[item[0]])[0]
        ),
    )
    chain = [current]
    endpoint = current[-1]
    for _, candidate in usable[current_pos + 1 :]:
        forward = float(np.linalg.norm(candidate[0] - endpoint))
        reverse = float(np.linalg.norm(candidate[-1] - endpoint))
        if min(forward, reverse) > _ROUTE_CHAIN_GAP_M:
            break
        if reverse < forward:
            candidate = candidate[::-1]
        chain.append(candidate)
        endpoint = candidate[-1]
    return np.asarray([segment for line in chain for segment in zip(line[:-1], line[1:])])


def _project_along_route(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Project points onto the ordered route segments; return longitudinal s."""
    start, end = segments[:, 0], segments[:, 1]
    direction = end - start
    length = np.linalg.norm(direction, axis=1)
    valid = length > 1e-8
    start, direction, length = start[valid], direction[valid], length[valid]
    cumulative = np.concatenate(([0.0], np.cumsum(length[:-1])))
    projected_s = []
    for point in np.asarray(points)[:, :2]:
        t = np.clip(np.einsum("ij,ij->i", point - start, direction) / (length * length), 0.0, 1.0)
        closest = start + direction * t[:, None]
        index = int(np.argmin(np.linalg.norm(closest - point, axis=1)))
        projected_s.append(float(cumulative[index] + t[index] * length[index]))
    return np.asarray(projected_s)


def _stop_position_s(xy: np.ndarray, speed: np.ndarray, segments: np.ndarray) -> tuple[float, bool]:
    """Return the median s of the final sustained-stop cluster, or terminal s."""
    positions = _project_along_route(xy, segments)
    stopped = np.asarray(speed) <= _STOP_SPEED_THRESHOLD_MPS
    width = max(1, round(_SUSTAINED_STOP_DURATION_S / _PREDICTION_TIMESTEP_SECONDS))
    runs = []
    start = None
    for index, is_stopped in enumerate(np.r_[stopped, False]):
        if is_stopped and start is None:
            start = index
        elif not is_stopped and start is not None:
            if index - start >= width:
                runs.append((start, index))
            start = None
    if not runs:
        return float(positions[-1]), False
    start, end = runs[-1]
    return float(np.median(positions[start:end])), True


def _sample_lanes(lanes: torch.Tensor, index: int) -> np.ndarray:
    """Select the route-lane tensor for one sample and return it as numpy."""
    if lanes.ndim == 5:
        if lanes.shape[1] != 1:
            raise ValueError(
                f"expected singleton route_lanes context axis, got {tuple(lanes.shape)}"
            )
        lanes = lanes[:, 0]
    if lanes.ndim == 3:
        scene = lanes
    elif lanes.ndim == 4:
        scene = lanes[0] if lanes.shape[0] == 1 else lanes[index]
    else:
        raise ValueError(
            f"route_lanes must have shape (S,P,D) or (N,S,P,D); got {tuple(lanes.shape)}"
        )
    return scene.detach().cpu().numpy()


@torch.no_grad()
def evaluate_stop_overshoot_with_details(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    parameters: dict,
) -> MetricEvaluation:
    """Evaluate stop-position overshoot vs. the GT stop position along the route.

    A sample fails when the predicted stop position (median s of the final
    sustained stop, or the terminal position if the ego never sustains a
    stop) is more than ``tolerance_m`` beyond the GT stop position, both
    projected onto the current chained route-lane centerline.
    """
    if ego_trajs.ndim != 3 or ego_trajs.shape[-1] < 2:
        raise ValueError(f"ego_trajs must have shape (N, T, D>=2), got {tuple(ego_trajs.shape)}")
    tolerance = float(parameters["tolerance_m"])
    if tolerance < 0:
        raise ValueError("stop_overshoot tolerance_m must be non-negative")

    lanes = data.get("route_lanes", data.get("lanes"))
    gt_future = data.get("ego_agent_future")
    if lanes is None or gt_future is None:
        raise ValueError(
            "stop_overshoot requires route_lanes (or lanes) and ego_agent_future in data"
        )

    batch_size = ego_trajs.shape[0]
    overshoot = torch.zeros(batch_size, dtype=ego_trajs.dtype, device=ego_trajs.device)
    gt_stop_s = torch.zeros(batch_size, dtype=ego_trajs.dtype, device=ego_trajs.device)
    pred_stop_s = torch.zeros(batch_size, dtype=ego_trajs.dtype, device=ego_trajs.device)
    gt_sustained_stop = torch.zeros(batch_size, dtype=torch.bool, device=ego_trajs.device)
    pred_sustained_stop = torch.zeros(batch_size, dtype=torch.bool, device=ego_trajs.device)

    for index in range(batch_size):
        segments = _current_route_segments(_sample_lanes(lanes, index))
        pred_xy = ego_trajs[index, :, :2].detach().cpu().numpy().astype(float)
        gt_xy = gt_future[index, :, :2].detach().cpu().numpy().astype(float)
        gt_s, gt_stopped = _stop_position_s(gt_xy, _speed_mps(gt_xy), segments)
        pred_s, pred_stopped = _stop_position_s(pred_xy, _speed_mps(pred_xy), segments)
        overshoot[index] = max(0.0, pred_s - gt_s)
        gt_stop_s[index] = gt_s
        pred_stop_s[index] = pred_s
        gt_sustained_stop[index] = gt_stopped
        pred_sustained_stop[index] = pred_stopped

    passed = overshoot <= tolerance
    return MetricEvaluation(
        scores={"failure_rate_percent": (~passed).to(ego_trajs.dtype) * 100.0},
        details={
            "stop_overshoot": {
                "overshoot_m": overshoot,
                "tolerance_m": torch.full_like(overshoot, tolerance),
                "gt_stop_position_s_m": gt_stop_s,
                "predicted_stop_position_s_m": pred_stop_s,
                "gt_sustained_stop": gt_sustained_stop.to(ego_trajs.dtype),
                "predicted_sustained_stop": pred_sustained_stop.to(ego_trajs.dtype),
            }
        },
    )


__all__ = ["evaluate_stop_overshoot_with_details"]
