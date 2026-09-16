"""Lane-change completion metric for scenario-based open-loop evaluation.

A lane-change scene is scored against one question: within the prediction
horizon, does the planner end up in the lane the recorded ego moved to?

Answering that needs a lateral reference that stays valid for the whole
horizon, which the NPZ lane tensor does not offer directly -- each lanelet is a
short polyline and the ego crosses several of them in 8 s.  So the metric first
rebuilds the *source lane*: the lanelet the ego starts in, extended forward
through its endpoint-chained successors.  Every lateral quantity below is then
measured against that path, which deliberately decouples the score from
longitudinal error -- a planner that is merely too slow still gets credit for
the lane change it did perform, and ``arrival``/``simple_turn`` already cover
longitudinal and path-shape accuracy.

Lane *identity* is never compared, because the NPZ carries no lanelet ids, only
geometry.  "Ended up in the GT lane" is therefore expressed as "ended up within
half a lane width of the GT's final lateral position", and "left the original
lane" as "crossed the source lane's own boundary", with the lane width read
from the lane tensor's boundary-offset columns rather than hard-coded.
"""

from __future__ import annotations

import torch

from planner_metrics.evaluation import MetricEvaluation
from planner_metrics.geometry import (
    _point_to_segments_min_dist,
    _point_to_segments_signed_lateral,
)
from planner_metrics.horizon import resolve_horizon_steps
from planner_metrics.lane_tensor import resolve_lane_tensor

_PREDICTION_TIMESTEP_SECONDS = 0.1
_LANE_POINT_MIN_NORM = 1e-6
_SEGMENT_MIN_LENGTH = 1e-6
# Chain far enough past the trajectories that their endpoints still project
# onto a real segment instead of the path's extrapolated tail.
_SOURCE_PATH_MARGIN_M = 20.0

_DEFAULT_HORIZON_SECONDS = 8.0
_DEFAULT_MINIMUM_LATERAL_SHIFT_M = 1.0
_DEFAULT_CHAIN_TOLERANCE_M = 1.0


def _valid_point_mask(lane: torch.Tensor) -> torch.Tensor:
    """Return which of a lanelet's ``(P, D)`` rows are real points, not padding."""
    return lane[:, :4].abs().sum(dim=-1) > _LANE_POINT_MIN_NORM


def _lane_centerline(lane: torch.Tensor) -> torch.Tensor:
    """Return a lanelet's valid centerline points, shape ``(K, 2)``."""
    return lane[_valid_point_mask(lane)][:, :2]


def _polyline_segments(polyline: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a polyline's non-degenerate segments as ``(p1, p2)``."""
    p1, p2 = polyline[:-1], polyline[1:]
    keep = (p2 - p1).norm(dim=-1) > _SEGMENT_MIN_LENGTH
    return p1[keep], p2[keep]


def _lane_half_widths(lane: torch.Tensor) -> tuple[float, float]:
    """Return the lanelet's ``(left, right)`` half width in meters.

    The lane tensor stores each boundary as an offset from the centerline
    point, so the half width is that offset's length.  The median over the
    lanelet's points keeps a single malformed boundary point from distorting
    the threshold.
    """
    points = lane[_valid_point_mask(lane)]
    return (
        float(points[:, 4:6].norm(dim=-1).median()),
        float(points[:, 6:8].norm(dim=-1).median()),
    )


def _nearest_lane_index(point: torch.Tensor, lanes: torch.Tensor) -> int:
    """Return the index of the lanelet whose centerline is closest to ``point``."""
    best_index = -1
    best_distance = float("inf")
    for index in range(lanes.shape[0]):
        centerline = _lane_centerline(lanes[index])
        if centerline.shape[0] < 2:
            continue
        seg_p1, seg_p2 = _polyline_segments(centerline)
        if seg_p1.shape[0] == 0:
            continue
        distance = float(_point_to_segments_min_dist(point.reshape(1, 2), seg_p1, seg_p2)[0])
        if distance < best_distance:
            best_distance = distance
            best_index = index
    if best_index < 0:
        raise ValueError("lane_change metric found no usable lanelet centerline")
    return best_index


def _build_source_lane_path(
    lanes: torch.Tensor,
    source_index: int,
    required_length_m: float,
    chain_tolerance_m: float,
) -> torch.Tensor:
    """Extend the source lanelet forward through its endpoint-chained successors.

    Two lanelets chain when the successor's first point coincides with the
    current tail (that is how lanelet2 successor links survive into the NPZ,
    which carries no explicit connectivity).  Where several successors chain --
    a fork -- the one whose initial tangent best matches the current heading
    wins, so the reconstructed path follows the ego's own lane rather than a
    turning branch.  Chaining stops once the path is long enough to cover the
    scored trajectories.
    """
    path = _lane_centerline(lanes[source_index])
    if path.shape[0] < 2:
        raise ValueError("lane_change metric needs at least two points in the source lanelet")

    used = {source_index}
    while float((path[1:] - path[:-1]).norm(dim=-1).sum()) < required_length_m:
        tail_direction = path[-1] - path[-2]
        tail_norm = float(tail_direction.norm())
        if tail_norm <= _SEGMENT_MIN_LENGTH:
            break
        tail_direction = tail_direction / tail_norm

        best_index = -1
        best_alignment = 0.0
        for index in range(lanes.shape[0]):
            if index in used:
                continue
            centerline = _lane_centerline(lanes[index])
            if centerline.shape[0] < 2:
                continue
            if float((centerline[0] - path[-1]).norm()) > chain_tolerance_m:
                continue
            start_direction = centerline[1] - centerline[0]
            start_norm = float(start_direction.norm())
            if start_norm <= _SEGMENT_MIN_LENGTH:
                continue
            alignment = float(tail_direction @ (start_direction / start_norm))
            if alignment > best_alignment:
                best_alignment = alignment
                best_index = index

        if best_index < 0:
            break
        used.add(best_index)
        path = torch.cat([path, _lane_centerline(lanes[best_index])[1:]], dim=0)

    return path


def _resolve_gt_future(ego_trajs: torch.Tensor, data: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return ``ego_agent_future`` broadcast to the prediction's batch size."""
    gt_future = data.get("ego_agent_future")
    if gt_future is None:
        raise ValueError("lane_change metric requires ego_agent_future in data")
    if gt_future.ndim == 2:
        gt_future = gt_future.unsqueeze(0)
    if gt_future.ndim != 3 or gt_future.shape[-1] < 2:
        raise ValueError(
            f"lane_change ground truth must have shape (N, T, D>=2), got {tuple(gt_future.shape)}"
        )
    if gt_future.shape[0] not in (1, ego_trajs.shape[0]):
        raise ValueError(
            "lane_change ground truth batch dimension must be 1 or match predictions; "
            f"got {gt_future.shape[0]} for N={ego_trajs.shape[0]}"
        )
    gt_future = gt_future.to(device=ego_trajs.device, dtype=ego_trajs.dtype)
    if gt_future.shape[0] == 1:
        gt_future = gt_future.expand(ego_trajs.shape[0], -1, -1)
    return gt_future


@torch.no_grad()
def compute_lane_change_components_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    horizon_steps: int | None = None,
    chain_tolerance_m: float = _DEFAULT_CHAIN_TOLERANCE_M,
) -> dict[str, torch.Tensor]:
    """Return per-sample lateral offsets from the source lane, plus its geometry.

    ``predicted_lateral_offset_m``/``gt_lateral_offset_m`` have shape
    ``(N, T_selected)`` and are signed, positive to the left of the source
    lane's direction of travel.  The remaining entries are per-sample ``(N,)``
    scene descriptors that the scoring layer turns into thresholds.
    """
    if ego_trajs.ndim != 3 or ego_trajs.shape[-1] < 2:
        raise ValueError(f"ego_trajs must have shape (N, T, D>=2), got {tuple(ego_trajs.shape)}")

    gt_future = _resolve_gt_future(ego_trajs, data)
    available_steps = min(ego_trajs.shape[1], gt_future.shape[1])
    if horizon_steps is None:
        horizon_steps = available_steps
    if not 1 <= horizon_steps <= available_steps:
        raise ValueError(f"horizon_steps must be in [1, {available_steps}]")

    # ``lanes`` is preferred over ``route_lanes`` here (the opposite of
    # centerline.py): a lane change leaves the route corridor's own lane, so the
    # metric needs the full surrounding lane set to rebuild the source lane and
    # to have the target lane's geometry present at all.
    lanes = data.get("lanes", data.get("route_lanes"))
    if lanes is None:
        raise ValueError("lane_change metric requires lanes or route_lanes in data")
    lanes = resolve_lane_tensor(lanes, ego_trajs.shape[0])
    if lanes.shape[-1] < 8:
        raise ValueError(
            "lane_change metric needs the lane boundary-offset columns (D>=8); "
            f"got D={lanes.shape[-1]}"
        )

    predicted_offsets = []
    gt_offsets = []
    half_width_left = []
    half_width_right = []
    source_lane_index = []

    origin = torch.zeros(2, device=ego_trajs.device, dtype=ego_trajs.dtype)
    for index in range(ego_trajs.shape[0]):
        scene_lanes = lanes[0 if lanes.shape[0] == 1 else index].to(ego_trajs)
        predicted_xy = ego_trajs[index, :horizon_steps, :2]
        gt_xy = gt_future[index, :horizon_steps, :2]

        # The scene is in the ego frame at t=0, so the ego starts at the origin.
        source_index = _nearest_lane_index(origin, scene_lanes)
        required_length = (
            float(torch.cat([predicted_xy, gt_xy], dim=0).norm(dim=-1).max())
            + _SOURCE_PATH_MARGIN_M
        )
        path = _build_source_lane_path(
            scene_lanes, source_index, required_length, chain_tolerance_m
        )
        seg_p1, seg_p2 = _polyline_segments(path)
        if seg_p1.shape[0] == 0:
            raise ValueError("lane_change metric found no usable source-lane segments")

        predicted_offsets.append(_point_to_segments_signed_lateral(predicted_xy, seg_p1, seg_p2))
        gt_offsets.append(_point_to_segments_signed_lateral(gt_xy, seg_p1, seg_p2))
        left, right = _lane_half_widths(scene_lanes[source_index])
        half_width_left.append(left)
        half_width_right.append(right)
        source_lane_index.append(source_index)

    def as_tensor(values: list[float]) -> torch.Tensor:
        return torch.tensor(values, device=ego_trajs.device, dtype=ego_trajs.dtype)

    return {
        "predicted_lateral_offset_m": torch.stack(predicted_offsets, dim=0),
        "gt_lateral_offset_m": torch.stack(gt_offsets, dim=0),
        "lane_half_width_left_m": as_tensor(half_width_left),
        "lane_half_width_right_m": as_tensor(half_width_right),
        "source_lane_index": as_tensor([float(value) for value in source_lane_index]),
    }


@torch.no_grad()
def evaluate_lane_change_with_details(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    parameters: dict,
) -> MetricEvaluation:
    """Score whether the prediction completes the GT's lane change within the horizon.

    A sample succeeds when, at the end of the horizon, the prediction has
    crossed the source lane's boundary in the same direction the GT went AND
    has landed within half a lane width of the GT's final lateral position.
    The first condition rejects staying put, drifting the wrong way, and
    starting a change without finishing it; the second rejects overshooting
    into the lane beyond the target.

    ``minimum_lateral_shift_m`` is a floor under the boundary threshold, so a
    map with missing or degenerate boundary offsets cannot make the crossing
    test trivially true.

    Every sample must be a genuine lane-change scene: if the recorded ego never
    leaves its own lane within the horizon, the NPZ does not belong in the
    ``lane_change`` list and evaluation fails loudly rather than scoring it.
    """
    horizon_seconds = float(parameters.get("horizon_seconds", _DEFAULT_HORIZON_SECONDS))
    minimum_lateral_shift_m = float(
        parameters.get("minimum_lateral_shift_m", _DEFAULT_MINIMUM_LATERAL_SHIFT_M)
    )
    chain_tolerance_m = float(parameters.get("chain_tolerance_m", _DEFAULT_CHAIN_TOLERANCE_M))

    gt_future = _resolve_gt_future(ego_trajs, data)
    steps = resolve_horizon_steps(
        horizon_seconds,
        min(ego_trajs.shape[1], gt_future.shape[1]),
        label="lane_change",
        timestep_seconds=_PREDICTION_TIMESTEP_SECONDS,
    )
    components = compute_lane_change_components_batch(ego_trajs, data, steps, chain_tolerance_m)
    predicted_offsets = components["predicted_lateral_offset_m"]
    gt_offsets = components["gt_lateral_offset_m"]

    successes = []
    completion_ratios = []
    offset_errors = []
    change_times = []
    directions = []
    crossed_flags = []
    reached_flags = []

    for index in range(ego_trajs.shape[0]):
        gt_shift = float(gt_offsets[index, -1])
        direction = 1.0 if gt_shift >= 0.0 else -1.0
        half_width = float(
            components["lane_half_width_left_m"][index]
            if direction > 0
            else components["lane_half_width_right_m"][index]
        )
        crossing_threshold = max(half_width, minimum_lateral_shift_m)

        if direction * gt_shift <= crossing_threshold:
            horizon_s = steps * _PREDICTION_TIMESTEP_SECONDS
            raise ValueError(
                f"lane_change sample {index} is not a lane-change scene: the recorded ego ends "
                f"{direction * gt_shift:.2f} m from its own lane center after {horizon_s:.1f} s, "
                f"which never clears the {crossing_threshold:.2f} m lane boundary"
            )

        signed_progress = direction * predicted_offsets[index]
        crossed = bool(signed_progress[-1] > crossing_threshold)
        predicted_shift = float(predicted_offsets[index, -1])
        reached = bool(abs(predicted_shift - gt_shift) <= half_width)

        beyond = (signed_progress > crossing_threshold).nonzero()
        change_time = (
            float(beyond[0]) * _PREDICTION_TIMESTEP_SECONDS
            if beyond.numel() > 0
            else steps * _PREDICTION_TIMESTEP_SECONDS
        )

        successes.append(float(crossed and reached))
        completion_ratios.append(min(max(predicted_shift / gt_shift, 0.0), 1.0))
        offset_errors.append(abs(predicted_shift - gt_shift))
        change_times.append(change_time)
        directions.append(direction)
        crossed_flags.append(float(crossed))
        reached_flags.append(float(reached))

    def as_tensor(values: list[float]) -> torch.Tensor:
        return torch.tensor(values, device=ego_trajs.device, dtype=ego_trajs.dtype)

    return MetricEvaluation(
        scores={
            "lane_change_success": as_tensor(successes),
            "lane_change_completion_ratio": as_tensor(completion_ratios),
            "final_lateral_offset_error_m": as_tensor(offset_errors),
            "lane_change_time_s": as_tensor(change_times),
        },
        details={
            "lane_change": {
                "predicted_lateral_shift_m": predicted_offsets[:, -1],
                "gt_lateral_shift_m": gt_offsets[:, -1],
                "initial_lateral_offset_m": gt_offsets[:, 0],
                "gt_direction": as_tensor(directions),
                "left_source_lane": as_tensor(crossed_flags),
                "reached_gt_lane": as_tensor(reached_flags),
                "lane_half_width_left_m": components["lane_half_width_left_m"],
                "lane_half_width_right_m": components["lane_half_width_right_m"],
                "source_lane_index": components["source_lane_index"],
            }
        },
    )


__all__ = [
    "compute_lane_change_components_batch",
    "evaluate_lane_change_with_details",
]
