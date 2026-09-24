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

# A lanelet only qualifies as the ego's own lane when its tangent at the ego is
# within ~37 deg of the ego heading; a crossing road's turn lanelet that passes
# through the ego position is steeper than that.
_MIN_SOURCE_HEADING_ALIGNMENT = 0.8
# Lanelets whose centerline distance from the ego differs by less than this are
# a tie (typically two branches of a fork starting at the same point).
_SOURCE_DISTANCE_TIE_M = 0.1

_DEFAULT_HORIZON_SECONDS = 8.0
_DEFAULT_MINIMUM_LATERAL_SHIFT_M = 1.0
_DEFAULT_CHAIN_TOLERANCE_M = 1.0


def _valid_point_mask(lane: torch.Tensor) -> torch.Tensor:
    """Return which of a lanelet's ``(P, D)`` rows are real points, not padding."""
    return lane[:, :4].abs().sum(dim=-1) > _LANE_POINT_MIN_NORM


def _polyline_segments(polyline: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a polyline's non-degenerate segments as ``(p1, p2)``."""
    p1, p2 = polyline[:-1], polyline[1:]
    keep = (p2 - p1).norm(dim=-1) > _SEGMENT_MIN_LENGTH
    return p1[keep], p2[keep]


def _scene_centerlines(lanes: torch.Tensor) -> dict[int, torch.Tensor]:
    """Return each usable lanelet's centerline ``(K, 2)``, keyed by lanelet index.

    Padded-out lanelets and those without a single real segment are dropped
    here, once per scene, so the lane search below never has to guard for them.
    """
    centerlines = {}
    for index in range(lanes.shape[0]):
        centerline = lanes[index][_valid_point_mask(lanes[index])][:, :2]
        if centerline.shape[0] >= 2 and _polyline_segments(centerline)[0].shape[0] > 0:
            centerlines[index] = centerline
    if not centerlines:
        raise ValueError("lane_change metric found no usable lanelet centerline")
    return centerlines


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


def _nearest_lane_index(
    point: torch.Tensor, centerlines: dict[int, torch.Tensor]
) -> tuple[int, bool]:
    """Return ``(index, heading_aligned)`` for the lanelet the ego is travelling on at ``point``.

    Candidates are first filtered by heading: the centerline tangent nearest
    the ego must be within ``_MIN_SOURCE_HEADING_ALIGNMENT`` of the ego heading
    (the ego frame puts the ego at the origin heading +x, so that is simply the
    tangent's x-component).  This rejects oncoming and crossing lanelets, and
    also a crossing road's turn lanelet that sweeps through the ego position at
    45-70 deg -- common at an intersection, and closer to the ego than its own
    lane's centerline often enough to matter.

    Among the aligned candidates the nearest centerline wins.  Distances within
    ``_SOURCE_DISTANCE_TIE_M`` of the minimum are a tie -- at a fork the
    straight and turning branches start at the same point -- and the tie goes
    to the lanelet whose remaining run (from the ego to its end) points most
    along the ego heading, i.e. the straight branch.

    When no lanelet passes the heading filter the nearest one is returned with
    ``heading_aligned=False``; the scorer surfaces that flag per sample so a
    scene scored against an oncoming or crossing reference lane can be told
    apart from a genuine planner failure.

    This does NOT disambiguate two parallel lanes: once the ego is past the
    midpoint of a change already in progress, the lane it is moving INTO is the
    nearest one. Such a scene is reported as a failed precondition by the
    scorer rather than mis-scored — see ``evaluate_lane_change_with_details``.
    """
    point = point.reshape(1, 2)
    nearest = (float("inf"), -1)
    candidates: list[tuple[float, float, int]] = []  # (distance, ahead alignment, index)
    for index, centerline in centerlines.items():
        seg_p1, seg_p2 = _polyline_segments(centerline)
        distance = float(_point_to_segments_min_dist(point, seg_p1, seg_p2)[0])
        nearest = min(nearest, (distance, index))
        segment = int((((seg_p1 + seg_p2) / 2) - point).norm(dim=-1).argmin())
        tangent = seg_p2[segment] - seg_p1[segment]
        if float(tangent[0] / tangent.norm()) < _MIN_SOURCE_HEADING_ALIGNMENT:
            continue
        ahead = centerline[-1] - seg_p1[segment]  # spans at least the nearest segment
        candidates.append((distance, float(ahead[0] / ahead.norm()), index))

    if not candidates:
        return nearest[1], False
    nearest_distance = min(distance for distance, _, _ in candidates)
    tied = [c for c in candidates if c[0] <= nearest_distance + _SOURCE_DISTANCE_TIE_M]
    return max(tied, key=lambda c: c[1])[2], True


def _arc_length_behind(path: torch.Tensor, point: torch.Tensor) -> float:
    """Return the path's arc length that lies behind ``point``'s projection."""
    seg_p1, seg_p2 = _polyline_segments(path)
    segment = seg_p2 - seg_p1
    lengths = segment.norm(dim=-1)
    offset = point.reshape(1, 2) - seg_p1
    t = ((offset * segment).sum(-1) / (lengths**2)).clamp(0, 1)
    closest = seg_p1 + t.unsqueeze(-1) * segment
    nearest = int((point.reshape(1, 2) - closest).norm(dim=-1).argmin())
    return float(lengths[:nearest].sum() + t[nearest] * lengths[nearest])


def _build_source_lane_path(
    centerlines: dict[int, torch.Tensor],
    source_index: int,
    required_length_m: float,
    chain_tolerance_m: float,
) -> torch.Tensor:
    """Extend the source lanelet forward through its endpoint-chained successors.

    Two lanelets chain when the successor's first point coincides with the
    current tail (that is how lanelet2 successor links survive into the NPZ,
    which carries no explicit connectivity).  Where several successors chain --
    a fork -- the one whose overall direction (last point minus first point)
    best matches the current heading wins, so the reconstructed path follows
    the ego's own lane rather than a turning branch.  The overall direction is
    used, not the initial tangent: a turn lanelet leaves the fork tangent to
    the straight one, so their first segments are indistinguishable and the
    pick would come down to floating-point noise.

    Chaining stops once the path reaches ``required_length_m`` AHEAD OF THE
    EGO: lanelets are resampled to a fixed point count regardless of length,
    so the ego often sits deep inside a long source lanelet, and measuring the
    budget against the whole path (including what is behind the ego) would
    stop chaining while the trajectories still run off the end.
    """
    path = centerlines[source_index]
    # The ego is at the origin in the scene frame; the prefix behind it never
    # changes as successors are appended, so it is measured once.
    behind_ego = _arc_length_behind(path, torch.zeros(2, device=path.device, dtype=path.dtype))

    used = {source_index}
    while float((path[1:] - path[:-1]).norm(dim=-1).sum()) - behind_ego < required_length_m:
        tail_direction = path[-1] - path[-2]
        if float(tail_direction.norm()) <= _SEGMENT_MIN_LENGTH:
            break
        tail_direction = tail_direction / tail_direction.norm()

        best_index, best_alignment = -1, 0.0
        for index, centerline in centerlines.items():
            if index in used or float((centerline[0] - path[-1]).norm()) > chain_tolerance_m:
                continue
            overall = centerline[-1] - centerline[0]
            alignment = float(tail_direction @ overall) / max(
                float(overall.norm()), _SEGMENT_MIN_LENGTH
            )
            if alignment > best_alignment:
                best_alignment, best_index = alignment, index
        if best_index < 0:
            break
        used.add(best_index)
        path = torch.cat([path, centerlines[best_index][1:]], dim=0)

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


_COMPONENT_KEYS = (
    "predicted_lateral_offset_m",
    "gt_lateral_offset_m",
    "lane_half_width_left_m",
    "lane_half_width_right_m",
    "source_lane_index",
    "source_lane_heading_aligned",
)


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

    offsets: dict[str, list] = {key: [] for key in _COMPONENT_KEYS}
    for index in range(ego_trajs.shape[0]):
        # The lane search is a Python loop over lanelets with scalar reads, so
        # it runs on the CPU copy; only the final offsets touch the ego device.
        scene_lanes = lanes[0 if lanes.shape[0] == 1 else index].to("cpu", ego_trajs.dtype)
        predicted_xy = ego_trajs[index, :horizon_steps, :2]
        gt_xy = gt_future[index, :horizon_steps, :2]

        # The scene is in the ego frame at t=0, so the ego starts at the origin.
        centerlines = _scene_centerlines(scene_lanes)
        source_index, heading_aligned = _nearest_lane_index(torch.zeros(2), centerlines)
        required_length = (
            float(torch.cat([predicted_xy, gt_xy], dim=0).norm(dim=-1).max())
            + _SOURCE_PATH_MARGIN_M
        )
        path = _build_source_lane_path(
            centerlines, source_index, required_length, chain_tolerance_m
        )
        seg_p1, seg_p2 = _polyline_segments(path.to(ego_trajs.device))
        left, right = _lane_half_widths(scene_lanes[source_index])

        offsets["predicted_lateral_offset_m"].append(
            _point_to_segments_signed_lateral(predicted_xy, seg_p1, seg_p2)
        )
        offsets["gt_lateral_offset_m"].append(
            _point_to_segments_signed_lateral(gt_xy, seg_p1, seg_p2)
        )
        offsets["lane_half_width_left_m"].append(left)
        offsets["lane_half_width_right_m"].append(right)
        offsets["source_lane_index"].append(float(source_index))
        offsets["source_lane_heading_aligned"].append(float(heading_aligned))

    return {
        key: torch.stack(values, dim=0)
        if torch.is_tensor(values[0])
        else torch.tensor(values, device=ego_trajs.device, dtype=ego_trajs.dtype)
        for key, values in offsets.items()
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

    ``minimum_lateral_shift_m`` is a floor under the lane tolerance, so a map
    with missing or degenerate boundary offsets cannot make the crossing test
    trivially true nor the reached test impossible to satisfy.

    The aggregate score is the success rate in percent, like the other
    scenario metrics. A sample whose recorded ego never leaves the
    reconstructed source lane carries no lane change to score. It counts as a
    failure, and the per-sample ``gt_lane_change_detected`` detail records
    which samples did carry one, so the details can separate "the planner
    failed" from "this list (or the source-lane reconstruction) is wrong" --
    the latter is what a scene captured after the change is already past its
    midpoint looks like, since the nearest lane is then the one being entered.
    Completion ratio, final lateral offset error and the time at which the
    prediction left the source lane are also reported per sample in the details.
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
    predicted = components["predicted_lateral_offset_m"]  # (N, T)
    gt = components["gt_lateral_offset_m"]  # (N, T)
    initial_shift, gt_shift, predicted_shift = gt[:, 0], gt[:, -1], predicted[:, -1]

    # +1 when the GT moved left of the source lane, -1 when right.  One floored
    # tolerance serves both tests below: a map whose boundary-offset columns are
    # missing or degenerate reports a zero half width, which would otherwise
    # make `crossed` trivially true and `reached` require an exact match.
    direction = (gt_shift >= initial_shift).to(gt.dtype) * 2 - 1
    half_width = torch.where(
        direction > 0,
        components["lane_half_width_left_m"],
        components["lane_half_width_right_m"],
    )
    lane_tolerance = half_width.clamp(min=minimum_lateral_shift_m)

    # The recorded ego must itself have left the source lane, or there is no
    # lane change to score; every judgement below is gated on that.
    detected = direction * gt_shift > lane_tolerance
    signed_progress = direction[:, None] * predicted  # (N, T), positive towards the GT's side
    crossed = detected & (signed_progress[:, -1] > lane_tolerance)
    reached = detected & ((predicted_shift - gt_shift).abs() <= lane_tolerance)

    # Prediction index i is the pose at t=(i+1)*dt; a sample that never leaves
    # the source lane reports the full horizon.
    beyond = signed_progress > lane_tolerance[:, None]
    first_beyond = (beyond.to(torch.int8).argmax(dim=1) + 1) * _PREDICTION_TIMESTEP_SECONDS
    change_time = torch.where(
        detected & beyond.any(dim=1), first_beyond, steps * _PREDICTION_TIMESTEP_SECONDS
    )

    # Progress is measured from where the ego started, not from the lane
    # center, so an ego that begins off-center and never moves scores 0.
    # Closeness to the GT's lateral target (rather than raw progress) keeps an
    # overshoot from reading as a completed change.  A GT with no lateral
    # progress at all (parked off-center) has nothing to complete.
    gt_progress = gt_shift - initial_shift
    predicted_progress = predicted_shift - initial_shift
    completion = 1 - (predicted_progress - gt_progress).abs() / gt_progress.abs().clamp(
        min=_SEGMENT_MIN_LENGTH
    )
    completion = torch.where(
        detected & (gt_progress.abs() > _SEGMENT_MIN_LENGTH), completion.clamp(0, 1), 0.0
    )

    to_score = gt.dtype
    return MetricEvaluation(
        scores={"success_rate_percent": (crossed & reached).to(to_score) * 100.0},
        details={
            "lane_change": {
                "gt_lane_change_detected": detected.to(to_score),
                "completion_ratio": completion,
                "final_lateral_offset_error_m": (predicted_shift - gt_shift).abs(),
                "lane_change_time_s": change_time,
                "predicted_lateral_shift_m": predicted_shift,
                "gt_lateral_shift_m": gt_shift,
                "initial_lateral_offset_m": initial_shift,
                "gt_direction": direction,
                "left_source_lane": crossed.to(to_score),
                "reached_gt_lane": reached.to(to_score),
                "lane_half_width_left_m": components["lane_half_width_left_m"],
                "lane_half_width_right_m": components["lane_half_width_right_m"],
                "source_lane_index": components["source_lane_index"],
                "source_lane_heading_aligned": components["source_lane_heading_aligned"],
            }
        },
    )


__all__ = [
    "compute_lane_change_components_batch",
    "evaluate_lane_change_with_details",
]
