"""Lane-change completion: does the planner end up in the lane the GT moved to?

Every lateral quantity is measured against the *source lane* -- the lanelet the
ego starts in, extended through its successors (see ``source_lane``). That
deliberately decouples the score from longitudinal error: a planner that is
merely too slow still gets credit for the lane change it performed, and
``arrival``/``simple_turn`` already cover longitudinal and path-shape accuracy.

Lane *identity* is never compared, since the NPZ carries no lanelet ids. "In
the GT lane" is "within half a lane width of the GT's final lateral position",
and "left the original lane" is "crossed the source lane's boundary", with the
width read from the tensor's boundary-offset columns rather than hard-coded.
"""

from __future__ import annotations

import torch

from planner_metrics.evaluation import MetricEvaluation
from planner_metrics.geometry import _point_to_segments_signed_lateral
from planner_metrics.horizon import resolve_horizon_steps
from planner_metrics.lane_tensor import resolve_lane_tensor
from planner_metrics.source_lane import reconstruct_source_lane

_PREDICTION_TIMESTEP_SECONDS = 0.1
_SEGMENT_MIN_LENGTH = 1e-6
# Chain far enough past the trajectories that their endpoints still project
# onto a real segment instead of the path's extrapolated tail.
_SOURCE_PATH_MARGIN_M = 20.0

_DEFAULT_HORIZON_SECONDS = 8.0
_DEFAULT_MINIMUM_LATERAL_SHIFT_M = 1.0
_DEFAULT_CHAIN_TOLERANCE_M = 1.0


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
        required_length = (
            float(torch.cat([predicted_xy, gt_xy], dim=0).norm(dim=-1).max())
            + _SOURCE_PATH_MARGIN_M
        )
        source = reconstruct_source_lane(
            scene_lanes, torch.zeros(2), required_length, chain_tolerance_m
        )
        seg_p1, seg_p2 = source.segments(ego_trajs.device)
        # The half width is the tolerance every threshold is built from, so it
        # is read where the scorer decides: at the GT's final position, the same
        # station its lateral offset is measured against.
        left, right = source.half_widths_at(scene_lanes, gt_xy[-1])

        offsets["predicted_lateral_offset_m"].append(
            _point_to_segments_signed_lateral(predicted_xy, seg_p1, seg_p2)
        )
        offsets["gt_lateral_offset_m"].append(
            _point_to_segments_signed_lateral(gt_xy, seg_p1, seg_p2)
        )
        offsets["lane_half_width_left_m"].append(left)
        offsets["lane_half_width_right_m"].append(right)
        offsets["source_lane_index"].append(float(source.index))
        offsets["source_lane_heading_aligned"].append(float(source.heading_aligned))

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
    """Score whether the prediction completes the GT's lane change, as a success rate.

    A sample succeeds when the prediction crossed the source lane's boundary in
    the GT's direction AND landed within the lane tolerance of the GT's final
    lateral position -- rejecting respectively "never changed / wrong way /
    unfinished" and "overshot past the target lane".

    Two things make a sample unscorable rather than failed on merit: the
    reference lane fell back to an oncoming or crossing lanelet
    (``source_lane_heading_aligned``), or the recorded ego performed no lane
    change (``gt_lane_change_detected``). Both count as failures, and since
    neither is aggregated, those two details are what distinguish an unscorable
    list from a weak planner.
    """
    horizon_seconds = float(parameters.get("horizon_seconds", _DEFAULT_HORIZON_SECONDS))
    minimum_lateral_shift_m = float(
        parameters.get("minimum_lateral_shift_m", _DEFAULT_MINIMUM_LATERAL_SHIFT_M)
    )
    chain_tolerance_m = float(parameters.get("chain_tolerance_m", _DEFAULT_CHAIN_TOLERANCE_M))
    if minimum_lateral_shift_m <= 0:
        raise ValueError("lane_change minimum_lateral_shift_m must be positive")

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

    # +1 when the GT moved left of the source lane, -1 when right. The floor
    # keeps a map with zeroed boundary offsets from making `crossed` trivially
    # true and `reached` impossible.
    direction = (gt_shift >= initial_shift).to(gt.dtype) * 2 - 1
    half_width = torch.where(
        direction > 0,
        components["lane_half_width_left_m"],
        components["lane_half_width_right_m"],
    )
    lane_tolerance = half_width.clamp(min=minimum_lateral_shift_m)
    # A substituted floor is indistinguishable in the output from a genuinely
    # narrow lane, so record which one it was.
    tolerance_from_map = half_width >= minimum_lateral_shift_m

    # Everything below is gated on both preconditions. An oncoming reference
    # lane reverses the lateral readings, so a mirrored one would score as a
    # success. And `detected` needs BOTH halves: ending up outside the source
    # lane is also true of an ego that STARTS outside it and drives dead
    # straight, so the GT must have moved sideways too.
    aligned = components["source_lane_heading_aligned"] > 0.5
    gt_progress = gt_shift - initial_shift
    moved = direction * gt_progress > minimum_lateral_shift_m
    detected = moved & (direction * gt_shift > lane_tolerance)
    scorable = aligned & detected
    signed_progress = direction[:, None] * predicted  # (N, T), positive towards the GT's side
    crossed = scorable & (signed_progress[:, -1] > lane_tolerance)
    reached = scorable & ((predicted_shift - gt_shift).abs() <= lane_tolerance)

    # Prediction index i is the pose at t=(i+1)*dt; a sample that never leaves
    # the source lane reports the full horizon.
    beyond = signed_progress > lane_tolerance[:, None]
    first_beyond = (
        beyond.to(torch.int8).argmax(dim=1).to(gt.dtype) + 1
    ) * _PREDICTION_TIMESTEP_SECONDS
    change_time = torch.where(
        scorable & beyond.any(dim=1), first_beyond, steps * _PREDICTION_TIMESTEP_SECONDS
    )

    # Closeness to the GT's lateral target, measured from where the ego
    # started: an overshoot then reads as far from complete as no motion does.
    predicted_progress = predicted_shift - initial_shift
    completion = 1 - (predicted_progress - gt_progress).abs() / gt_progress.abs().clamp(
        min=minimum_lateral_shift_m
    )
    completion = torch.where(scorable, completion.clamp(0, 1), 0.0)

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
                "lane_tolerance_m": lane_tolerance,
                "lane_tolerance_from_map": tolerance_from_map.to(to_score),
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
