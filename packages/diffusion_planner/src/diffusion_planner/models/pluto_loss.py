"""PLUTO-style training loss for :class:`PlutoPlanner`.

Follows ``pluto_trainer.get_planning_loss`` / ``get_prediction_loss``:
smooth-L1 regression on the single candidate whose progress bin matches the
ground truth, cross-entropy over all candidate logits, smooth-L1 on neighbor
futures, and a yaw-norm regularizer. The existing turn-indicator loss is reused
unchanged.

The progress target is measured along the preferred route (the chained
``route_lanes`` centerline) at the 8 s point, in meters.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from diffusion_planner.data.dimensions import TRAJECTORY_DIM

from .loss import compute_turn_indicator_loss
from .pluto_decoder import PADDED_LOGIT, select_candidate
from .pluto_planner import PlutoPlanner


def chain_route_centerline(
    route_lanes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chain ``(B, S, P, C)`` route segments into one ``(B, S * P, 2)`` polyline.

    Returns the points and a ``(B, S * P)`` validity mask (all points of an
    all-zero segment are invalid).
    """
    batch, segments, points, _ = route_lanes.shape
    valid_segment = route_lanes.abs().sum(dim=(-2, -1)) > 0
    xy = route_lanes[..., :2].reshape(batch, segments * points, 2)
    valid = valid_segment.unsqueeze(-1).expand(-1, -1, points)
    return xy, valid.reshape(batch, segments * points)


def project_progress(
    points: torch.Tensor,
    valid: torch.Tensor,
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Arc length along a masked polyline at the foot point of ``query``.

    Args:
        points: Polyline points ``(B, N, 2)``.
        valid: Point validity ``(B, N)``.
        query: Query points ``(B, 2)``.

    Returns:
        Progress ``(B,)`` in the polyline's units and ``(B,)`` flags that are
        False when the polyline has no valid segment.
    """
    start = points[:, :-1]
    end = points[:, 1:]
    segment_valid = valid[:, :-1] & valid[:, 1:]
    direction = end - start
    length_sq = (direction * direction).sum(dim=-1)
    length = length_sq.sqrt() * segment_valid
    offset = query.unsqueeze(1) - start
    t = ((offset * direction).sum(dim=-1) / length_sq.clamp_min(1e-9)).clamp(0.0, 1.0)
    foot = start + t.unsqueeze(-1) * direction
    distance_sq = ((query.unsqueeze(1) - foot) ** 2).sum(dim=-1)
    distance_sq = distance_sq.masked_fill(~segment_valid, float("inf"))
    nearest = distance_sq.argmin(dim=1)
    cumulative = torch.cumsum(length, dim=1) - length
    rows = torch.arange(points.shape[0], device=points.device)
    progress = cumulative[rows, nearest] + t[rows, nearest] * length[rows, nearest]
    return progress, segment_valid.any(dim=1)


def ground_truth_path_length(ego_future_xy: torch.Tensor) -> torch.Tensor:
    """Path length ``(B,)`` from the current pose (origin) through the future xy."""
    steps = torch.cat(
        (ego_future_xy[:, :1], ego_future_xy[:, 1:] - ego_future_xy[:, :-1]), dim=1
    )
    return torch.linalg.vector_norm(steps, dim=-1).sum(dim=1)


def assign_mode_targets(
    ego_future_xy: torch.Tensor,
    route_lanes: torch.Tensor,
    *,
    position_scale: float,
    num_modes: int,
    mode_interval_m: float,
    bin_edges_m: Sequence[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the target progress bin ``(B,)`` and the progress in meters ``(B,)``.

    The ground-truth position at the last step is projected onto the chained
    route centerline (PLUTO's ``future_projection`` at 8 s). Frames without a
    valid route fall back to the ground-truth path length.
    """
    xy = ego_future_xy.float()
    points, valid = chain_route_centerline(route_lanes.float())
    progress, has_route = project_progress(points, valid, xy[:, -1])
    progress_m = torch.where(has_route, progress, ground_truth_path_length(xy))
    progress_m = progress_m * position_scale
    if bin_edges_m is not None:
        edges = torch.as_tensor(
            list(bin_edges_m), dtype=progress_m.dtype, device=progress_m.device
        )
        if edges.numel() != num_modes - 1:
            raise ValueError(
                f"bin_edges_m must have num_modes - 1 = {num_modes - 1} entries, "
                f"got {edges.numel()}"
            )
        target = torch.bucketize(progress_m, edges)
    else:
        target = torch.floor(progress_m / mode_interval_m).long()
    return target.clamp(0, num_modes - 1), progress_m


def classification_targets(
    target_line: torch.Tensor,
    target_mode: torch.Tensor,
    *,
    num_lines: int,
    num_modes: int,
    label_smoothing: float,
) -> torch.Tensor:
    """One-hot ``(B, R * M)`` targets, optionally smoothed onto the neighbor bins."""
    batch = target_mode.shape[0]
    rows = torch.arange(batch, device=target_mode.device)
    soft = target_mode.new_zeros((batch, num_lines, num_modes), dtype=torch.float32)
    soft[rows, target_line, target_mode] = 1.0 - label_smoothing
    if label_smoothing > 0.0:
        lower = (target_mode - 1).clamp_min(0)
        upper = (target_mode + 1).clamp_max(num_modes - 1)
        soft[rows, target_line, lower] += 0.5 * label_smoothing
        soft[rows, target_line, upper] += 0.5 * label_smoothing
    return soft.reshape(batch, num_lines * num_modes)


def masked_smooth_l1_meters(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    position_scale: float,
) -> torch.Tensor:
    """Smooth-L1 (``beta`` = 1 m) on xy in meters and raw ``[cos, sin]``.

    Summed over the four channels, averaged over valid ``(..., T)`` entries.
    """
    scale = prediction.new_tensor([position_scale, position_scale, 1.0, 1.0])
    loss = F.smooth_l1_loss(
        prediction.float() * scale,
        target.float() * scale,
        reduction="none",
        beta=1.0,
    ).sum(dim=-1)
    weight = valid.to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def compute_pluto_loss(
    model: PlutoPlanner,
    input_data: dict[str, torch.Tensor],
    *,
    position_scale: float = 50.0,
    num_modes: int = 12,
    mode_interval_m: float = 10.0,
    bin_edges_m: Sequence[float] | None = None,
    label_smoothing: float = 0.0,
    regression_weight: float = 1.0,
    classification_weight: float = 1.0,
    neighbor_weight: float = 1.0,
    yaw_regularization_weight: float = 1.0,
    turn_indicator_loss_weight: float = 1.0,
    turn_indicator_transition_loss_weight: float = 5.0,
) -> dict[str, torch.Tensor]:
    """Compute the PLUTO planning loss and monitoring values for one batch.

    Returns the keys the training loop expects from the flow-matching loss
    (``total``, ``trajectory``, ``turn_indicator``, ``turn_indicator_correct``,
    ``turn_indicator_valid_count``) plus the PLUTO terms and metrics in meters.
    """
    outputs = model(input_data)
    candidates = outputs["candidates"]
    logits = outputs["logits"]
    line_padding = outputs["line_padding"]
    batch, num_lines, decoder_modes, _, _ = candidates.shape
    if num_lines != 1:
        raise NotImplementedError(
            "lateral target assignment for more than one reference line is not "
            "implemented (Option A)"
        )
    if decoder_modes != num_modes:
        raise ValueError(
            f"loss num_modes={num_modes} does not match the decoder ({decoder_modes})"
        )

    ego_target = input_data["ego_agent_future"][..., :TRAJECTORY_DIM]
    ego_valid = ego_target[..., 2:4].abs().sum(dim=-1) > 0
    with torch.no_grad():
        target_mode, progress_m = assign_mode_targets(
            ego_target[..., :2],
            input_data["route_lanes"],
            position_scale=position_scale,
            num_modes=num_modes,
            mode_interval_m=mode_interval_m,
            bin_edges_m=bin_edges_m,
        )
    target_line = torch.zeros_like(target_mode)
    rows = torch.arange(batch, device=candidates.device)
    assigned = candidates[rows, target_line, target_mode]  # (B, T, 4)

    regression = masked_smooth_l1_meters(
        assigned, ego_target, ego_valid, position_scale
    )
    yaw_norm = torch.linalg.vector_norm(assigned[..., 2:4].float(), dim=-1)
    yaw_regularization = F.l1_loss(yaw_norm, torch.ones_like(yaw_norm))

    masked_logits = logits.masked_fill(line_padding.unsqueeze(-1), PADDED_LOGIT)
    masked_logits = masked_logits.reshape(batch, num_lines * num_modes).float()
    soft_targets = classification_targets(
        target_line,
        target_mode,
        num_lines=num_lines,
        num_modes=num_modes,
        label_smoothing=label_smoothing,
    )
    classification = (
        -(soft_targets * F.log_softmax(masked_logits, dim=-1)).sum(-1).mean()
    )

    neighbor_target = input_data["neighbor_agents_future"]
    neighbor_valid = neighbor_target[..., 2:4].abs().sum(dim=-1) > 0
    neighbor = masked_smooth_l1_meters(
        outputs["neighbors"], neighbor_target, neighbor_valid, position_scale
    )

    turn_indicator, correct, valid_count = compute_turn_indicator_loss(
        outputs["turn_indicator_logits"],
        input_data,
        transition_weight=turn_indicator_transition_loss_weight,
    )

    trajectory = (
        regression_weight * regression
        + classification_weight * classification
        + neighbor_weight * neighbor
        + yaw_regularization_weight * yaw_regularization
    )
    total = trajectory + turn_indicator_loss_weight * turn_indicator

    with torch.no_grad():
        _, selected_index = select_candidate(candidates, logits, line_padding)
        target_index = target_line * num_modes + target_mode
        selection_accuracy = (selected_index == target_index).float().mean()
        xy_error = (
            candidates[..., :2].float() - ego_target[:, None, None, :, :2].float()
        ) * position_scale
        distance = torch.linalg.vector_norm(xy_error, dim=-1)  # (B, R, M, T)
        ade = distance.mean(dim=-1).reshape(batch, num_lines * num_modes)
        fde = distance[..., -1].reshape(batch, num_lines * num_modes)
        selected_ade = ade[rows, selected_index].mean()
        selected_fde = fde[rows, selected_index].mean()
        oracle_ade = ade.min(dim=-1).values.mean()
        oracle_fde = fde.min(dim=-1).values.mean()
        mode_target_counts = torch.bincount(target_mode, minlength=num_modes).float()

    return {
        "total": total,
        "trajectory": trajectory,
        "regression": regression,
        "classification": classification,
        "neighbor": neighbor,
        "yaw_regularization": yaw_regularization,
        "turn_indicator": turn_indicator,
        "turn_indicator_correct": correct,
        "turn_indicator_valid_count": valid_count,
        "selection_accuracy": selection_accuracy,
        "selected_ade_m": selected_ade,
        "selected_fde_m": selected_fde,
        "oracle_ade_m": oracle_ade,
        "oracle_fde_m": oracle_fde,
        "mean_progress_m": progress_m.mean(),
        "mode_target_counts": mode_target_counts,
    }
