from argparse import Namespace
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_planner.dimensions import TURN_INDICATOR_OUTPUT_KEEP
from diffusion_planner.model.guidance.collision import (
    batch_signed_distance_rect,
    center_rect_to_points,
)


def make_turn_indicator_gt(
    turn_indicators: torch.Tensor,  # # [B, INPUT_T + 1]
) -> torch.Tensor:
    turn_indicators_gt = turn_indicators.long()  # [B, INPUT_T + 1]
    turn_indicators_gt_keep = turn_indicators_gt[:, -1] == turn_indicators_gt[:, -2]  # [B,]
    turn_indicators_gt = turn_indicators_gt[:, -1] * ~turn_indicators_gt_keep  # change to 0 if keep
    turn_indicators_gt = turn_indicators_gt + turn_indicators_gt_keep * TURN_INDICATOR_OUTPUT_KEEP
    return turn_indicators_gt


def diffusion_loss_func(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    marginal_prob: Callable[[torch.Tensor], torch.Tensor],
    futures: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args: Namespace,
    eps: float = 1e-3,
):
    norm = args.state_normalizer
    model_type = args.diffusion_model_type

    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask  # [B, P, V]

    B, Pn, T, _ = neighbors_future.shape
    ego_current, neighbors_current = (
        inputs["ego_current_state"][:, :4],
        inputs["neighbor_agents_past"][:, :Pn, -1, :4],
    )
    longitudinal_velocity = inputs["ego_current_state"][:, 4:5]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_mask = torch.concat(
        (neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1
    )

    gt_future = torch.cat(
        [ego_future[:, None, :, :], neighbors_future[..., :]], dim=1
    )  # [B, P = 1 + 1 + neighbor, T, 4]
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)  # [B, P, 4]

    P = gt_future.shape[1]
    t = torch.rand(B, device=gt_future.device) * (1 - eps) + eps  # [B,]
    z = torch.randn_like(gt_future, device=gt_future.device)  # [B, P, T, 4]

    all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)
    all_gt[:, 1:][neighbor_mask] = 0.0

    mean, std = marginal_prob(all_gt[..., 1:, :], t)
    std = std.view(-1, *([1] * (len(all_gt[..., 1:, :].shape) - 1)))

    if model_type == "flow_matching":
        # t=0 is noise, t=1 is data
        t = t.reshape(-1, *([1] * (len(all_gt.shape) - 1)))  # [B, 1, 1, 1]
        xT = (1 - t) * z + t * all_gt[:, :, 1:, :]  # [B, P, T, 4]
        t = t.reshape(-1)  # [B,]
    else:
        xT = mean + std * z

    xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)

    merged_inputs = {
        **inputs,
        "gt_trajectories": all_gt,
        "sampled_trajectories": xT,
        "diffusion_time": t,
    }

    _, decoder_output = model(merged_inputs)  # [B, P, 1 + T, 4]
    model_output = decoder_output["model_output"][:, :, 1:, :]  # [B, P, T, 4]
    safety_loss_terms: Dict[str, torch.Tensor] = {}

    if model_type == "score":
        dpm_loss = torch.sum((model_output * std + z) ** 2, dim=-1)
    elif model_type == "x_start":
        # dpm_loss = torch.sum((model_output - all_gt[:, :, 1:, :]) ** 2, dim=-1)
        loss_dict = loss_func(model_output, all_gt[:, :, 1:, :])
        heading_l2_loss = loss_dict["heading_l2_loss"]
        position_lat_loss = loss_dict["position_lat_loss"]
        position_lon_loss = loss_dict["position_lon_loss"]

        # velocity weight
        velocity_weight = longitudinal_velocity * args.coeff_velocity
        velocity_weight = torch.abs(velocity_weight)
        velocity_weight = torch.clamp_min(velocity_weight, 1.0)
        velocity_weight = velocity_weight.unsqueeze(-1)  # [B, 1, 1]
        position_lon_loss = position_lon_loss / velocity_weight

        # timestep weight
        timestep_weight = args.coeff_timestep
        assert T % len(timestep_weight) == 0, (
            f"Timestep {T} is not divisible by the number of timestep weights {len(timestep_weight)}"
        )
        unit = T // len(timestep_weight)
        for i in range(len(timestep_weight)):
            position_lat_loss[:, :, (i + 0) * unit : (i + 1) * unit] *= timestep_weight[i]
            position_lon_loss[:, :, (i + 0) * unit : (i + 1) * unit] *= timestep_weight[i]
            heading_l2_loss[:, :, (i + 0) * unit : (i + 1) * unit] *= timestep_weight[i]

        # mask neighbors' heading loss
        # heading_l2_loss[:, 1:] *= 0.0

        dpm_loss = (
            args.coeff_position_lat_loss * position_lat_loss
            + args.coeff_position_lon_loss * position_lon_loss
            + args.coeff_heading_l2_loss * heading_l2_loss
        )
        # safety_penalty, safety_logs, _ = compute_safety_penalty(
        #     model_output,
        #     inputs,
        #     neighbors_future,
        #     neighbors_future_valid,
        #     args,
        #     return_components=True,
        # )
        # if safety_penalty is not None:
        #     dpm_loss[:, 0, :] = dpm_loss[:, 0, :] + safety_penalty
        #     safety_loss_terms.update(safety_logs)
    elif model_type == "flow_matching":
        target_v = all_gt[:, :, 1:, :] - z
        dpm_loss = torch.sum((model_output - target_v) ** 2, dim=-1)

    masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]

    loss = {}

    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

    loss["ego_planning_loss"] = dpm_loss[:, 0, :].mean()
    if safety_loss_terms:
        loss.update(safety_loss_terms)

    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    turn_indicator_logit = decoder_output["turn_indicator_logit"]  # [B, TURN_INDICATOR_OUTPUT_KEEP]
    turn_indicator_gt = make_turn_indicator_gt(inputs["turn_indicators"])  # [B,]
    turn_indicator_loss = nn.functional.cross_entropy(
        turn_indicator_logit, turn_indicator_gt, reduction="none"
    )
    turn_indicator_change = inputs["turn_indicators"][:, -2] != inputs["turn_indicators"][:, -1]
    turn_indicator_coeff = torch.where(turn_indicator_change, 1.0, 0.05)
    turn_indicator_loss = (turn_indicator_loss * turn_indicator_coeff).mean()
    loss["turn_indicator_loss"] = turn_indicator_loss

    with torch.no_grad():
        turn_indicator_accuracy = (
            (turn_indicator_logit.argmax(dim=-1) == turn_indicator_gt).float().mean()
        )
        loss["turn_indicator_accuracy"] = turn_indicator_accuracy

    return loss


def loss_func(
    trajectory_pred: torch.Tensor, trajectory_gt: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Calculate the loss between predicted and ground truth trajectories.

    Args:
        trajectory_pred (torch.Tensor): Predicted trajectory of shape [..., T, D].
        trajectory_gt (torch.Tensor): Ground truth trajectory of shape [..., T, D].
        where, D=4 (x, y, cos, sin).

    Returns:
        dict[str, torch.Tensor]: A dictionary containing the loss values.
        where, each loss' shape is [..., T].
    """
    result_dict = {}

    ###################
    # Basic L2 Losses #
    ###################
    # simple L2 loss
    result_dict["simple_l2_loss"] = torch.mean((trajectory_pred - trajectory_gt) ** 2, dim=-1)

    # Position loss (x, y coordinates)
    position_pred = trajectory_pred[..., :2]  # [..., T, 2]
    position_gt = trajectory_gt[..., :2]  # [..., T, 2]

    # Calculate L2 distance for each time step
    position_diff = position_pred - position_gt  # [..., T, 2]
    position_error = torch.sum(position_diff**2, dim=-1)  # [..., T]
    result_dict["position_l2_loss"] = position_error

    # Heading loss (cos, sin components)
    cos_sin_pred = trajectory_pred[..., 2:]  # [..., T, 2]
    cos_sin_gt = trajectory_gt[..., 2:]  # [..., T, 2]

    # heading l2 loss
    heading_loss = torch.sum((cos_sin_pred - cos_sin_gt) ** 2, dim=-1)  # [..., T]
    result_dict["heading_l2_loss"] = heading_loss

    ######################
    # Specialized Losses #
    ######################
    # Lateral or longitudinal error (along vehicle direction)
    cos_gt = cos_sin_gt[..., 0]  # [..., T]
    sin_gt = cos_sin_gt[..., 1]  # [..., T]
    lon_diff = +position_diff[..., 0] * cos_gt + position_diff[..., 1] * sin_gt  # [..., T]
    lat_diff = -position_diff[..., 0] * sin_gt + position_diff[..., 1] * cos_gt  # [..., T]
    lat_error = torch.abs(lat_diff)  # [..., T]
    lon_error = torch.abs(lon_diff)  # [..., T]
    result_dict["position_lat_loss"] = lat_error
    result_dict["position_lon_loss"] = lon_error

    # Cosine similarity loss
    cosine_similarity = torch.sum(cos_sin_pred * cos_sin_gt, dim=-1)  # [..., T]
    result_dict["cosine_similarity_loss"] = 1.0 - cosine_similarity  # [..., T]

    return result_dict


def _gather_feature(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    expanded_index = index.unsqueeze(-1).expand(-1, -1, values.size(-1))
    return torch.gather(values, 1, expanded_index)


def _lane_corner_clearance(
    points_xy: torch.Tensor,
    center: torch.Tensor,
    n_left: torch.Tensor,
    width_left: torch.Tensor,
    width_right: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate lateral distances from points to the nearest lane boundary.

    Args:
        points_xy: [B, M, 2] points to evaluate.
        center: [B, N, 2] lane center samples.
        n_left: [B, N, 2] left-direction normals.
        width_left: [B, N] distance from center to left boundary along n_left.
        width_right: [B, N] distance from center to right boundary along n_left.
        valid_mask: [B, N] mask of valid lane samples.

    Returns:
        dist_left: [B, M] signed distance to left boundary.
        dist_right: [B, M] signed distance to right boundary.
        valid_point_mask: [B, M] boolean mask indicating valid proximity.
    """
    diff = points_xy.unsqueeze(2) - center.unsqueeze(1)  # [B, M, N, 2]
    dist2 = (diff**2).sum(-1)
    inf = torch.full_like(dist2, float("inf"))
    dist2 = torch.where(valid_mask[:, None, :], dist2, inf)

    min_dist, nearest_idx = torch.min(dist2, dim=-1)
    valid_point_mask = torch.isfinite(min_dist)

    nearest_idx = nearest_idx.clamp_min(0)
    selected_center = _gather_feature(center, nearest_idx)
    selected_n_left = _gather_feature(n_left, nearest_idx)
    selected_width_left = _gather_feature(width_left.unsqueeze(-1), nearest_idx).squeeze(-1)
    selected_width_right = _gather_feature(width_right.unsqueeze(-1), nearest_idx).squeeze(-1)

    lat = ((points_xy - selected_center) * selected_n_left).sum(-1)
    dist_left = selected_width_left - lat
    dist_right = lat - selected_width_right

    return dist_left, dist_right, valid_point_mask


def lane_boundary_penalty(
    ego_bbox_corners: torch.Tensor,
    route_lanes: torch.Tensor,
) -> torch.Tensor | None:
    """Compute lane boundary clearance penalty for ego BBox.

    Args:
        ego_bbox_corners: [B, T, 4, 2] ego rectangle corners per timestep.
        route_lanes: [B, num_segments, num_points, dim] lane tensor.

    Returns:
        penalty: [B, T] penalty per sample & timestep, or None if lane tensor missing.
    """
    if route_lanes is None:
        return None

    B, T, _, _ = ego_bbox_corners.shape
    device = ego_bbox_corners.device
    dtype = ego_bbox_corners.dtype

    lane_tensor = route_lanes
    center = lane_tensor[..., :2].reshape(B, -1, 2)
    direction = lane_tensor[..., 2:4].reshape(B, -1, 2)
    left_offset = lane_tensor[..., 4:6].reshape(B, -1, 2)
    right_offset = lane_tensor[..., 6:8].reshape(B, -1, 2)

    direction_norm = torch.linalg.norm(direction, dim=-1, keepdim=True)
    valid_mask = direction_norm.squeeze(-1) > 1e-6
    direction_norm = direction_norm.clamp_min(1e-6)
    n_left = torch.stack([-direction[..., 1], direction[..., 0]], dim=-1) / direction_norm

    width_left = (left_offset * n_left).sum(-1)
    width_right = (right_offset * n_left).sum(-1)

    corners_flat = ego_bbox_corners.reshape(B, -1, 2)  # [B, T*4, 2]

    dist_left, dist_right, valid_corner_mask = _lane_corner_clearance(
        corners_flat,
        center,
        n_left,
        width_left,
        width_right,
        valid_mask,
    )

    dist_left = dist_left.view(B, T, 4)
    dist_right = dist_right.view(B, T, 4)
    valid_corner_mask = valid_corner_mask.view(B, T, 4)

    distance_to_boundary = torch.minimum(dist_left, dist_right)
    inf_tensor = torch.full_like(distance_to_boundary, float("inf"))
    distance_to_boundary = torch.where(valid_corner_mask, distance_to_boundary, inf_tensor)

    min_corner_distance = distance_to_boundary.min(dim=-1).values
    valid_time_mask = valid_corner_mask.any(dim=-1)

    lane_margin = 0.3
    lane_weight = 1.0
    lane_violation = torch.where(
        valid_time_mask,
        F.relu(lane_margin - min_corner_distance),
        torch.zeros_like(min_corner_distance),
    )
    lane_penalty = lane_weight * lane_violation

    return lane_penalty


def neighbor_clearance_penalty(
    ego_bbox_corners: torch.Tensor,
    neighbors_future: torch.Tensor,
    neighbors_future_valid: torch.Tensor,
    denorm_inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute neighbor clearance penalty using oriented rectangles."""

    B, T, _, _ = ego_bbox_corners.shape
    P = neighbors_future.shape[1]

    neighbor_sizes = denorm_inputs["neighbor_agents_past"][:, :P, -1, :]
    neighbor_width = torch.clamp(neighbor_sizes[..., 6], min=1e-3)
    neighbor_length = torch.clamp(neighbor_sizes[..., 7], min=1e-3)

    neighbor_pos = neighbors_future[..., :2]
    neighbor_cos = neighbors_future[..., 2]
    neighbor_sin = neighbors_future[..., 3]
    orientation_norm = torch.sqrt(neighbor_cos**2 + neighbor_sin**2).clamp_min(1e-6)
    neighbor_cos = neighbor_cos / orientation_norm
    neighbor_sin = neighbor_sin / orientation_norm

    neighbor_length = neighbor_length[..., None].expand(-1, -1, T)
    neighbor_width = neighbor_width[..., None].expand(-1, -1, T)

    neighbor_rect = torch.stack(
        [
            neighbor_pos[..., 0],
            neighbor_pos[..., 1],
            neighbor_cos,
            neighbor_sin,
            neighbor_length,
            neighbor_width,
        ],
        dim=-1,
    )  # [B, P, T, 6]

    ego_flat = ego_bbox_corners.reshape(B * T, 4, 2)
    neighbor_flat = neighbor_rect.reshape(B * T, P, 6)
    valid_mask = neighbors_future_valid.reshape(B * T, P)

    ego_pts = ego_flat.unsqueeze(1).expand(-1, P, 4, 2).reshape(-1, 4, 2)
    neighbor_pts = center_rect_to_points(neighbor_flat.reshape(-1, 6))
    distances = batch_signed_distance_rect(ego_pts, neighbor_pts)
    distances = distances.reshape(B * T, P)

    inf_tensor = torch.full_like(distances, float("inf"))
    distances = torch.where(valid_mask, distances, inf_tensor)

    min_distance, _ = distances.min(dim=1)
    min_distance = min_distance.reshape(B, T)

    neighbor_margin = 0.5
    penalty = torch.zeros_like(min_distance)
    finite_mask = torch.isfinite(min_distance)
    penalty[finite_mask] = F.relu(neighbor_margin - min_distance[finite_mask])

    return penalty


def compute_safety_penalty(
    trajectory_pred: torch.Tensor,
    inputs: Dict[str, torch.Tensor],
    neighbors_future: torch.Tensor,
    neighbors_future_valid: torch.Tensor,
    args: Namespace,
    return_components: bool = False,
) -> tuple:
    state_normalizer = args.state_normalizer
    observation_normalizer = args.observation_normalizer
    denorm_inputs = observation_normalizer.inverse(inputs)

    traj_world = state_normalizer.inverse(trajectory_pred)
    ego_traj = traj_world[:, 0]  # [B, T, 4]
    B, T, _ = ego_traj.shape
    device = ego_traj.device
    dtype = ego_traj.dtype

    heading = ego_traj[..., 2:]
    heading_unit = heading / torch.linalg.norm(heading, dim=-1, keepdim=True).clamp_min(1e-6)
    ego_xy = ego_traj[..., :2]

    ego_shape = denorm_inputs["ego_shape"]
    wheel_base = ego_shape[:, 0]
    ego_length = ego_shape[:, 1]
    ego_width = ego_shape[:, 2]

    cog_to_rear = 0.5 * wheel_base[:, None, None]

    ego_center_xy = ego_xy + heading_unit * cog_to_rear

    half_length = (ego_length / 2.0).unsqueeze(-1).expand(-1, T)
    half_width = (ego_width / 2.0).unsqueeze(-1).expand(-1, T)
    half_sizes = torch.stack([half_length, half_width], dim=-1)  # [B, T, 2]
    corner_signs = torch.tensor(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    local_corners = corner_signs[None, None, :, :] * half_sizes[:, :, None, :]
    rot = torch.stack(
        [
            heading_unit[..., 0],
            -heading_unit[..., 1],
            heading_unit[..., 1],
            heading_unit[..., 0],
        ],
        dim=-1,
    ).reshape(B, T, 2, 2)
    rotated_corners = torch.einsum("btij,btkj->btki", rot, local_corners)
    ego_bbox_corners = ego_center_xy[:, :, None, :] + rotated_corners

    lane_penalty = lane_boundary_penalty(ego_bbox_corners, denorm_inputs["route_lanes"])
    neighbor_penalty = neighbor_clearance_penalty(
        ego_bbox_corners,
        neighbors_future,
        neighbors_future_valid,
        denorm_inputs,
    )

    total_penalty = lane_penalty + neighbor_penalty
    logs: Dict[str, torch.Tensor] = {}
    logs["ego_safety_margin_loss"] = total_penalty.mean()
    logs["lane_boundary_margin_loss"] = lane_penalty.mean()
    logs["neighbor_margin_loss"] = neighbor_penalty.mean()

    if return_components:
        return total_penalty, logs, (lane_penalty, neighbor_penalty)
    return total_penalty, logs
