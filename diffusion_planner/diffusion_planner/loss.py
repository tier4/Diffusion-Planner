from argparse import Namespace
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn


def diffusion_loss_func(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    marginal_prob: Callable[[torch.Tensor], torch.Tensor],
    futures: Tuple[torch.Tensor, torch.Tensor],
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
    longtitudinal_velocity = inputs["ego_current_state"][:, 4:5]
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

    if model_type == "score":
        dpm_loss = torch.sum((model_output * std + z) ** 2, dim=-1)
    elif model_type == "x_start":
        # dpm_loss = torch.sum((model_output - all_gt[:, :, 1:, :]) ** 2, dim=-1)
        loss_dict = loss_func(model_output, all_gt[:, :, 1:, :])
        heading_l2_loss = loss_dict["heading_l2_loss"]
        position_lat_loss = loss_dict["position_lat_loss"]
        position_lon_loss = loss_dict["position_lon_loss"]
        velocity_weight = longtitudinal_velocity * args.coeff_velocity
        velocity_weight = torch.clamp_min(velocity_weight, 1.0)
        # apply velocity weight only to longitudinal position loss
        velocity_weight = velocity_weight.unsqueeze(-1)  # [B, 1, 1]
        position_lon_loss = position_lon_loss * velocity_weight
        dpm_loss = (
            args.coeff_position_lat_loss * position_lat_loss
            + args.coeff_position_lon_loss * position_lon_loss
            + args.coeff_heading_l2_loss * heading_l2_loss
        )
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

    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    turn_indicator_logit = decoder_output["turn_indicator_logit"]  # [B, 4]
    turn_indicator_gt = inputs["turn_indicator"]
    turn_indicator_loss = nn.functional.cross_entropy(
        turn_indicator_logit, turn_indicator_gt, reduction="mean"
    )
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
