"""GRPO (Group Relative Policy Optimization) building blocks for the diffusion planner.

This module is the GRPO counterpart of ``train_epoch.py``/``decoder.compute_training_loss``.
The pipeline is:

  1. ``expand_batch``        - replicate every scene in the batch ``N`` times so we can
                               draw a *group* of ``N`` trajectories per scene in a single
                               multi-batch inference pass.
  2. ``sample_group``        - run the model in inference mode with random initial noise to
                               produce ``N`` diverse ego trajectories per scene.
  3. ``compute_collision_reward`` - score each trajectory with a collision-based reward
                               (neighbor collision + optional road-border), reusing the same
                               penalty functions used by the supervised trainer.
  4. ``compute_group_advantages`` - normalise rewards within each group of ``N``.
  5. ``compute_grpo_loss``   - advantage-weighted diffusion (denoising) loss. Treating the
                               diffusion reconstruction loss of a generated trajectory as a
                               proxy for its negative log-likelihood, minimising
                               ``mean(advantage_i * loss_i)`` pushes probability mass toward
                               higher-reward (lower-collision) trajectories.

Only the ``x_start`` diffusion model type (the default) without velocity representation is
supported; the helpers raise a clear error otherwise.
"""

import random

import torch

from diffusion_planner.dimensions import MAX_NUM_AGENTS, OUTPUT_T, POSE_DIM
from diffusion_planner.loss import (
    compute_ego_edge_points,
    compute_neighbor_collision_penalty,
    compute_road_border_penalty,
    loss_func,
)
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.module.decoder import generate_prefix_mask
from diffusion_planner.utils.unicycle_accel_curvature import smoothing_future_trajectory


def expand_batch(inputs: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    """Replicate every scene ``n`` times along the batch dimension.

    Scene ``b`` ends up occupying rows ``[b * n, (b + 1) * n)`` so a group of ``n`` samples
    for the same scene is contiguous (handy for the group-relative advantage reshape).
    """
    return {key: value.repeat_interleave(n, dim=0) for key, value in inputs.items()}


@torch.no_grad()
def sample_group(
    model,
    norm_inputs: dict[str, torch.Tensor],
    noise_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """Generate one ego trajectory per (already replicated) row via inference sampling.

    Args:
        model: the Diffusion_Planner (or DDP-wrapped) model.
        norm_inputs: observation-normalized inputs, batch dimension already expanded.
        noise_scale: multiplier on the standard-normal initial diffusion noise. Larger
            values give more diverse trajectories within a group.
        device: target device.

    Returns:
        ego_world: [B*N, T, 4] ego trajectories in the ego-centric world frame
            (x, y, cos_yaw, sin_yaw).
    """
    was_training = model.training
    model.eval()

    B = norm_inputs["ego_current_state"].shape[0]
    inference_inputs = dict(norm_inputs)
    inference_inputs["sampled_trajectories"] = (
        torch.randn(B, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM, device=device) * noise_scale
    )
    inference_inputs["delay"] = torch.zeros(B, dtype=torch.float32, device=device)

    _, outputs = model(inference_inputs)
    ego_world = outputs["prediction"][:, 0].detach()  # [B*N, T, 4]

    if was_training:
        model.train()
    return ego_world


@torch.no_grad()
def compute_collision_reward(
    ego_world: torch.Tensor,
    norm_inputs: dict[str, torch.Tensor],
    neighbors_future: torch.Tensor,
    neighbors_future_valid: torch.Tensor,
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collision-based reward for a batch of generated ego trajectories.

    Higher reward == fewer/less-severe collisions. The reward is the negative of the
    accumulated hinge penalties used by the supervised trainer, so the units are directly
    comparable to the ``neighbor_collision_loss`` / ``road_border_loss`` metrics.

    Args:
        ego_world: [B*N, T, 4] generated ego trajectories (world frame).
        norm_inputs: observation-normalized inputs (batch already expanded).
        neighbors_future: [B*N, Pn, T, 4] neighbor futures (world frame, cos/sin).
        neighbors_future_valid: [B*N, Pn, T] validity mask.
        args: namespace with reward weights / margins.

    Returns:
        reward: [B*N] scalar reward per trajectory.
        nc_penalty: [B*N, T] neighbor-collision penalty per timestep (for logging).
        rb_penalty: [B*N, T] road-border penalty per timestep (for logging).
    """
    denorm_inputs = args.observation_normalizer.inverse(norm_inputs)

    ego_edge_points = compute_ego_edge_points(
        ego_world, norm_inputs["ego_shape"], n_interp=args.road_border_n_interp
    )

    nc_penalty = compute_neighbor_collision_penalty(
        ego_edge_points,
        neighbors_future,
        neighbors_future_valid,
        denorm_inputs["neighbor_agents_past"],
        margin_vehicle=args.neighbor_collision_margin_vehicle,
        margin_pedestrian=args.neighbor_collision_margin_pedestrian,
        margin_bicycle=args.neighbor_collision_margin_bicycle,
    )  # [B*N, T]

    if args.w_road_border > 0.0:
        rb_penalty = compute_road_border_penalty(
            ego_edge_points,
            denorm_inputs["line_strings"],
            margin=args.road_border_margin,
        )  # [B*N, T]
    else:
        rb_penalty = torch.zeros_like(nc_penalty)

    reward = -(
        args.w_collision * nc_penalty.sum(dim=-1) + args.w_road_border * rb_penalty.sum(dim=-1)
    )
    return reward, nc_penalty, rb_penalty


@torch.no_grad()
def compute_gt_l2_distance(
    ego_world: torch.Tensor,
    ego_future_gt: torch.Tensor,
) -> torch.Tensor:
    """ADE (mean per-waypoint L2) from each generated trajectory to the scene's own GT future.

    Penalising this distance keeps the generated ego trajectory close to the real recorded
    maneuver, so the policy cannot reward-hack the collision term by, e.g., standing still.

    Args:
        ego_world: [B, T, 4] generated ego trajectories (x, y, cos, sin), ego frame, metres.
        ego_future_gt: [B, Tg, >=2] GT ego future (x, y, ...), same ego-centric frame.

    Returns:
        ade: [B] mean L2 distance over valid (non-zero-padded) GT waypoints.
    """
    gen_xy = ego_world[..., :2]
    gt_xy = ego_future_gt[..., :2]
    T = min(gen_xy.shape[1], gt_xy.shape[1])
    gen_xy, gt_xy = gen_xy[:, :T], gt_xy[:, :T]
    dist = torch.linalg.norm(gen_xy - gt_xy, dim=-1)  # [B, T]
    valid = (gt_xy.abs().sum(dim=-1) > 1e-6).float()  # mask zero-padded GT waypoints
    return (dist * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)  # [B]


@torch.no_grad()
def compute_kinematic_consistency_penalty(
    ego_world: torch.Tensor,
    ego_agent_past_4d: torch.Tensor,
    ego_current_state: torch.Tensor,
) -> torch.Tensor:
    """Round-trip drift of a generated trajectory through the (accel, curvature) action space.

    A kinematically feasible trajectory survives a trajectory -> action -> trajectory round-trip
    almost unchanged, because the unicycle action space can represent it exactly. An infeasible /
    jerky trajectory (e.g. teleporting, instantaneous heading flips) cannot be reproduced from any
    smooth control sequence, so the reconstruction drifts away. Penalising this drift pushes the
    policy toward dynamically realisable trajectories without needing a reference GT.

    Args:
        ego_world: [B, T, 4] generated ego trajectory (x, y, cos, sin), ego-centric frame, metres.
        ego_agent_past_4d: [B, Th, 4] ego past trajectory (x, y, cos, sin), same frame. The last
            history step is assumed to be the (zeroed) current pose.
        ego_current_state: [B, >=5] ego current state; index 4 is the longitudinal velocity used
            as the integration's initial speed.

    Returns:
        drift: [B] mean per-waypoint L2 distance between the trajectory and its round-trip.
    """
    smoothed = smoothing_future_trajectory(ego_agent_past_4d, ego_current_state, ego_world)
    drift = torch.linalg.norm(ego_world[..., :2] - smoothed[..., :2], dim=-1)  # [B, T]
    return drift.mean(dim=-1)  # [B]


def compute_group_advantages(reward: torch.Tensor, num_scenes: int, n: int, eps: float):
    """Group-relative advantages: normalise rewards within each group of ``n`` samples.

    Args:
        reward: [B*N] rewards laid out as ``num_scenes`` contiguous groups of ``n``.
        num_scenes: number of distinct scenes (B).
        n: group size (N).
        eps: numerical floor for the per-group std.

    Returns:
        advantages: [B*N] normalised advantages (mean 0, std ~1 within each group).
    """
    grouped = reward.view(num_scenes, n)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    advantages = (grouped - mean) / (std + eps)
    return advantages.reshape(-1)


def compute_grpo_loss(
    model,
    norm_inputs: dict[str, torch.Tensor],
    ego_pseudo_gt: torch.Tensor,
    neighbors_future: torch.Tensor,
    neighbor_future_mask: torch.Tensor,
    advantages: torch.Tensor,
    args,
) -> dict[str, torch.Tensor]:
    """Advantage-weighted diffusion loss (the GRPO policy-gradient surrogate).

    This mirrors the ``x_start`` branch of :func:`decoder.compute_training_loss`, but:
      * the ego target is the *generated* trajectory (``ego_pseudo_gt``) rather than GT,
      * the per-sample ego loss is weighted by its group-relative advantage,
      * neighbor / turn-indicator / penalty terms are dropped (GRPO only shapes the ego policy).

    Args:
        model: the Diffusion_Planner (training mode forward).
        norm_inputs: observation-normalized inputs (batch already expanded to B*N).
        ego_pseudo_gt: [B*N, T, 4] generated ego trajectory (world frame), detached.
        neighbors_future: [B*N, Pn, T, 4] neighbor futures (world frame).
        neighbor_future_mask: [B*N, Pn, T] True where a neighbor timestep is invalid.
        advantages: [B*N] group-relative advantages.
        args: namespace with normalizers, loss coefficients, horizon.

    Returns:
        dict with ``loss`` (scalar, to backprop) plus detached scalar diagnostics.
    """
    if args.diffusion_model_type != "x_start":
        raise NotImplementedError(
            f"GRPO loss only supports diffusion_model_type='x_start', got "
            f"'{args.diffusion_model_type}'."
        )
    if args.use_velocity_representation:
        raise NotImplementedError("GRPO loss does not support velocity representation.")

    norm = args.state_normalizer
    ego_target = ego_pseudo_gt.detach()

    B, Pn, T, _ = neighbors_future.shape
    P = 1 + Pn
    device = ego_pseudo_gt.device

    ego_current = norm_inputs["ego_current_state"][:, :4]
    neighbors_current = norm_inputs["neighbor_agents_past"][:, :Pn, -1, :4]
    longitudinal_velocity = norm_inputs["ego_current_state"][:, 4:5]

    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_mask = torch.concat(
        (neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1
    )  # [B, Pn, T+1]

    # ego row uses the generated (pseudo-GT) trajectory; neighbor rows keep their GT futures.
    gt_future = torch.cat([ego_target[:, None, :, :], neighbors_future], dim=1)  # [B, P, T, 4]
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)  # [B, P, 4]

    eps = 1e-3
    t = torch.rand(B, device=device) * (1 - eps) + eps
    t = t.view(B, 1, 1, 1).expand(B, P, T + 1, 1)
    z = torch.randn_like(gt_future)

    max_delay = 5
    delay = torch.randint(0, max_delay + 1, (B,), device=device)
    prefix_mask = generate_prefix_mask(delay, P, T + 1)  # [B, P, T+1, 1]
    mask_coeff = random.uniform(0.0, 1.0)
    curr_mask_time = torch.maximum(t * mask_coeff, torch.tensor(eps, device=device))
    t = torch.where(prefix_mask, curr_mask_time, t)

    all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)  # [B, P, T+1, 4]
    all_gt[:, 1:][neighbor_mask] = 0.0

    mean, std = VPSDE_linear().marginal_prob(all_gt[..., 1:, :], t[..., 1:, :])
    xT = mean + std * z
    xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
    xT = torch.where(prefix_mask, all_gt, xT)

    merged_inputs = {
        **norm_inputs,
        "gt_trajectories": all_gt,
        "sampled_trajectories": xT,
        "diffusion_time": t,
        "prefix_mask": prefix_mask,
    }
    _, decoder_output = model(merged_inputs)
    model_output = decoder_output["model_output"][:, :, 1:, :]  # [B, P, T, 4]
    gt_target = all_gt[:, :, 1:, :]

    loss_dict = loss_func(model_output, gt_target)
    heading_l2_loss = loss_dict["heading_l2_loss"]
    position_lat_loss = loss_dict["position_lat_loss"]
    position_lon_loss = loss_dict["position_lon_loss"]

    velocity_weight = torch.abs(longitudinal_velocity * args.coeff_velocity)
    velocity_weight = torch.clamp_min(velocity_weight, 1.0).unsqueeze(-1)  # [B, 1, 1]
    position_lon_loss = position_lon_loss / velocity_weight

    timestep_weight = args.coeff_timestep
    unit = T // len(timestep_weight)
    for i in range(len(timestep_weight)):
        position_lat_loss[:, :, i * unit : (i + 1) * unit] *= timestep_weight[i]
        position_lon_loss[:, :, i * unit : (i + 1) * unit] *= timestep_weight[i]
        heading_l2_loss[:, :, i * unit : (i + 1) * unit] *= timestep_weight[i]

    dpm_loss = (
        args.coeff_position_lat_loss * position_lat_loss
        + args.coeff_position_lon_loss * position_lon_loss
        + args.coeff_heading_l2_loss * heading_l2_loss
    )  # [B, P, T]

    # Per-sample ego diffusion loss (proxy for negative log-likelihood of the trajectory).
    ego_loss_per_sample = dpm_loss[:, 0, : args.ego_prediction_horizon].mean(dim=-1)  # [B]

    # GRPO policy-gradient surrogate: minimise advantage-weighted reconstruction loss.
    grpo_loss = (advantages * ego_loss_per_sample).mean()

    return {
        "loss": grpo_loss,
        "ego_diffusion_loss": ego_loss_per_sample.mean().detach(),
        "mean_advantage": advantages.mean().detach(),
        "abs_advantage": advantages.abs().mean().detach(),
    }
