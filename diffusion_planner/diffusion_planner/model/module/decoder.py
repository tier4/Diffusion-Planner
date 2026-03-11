import random
from argparse import Namespace
from functools import partial

import torch
import torch.nn as nn

import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm
from diffusion_planner.dimensions import (
    OUTPUT_MODE_CONTROL,
    OUTPUT_MODE_TRAJECTORY,
    OUTPUT_MODE_TRAJECTORY_AND_CONTROL,
    POSE_DIM,
    TURN_INDICATOR_OUTPUT_DIM,
    output_dim_for_mode,
)
from diffusion_planner.loss import (
    compute_ego_edge_points,
    compute_neighbor_collision_penalty,
    compute_road_border_penalty,
    control_to_waypoints,
    hybrid_loss,
    loss_func,
    make_turn_indicator_gt,
    velocity_to_waypoints,
    waypoints_to_control,
    waypoints_to_velocity,
)
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.flow_matching_utils.ode_solver import (
    euler_integration,
    heun_integration,
    rk4_integration,
)
from diffusion_planner.model.module.dit import DiT
from diffusion_planner.utils.normalizer import (
    ControlNormalizer,
    ObservationNormalizer,
    StateNormalizer,
)


def generate_prefix_mask(delay: torch.Tensor, num_agents: int, max_len: int) -> torch.Tensor:
    """Generates a prefix mask based on a delay tensor.

    Args:
        delay: A 1D tensor of shape (B,) with delay values.
        num_agents: The number of agents (P).
        max_len: The maximum length of the sequence (T+1 or T_plus_1).

    Returns:
        A 4D boolean tensor of shape (B, num_agents, max_len, 1) where mask[i, :, j, 0] is True if j <= delay[i].
    """
    # Create steps tensor (1, 1, max_len, 1)
    steps = torch.arange(max_len, device=delay.device).view(1, 1, -1, 1)
    # Reshape delay to (B, 1, 1, 1) for broadcasting
    reshaped_delay = delay.reshape(delay.shape[0], 1, 1, 1)
    # Perform the comparison, result is (B, 1, max_len, 1)
    mask = steps <= reshaped_delay
    # Expand to include the num_agents dimension
    mask = mask.expand(-1, num_agents, -1, -1)  # (B, num_agents, max_len, 1)

    # Always predict for neighbors by setting their mask to False
    result = torch.zeros_like(mask, dtype=torch.bool)
    result[:, 0, :, :] = mask[:, 0, :, :]

    return result


def _compute_trajectory_loss(
    model_output: torch.Tensor,
    gt_target: torch.Tensor,
    use_velocity: bool,
    hybrid_omega: float,
    hybrid_window: int,
    longitudinal_velocity: torch.Tensor,
    args: Namespace,
    T: int,
) -> torch.Tensor:
    """Compute trajectory-space loss. Returns [B, P, T]."""
    if use_velocity:
        return hybrid_loss(model_output, gt_target, omega=hybrid_omega, W=hybrid_window)

    loss_dict = loss_func(model_output, gt_target)
    heading_l2_loss = loss_dict["heading_l2_loss"]
    position_lat_loss = loss_dict["position_lat_loss"]
    position_lon_loss = loss_dict["position_lon_loss"]

    velocity_weight = longitudinal_velocity * args.coeff_velocity
    velocity_weight = torch.abs(velocity_weight)
    velocity_weight = torch.clamp_min(velocity_weight, 1.0)
    velocity_weight = velocity_weight.unsqueeze(-1)
    position_lon_loss = position_lon_loss / velocity_weight

    timestep_weight = args.coeff_timestep
    assert T % len(timestep_weight) == 0, (
        f"Timestep {T} is not divisible by the number of timestep weights {len(timestep_weight)}"
    )
    unit = T // len(timestep_weight)
    for i in range(len(timestep_weight)):
        position_lat_loss[:, :, i * unit : (i + 1) * unit] *= timestep_weight[i]
        position_lon_loss[:, :, i * unit : (i + 1) * unit] *= timestep_weight[i]
        heading_l2_loss[:, :, i * unit : (i + 1) * unit] *= timestep_weight[i]

    return (
        args.coeff_position_lat_loss * position_lat_loss
        + args.coeff_position_lon_loss * position_lon_loss
        + args.coeff_heading_l2_loss * heading_l2_loss
    )


def build_gt_representation(
    gt_future: torch.Tensor,
    current_states: torch.Tensor,
    inputs: dict[str, torch.Tensor],
    output_mode: str,
    use_velocity: bool,
    norm: StateNormalizer,
    control_norm: ControlNormalizer,
    obs_norm: ObservationNormalizer,
    Pn: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build GT and current state in the target representation.

    Returns:
        all_gt: [B, P, T+1, D] where D depends on output_mode.
        all_gt_pose: [B, P, T+1, 4] trajectory in pose space (for turn indicator / edge points).
            Only differs from all_gt when output_mode includes control.
    """
    has_traj = output_mode in (OUTPUT_MODE_TRAJECTORY, OUTPUT_MODE_TRAJECTORY_AND_CONTROL)
    has_ctrl = output_mode in (OUTPUT_MODE_CONTROL, OUTPUT_MODE_TRAJECTORY_AND_CONTROL)

    # --- Trajectory part [B, P, T, 4] ---
    if has_traj:
        if use_velocity:
            full_traj = torch.cat([current_states[:, :, None, :], gt_future], dim=2)
            traj_gt = waypoints_to_velocity(full_traj)  # [B, P, T, 4]
        else:
            traj_gt = norm(gt_future)  # [B, P, T, 4]
        traj_current = current_states  # [B, P, 4]

    # --- Control part [B, P, T, 2] ---
    # Control is only meaningful for ego. Neighbor control in ego-centric frame
    # is ill-defined (unicycle model requires origin position and zero heading).
    # Neighbor control channels are filled with zeros.
    if has_ctrl:
        raw_inputs = obs_norm.inverse(inputs)
        B, P, T = gt_future.shape[:3]

        # Ego control
        ego_history = raw_inputs["ego_agent_past"]  # [B, T_hist, 4] raw
        ego_v0 = raw_inputs["ego_current_state"][:, 4:5]  # [B, 1] raw velocity
        ego_ctrl = waypoints_to_control(
            ego_history, gt_future[:, 0], t0_states={"v": ego_v0.squeeze(-1)}
        )  # [B, T, 2]
        ego_ctrl_norm = control_norm(ego_ctrl)  # [B, T, 2]

        # Assemble: ego control + zeros for neighbors
        ctrl_gt = torch.zeros(B, P, T, 2, device=gt_future.device)
        ctrl_gt[:, 0] = ego_ctrl_norm

        # Current state: [normalized_v0, 0] for ego, zeros for neighbors
        ego_ctrl_current = control_norm(
            torch.cat([ego_v0, torch.zeros_like(ego_v0)], dim=-1)
        )  # [B, 2]
        ctrl_current = torch.zeros(B, P, 2, device=gt_future.device)
        ctrl_current[:, 0] = ego_ctrl_current

    # --- Assemble ---
    if output_mode == OUTPUT_MODE_TRAJECTORY:
        gt_converted = traj_gt
        current_D = traj_current
    elif output_mode == OUTPUT_MODE_CONTROL:
        gt_converted = ctrl_gt
        current_D = ctrl_current
    else:  # trajectory_and_control
        gt_converted = torch.cat([traj_gt, ctrl_gt], dim=-1)  # [B, P, T, 6]
        current_D = torch.cat([traj_current, ctrl_current], dim=-1)  # [B, P, 6]

    all_gt = torch.cat([current_D[:, :, None, :], gt_converted], dim=2)  # [B, P, T+1, D]

    # Pose-space GT (always 4D) for turn indicator and edge point computation
    if use_velocity:
        full_traj = torch.cat([current_states[:, :, None, :], gt_future], dim=2)
        vel_gt = waypoints_to_velocity(full_traj)
        all_gt_pose = torch.cat([current_states[:, :, None, :], vel_gt], dim=2)
    else:
        all_gt_pose = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)

    return all_gt, all_gt_pose


def compute_training_loss(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    futures: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args: Namespace,
):
    norm = args.state_normalizer
    control_norm = args.control_normalizer
    obs_norm = args.observation_normalizer
    model_type = args.diffusion_model_type
    use_velocity = args.use_velocity_representation
    hybrid_omega = args.hybrid_loss_omega
    hybrid_window = args.hybrid_loss_window
    output_mode = args.output_mode
    D = output_dim_for_mode(output_mode)

    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask  # [B, Pn, V]

    B, Pn, T, _ = neighbors_future.shape
    P = 1 + Pn
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
    )  # [B, P, T, 4]
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)  # [B, P, 4]

    # Build GT in the target representation
    all_gt, all_gt_pose = build_gt_representation(
        gt_future, current_states, inputs, output_mode, use_velocity, norm, control_norm, obs_norm, Pn
    )
    all_gt[:, 1:][neighbor_mask] = 0.0
    all_gt_pose[:, 1:][neighbor_mask] = 0.0

    eps = 1e-3
    t = torch.rand(B, device=gt_future.device) * (1 - eps) + eps  # [B,]
    t = t.view(B, 1, 1, 1)
    t = t.expand(B, P, T + 1, 1)
    z = torch.randn(B, P, T, D, device=gt_future.device)  # [B, P, T, D]

    max_delay = 5
    delay = torch.randint(0, max_delay + 1, (B,), device=gt_future.device)  # [B,]
    prefix_mask = generate_prefix_mask(delay, 1 + Pn, T + 1)  # (B, P, T+1, 1)
    mask_coeff = random.uniform(0.0, 1.0)
    curr_mask_time = torch.maximum(t * mask_coeff, torch.tensor(eps, device=gt_future.device))
    t = torch.where(prefix_mask, curr_mask_time, t)

    if model_type == "x_start":
        mean, std = VPSDE_linear().marginal_prob(all_gt[..., 1:, :], t[..., 1:, :])
        xT = mean + std * z

        xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
        xT = torch.where(prefix_mask, all_gt, xT)  # [B, P, 1 + T, D]

        merged_inputs = {
            **inputs,
            "gt_trajectories": all_gt,
            "sampled_trajectories": xT,
            "diffusion_time": t,
            "prefix_mask": prefix_mask,
        }
        # Pass pose-space GT for turn indicator when not in trajectory mode
        if output_mode != OUTPUT_MODE_TRAJECTORY:
            merged_inputs["gt_trajectories_pose"] = all_gt_pose

        _, decoder_output = model(merged_inputs)
        model_output = decoder_output["model_output"][:, :, 1:, :]  # [B, P, T, D]

        gt_target = all_gt[:, :, 1:, :]  # [B, P, T, D]

        # --- Loss computation per output_mode ---
        if output_mode == OUTPUT_MODE_TRAJECTORY:
            dpm_loss = _compute_trajectory_loss(
                model_output, gt_target, use_velocity, hybrid_omega, hybrid_window,
                longitudinal_velocity, args, T,
            )
        elif output_mode == OUTPUT_MODE_CONTROL:
            dpm_loss = torch.sum((model_output - gt_target) ** 2, dim=-1)  # [B, P, T]
        else:  # trajectory_and_control
            traj_out = model_output[..., :POSE_DIM]
            traj_gt = gt_target[..., :POSE_DIM]
            ctrl_out = model_output[..., POSE_DIM:]
            ctrl_gt = gt_target[..., POSE_DIM:]

            traj_loss = _compute_trajectory_loss(
                traj_out, traj_gt, use_velocity, hybrid_omega, hybrid_window,
                longitudinal_velocity, args, T,
            )
            # Control loss for ego only — neighbor control is not well-defined
            # in ego-centric frame (position/heading offset violates unicycle assumptions).
            ego_ctrl_loss = torch.sum(
                (ctrl_out[:, 0] - ctrl_gt[:, 0]) ** 2, dim=-1
            )  # [B, T]
            coeff_ctrl = args.coeff_control_loss
            dpm_loss = traj_loss
            dpm_loss[:, 0] = dpm_loss[:, 0] + coeff_ctrl * ego_ctrl_loss

    elif model_type == "flow_matching":
        # t=0 is noise, t=1 is data
        t = t.reshape(-1, *([1] * (len(all_gt.shape) - 1)))  # [B, 1, 1, 1]
        xT = (1 - t) * z + t * all_gt[:, :, 1:, :]  # [B, P, T, D]
        t = t.reshape(-1)  # [B,]

        xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
        merged_inputs = {
            **inputs,
            "gt_trajectories": all_gt,
            "sampled_trajectories": xT,
            "diffusion_time": t,
            "prefix_mask": prefix_mask,
        }
        if output_mode != OUTPUT_MODE_TRAJECTORY:
            merged_inputs["gt_trajectories_pose"] = all_gt_pose

        _, decoder_output = model(merged_inputs)
        model_output = decoder_output["model_output"][:, :, 1:, :]  # [B, P, T, D]

        target_v = all_gt[:, :, 1:, :] - z
        dpm_loss = torch.sum((model_output - target_v) ** 2, dim=-1)
    else:
        raise NotImplementedError(f"Unknown diffusion model type: {model_type}")

    masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]

    loss = {}

    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

    loss["ego_planning_loss"] = dpm_loss[:, 0, : args.ego_prediction_horizon].mean()

    # Compute ego edge points for penalty losses
    need_ego_edge = model_type == "x_start" and (
        args.coeff_road_border_loss > 0 or args.coeff_neighbor_collision_loss > 0
    )
    if need_ego_edge:
        # For control/mixed modes, reconstruct trajectory from the trajectory part or
        # from the pose-space GT. Edge losses always operate in trajectory space.
        if output_mode == OUTPUT_MODE_TRAJECTORY:
            ego_pred = model_output[:, 0]  # [B, T, 4]
            if use_velocity:
                ego_current_raw = current_states[:, 0]  # [B, 4]
                ego_pred_world = velocity_to_waypoints(ego_pred)
                ego_pred_world[..., :2] = ego_pred_world[..., :2] + ego_current_raw[:, None, :2]
            else:
                ego_pred_world = ego_pred * norm.std[0].to(model_output.device) + norm.mean[0].to(
                    model_output.device
                )  # [B, T, 4]
        elif output_mode == OUTPUT_MODE_CONTROL:
            ego_ctrl_pred = model_output[:, 0]  # [B, T, 2]
            ego_pred_world = control_to_waypoints(
                ego_ctrl_pred,
                inputs["ego_agent_past"],
                t0_states={"v": longitudinal_velocity.squeeze(-1)},
            )  # [B, T, 4]
        else:  # trajectory_and_control
            ego_pred = model_output[:, 0, :, :POSE_DIM]  # [B, T, 4]
            if use_velocity:
                ego_current_raw = current_states[:, 0]
                ego_pred_world = velocity_to_waypoints(ego_pred)
                ego_pred_world[..., :2] = ego_pred_world[..., :2] + ego_current_raw[:, None, :2]
            else:
                ego_pred_world = ego_pred * norm.std[0].to(model_output.device) + norm.mean[0].to(
                    model_output.device
                )

        ego_edge_points = compute_ego_edge_points(
            ego_pred_world, inputs["ego_shape"], n_interp=args.road_border_n_interp
        )
        denorm_inputs = args.observation_normalizer.inverse(inputs)

    # Road border collision loss (ego only, x_start mode)
    if args.coeff_road_border_loss > 0 and model_type == "x_start":
        rb_loss = compute_road_border_penalty(
            ego_edge_points,
            denorm_inputs["line_strings"],
            margin=args.road_border_margin,
        )  # [B, T]
        loss["road_border_loss"] = rb_loss.mean()
    else:
        loss["road_border_loss"] = torch.tensor(0.0, device=dpm_loss.device)

    # Neighbor collision loss (ego only, x_start mode)
    if args.coeff_neighbor_collision_loss > 0 and model_type == "x_start":
        nc_loss = compute_neighbor_collision_penalty(
            ego_edge_points,
            neighbors_future,
            neighbors_future_valid,
            denorm_inputs["neighbor_agents_past"],
            margin=args.neighbor_collision_margin,
        )  # [B, T]
        loss["neighbor_collision_loss"] = nc_loss.mean()
    else:
        loss["neighbor_collision_loss"] = torch.tensor(0.0, device=dpm_loss.device)

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


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len

        self._output_mode = config.output_mode
        self._D = output_dim_for_mode(self._output_mode)

        self.dit = DiT(
            depth=config.decoder_depth,
            output_dim=(config.future_len + 1) * self._D,
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
            T=config.future_len + 1,
            D=self._D,
        )
        self.turn_indicator_predictor = nn.Linear(
            2 * (self._future_len // 10) + config.hidden_dim, TURN_INDICATOR_OUTPUT_DIM
        )

        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer
        self._control_normalizer: ControlNormalizer = config.control_normalizer
        self._neighbor_control_normalizer: ControlNormalizer = config.neighbor_control_normalizer

        # self._guidance_fn = config.guidance_fn
        self._guidance_fn = (
            config.guidance_fn if config.__dict__.get("guidance_fn") is not None else None
        )
        self._guidance_scale = config.guidance_scale
        self._model_type = config.diffusion_model_type
        self._use_velocity = config.use_velocity_representation

        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

        # Zero-out output layers:
        nn.init.constant_(self.dit.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.dit.final_layer.proj[-1].bias, 0)

    def _prepare_current_states(self, inputs):
        """Extract and prepare current states for ego and neighbors.

        Args:
            inputs: Dict containing ego_current_state and neighbor_agents_past

        Returns:
            Tuple of (current_states, neighbor_current_mask, ego_current, neighbors_current)
                - current_states: [B, P, 4] concatenated ego and neighbor current states
                - neighbor_current_mask: [B, Pn] mask for invalid neighbors
                - ego_current: [B, 1, 4] ego current state
                - neighbors_current: [B, Pn, 4] neighbor current states
        """
        ego_current = inputs["ego_current_state"][:, None, :4]
        neighbors_current = inputs["neighbor_agents_past"][
            :, : self._predicted_neighbor_num, -1, :4
        ]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask

        current_states = torch.cat([ego_current, neighbors_current], dim=1)  # [B, P, 4]

        return current_states, neighbor_current_mask, ego_current, neighbors_current

    def _build_current_states_D(self, inputs, current_states):
        """Build current states in D-dimensional space for the diffusion process.

        Args:
            inputs: Dict containing ego_current_state and neighbor_agents_past.
            current_states: [B, P, 4] pose-space current states.

        Returns:
            current_states_D: [B, P, D] current states in the target representation.
        """
        if self._output_mode == OUTPUT_MODE_TRAJECTORY:
            return current_states  # [B, P, 4]

        B = current_states.shape[0]
        Pn = self._predicted_neighbor_num
        P = 1 + Pn

        # Control current state: ego = normalized [v0, 0], neighbors = zeros
        raw_inputs = self._observation_normalizer.inverse(inputs)
        ego_v0 = raw_inputs["ego_current_state"][:, 4:5]  # [B, 1]
        ego_ctrl_current = self._control_normalizer(
            torch.cat([ego_v0, torch.zeros_like(ego_v0)], dim=-1)
        )  # [B, 2]
        ctrl_current = torch.zeros(B, P, 2, device=current_states.device)
        ctrl_current[:, 0] = ego_ctrl_current

        if self._output_mode == OUTPUT_MODE_CONTROL:
            return ctrl_current
        else:  # trajectory_and_control
            return torch.cat([current_states, ctrl_current], dim=-1)  # [B, P, 6]

    def _compute_turn_indicator(self, ego_trajectory, encoding_pooled):
        """Compute turn indicator logit from ego trajectory and encoding.

        Args:
            ego_trajectory: [B, 2 * (T // 10)] flattened ego trajectory positions
            encoding_pooled: [B, D] pooled encoding

        Returns:
            turn_indicator_logit: [B, TURN_INDICATOR_OUTPUT_DIM]
        """
        turn_indicator_input = torch.cat([ego_trajectory, encoding_pooled], dim=-1)
        return self.turn_indicator_predictor(turn_indicator_input)

    def _forward_training(self, encoding, inputs, neighbor_current_mask, encoding_pooled):
        """Forward pass for training mode.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing sampled_trajectories, gt_trajectories, diffusion_time, etc.
            neighbor_current_mask: [B, Pn] mask for invalid neighbors
            encoding_pooled: [B, D] pooled encoding

        Returns:
            Dict containing model_output and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num
        D = self._D

        sampled_trajectories = inputs["sampled_trajectories"].reshape(
            B, P, (1 + self._future_len), D
        )
        diffusion_time = inputs["diffusion_time"]

        gt_trajectories = inputs["gt_trajectories"].reshape(B, P, (1 + self._future_len), D)
        # Turn indicator uses pose (x,y) from gt_trajectories_pose if available,
        # otherwise fall back to first 2 channels of gt_trajectories.
        if "gt_trajectories_pose" in inputs:
            gt_traj_pose = inputs["gt_trajectories_pose"].reshape(
                B, P, (1 + self._future_len), POSE_DIM
            )
            ego_trajectory = gt_traj_pose[:, 0, 1::10, :2].reshape(
                B, 2 * (self._future_len // 10)
            )
        else:
            ego_trajectory = gt_trajectories[:, 0, 1::10, :2].reshape(
                B, 2 * (self._future_len // 10)
            )
        turn_indicator_logit = self._compute_turn_indicator(ego_trajectory, encoding_pooled)

        return {
            "model_output": self.dit(
                sampled_trajectories,
                diffusion_time,
                encoding,
                neighbor_current_mask,
            ).reshape(B, P, -1, D),
            "turn_indicator_logit": turn_indicator_logit,
        }

    def denoised_to_trajectory(self, x, inputs, current_states):
        """Convert denoised output [B, P, T+1, D] to trajectory [B, P, T, 4].

        Handles all output modes (trajectory, control, trajectory_and_control)
        and velocity representation.
        """
        B, P = x.shape[:2]
        Pn = self._predicted_neighbor_num

        if self._output_mode == OUTPUT_MODE_CONTROL:
            # x is [B, P, T+1, 2] — normalized control (accel, curvature)
            # Denormalize ego and neighbor with separate normalizers
            ctrl_raw = x[:, :, 1:, :]  # [B, P, T, 2]
            ctrl = torch.empty_like(ctrl_raw)
            ctrl[:, 0:1] = self._control_normalizer.inverse(ctrl_raw[:, 0:1])
            ctrl[:, 1:] = self._neighbor_control_normalizer.inverse(ctrl_raw[:, 1:])

            # Denormalize inputs to get raw history/velocity for control→trajectory conversion
            raw_inputs = self._observation_normalizer.inverse(inputs)

            # Ego: convert control → trajectory
            ego_v0 = raw_inputs["ego_current_state"][:, 4:5]  # [B, 1] raw velocity
            ego_traj = control_to_waypoints(
                ctrl[:, 0], raw_inputs["ego_agent_past"],
                t0_states={"v": ego_v0.squeeze(-1)},
            )  # [B, T, 4]

            # Neighbors: convert control → trajectory
            # Control signals (accel, curvature) are frame-invariant, so integrating
            # from ego-centric history directly produces ego-centric trajectories.
            neighbor_history = raw_inputs["neighbor_agents_past"][:, :Pn, :, :4]
            neighbor_traj = control_to_waypoints(
                ctrl[:, 1:], neighbor_history,
            )  # [B, Pn, T, 4] in ego-centric frame

            return torch.cat([ego_traj[:, None], neighbor_traj], dim=1)

        elif self._output_mode == OUTPUT_MODE_TRAJECTORY_AND_CONTROL:
            # x is [B, P, T+1, 6] — use trajectory part [B, P, T+1, 4]
            x_traj = x[..., :POSE_DIM]  # [B, P, T+1, 4]
        else:
            x_traj = x  # [B, P, T+1, 4]

        # Convert trajectory/velocity representation to world waypoints
        if self._use_velocity:
            future = velocity_to_waypoints(x_traj[:, :, 1:, :])
            future[..., :2] = future[..., :2] + current_states[:, :, None, :2]
            return future
        else:
            return self._state_normalizer.inverse(x_traj)[:, :, 1:]

    def _compute_turn_indicator_from_denoised(self, x, encoding_pooled):
        """Extract ego trajectory (x,y) from denoised output for turn indicator."""
        B = x.shape[0]
        if self._output_mode == OUTPUT_MODE_CONTROL:
            # Control mode: first 2 channels are (accel, curvature), not (x,y).
            # Use zeros as fallback — turn indicator relies mainly on encoding_pooled.
            ego_xy = torch.zeros(
                B, 2 * (self._future_len // 10), device=x.device, dtype=x.dtype
            )
        else:
            # trajectory or trajectory_and_control: first 2 channels are (x,y)
            ego_xy = x[:, 0, 1::10, :2].reshape(B, 2 * (self._future_len // 10))
        return self._compute_turn_indicator(ego_xy, encoding_pooled)

    def _inference_flow_matching(
        self, encoding, inputs, current_states, neighbor_current_mask, encoding_pooled, sampled_trajectories
    ):
        """Inference using Flow Matching approach."""
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num
        D = self._D

        x = sampled_trajectories
        NUM_STEP = 10
        func = partial(
            self.dit,
            cross_c=encoding,
            neighbor_current_mask=neighbor_current_mask,
        )
        x = euler_integration(func, x, NUM_STEP)
        x = x.reshape(B, P, (1 + self._future_len), D)

        turn_indicator_logit = self._compute_turn_indicator_from_denoised(x, encoding_pooled)
        prediction = self.denoised_to_trajectory(x, inputs, current_states)

        return {"prediction": prediction, "turn_indicator_logit": turn_indicator_logit, "denoised": x}

    def _inference_x_start(
        self,
        encoding,
        inputs,
        current_states,
        neighbor_current_mask,
        encoding_pooled,
        sampled_trajectories,
    ):
        """Inference using X-Start (DPM Solver) approach."""
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num
        D = self._D

        xT = sampled_trajectories
        action_prefix = sampled_trajectories.reshape(B, P, -1, D)

        # Build current state in D-space for prefix constraint
        current_states_D = self._build_current_states_D(inputs, current_states)  # [B, P, D]
        action_prefix[:, :, 0, :] = current_states_D

        B, P, T_plus_1, _ = action_prefix.shape

        delay = inputs["delay"].to(device=action_prefix.device)
        mask = generate_prefix_mask(delay, P, T_plus_1)  # (B, P, T_plus_1, 1)

        def prefix_constraint(xt, t, step):
            xt = xt.reshape(B, P, -1, D)
            xt[:, :, 0, :] = current_states_D
            return xt.reshape(B, P, -1)

        model_wrapper_params = {
            "classifier_fn": self._guidance_fn,
            "classifier_kwargs": {
                "model": self.dit,
                "model_condition": {
                    "cross_c": encoding,
                    "neighbor_current_mask": neighbor_current_mask,
                },
                "inputs": inputs,
                "observation_normalizer": self._observation_normalizer,
                "state_normalizer": self._state_normalizer,
            },
            "guidance_scale": self._guidance_scale,
            "guidance_type": "classifier" if self._guidance_fn is not None else "uncond",
        }

        noise_schedule = dpm.NoiseScheduleVP()

        model_fn = dpm.model_wrapper(
            self.dit,
            noise_schedule,
            model_type=self._model_type,
            model_kwargs={
                "cross_c": encoding,
                "neighbor_current_mask": neighbor_current_mask,
            },
            D=D,
            **model_wrapper_params,
        )

        dpm_solver = dpm.DPM_Solver(model_fn, noise_schedule, correcting_xt_fn=prefix_constraint, D=D)

        x0 = dpm_solver.sample(xT, steps=10, prefix_mask=mask, skip_type="logSNR")

        x0 = x0.reshape(B, P, (1 + self._future_len), D)

        turn_indicator_logit = self._compute_turn_indicator_from_denoised(x0, encoding_pooled)
        prediction = self.denoised_to_trajectory(x0, inputs, current_states)

        return {"prediction": prediction, "turn_indicator_logit": turn_indicator_logit, "denoised": x0}

    def _forward_inference(
        self, encoding, inputs, current_states, neighbor_current_mask, encoding_pooled
    ):
        """Forward pass for inference mode.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing input data
            current_states: [B, P, 4] current states
            neighbor_current_mask: [B, Pn] mask for invalid neighbors
            encoding_pooled: [B, D] pooled encoding

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num
        D = self._D

        sampled_trajectories = inputs["sampled_trajectories"].reshape(
            B, P, (1 + self._future_len) * D
        )

        if self._model_type == "flow_matching":
            return self._inference_flow_matching(
                encoding, inputs, current_states, neighbor_current_mask, encoding_pooled, sampled_trajectories
            )
        elif self._model_type == "x_start":
            return self._inference_x_start(
                encoding,
                inputs,
                current_states,
                neighbor_current_mask,
                encoding_pooled,
                sampled_trajectories,
            )
        else:
            raise NotImplementedError(f"Unknown model type {self._model_type}")

    def forward(self, encoding, inputs):
        """
        Diffusion decoder process.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict
                {
                    ...
                    "ego_current_state": current ego states,
                    "neighbor_agent_past": past and current neighbor states,

                    "sampled_trajectories": sampled current-future ego & neighbor states,        [B, P, 1 + self._future_len, 4]
                    "delay": number of initial steps to keep fixed (>=0),
                    [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "model_output": Predicted future states, [B, P, 1 + self._future_len, 4]
                    [inference-only] "prediction": Predicted future states, [B, P, self._future_len, 4]
                    "turn_indicator_logit": Turn indicator prediction, [B, TURN_INDICATOR_OUTPUT_DIM]
                    ...
                }

        """
        # Common preprocessing
        current_states, neighbor_current_mask, ego_current, neighbors_current = (
            self._prepare_current_states(inputs)
        )

        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)

        # Pool encoding to get a fixed-size representation
        encoding_pooled = torch.mean(encoding, dim=1)  # [B, D]

        # Dispatch to training or inference
        if self.training:
            return self._forward_training(encoding, inputs, neighbor_current_mask, encoding_pooled)
        else:
            return self._forward_inference(
                encoding, inputs, current_states, neighbor_current_mask, encoding_pooled
            )
