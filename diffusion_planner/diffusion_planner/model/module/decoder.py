from argparse import Namespace
from functools import partial

import torch
import torch.nn as nn

import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm
from diffusion_planner.loss import (
    compute_ego_edge_points,
    compute_neighbor_collision_penalty,
    compute_road_border_penalty,
    hybrid_loss,
    loss_func,
    make_turn_indicator_gt,
    velocity_to_waypoints,
    waypoints_to_velocity,
)
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.flow_matching_utils.ode_solver import (
    euler_integration,
    heun_integration,
    rk4_integration,
)
from diffusion_planner.model.module.dit import DiT
from diffusion_planner.model.module.turn_indicator import TurnIndicatorNetwork
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


def replace_current_state(x: torch.Tensor, current_states: torch.Tensor) -> torch.Tensor:
    """Return a trajectory tensor with the first timestep replaced."""
    return torch.cat([current_states[:, :, None, :], x[:, :, 1:, :]], dim=2)


def add_current_xy(future: torch.Tensor, current_states: torch.Tensor) -> torch.Tensor:
    """Add current xy position to future xy channels without mutating the input."""
    xy = future[..., :2] + current_states[:, :, None, :2]
    return torch.cat([xy, future[..., 2:]], dim=-1)


def compute_training_loss(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    futures: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args: Namespace,
):
    norm = args.state_normalizer
    model_type = args.diffusion_model_type
    use_velocity = args.use_velocity_representation
    hybrid_omega = args.hybrid_loss_omega
    hybrid_window = args.hybrid_loss_window

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

    eps = 1e-3
    t = torch.rand(B, device=gt_future.device) * (1 - eps) + eps  # [B,]
    t = t.view(B, 1, 1, 1)
    t = t.expand(B, P, T + 1, 1)
    z = torch.randn_like(gt_future, device=gt_future.device)  # [B, P, T, 4]

    if use_velocity:
        full_traj = torch.cat([current_states[:, :, None, :], gt_future], dim=2)  # [B, P, T+1, 4]
        gt_velocity = waypoints_to_velocity(full_traj)  # [B, P, T, 4]
        all_gt = torch.cat([current_states[:, :, None, :], gt_velocity], dim=2)
    else:
        all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)
    all_gt[:, 1:][neighbor_mask] = 0.0

    if model_type == "x_start":
        mean, std = VPSDE_linear().marginal_prob(all_gt[..., 1:, :], t[..., 1:, :])
        # mean([B, P, T, D]), std([B, 1, T, 1]), z([B, P, T, D])
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

        gt_target = all_gt[:, :, 1:, :]  # [B, P, T, 4]

        if use_velocity:
            # Hybrid loss: velocity L2 + omega * waypoint L2 (with detach window)
            dpm_loss = hybrid_loss(
                model_output,
                gt_target,
                omega=hybrid_omega,
                W=hybrid_window,
            )  # [B, P, T]
        else:
            loss_dict = loss_func(model_output, gt_target)
            heading_l2_loss = loss_dict["heading_l2_loss"]  # [B, P, T]
            position_lat_loss = loss_dict["position_lat_loss"]  # [B, P, T]
            position_lon_loss = loss_dict["position_lon_loss"]  # [B, P, T]

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

            dpm_loss = (
                args.coeff_position_lat_loss * position_lat_loss
                + args.coeff_position_lon_loss * position_lon_loss
                + args.coeff_heading_l2_loss * heading_l2_loss
            )  # [B, P, T]

    elif model_type == "flow_matching":
        # t=0 is noise, t=1 is data
        t = t.reshape(-1, *([1] * (len(all_gt.shape) - 1)))  # [B, 1, 1, 1]
        xT = (1 - t) * z + t * all_gt[:, :, 1:, :]  # [B, P, T, 4]
        t = t.reshape(-1)  # [B,]

        xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
        merged_inputs = {
            **inputs,
            "gt_trajectories": all_gt,
            "sampled_trajectories": xT,
            "diffusion_time": t,
        }
        _, decoder_output = model(merged_inputs)  # [B, P, 1 + T, 4]
        model_output = decoder_output["model_output"][:, :, 1:, :]  # [B, P, T, 4]

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
        ego_pred = model_output[:, 0]  # [B, T, 4]
        if use_velocity:
            ego_current_raw = current_states[:, 0]  # [B, 4]
            ego_pred_world = velocity_to_waypoints(ego_pred)
            ego_pred_world[..., :2] = ego_pred_world[..., :2] + ego_current_raw[:, None, :2]
        else:
            ego_pred_world = ego_pred * norm.std[0].to(model_output.device) + norm.mean[0].to(
                model_output.device
            )  # [B, T, 4]
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
            margin_vehicle=args.neighbor_collision_margin_vehicle,
            margin_pedestrian=args.neighbor_collision_margin_pedestrian,
            margin_bicycle=args.neighbor_collision_margin_bicycle,
        )  # [B, T]
        loss["neighbor_collision_loss"] = nc_loss.mean()
    else:
        loss["neighbor_collision_loss"] = torch.tensor(0.0, device=dpm_loss.device)

    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    turn_indicator_logit = decoder_output["turn_indicator_logit"]  # [B, TURN_INDICATOR_OUTPUT_KEEP]
    # TurnIndicatorDropoutAugmentation zeroes the encoder input but stashes the
    # original sequence under turn_indicators_gt_source so the GT stays intact.
    turn_indicators_for_gt = inputs.get("turn_indicators_gt_source", inputs["turn_indicators"])
    turn_indicator_gt = make_turn_indicator_gt(turn_indicators_for_gt)  # [B,]
    turn_indicator_loss = nn.functional.cross_entropy(
        turn_indicator_logit, turn_indicator_gt, reduction="none"
    )
    turn_indicator_change = turn_indicators_for_gt[:, -2] != turn_indicators_for_gt[:, -1]
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

        self.dit = DiT(
            depth=config.decoder_depth,
            output_dim=(config.future_len + 1) * 4,  # x, y, cos, sin
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
        )
        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer

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

        # The turn-indicator predictor is parameter- and feature-independent
        # from the diffusion encoder and denoising network.
        # Halve the encoder hyperparameters to keep this auxiliary predictor light.
        self.turn_indicator_predictor = TurnIndicatorNetwork(
            hidden_dim=config.hidden_dim // 2,
            num_heads=config.num_heads // 2,
            mixer_depth=config.encoder_mixer_depth // 2,
            fusion_depth=config.encoder_fusion_depth // 2,
            drop_path_rate=config.encoder_drop_path_rate,
        )

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

    def _compute_turn_indicator(self, ego_trajectory, inputs):
        """Compute turn-indicator logits without diffusion-network features."""
        return self.turn_indicator_predictor(ego_trajectory, inputs)

    def _forward_training(self, encoding, inputs, neighbor_current_mask):
        """Forward pass for training mode.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing sampled_trajectories, gt_trajectories, diffusion_time, etc.
            neighbor_current_mask: [B, Pn] mask for invalid neighbors

        Returns:
            Dict containing model_output and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        sampled_trajectories = inputs["sampled_trajectories"].reshape(
            B, P, (1 + self._future_len), 4
        )
        diffusion_time = inputs["diffusion_time"]

        gt_trajectories = inputs["gt_trajectories"].reshape(B, P, (1 + self._future_len), 4)
        turn_indicator_logit = self._compute_turn_indicator(gt_trajectories[:, 0, 1:], inputs)

        return {
            "model_output": self.dit(
                sampled_trajectories,
                diffusion_time,
                encoding,
                neighbor_current_mask,
            ).reshape(B, P, -1, 4),
            "turn_indicator_logit": turn_indicator_logit,
        }

    def _inference_flow_matching(
        self,
        encoding,
        inputs,
        current_states,
        neighbor_current_mask,
        sampled_trajectories,
    ):
        """Inference using Flow Matching approach.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing input data
            neighbor_current_mask: [B, Pn] mask for invalid neighbors
            sampled_trajectories: [B, P, (1 + T) * 4] sampled trajectories

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        x = sampled_trajectories
        NUM_STEP = 10
        func = partial(
            self.dit,
            cross_c=encoding,
            neighbor_current_mask=neighbor_current_mask,
        )
        x = euler_integration(func, x, NUM_STEP)
        # x = heun_integration(func, x, NUM_STEP)
        # x = rk4_integration(func, x, NUM_STEP)
        x = x.reshape(B, P, (1 + self._future_len), 4)
        turn_indicator_logit = self._compute_turn_indicator(x[:, 0, 1:], inputs)
        if self._use_velocity:
            future = velocity_to_waypoints(x[:, :, 1:, :])
            future = add_current_xy(future, current_states)
            x = future  # [B, P, T, 4]
        else:
            x = self._state_normalizer.inverse(x)[:, :, 1:]
        return {
            "prediction": x,
            "turn_indicator_logit": turn_indicator_logit,
        }

    def _inference_x_start(
        self,
        encoding,
        inputs,
        current_states,
        neighbor_current_mask,
        sampled_trajectories,
    ):
        """Inference using X-Start (DPM Solver) approach.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing input data
            current_states: [B, P, 4] current states
            neighbor_current_mask: [B, Pn] mask for invalid neighbors
            sampled_trajectories: [B, P, (1 + T) * 4] sampled trajectories

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        action_prefix = sampled_trajectories.reshape(B, P, 1 + self._future_len, 4)
        action_prefix = replace_current_state(action_prefix, current_states)
        xT = action_prefix.reshape(B, P, (1 + self._future_len) * 4)

        def prefix_constraint(xt, t, step):
            xt = xt.reshape(B, P, 1 + self._future_len, 4)
            xt = replace_current_state(xt, current_states)
            return xt

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
            **model_wrapper_params,
        )

        dpm_solver = dpm.DPM_Solver(model_fn, noise_schedule, correcting_xt_fn=prefix_constraint)

        x0 = dpm_solver.sample(xT, steps=10, skip_type="logSNR")

        x0 = x0.reshape(B, P, (1 + self._future_len), 4)
        turn_indicator_logit = self._compute_turn_indicator(x0[:, 0, 1:], inputs)
        if self._use_velocity:
            future = velocity_to_waypoints(x0[:, :, 1:, :])
            future = add_current_xy(future, current_states)
            x0 = future  # [B, P, T, 4]
        else:
            x0 = self._state_normalizer.inverse(x0)[:, :, 1:]
        return {
            "prediction": x0,
            "turn_indicator_logit": turn_indicator_logit,
        }

    def _forward_inference(self, encoding, inputs, current_states, neighbor_current_mask):
        """Forward pass for inference mode.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing input data
            current_states: [B, P, 4] current states
            neighbor_current_mask: [B, Pn] mask for invalid neighbors

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        sampled_trajectories = inputs["sampled_trajectories"].reshape(
            B, P, (1 + self._future_len) * 4
        )

        if self._model_type == "flow_matching":
            return self._inference_flow_matching(
                encoding,
                inputs,
                current_states,
                neighbor_current_mask,
                sampled_trajectories,
            )
        elif self._model_type == "x_start":
            return self._inference_x_start(
                encoding,
                inputs,
                current_states,
                neighbor_current_mask,
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

        # Dispatch to training or inference
        if self.training:
            return self._forward_training(encoding, inputs, neighbor_current_mask)
        else:
            return self._forward_inference(encoding, inputs, current_states, neighbor_current_mask)
