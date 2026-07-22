"""PlanTF-style regression decoder head on top of the Diffusion-Planner encoder.

Ported from planTF (https://github.com/jchengai/planTF, Cheng et al., ICRA 2024)
and adapted to the Diffusion-Planner interface:

- consumes the shared encoder output ``[B, token_num, hidden_dim]`` and the raw
  ``inputs`` dict, like :class:`diffusion_planner.model.module.decoder.Decoder`
- outputs (x, y, cos, sin) trajectories in the ``StateNormalizer`` space during
  training and denormalized ``prediction`` ``[B, P, T, 4]`` at inference, so
  validation / simulation / ROS consumers work unchanged
- diffusion-only inputs (``sampled_trajectories``, ``diffusion_time``,
  ``delay``) are accepted but ignored

``compute_plantf_training_loss`` is the planTF counterpart of
``decoder.compute_training_loss`` (same convention as ``grpo_utils`` mirroring
it): winner-takes-all regression over ``num_modes`` ego candidates + mode
classification, with DP's lat/lon/heading decomposition, velocity/timestep
weighting and penalty losses reused as-is.
"""

from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_planner.dimensions import TURN_INDICATOR_OUTPUT_DIM
from diffusion_planner.loss import (
    compute_ego_edge_points,
    compute_neighbor_collision_penalty,
    compute_road_border_penalty,
    loss_func,
    make_turn_indicator_gt,
)
from diffusion_planner.model.module.turn_indicator import TurnIndicatorNetwork
from diffusion_planner.utils.normalizer import StateNormalizer


class PlanTFTrajectoryHead(nn.Module):
    """Multi-modal ego trajectory head (planTF ``TrajectoryDecoder``)."""

    def __init__(self, embed_dim, num_modes, future_steps, out_channels=4):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.out_channels = out_channels

        self.multimodal_proj = nn.Linear(embed_dim, num_modes * embed_dim)

        hidden = 2 * embed_dim
        self.loc = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, future_steps * out_channels),
        )
        self.pi = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        """
        Args:
            x: [B, embed_dim] ego token from the encoder.

        Returns:
            loc: [B, num_modes, future_steps, out_channels] mode trajectories.
            pi: [B, num_modes] mode logits.
        """
        x = self.multimodal_proj(x).view(-1, self.num_modes, self.embed_dim)
        loc = self.loc(x).view(-1, self.num_modes, self.future_steps, self.out_channels)
        pi = self.pi(x).squeeze(-1)
        return loc, pi


class PlanTFDecoder(nn.Module):
    """One-shot regression decoder with the same forward contract as ``Decoder``."""

    def __init__(self, config):
        super().__init__()

        if getattr(config, "use_velocity_representation", False):
            raise NotImplementedError(
                "decoder_type='plantf' does not support use_velocity_representation"
            )

        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._num_modes = getattr(config, "num_modes", 6)

        hidden_dim = config.hidden_dim
        self.trajectory_head = PlanTFTrajectoryHead(
            embed_dim=hidden_dim,
            num_modes=self._num_modes,
            future_steps=self._future_len,
            out_channels=4,  # x, y, cos, sin
        )
        # planTF's agent_predictor, widened from (x, y) to DP's (x, y, cos, sin)
        self.neighbor_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * hidden_dim, self._future_len * 4),
        )
        self.turn_indicator_predictor = nn.Linear(
            2 * (self._future_len // 10) + hidden_dim, TURN_INDICATOR_OUTPUT_DIM
        )
        self._state_normalizer: StateNormalizer = config.state_normalizer

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

        self.independent_turn_indicator_predictor = TurnIndicatorNetwork(
            hidden_dim=config.hidden_dim // 2,
            num_heads=config.num_heads // 2,
            mixer_depth=config.encoder_mixer_depth // 2,
            fusion_depth=config.encoder_fusion_depth // 2,
            drop_path_rate=config.encoder_drop_path_rate,
        )

    def _compute_turn_indicator(self, ego_trajectory, encoding_pooled):
        turn_indicator_input = torch.cat([ego_trajectory, encoding_pooled], dim=-1)
        return self.turn_indicator_predictor(turn_indicator_input)

    def _decode(self, encoding):
        """Run both heads on the encoder tokens.

        Returns (trajectory [B, K, T, 4], probability [B, K],
        neighbor_prediction [B, Pn, T, 4]), all in normalized space.
        """
        B = encoding.shape[0]
        Pn = self._predicted_neighbor_num
        trajectory, probability = self.trajectory_head(encoding[:, 0])
        neighbor_prediction = self.neighbor_predictor(encoding[:, 1 : 1 + Pn]).view(
            B, Pn, self._future_len, 4
        )
        return trajectory, probability, neighbor_prediction

    def _best_mode_trajectory(self, trajectory, probability):
        """Select the argmax-probability mode. gather keeps the batch axis
        dynamic for ONNX tracing (unlike arange-based indexing)."""
        index = probability.argmax(dim=-1).view(-1, 1, 1, 1).expand(-1, 1, self._future_len, 4)
        return trajectory.gather(1, index).squeeze(1)

    @staticmethod
    def _pool_encoding(encoding):
        # Pool only valid encoder tokens. The encoder zero-fills masked tokens.
        encoding_valid = torch.any(encoding != 0, dim=-1)  # [B, N]
        encoding_count = encoding_valid.sum(dim=1).clamp_min(1).unsqueeze(-1)
        return (encoding * encoding_valid.unsqueeze(-1)).sum(dim=1) / encoding_count

    def _subsampled_ego_xy(self, ego_trajectory):
        """Every-10th-step xy, matching gt_trajectories[:, 0, 1::10, :2] used in
        training (index 1 on the current-state-prepended axis = future step 0)."""
        return ego_trajectory[:, ::10, :2].reshape(-1, 2 * (self._future_len // 10))

    def forward_deploy(self, encoding):
        """One-shot deployment path for the split ONNX export.

        Unlike the diffusion decoder there is no external denoising loop, so a
        single call maps the encoder output to the final prediction. The
        independent turn-indicator head is excluded (it re-encodes raw map
        inputs itself and is a training-time auxiliary, not a deploy output).

        Returns:
            prediction: [B, 1 + Pn, T, 4] best-mode ego + neighbors, denormalized.
            probability: [B, K] mode logits.
            turn_indicator_logit: [B, TURN_INDICATOR_OUTPUT_DIM]
        """
        trajectory, probability, neighbor_prediction = self._decode(encoding)
        best_trajectory = self._best_mode_trajectory(trajectory, probability)
        turn_indicator_logit = self._compute_turn_indicator(
            self._subsampled_ego_xy(best_trajectory), self._pool_encoding(encoding)
        )
        prediction = torch.cat([best_trajectory[:, None], neighbor_prediction], dim=1)
        prediction = self._state_normalizer.inverse(prediction)
        return prediction, probability, turn_indicator_logit

    def forward(self, encoding, inputs):
        """
        Args:
            encoding: [B, token_num, D] encoder output (token 0 = ego,
                tokens 1 .. 1+predicted_neighbor_num = neighbors).
            inputs: Dict. Training additionally requires "gt_trajectories"
                [B, P, 1 + future_len, 4] (normalized, current state prepended),
                injected by ``compute_plantf_training_loss``.

        Returns:
            Dict:
                [both] "trajectory": [B, K, T, 4] ego mode trajectories (normalized)
                [both] "probability": [B, K] mode logits
                [both] "neighbor_prediction": [B, Pn, T, 4] (normalized)
                [both] "turn_indicator_logit" / "independent_turn_indicator_logit"
                [inference-only] "prediction": [B, 1 + Pn, T, 4] best-mode ego +
                    neighbors, denormalized — same contract as ``Decoder``.
        """
        B = encoding.shape[0]
        Pn = self._predicted_neighbor_num

        trajectory, probability, neighbor_prediction = self._decode(encoding)
        encoding_pooled = self._pool_encoding(encoding)

        outputs = {
            "trajectory": trajectory,
            "probability": probability,
            "neighbor_prediction": neighbor_prediction,
        }

        if self.training:
            gt_trajectories = inputs["gt_trajectories"].reshape(B, 1 + Pn, 1 + self._future_len, 4)
            ego_trajectory = gt_trajectories[:, 0, 1::10, :2].reshape(
                B, 2 * (self._future_len // 10)
            )
            outputs["turn_indicator_logit"] = self._compute_turn_indicator(
                ego_trajectory, encoding_pooled
            )
            outputs["independent_turn_indicator_logit"] = self.independent_turn_indicator_predictor(
                gt_trajectories[:, 0, 1:], inputs
            )
            return outputs

        best_trajectory = self._best_mode_trajectory(trajectory, probability)

        outputs["turn_indicator_logit"] = self._compute_turn_indicator(
            self._subsampled_ego_xy(best_trajectory), encoding_pooled
        )
        outputs["independent_turn_indicator_logit"] = self.independent_turn_indicator_predictor(
            best_trajectory, inputs
        )

        prediction = torch.cat([best_trajectory[:, None], neighbor_prediction], dim=1)
        outputs["prediction"] = self._state_normalizer.inverse(prediction)
        return outputs


def compute_plantf_training_loss(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    futures: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args: Namespace,
):
    """PlanTF-head counterpart of ``decoder.compute_training_loss``.

    Returns the same loss keys plus ``mode_cls_loss`` so ``train_epoch`` can
    combine them with the shared coefficients.
    """
    norm = args.state_normalizer

    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask  # [B, Pn, T]

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
    )  # [B, P, T, 4]
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)  # [B, P, 4]

    all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)
    all_gt[:, 1:][neighbor_mask] = 0.0

    merged_inputs = {**inputs, "gt_trajectories": all_gt}
    _, decoder_output = model(merged_inputs)

    trajectory = decoder_output["trajectory"]  # [B, K, T, 4], normalized
    probability = decoder_output["probability"]  # [B, K]
    neighbor_prediction = decoder_output["neighbor_prediction"]  # [B, Pn, T, 4]

    gt_norm = all_gt[:, :, 1:, :]  # [B, P, T, 4]
    ego_gt = gt_norm[:, 0]  # [B, T, 4]

    # Winner-takes-all mode selection on xy ADE in meters (the normalized x/y
    # scales differ, so rescale by std; the mean shift cancels in the diff).
    with torch.no_grad():
        ego_std_xy = norm.std[0][..., :2].to(trajectory.device)  # [1, 2]
        xy_diff = (trajectory[..., :2] - ego_gt[:, None, :, :2]) * ego_std_xy
        ade = torch.norm(xy_diff, dim=-1).mean(-1)  # [B, K]
        best_mode = ade.argmin(dim=-1)  # [B,]
    best_trajectory = trajectory[torch.arange(B, device=trajectory.device), best_mode]

    prediction = torch.cat([best_trajectory[:, None], neighbor_prediction], dim=1)  # [B, P, T, 4]

    loss_dict = loss_func(prediction, gt_norm)
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

    traj_loss = (
        args.coeff_position_lat_loss * position_lat_loss
        + args.coeff_position_lon_loss * position_lon_loss
        + args.coeff_heading_l2_loss * heading_l2_loss
    )  # [B, P, T]

    masked_prediction_loss = traj_loss[:, 1:, :][neighbors_future_valid]

    loss = {}

    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

    loss["ego_planning_loss"] = traj_loss[:, 0, : args.ego_prediction_horizon].mean()
    loss["mode_cls_loss"] = F.cross_entropy(probability, best_mode)

    # Compute ego edge points for penalty losses (best mode only)
    need_ego_edge = args.coeff_road_border_loss > 0 or args.coeff_neighbor_collision_loss > 0
    if need_ego_edge:
        ego_pred_world = best_trajectory * norm.std[0].to(trajectory.device) + norm.mean[0].to(
            trajectory.device
        )  # [B, T, 4]
        ego_edge_points = compute_ego_edge_points(
            ego_pred_world, inputs["ego_shape"], n_interp=args.road_border_n_interp
        )
        denorm_inputs = args.observation_normalizer.inverse(inputs)

    if args.coeff_road_border_loss > 0:
        rb_loss = compute_road_border_penalty(
            ego_edge_points,
            denorm_inputs["line_strings"],
            margin=args.road_border_margin,
        )  # [B, T]
        loss["road_border_loss"] = rb_loss.mean()
    else:
        loss["road_border_loss"] = torch.tensor(0.0, device=traj_loss.device)

    if args.coeff_neighbor_collision_loss > 0:
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
        loss["neighbor_collision_loss"] = torch.tensor(0.0, device=traj_loss.device)

    assert not torch.isnan(traj_loss).sum(), "loss cannot be nan"

    turn_indicator_logit = decoder_output["turn_indicator_logit"]
    turn_indicator_gt = make_turn_indicator_gt(inputs["turn_indicators"])  # [B,]
    turn_indicator_loss = nn.functional.cross_entropy(
        turn_indicator_logit, turn_indicator_gt, reduction="none"
    )
    turn_indicator_change = inputs["turn_indicators"][:, -2] != inputs["turn_indicators"][:, -1]
    turn_indicator_coeff = torch.where(turn_indicator_change, 1.0, 0.05)
    turn_indicator_loss = (turn_indicator_loss * turn_indicator_coeff).mean()
    loss["turn_indicator_loss"] = turn_indicator_loss

    independent_turn_indicator_logit = decoder_output["independent_turn_indicator_logit"]
    independent_turn_indicator_loss = nn.functional.cross_entropy(
        independent_turn_indicator_logit, turn_indicator_gt, reduction="none"
    )
    independent_turn_indicator_loss = (
        independent_turn_indicator_loss * turn_indicator_coeff
    ).mean()
    loss["independent_turn_indicator_loss"] = independent_turn_indicator_loss

    with torch.no_grad():
        turn_indicator_accuracy = (
            (turn_indicator_logit.argmax(dim=-1) == turn_indicator_gt).float().mean()
        )
        loss["turn_indicator_accuracy"] = turn_indicator_accuracy

    return loss
