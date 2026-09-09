"""Conditional flow-matching planner model."""

from __future__ import annotations

import torch
from torch import nn

from diffusion_planner.data.dimensions import (
    EGO_VELOCITY_INDEX,
    TRAJECTORY_DIM,
)

from .control import ControlNormalizer, control_to_waypoints, denormalize_positions
from .decoder import TrajectoryDecoder
from .encoder import SceneEncoder
from .flow_matching import sample
from .turn_indicator import TurnIndicatorDecoder


class DiffusionPlanner(nn.Module):
    """Predict the ego trajectory with conditional flow matching.

    The model predicts ego control only. Neighbors are inputs, not outputs: they
    reach the model as scene tokens through the encoder, and the decoder carries
    no agent axis at all.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        scene_fusion_depth: int = 4,
        element_encoder_depth: int = 2,
        decoder_depth: int = 6,
        trajectory_encoder_depth: int = 2,
        trajectory_mixer_hidden_dim: int = 128,
        feedforward_dim: int = 1024,
        element_mixer_hidden_dim: int = 128,
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
        velocity_threshold: float = 0.1,
        goal_max_distance: float = 2.0,
        position_scale: float = 50.0,
        control_mean: tuple[float, float] = (-0.030108, -0.001032),
        control_std: tuple[float, float] = (2.177840, 0.033305),
    ) -> None:
        super().__init__()
        self.position_scale = position_scale
        self.control_normalizer = ControlNormalizer(
            list(control_mean), list(control_std)
        )
        self.scene_encoder = SceneEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            fusion_depth=scene_fusion_depth,
            encoder_depth=element_encoder_depth,
            drop_path_rate=drop_path_rate,
            dropout=dropout,
            mixer_hidden_dim=element_mixer_hidden_dim,
            velocity_threshold=velocity_threshold,
            goal_max_distance=goal_max_distance,
        )
        self.trajectory_decoder = TrajectoryDecoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            depth=decoder_depth,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            trajectory_encoder_depth=trajectory_encoder_depth,
            trajectory_mixer_hidden_dim=trajectory_mixer_hidden_dim,
        )
        self.turn_indicator_decoder = TurnIndicatorDecoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            trajectory_encoder_depth=trajectory_encoder_depth,
            trajectory_mixer_hidden_dim=trajectory_mixer_hidden_dim,
        )

    def predict_turn_indicator(
        self,
        input_data: dict[str, torch.Tensor],
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next-indicator logits from a predicted ego trajectory."""
        with torch.no_grad():
            scene, scene_mask = self.scene_encoder(input_data)
        return self.turn_indicator_decoder(
            scene.detach(),
            scene_mask,
            input_data["turn_indicators"][:, -1],
            trajectory,
        )

    @staticmethod
    def create_ego_pose(input_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the current `[x, y, cos_yaw, sin_yaw]` ego pose `(B, 4)`."""
        return input_data["ego_agent_past"][:, -1, :TRAJECTORY_DIM]

    def forward(
        self,
        x: torch.Tensor,
        input_data: dict[str, torch.Tensor],
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the clean ego control and the next turn indicator.

        Args:
            x: Normalized flow-state control with shape `(B, T, CONTROL_DIM)`.
            input_data: Batched planner input tensors, including traffic-light future.
            time: Flow times with shape `(B,)` or `(B, 1)`.

        Returns:
            Clean ego control `(B, T, CONTROL_DIM)` and turn-indicator logits `(B, 3)`.
        """
        scene, scene_mask = self.scene_encoder(input_data)
        ego_pose = self.create_ego_pose(input_data)
        control = self.trajectory_decoder(x, scene, scene_mask, ego_pose, time)
        turn_indicator_logits = self.turn_indicator_decoder(
            scene,
            scene_mask,
            input_data["turn_indicators"][:, -1],
            input_data["ego_agent_future"][..., :TRAJECTORY_DIM],
        )
        return control, turn_indicator_logits

    def ego_poses_from_control(
        self, control: torch.Tensor, input_data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Roll normalized ego control out into normalized ego poses.

        Args:
            control: `(B, T, 2)` ego control in `ControlNormalizer` space.
            input_data: Batched planner inputs, normalized as the model sees them.

        Returns:
            `(B, T, 4)` ego poses in the same normalized units as the pose head.
        """
        ego_past = input_data["ego_agent_past"]
        history = denormalize_positions(
            ego_past[..., :TRAJECTORY_DIM], self.position_scale
        )
        initial_speed = ego_past[:, -1, EGO_VELOCITY_INDEX]
        poses = control_to_waypoints(
            self.control_normalizer.inverse(control), history, initial_speed
        )
        return torch.cat((poses[..., :2] / self.position_scale, poses[..., 2:]), dim=-1)

    @torch.no_grad()
    def sample(
        self,
        input_data: dict[str, torch.Tensor],
        initial_noise: torch.Tensor,
        num_steps: int = 20,
        time_epsilon: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate the ego trajectory and predict the next turn indicator.

        `input_data` must already contain training ground-truth or inference-time
        heuristic traffic-light future tensors. `initial_noise` has shape
        `(B, T, CONTROL_DIM)` and completely determines the initial flow state.
        The returned trajectory is poses `(B, T, 4)`, reconstructed by integrating
        the sampled control through the unicycle model.
        """
        scene, scene_mask = self.scene_encoder(input_data)
        ego_pose = self.create_ego_pose(input_data)
        control = sample(
            x0_model=lambda state, time: self.trajectory_decoder(
                state, scene, scene_mask, ego_pose, time
            ),
            initial_state=initial_noise,
            num_steps=num_steps,
            epsilon=time_epsilon,
            # Nothing to constrain: every channel of the ego control is free, and
            # there is no agent axis whose padding would have to be held at zero.
            project_state=lambda state: state,
        )
        trajectory = self.ego_poses_from_control(control, input_data)
        yaw = trajectory[..., 2:4]
        yaw = yaw / torch.linalg.vector_norm(yaw, dim=-1, keepdim=True).clamp_min(1e-6)
        trajectory = torch.cat((trajectory[..., :2], yaw), dim=-1)
        turn_indicator_logits = self.turn_indicator_decoder(
            scene,
            scene_mask,
            input_data["turn_indicators"][:, -1],
            trajectory,
        )
        return trajectory, turn_indicator_logits
