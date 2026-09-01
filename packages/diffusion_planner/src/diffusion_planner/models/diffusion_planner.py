"""Conditional flow-matching planner model."""

from __future__ import annotations

import torch
from torch import nn

from diffusion_planner.data.dimensions import TRAJECTORY_DIM

from .decoder import TrajectoryDecoder
from .encoder import SceneEncoder
from .flow_matching import sample
from .turn_indicator import TurnIndicatorDecoder


class DiffusionPlanner(nn.Module):
    """Predict joint ego and neighbor trajectories with conditional flow matching."""

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
    ) -> None:
        super().__init__()
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
    def create_agent_pose(
        input_data: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Create current `[x, y, cos_yaw, sin_yaw]` poses with shape `(B, A, 4)`."""
        ego_pose = input_data["ego_agent_past"][:, -1, :TRAJECTORY_DIM].unsqueeze(1)
        neighbor_pose = input_data["neighbor_agents_past"][:, :, -1, :TRAJECTORY_DIM]
        return torch.cat((ego_pose, neighbor_pose), dim=1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        input_data: dict[str, torch.Tensor],
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict a clean trajectory and the next turn indicator.

        Args:
            x: Normalized flow-state trajectories with shape `(B, A, T, 4)`.
            x_mask: Invalid-agent mask with shape `(B, A)`.
            input_data: Batched planner input tensors, including traffic-light future.
            time: Flow times with shape `(B,)` or `(B, 1)`.

        Returns:
            Normalized clean trajectories `(B, A, T, 4)` and turn-indicator
            logits `(B, 3)`.
        """
        scene, scene_mask = self.scene_encoder(input_data)
        agent_pose = self.create_agent_pose(input_data)
        trajectory = self.trajectory_decoder(
            x, x_mask, scene, scene_mask, agent_pose, time
        )
        turn_indicator_logits = self.turn_indicator_decoder(
            scene.detach(),
            scene_mask,
            input_data["turn_indicators"][:, -1],
            input_data["ego_agent_future"][..., :TRAJECTORY_DIM],
        )
        return trajectory, turn_indicator_logits

    @torch.no_grad()
    def sample(
        self,
        input_data: dict[str, torch.Tensor],
        initial_noise: torch.Tensor,
        num_steps: int = 20,
        time_epsilon: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate trajectories and predict the next turn indicator.

        `input_data` must already contain training ground-truth or inference-time
        heuristic traffic-light future tensors. `initial_noise` has shape
        `(B, A, T, 4)` and completely determines the initial flow state.
        """
        scene, scene_mask = self.scene_encoder(input_data)
        agent_pose = self.create_agent_pose(input_data)
        neighbor_mask = input_data["neighbor_agents_past"].abs().sum(dim=(-2, -1)) == 0
        ego_mask = torch.zeros(
            neighbor_mask.shape[0],
            1,
            dtype=torch.bool,
            device=neighbor_mask.device,
        )
        agent_mask = torch.cat((ego_mask, neighbor_mask), dim=1)
        trajectory = sample(
            x0_model=lambda state, time: self.trajectory_decoder(
                state, agent_mask, scene, scene_mask, agent_pose, time
            ),
            initial_state=initial_noise,
            num_steps=num_steps,
            epsilon=time_epsilon,
            project_state=lambda state: state.masked_fill(
                agent_mask[:, :, None, None], 0.0
            ),
        )
        yaw = trajectory[..., 2:4]
        yaw = yaw / torch.linalg.vector_norm(yaw, dim=-1, keepdim=True).clamp_min(1e-6)
        trajectory = torch.cat((trajectory[..., :2], yaw), dim=-1)
        trajectory = trajectory.masked_fill(agent_mask[:, :, None, None], 0.0)
        turn_indicator_logits = self.turn_indicator_decoder(
            scene,
            scene_mask,
            input_data["turn_indicators"][:, -1],
            trajectory[:, 0],
        )
        return trajectory, turn_indicator_logits
