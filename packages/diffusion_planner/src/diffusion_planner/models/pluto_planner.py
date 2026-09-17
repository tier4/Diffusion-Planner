"""PLUTO-style one-shot planner that shares the diffusion planner's scene encoder."""

from __future__ import annotations

import torch
from torch import nn

from diffusion_planner.data.dimensions import TRAJECTORY_DIM

from .encoder import SceneEncoder
from .pluto_decoder import (
    EGO_TOKEN_INDEX,
    NeighborPredictor,
    PlutoDecoder,
    PreferredRouteQuery,
    select_candidate,
)
from .turn_indicator import TurnIndicatorDecoder


def normalize_yaw(trajectory: torch.Tensor) -> torch.Tensor:
    """Project the ``[cos, sin]`` channels onto the unit circle (zeros stay zero)."""
    yaw = trajectory[..., 2:4]
    yaw = yaw / torch.linalg.vector_norm(yaw, dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.cat((trajectory[..., :2], yaw), dim=-1)


class PlutoPlanner(nn.Module):
    """Predict ``R x M`` ego candidates and neighbor futures in one forward pass.

    The public surface mirrors :class:`DiffusionPlanner` (``sample``,
    ``predict_turn_indicator``) so the export script and the dashboard work
    with either model. ``sample`` accepts and ignores the flow-matching noise
    arguments.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        scene_fusion_depth: int = 4,
        element_encoder_depth: int = 2,
        element_mixer_hidden_dim: int = 128,
        decoder_depth: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        encoder_dropout: float = 0.0,
        num_modes: int = 12,
        cat_ego_token: bool = False,
        trajectory_encoder_depth: int = 2,
        trajectory_mixer_hidden_dim: int = 128,
        drop_path_rate: float = 0.0,
        velocity_threshold: float = 0.1,
        goal_max_distance: float = 2.0,
        agent_label_encoder: str | None = None,
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        # Only branches with the four-class agent label know this argument.
        encoder_options = (
            {}
            if agent_label_encoder is None
            else {"agent_label_encoder": agent_label_encoder}
        )
        self.scene_encoder = SceneEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            fusion_depth=scene_fusion_depth,
            encoder_depth=element_encoder_depth,
            drop_path_rate=drop_path_rate,
            dropout=encoder_dropout,
            mixer_hidden_dim=element_mixer_hidden_dim,
            velocity_threshold=velocity_threshold,
            goal_max_distance=goal_max_distance,
            **encoder_options,
        )
        self.route_query = PreferredRouteQuery(hidden_dim)
        self.pluto_decoder = PlutoDecoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            depth=decoder_depth,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            num_modes=num_modes,
            cat_ego_token=cat_ego_token,
        )
        self.neighbor_predictor = NeighborPredictor(hidden_dim)
        self.turn_indicator_decoder = TurnIndicatorDecoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=encoder_dropout,
            trajectory_encoder_depth=trajectory_encoder_depth,
            trajectory_mixer_hidden_dim=trajectory_mixer_hidden_dim,
        )

    def output_layers(self) -> tuple[nn.Module, ...]:
        """Head output layers that the optimizer keeps in AdamW."""
        return (
            self.pluto_decoder.location_head[-1],
            self.pluto_decoder.yaw_head[-1],
            self.pluto_decoder.probability_head[-1],
            self.neighbor_predictor.location_head[-1],
            self.neighbor_predictor.yaw_head[-1],
            self.turn_indicator_decoder.classifier,
        )

    def decode(
        self,
        input_data: dict[str, torch.Tensor],
        scene: torch.Tensor,
        scene_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the route query, the PLUTO decoder, and the neighbor predictor."""
        line_queries, line_padding = self.route_query(
            scene, scene_mask, input_data["route_lanes"]
        )
        ego_token = (
            scene[:, EGO_TOKEN_INDEX] if self.pluto_decoder.cat_ego_token else None
        )
        candidates, logits = self.pluto_decoder(
            line_queries, line_padding, scene, scene_mask, ego_token
        )
        neighbors = self.neighbor_predictor(scene, scene_mask)
        return {
            "candidates": candidates,
            "logits": logits,
            "line_padding": line_padding,
            "neighbors": neighbors,
        }

    def forward(self, input_data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Returns ``candidates (B, R, M, T, 4)``, ``logits (B, R, M)``,
        ``line_padding (B, R)``, ``neighbors (B, A, T, 4)`` (all normalized units)
        and ``turn_indicator_logits (B, 3)`` computed from the ground-truth ego
        future, as in :class:`DiffusionPlanner`.
        """
        scene, scene_mask = self.scene_encoder(input_data)
        outputs = self.decode(input_data, scene, scene_mask)
        outputs["turn_indicator_logits"] = self.turn_indicator_decoder(
            scene,
            scene_mask,
            input_data["turn_indicators"][:, -1],
            input_data["ego_agent_future"][..., :TRAJECTORY_DIM],
        )
        return outputs

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

    @torch.no_grad()
    def sample(
        self,
        input_data: dict[str, torch.Tensor],
        initial_noise: torch.Tensor | None = None,
        num_steps: int | None = None,
        time_epsilon: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plan once and return ``(B, 1 + A, T, 4)`` trajectories and turn logits.

        ``initial_noise``, ``num_steps`` and ``time_epsilon`` exist only for
        signature compatibility with :meth:`DiffusionPlanner.sample`; the
        decoder is deterministic.
        """
        del num_steps, time_epsilon
        scene, scene_mask = self.scene_encoder(input_data)
        outputs = self.decode(input_data, scene, scene_mask)
        ego, _ = select_candidate(
            outputs["candidates"], outputs["logits"], outputs["line_padding"]
        )
        ego = normalize_yaw(ego)
        neighbors = normalize_yaw(outputs["neighbors"])
        trajectory = torch.cat((ego.unsqueeze(1), neighbors), dim=1)
        if initial_noise is not None:
            # The ONNX exporter drops unused inputs, but the Autoware node always
            # feeds `initial_noise`. A shape-only dependency keeps it in the graph
            # without letting its values (possibly NaN) touch the output.
            trajectory = trajectory + torch.zeros_like(initial_noise[:, :1, :1, :1])
        turn_indicator_logits = self.turn_indicator_decoder(
            scene, scene_mask, input_data["turn_indicators"][:, -1], ego
        )
        return trajectory, turn_indicator_logits
