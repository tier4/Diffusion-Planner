"""Tests for Muon and auxiliary AdamW construction."""

from __future__ import annotations

import unittest

import torch

from diffusion_planner.models.diffusion_planner import DiffusionPlanner
from diffusion_planner.utils.optimizer import build_optimizer, classify_params


def _model() -> DiffusionPlanner:
    return DiffusionPlanner(
        hidden_dim=16,
        num_heads=4,
        scene_fusion_depth=1,
        element_encoder_depth=1,
        decoder_depth=1,
        trajectory_encoder_depth=1,
        feedforward_dim=32,
        element_mixer_hidden_dim=8,
    )


class OptimizerTest(unittest.TestCase):
    def test_linear_embedding_modules_use_muon(self) -> None:
        model = _model()
        groups = classify_params(
            model, (model.trajectory_decoder.output_projection.fc2,)
        )
        muon_names = {name for name, _ in groups["muon"]}

        self.assertIn("trajectory_decoder.ego_pose_embedding.weight", muon_names)
        self.assertIn("trajectory_decoder.time_embedding.mlp.fc1.weight", muon_names)
        self.assertIn("trajectory_decoder.time_embedding.mlp.fc2.weight", muon_names)

    def test_bare_embeddings_use_no_decay_adamw(self) -> None:
        model = _model()
        groups = classify_params(
            model, (model.trajectory_decoder.output_projection.fc2,)
        )
        no_decay_names = {name for name, _ in groups["adamw_no_decay"]}

        self.assertIn("scene_encoder.lane_encoder.element_embedding", no_decay_names)
        self.assertIn(
            "scene_encoder.neighbor_agent_encoder.element_embedding", no_decay_names
        )

    def test_explicit_output_layer_uses_decayed_adamw(self) -> None:
        model = _model()
        groups = classify_params(
            model, (model.trajectory_decoder.output_projection.fc2,)
        )
        decay_names = {name for name, _ in groups["adamw_decay"]}

        self.assertIn("trajectory_decoder.output_projection.fc2.weight", decay_names)

    def test_wrapper_is_optimizer_and_restores_inner_state(self) -> None:
        model = _model()
        output_layers = (model.trajectory_decoder.output_projection.fc2,)
        optimizer = build_optimizer(
            model,
            output_layers=output_layers,
            learning_rate=1e-4,
            weight_decay=0.01,
        )
        self.assertIsInstance(optimizer, torch.optim.Optimizer)

        loss = sum(parameter.square().sum() for parameter in model.parameters())
        loss.backward()
        optimizer.step()
        state = optimizer.state_dict()

        restored = build_optimizer(
            model,
            output_layers=output_layers,
            learning_rate=1e-4,
            weight_decay=0.01,
        )
        restored.load_state_dict(state)
        self.assertGreater(len(restored.state), 0)


if __name__ == "__main__":
    unittest.main()
