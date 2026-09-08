"""Tests for Muon and auxiliary AdamW construction."""

from __future__ import annotations

import unittest
from collections import Counter

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from diffusion_planner.models.diffusion_planner import DiffusionPlanner
from diffusion_planner.utils.optimizer import (
    BatchedMuon,
    build_optimizer,
    classify_params,
)

MATMUL_OPS = ("mm", "addmm", "bmm", "baddbmm")


class _OperatorCounter(TorchDispatchMode):
    """Count dispatched aten operators so launch counts can be asserted."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
        self.counts[func._schema.name.removeprefix("aten::")] += 1
        return func(*args, **(kwargs or {}))

    @property
    def matmuls(self) -> int:
        return sum(self.counts[name] for name in MATMUL_OPS)


def _matrices(shapes: tuple[tuple[int, int], ...]) -> list[torch.nn.Parameter]:
    """Build parameters with deterministic values and gradients."""
    generator = torch.Generator().manual_seed(0)
    matrices = []
    for index, shape in enumerate(shapes):
        parameter = torch.nn.Parameter(torch.randn(shape, generator=generator))
        parameter.grad = torch.randn(shape, generator=generator) * (1 + index % 3)
        matrices.append(parameter)
    return matrices


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

        self.assertIn("trajectory_decoder.agent_pose_embedding.weight", muon_names)
        self.assertIn("trajectory_decoder.time_embedding.mlp.fc1.weight", muon_names)
        self.assertIn("trajectory_decoder.time_embedding.mlp.fc2.weight", muon_names)

    def test_bare_embeddings_use_no_decay_adamw(self) -> None:
        model = _model()
        groups = classify_params(
            model, (model.trajectory_decoder.output_projection.fc2,)
        )
        no_decay_names = {name for name, _ in groups["adamw_no_decay"]}

        self.assertIn("trajectory_decoder.ego_embedding", no_decay_names)
        self.assertIn("trajectory_decoder.neighbor_embedding", no_decay_names)

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


class BatchedMuonTest(unittest.TestCase):
    """The batched update matches `torch.optim.Muon` with far fewer launches."""

    # Six shapes, three of them repeated, mirroring the planner's mix.
    SHAPES = (
        (32, 64),
        (32, 64),
        (32, 64),
        (64, 32),
        (64, 32),
        (16, 16),
        (48, 12),
        (12, 48),
    )

    def _run(
        self, optimizer_class: type[torch.optim.Optimizer], steps: int = 5, **options
    ) -> list[torch.Tensor]:
        torch.manual_seed(0)
        parameters = _matrices(self.SHAPES)
        settings = {
            "lr": 1e-2,
            "weight_decay": 0.01,
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 5,
            "eps": 1e-7,
            "adjust_lr_fn": "match_rms_adamw",
            **options,
        }
        optimizer = optimizer_class(parameters, **settings)
        generator = torch.Generator().manual_seed(7)
        for _ in range(steps):
            optimizer.step()
            for parameter in parameters:
                parameter.grad = torch.randn(
                    parameter.shape, generator=generator, dtype=parameter.dtype
                )
        return [parameter.detach().clone() for parameter in parameters]

    def test_matches_reference_muon_across_steps(self) -> None:
        expected = self._run(torch.optim.Muon)
        actual = self._run(BatchedMuon)

        for index, (reference, batched) in enumerate(
            zip(expected, actual, strict=True)
        ):
            # The iteration runs in bfloat16 in both, so batched matmuls differ
            # in the last bits; the scale here is that of bfloat16 rounding.
            difference = (reference - batched).abs().max().item()
            scale = reference.abs().max().item()
            self.assertLess(
                difference,
                5e-3 * scale,
                f"parameter {index} with shape {tuple(reference.shape)} diverged",
            )

    def test_matches_reference_muon_without_nesterov(self) -> None:
        expected = self._run(torch.optim.Muon, nesterov=False)
        actual = self._run(BatchedMuon, nesterov=False)

        for reference, batched in zip(expected, actual, strict=True):
            self.assertLess(
                (reference - batched).abs().max().item(),
                5e-3 * reference.abs().max().item(),
            )

    def test_issues_far_fewer_matmuls_than_reference_muon(self) -> None:
        counts = {}
        for name, optimizer_class in (
            ("reference", torch.optim.Muon),
            ("batched", BatchedMuon),
        ):
            parameters = _matrices(self.SHAPES)
            optimizer = optimizer_class(parameters, lr=1e-2, ns_steps=5)
            with _OperatorCounter() as counter:
                optimizer.step()
            counts[name] = counter.matmuls

        # One matrix at a time costs 3 matmuls per iteration; batching by shape
        # pays that once for each of the 5 distinct shapes here.
        self.assertEqual(counts["reference"], 3 * 5 * len(self.SHAPES))
        self.assertEqual(counts["batched"], 3 * 5 * 5)

    def test_state_is_interchangeable_with_reference_muon(self) -> None:
        parameters = _matrices(self.SHAPES)
        reference = torch.optim.Muon(
            parameters, lr=1e-2, adjust_lr_fn="match_rms_adamw"
        )
        reference.step()
        self.assertEqual(
            {key for state in reference.state.values() for key in state},
            {"momentum_buffer"},
        )

        batched = BatchedMuon(parameters, lr=1e-2, adjust_lr_fn="match_rms_adamw")
        batched.load_state_dict(reference.state_dict())
        for parameter in parameters:
            torch.testing.assert_close(
                batched.state[parameter]["momentum_buffer"],
                reference.state[parameter]["momentum_buffer"],
            )

    def test_skips_parameters_without_gradients(self) -> None:
        parameters = _matrices(((16, 16), (16, 16)))
        parameters[1].grad = None
        before = parameters[1].detach().clone()
        optimizer = BatchedMuon(parameters, lr=1e-2, weight_decay=0.1)

        optimizer.step()

        torch.testing.assert_close(parameters[1].detach(), before)
        self.assertNotIn(parameters[1], optimizer.state)

    def test_rejects_non_matrix_parameters(self) -> None:
        with self.assertRaises(ValueError):
            BatchedMuon([torch.nn.Parameter(torch.zeros(4))], lr=1e-2)


if __name__ == "__main__":
    unittest.main()
