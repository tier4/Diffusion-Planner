"""Tests for the PLUTO decoder, planner, and loss."""

from __future__ import annotations

import unittest

import torch

from diffusion_planner.data.dimensions import (
    MAX_NUM_NEIGHBORS,
    NUM_ROUTE_SEGMENTS,
    PLANNER_INPUT_SHAPES,
    TRAJECTORY_DIM,
    TRAJECTORY_LENGTH,
)
from diffusion_planner.models.pluto_decoder import (
    NUM_SCENE_TOKENS,
    ROUTE_TOKEN_START,
    PlutoDecoder,
    PreferredRouteQuery,
    select_candidate,
)
from diffusion_planner.models.pluto_loss import (
    assign_mode_targets,
    compute_pluto_loss,
    project_progress,
)
from diffusion_planner.models.pluto_planner import PlutoPlanner


def _random_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    batch = {
        name: torch.randn(batch_size, *shape, generator=generator)
        for name, shape in PLANNER_INPUT_SHAPES.items()
    }
    batch["turn_indicators"] = torch.randint(
        0, 4, (batch_size, 31), generator=generator
    ).float()
    batch["turn_indicators_future"] = torch.randint(
        1, 4, (batch_size, TRAJECTORY_LENGTH), generator=generator
    ).float()
    batch["ego_agent_future"] = torch.randn(
        batch_size, TRAJECTORY_LENGTH, 6, generator=generator
    )
    batch["neighbor_agents_future"] = torch.randn(
        batch_size,
        MAX_NUM_NEIGHBORS,
        TRAJECTORY_LENGTH,
        TRAJECTORY_DIM,
        generator=generator,
    )
    # Second sample: no neighbors, no route, mostly padded map.
    batch["neighbor_agents_past"][1] = 0.0
    batch["neighbor_agents_future"][1] = 0.0
    batch["route_lanes"][1] = 0.0
    return batch


class PlutoDecoderTest(unittest.TestCase):
    def test_shapes_and_backward_with_padded_lines(self) -> None:
        decoder = PlutoDecoder(
            hidden_dim=16, num_heads=4, depth=2, feedforward_dim=32, num_modes=3
        )
        line_queries = torch.randn(2, 4, 16, requires_grad=True)
        line_padding = torch.tensor(
            [[False, False, True, True], [True, True, True, True]]
        )
        scene = torch.randn(2, 6, 16)
        scene_mask = torch.tensor(
            [[False] * 6, [False, False, False, True, True, True]]
        )

        candidates, logits = decoder(line_queries, line_padding, scene, scene_mask)

        self.assertEqual(candidates.shape, (2, 4, 3, TRAJECTORY_LENGTH, 4))
        self.assertEqual(logits.shape, (2, 4, 3))
        self.assertTrue(torch.isfinite(candidates).all())
        self.assertTrue(torch.isfinite(logits).all())
        (candidates.sum() + logits.sum()).backward()
        self.assertIsNotNone(line_queries.grad)

        best, index = select_candidate(candidates, logits, line_padding)
        self.assertEqual(best.shape, (2, TRAJECTORY_LENGTH, 4))
        self.assertLess(int(index[0]), 2 * 3)  # never a padded line

    def test_single_line_skips_line_attention(self) -> None:
        decoder = PlutoDecoder(
            hidden_dim=16, num_heads=4, depth=1, feedforward_dim=32, num_modes=5
        )
        candidates, logits = decoder(
            torch.randn(3, 1, 16),
            torch.zeros(3, 1, dtype=torch.bool),
            torch.randn(3, 7, 16),
            torch.zeros(3, 7, dtype=torch.bool),
        )
        self.assertEqual(candidates.shape, (3, 1, 5, TRAJECTORY_LENGTH, 4))
        self.assertEqual(logits.shape, (3, 1, 5))


class PreferredRouteQueryTest(unittest.TestCase):
    def test_pools_route_tokens_and_flags_missing_route(self) -> None:
        query = PreferredRouteQuery(hidden_dim=8)
        scene = torch.randn(2, NUM_SCENE_TOKENS, 8)
        scene_mask = torch.zeros(2, NUM_SCENE_TOKENS, dtype=torch.bool)
        scene_mask[1, ROUTE_TOKEN_START : ROUTE_TOKEN_START + NUM_ROUTE_SEGMENTS] = True
        route_lanes = torch.randn(2, *PLANNER_INPUT_SHAPES["route_lanes"])
        route_lanes[1] = 0.0

        queries, padding = query(scene, scene_mask, route_lanes)

        self.assertEqual(queries.shape, (2, 1, 8))
        self.assertEqual(padding.tolist(), [[False], [True]])
        self.assertTrue(torch.all(queries[1] == 0.0))


class ProgressTargetTest(unittest.TestCase):
    def test_progress_along_straight_route(self) -> None:
        # Route: 25 segments of 20 points along +x, 1 m spacing in meters (0.02 normalized).
        spacing = 1.0 / 50.0
        route_lanes = torch.zeros(1, *PLANNER_INPUT_SHAPES["route_lanes"])
        x = torch.arange(NUM_ROUTE_SEGMENTS * 20, dtype=torch.float32) * spacing
        route_lanes[0, :, :, 0] = x.reshape(NUM_ROUTE_SEGMENTS, 20)
        route_lanes[0, :, :, 2] = 1.0  # any non-zero channel keeps the segment valid
        route_lanes[0, 10:] = 0.0  # only the first 10 segments (200 m) are valid

        ego_future = torch.zeros(1, TRAJECTORY_LENGTH, 2)
        ego_future[0, :, 0] = torch.linspace(0.0, 57.5 / 50.0, TRAJECTORY_LENGTH)
        ego_future[0, :, 1] = 0.3 / 50.0  # 30 cm lateral offset

        target, progress_m = assign_mode_targets(
            ego_future,
            route_lanes,
            position_scale=50.0,
            num_modes=12,
            mode_interval_m=10.0,
        )
        self.assertAlmostEqual(float(progress_m[0]), 57.5, places=3)
        self.assertEqual(int(target[0]), 5)

        edges = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 105.0]
        target_edges, _ = assign_mode_targets(
            ego_future,
            route_lanes,
            position_scale=50.0,
            num_modes=12,
            mode_interval_m=10.0,
            bin_edges_m=edges,
        )
        self.assertEqual(int(target_edges[0]), 6)

    def test_missing_route_falls_back_to_path_length(self) -> None:
        points = torch.zeros(1, 4, 2)
        valid = torch.zeros(1, 4, dtype=torch.bool)
        progress, has_route = project_progress(points, valid, torch.ones(1, 2))
        self.assertFalse(bool(has_route[0]))
        self.assertTrue(torch.isfinite(progress).all())

        ego_future = torch.zeros(1, TRAJECTORY_LENGTH, 2)
        ego_future[0, :, 0] = torch.linspace(0.0, 1.0, TRAJECTORY_LENGTH)  # 50 m
        target, progress_m = assign_mode_targets(
            ego_future,
            torch.zeros(1, *PLANNER_INPUT_SHAPES["route_lanes"]),
            position_scale=50.0,
            num_modes=12,
            mode_interval_m=10.0,
        )
        self.assertAlmostEqual(float(progress_m[0]), 50.0, places=3)
        self.assertEqual(int(target[0]), 5)


class PlutoPlannerTest(unittest.TestCase):
    def _planner(self) -> PlutoPlanner:
        return PlutoPlanner(
            hidden_dim=16,
            num_heads=4,
            scene_fusion_depth=1,
            element_encoder_depth=1,
            element_mixer_hidden_dim=8,
            decoder_depth=1,
            feedforward_dim=32,
            num_modes=3,
            trajectory_encoder_depth=1,
            trajectory_mixer_hidden_dim=8,
        )

    def test_loss_and_sample(self) -> None:
        planner = self._planner()
        batch = _random_batch()

        losses = compute_pluto_loss(planner, batch, num_modes=3)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertEqual(losses["mode_target_counts"].shape, (3,))
        grads = [
            p.grad for p in planner.pluto_decoder.parameters() if p.grad is not None
        ]
        self.assertTrue(grads)

        planner.eval()
        # The yaw head is zero-initialized; perturb it so the unit-circle check is meaningful.
        with torch.no_grad():
            planner.pluto_decoder.yaw_head[-1].bias.normal_()
        trajectory, turn_logits = planner.sample(batch)
        self.assertEqual(
            trajectory.shape,
            (2, MAX_NUM_NEIGHBORS + 1, TRAJECTORY_LENGTH, TRAJECTORY_DIM),
        )
        self.assertEqual(turn_logits.shape, (2, 3))
        self.assertTrue(torch.isfinite(trajectory).all())
        yaw_norm = torch.linalg.vector_norm(trajectory[:, 0, :, 2:4], dim=-1)
        self.assertTrue(torch.allclose(yaw_norm, torch.ones_like(yaw_norm), atol=1e-5))
        self.assertTrue(
            torch.all(trajectory[1, 1:] == 0.0)
        )  # padded neighbors stay zero
        self.assertEqual(len(planner.output_layers()), 6)


if __name__ == "__main__":
    unittest.main()
