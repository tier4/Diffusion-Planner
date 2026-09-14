"""Tests for the diffusion planner training loss."""

from __future__ import annotations

import unittest

import torch

from diffusion_planner.data.dimensions import AGENT_LABEL_DIM
from diffusion_planner.models.loss import (
    build_agent_loss_weights,
    trajectory_huber_loss,
)


class AgentLossWeightsTest(unittest.TestCase):
    """Unknown agents can be down-weighted in the trajectory loss but not in the encoder."""

    def _label(
        self, batch: int, neighbours: int, unknown_rows: list[int]
    ) -> torch.Tensor:
        label = torch.zeros(batch, neighbours, AGENT_LABEL_DIM)
        label[..., 0] = 1.0
        for row in unknown_rows:
            label[:, row] = 0.0
            label[:, row, AGENT_LABEL_DIM - 1] = 1.0
        return label

    def test_scale_of_one_matches_the_scalar_weights(self) -> None:
        like = torch.zeros(2, 5, 3, 4)
        weights = build_agent_loss_weights(
            self._label(2, 4, [1]),
            5,
            ego_loss_weight=100.0,
            neighbor_loss_weight=1.0,
            unknown_loss_scale=1.0,
            like=like,
        )
        self.assertTrue(torch.equal(weights[:, 0], torch.full((2,), 100.0)))
        self.assertTrue(torch.equal(weights[:, 1:], torch.ones(2, 4)))

    def test_unknown_neighbours_are_scaled_and_others_are_not(self) -> None:
        like = torch.zeros(2, 5, 3, 4)
        weights = build_agent_loss_weights(
            self._label(2, 4, [0, 2]),
            5,
            ego_loss_weight=100.0,
            neighbor_loss_weight=2.0,
            unknown_loss_scale=0.25,
            like=like,
        )
        # ego untouched, unknown neighbours at 2.0 * 0.25, known neighbours at 2.0
        self.assertTrue(torch.equal(weights[:, 0], torch.full((2,), 100.0)))
        expected = torch.tensor([0.5, 2.0, 0.5, 2.0]).expand(2, 4)
        self.assertTrue(torch.equal(weights[:, 1:], expected))

    def test_three_column_labels_are_left_alone(self) -> None:
        """Shards written before the unknown class have no column to weight."""
        like = torch.zeros(1, 3, 3, 4)
        weights = build_agent_loss_weights(
            torch.ones(1, 2, 3),
            3,
            ego_loss_weight=1.0,
            neighbor_loss_weight=1.0,
            unknown_loss_scale=0.0,
            like=like,
        )
        self.assertTrue(torch.equal(weights, torch.ones(1, 3)))

    def test_huber_loss_applies_the_per_agent_weights(self) -> None:
        prediction = torch.zeros(1, 3, 2, 4)
        target = torch.zeros(1, 3, 2, 4)
        target[..., 0] = 1.0
        target[..., 2] = 1.0
        time = torch.full((1,), 0.5)
        weights = torch.tensor([[10.0, 1.0, 0.0]])

        weighted = trajectory_huber_loss(
            prediction, target, time, 1e-5, agent_weights=weights
        )

        self.assertEqual(weighted.shape, (1, 3, 2, 4))
        self.assertEqual(float(weighted[0, 2].abs().sum()), 0.0)  # weight 0 silences it
        self.assertGreater(
            float(weighted[0, 0].abs().sum()), float(weighted[0, 1].abs().sum())
        )
