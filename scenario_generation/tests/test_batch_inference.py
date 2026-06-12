"""Tests for batched model inference in simulate.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from scenario_generation.simulate import _cat_tensor_dicts, _predict_batch


class TestCatTensorDicts:
    def test_shapes(self):
        d1 = {"a": torch.randn(1, 4, 3), "b": torch.randn(1, 10)}
        d2 = {"a": torch.randn(1, 4, 3), "b": torch.randn(1, 10)}
        d3 = {"a": torch.randn(1, 4, 3), "b": torch.randn(1, 10)}
        out = _cat_tensor_dicts([d1, d2, d3])
        assert out["a"].shape == (3, 4, 3)
        assert out["b"].shape == (3, 10)

    def test_preserves_dtypes(self):
        d1 = {
            "float": torch.randn(1, 5),
            "bool": torch.tensor([[True, False]]),
            "long": torch.tensor([[1, 2, 3]]),
        }
        d2 = {
            "float": torch.randn(1, 5),
            "bool": torch.tensor([[False, True]]),
            "long": torch.tensor([[4, 5, 6]]),
        }
        out = _cat_tensor_dicts([d1, d2])
        assert out["float"].dtype == torch.float32
        assert out["bool"].dtype == torch.bool
        assert out["long"].dtype == torch.int64

    def test_values_preserved(self):
        d1 = {"x": torch.tensor([[1.0, 2.0]])}
        d2 = {"x": torch.tensor([[3.0, 4.0]])}
        out = _cat_tensor_dicts([d1, d2])
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        torch.testing.assert_close(out["x"], expected)

    def test_single_dict(self):
        d = {"a": torch.randn(1, 3)}
        out = _cat_tensor_dicts([d])
        assert out["a"].shape == (1, 3)
        torch.testing.assert_close(out["a"], d["a"])


class TestPredictBatch:
    @staticmethod
    def _make_mock_model(n_agents: int, n_neighbors: int = 5):
        """Create a mock model that returns deterministic predictions."""
        model = MagicMock()
        model.decoder = MagicMock()

        def forward(data: dict[str, torch.Tensor]):
            B = data["ego_agent_past"].shape[0]
            P = 1 + n_neighbors
            pred = torch.arange(B, dtype=torch.float32).view(B, 1, 1, 1).expand(B, P, 80, 4)
            return None, {"prediction": pred}

        model.side_effect = forward
        return model

    @staticmethod
    def _make_mock_model_args(n_neighbors: int = 5):
        args = MagicMock()
        args.predicted_neighbor_num = n_neighbors
        args.future_len = 80
        args.observation_normalizer = lambda x: x
        return args

    def test_empty_ids(self, synthetic_scene):
        model = self._make_mock_model(0)
        args = self._make_mock_model_args()
        result = _predict_batch(model, args, synthetic_scene, [], "cpu")
        assert result == {}

    def test_single_agent(self, synthetic_scene):
        model = self._make_mock_model(1)
        args = self._make_mock_model_args()
        result = _predict_batch(model, args, synthetic_scene, ["ego"], "cpu")
        assert "ego" in result
        assert result["ego"].shape == (80, 4)

    def test_multi_agent_keys(self, synthetic_scene):
        ids = ["ego", "nb_1", "nb_2"]
        model = self._make_mock_model(len(ids))
        args = self._make_mock_model_args()
        result = _predict_batch(model, args, synthetic_scene, ids, "cpu")
        assert set(result.keys()) == set(ids)
        for aid in ids:
            assert result[aid].shape == (80, 4)

    def test_batch_vs_sequential(self, synthetic_scene):
        """Verify batched results match sequential single-agent calls."""
        ids = ["ego", "nb_1", "nb_2"]

        def deterministic_forward(data):
            B = data["ego_agent_past"].shape[0]
            ego_x = data["ego_current_state"][:, 0]
            pred = ego_x.view(B, 1, 1, 1).expand(B, 6, 80, 4).clone()
            return None, {"prediction": pred}

        model = MagicMock()
        model.decoder = MagicMock()
        model.side_effect = deterministic_forward

        args = self._make_mock_model_args()

        sequential = {}
        for aid in ids:
            from scenario_generation.simulate import _predict_as_ego

            sequential[aid] = _predict_as_ego(model, args, synthetic_scene, aid, "cpu")

        batched = _predict_batch(model, args, synthetic_scene, ids, "cpu")

        for aid in ids:
            np.testing.assert_allclose(
                batched[aid],
                sequential[aid],
                atol=1e-5,
                err_msg=f"Mismatch for agent {aid}",
            )
