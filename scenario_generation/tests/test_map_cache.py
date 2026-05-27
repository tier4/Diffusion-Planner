"""Tests for MapTensorCache in tensor_converter.py."""

from __future__ import annotations

import numpy as np

from scenario_generation.tensor_converter import (
    _NUM_LANES,
    _NUM_LINE_STRINGS,
    _NUM_POLYGONS,
    _NUM_STATIC,
    _POINTS_PER_LANELET,
    _POINTS_PER_LINE_STRING,
    _POINTS_PER_POLYGON,
    _SEGMENT_POINT_DIM,
    MapTensorCache,
    _build_lanes,
    _build_line_strings,
    _build_polygons,
    _build_static_objects,
    dump_step_npz,
    to_model_tensors,
)
from scenario_generation.transforms import _rotation_matrix


class TestMapTensorCache:
    def test_cache_lanes_matches_uncached(self, synthetic_scene):
        cache = MapTensorCache(synthetic_scene.map_data)
        R = _rotation_matrix(0.3)
        ego_xy = np.array([5.0, 2.0], dtype=np.float64)

        cached = cache.get_lanes_ego(R, ego_xy)
        uncached = _build_lanes(synthetic_scene.map_data.lanes, R, ego_xy, _NUM_LANES)

        assert cached.shape == uncached.shape
        np.testing.assert_allclose(cached, uncached, atol=1e-6)

    def test_cache_static_objects_matches_uncached(self, synthetic_scene):
        cache = MapTensorCache(synthetic_scene.map_data)
        R = _rotation_matrix(-0.5)
        ego_xy = np.array([1.0, -3.0], dtype=np.float64)

        cached = cache.get_static_objects_ego(R, ego_xy)
        uncached = _build_static_objects(synthetic_scene.map_data.static_objects, R, ego_xy)

        np.testing.assert_allclose(cached, uncached, atol=1e-6)

    def test_cache_polygons_matches_uncached(self, synthetic_scene):
        cache = MapTensorCache(synthetic_scene.map_data)
        R = _rotation_matrix(1.2)
        ego_xy = np.array([10.0, 5.0], dtype=np.float64)

        cached = cache.get_polygons_ego(R, ego_xy)
        uncached = _build_polygons(synthetic_scene.map_data.polygons, R, ego_xy)

        np.testing.assert_allclose(cached, uncached, atol=1e-6)

    def test_cache_line_strings_matches_uncached(self, synthetic_scene):
        cache = MapTensorCache(synthetic_scene.map_data)
        R = _rotation_matrix(0.0)
        ego_xy = np.array([0.0, 0.0], dtype=np.float64)

        cached = cache.get_line_strings_ego(R, ego_xy)
        uncached = _build_line_strings(synthetic_scene.map_data.line_strings, R, ego_xy)

        np.testing.assert_allclose(cached, uncached, atol=1e-6)

    def test_reuse_different_agents(self, synthetic_scene):
        """Same cache, two different ego frames produce different results."""
        cache = MapTensorCache(synthetic_scene.map_data)

        R1 = _rotation_matrix(0.0)
        xy1 = np.array([0.0, 0.0], dtype=np.float64)
        R2 = _rotation_matrix(1.0)
        xy2 = np.array([10.0, 5.0], dtype=np.float64)

        lanes1 = cache.get_lanes_ego(R1, xy1)
        lanes2 = cache.get_lanes_ego(R2, xy2)

        # Non-zero lanes should differ
        assert not np.allclose(lanes1, lanes2)

    def test_preserves_mask(self, synthetic_scene):
        """Zero-padded entries remain zero after ego transform."""
        cache = MapTensorCache(synthetic_scene.map_data)
        R = _rotation_matrix(0.7)
        ego_xy = np.array([3.0, -1.0], dtype=np.float64)

        lanes = cache.get_lanes_ego(R, ego_xy)

        # The fixture only populates lanes 0-4; lanes 5+ should be all zeros
        assert np.all(lanes[0, 5:] == 0.0)

    def test_speed_limit_passthrough(self, synthetic_scene):
        """Speed limits are returned unchanged regardless of agent pose."""
        cache = MapTensorCache(synthetic_scene.map_data)

        sl = cache.lanes_speed_limit
        hsl = cache.lanes_has_speed_limit

        assert sl.shape == (1, _NUM_LANES, 1)
        assert hsl.shape == (1, _NUM_LANES, 1)
        np.testing.assert_allclose(sl[0, 0, 0], 8.33, atol=1e-2)

    def test_full_to_model_tensors_cached_vs_uncached(self, synthetic_scene):
        """End-to-end: to_model_tensors with and without cache match."""
        from unittest.mock import MagicMock
        args = MagicMock()
        args.predicted_neighbor_num = 5
        args.future_len = 80
        args.observation_normalizer = lambda x: x

        cache = MapTensorCache(synthetic_scene.map_data)

        cached = to_model_tensors(synthetic_scene, "ego", args, "cpu", map_cache=cache)
        uncached = to_model_tensors(synthetic_scene, "ego", args, "cpu", map_cache=None)

        for key in ["lanes", "static_objects", "polygons", "line_strings",
                     "lanes_speed_limit", "lanes_has_speed_limit"]:
            np.testing.assert_allclose(
                cached[key].numpy(), uncached[key].numpy(), atol=1e-6,
                err_msg=f"Mismatch for {key}",
            )


class TestInferenceDelay:
    """``to_model_tensors`` threads ``inference_delay`` into the ``delay`` tensor."""

    def _args(self):
        from unittest.mock import MagicMock
        args = MagicMock()
        args.predicted_neighbor_num = 5
        args.future_len = 80
        args.observation_normalizer = lambda x: x
        return args

    def test_delay_defaults_to_zero(self, synthetic_scene):
        import torch
        out = to_model_tensors(synthetic_scene, "ego", self._args(), "cpu")
        assert "delay" in out
        assert out["delay"].dtype == torch.long
        assert out["delay"].shape == (1,)
        assert int(out["delay"].item()) == 0

    def test_delay_matches_inference_delay(self, synthetic_scene):
        import torch
        out = to_model_tensors(
            synthetic_scene, "ego", self._args(), "cpu", inference_delay=7,
        )
        assert out["delay"].dtype == torch.long
        assert out["delay"].shape == (1,)
        assert int(out["delay"].item()) == 7

    def test_delay_unaffected_by_map_cache(self, synthetic_scene):
        """Cache path shouldn't mutate delay — it must match the configured value."""
        cache = MapTensorCache(synthetic_scene.map_data)
        cached = to_model_tensors(
            synthetic_scene, "ego", self._args(), "cpu",
            map_cache=cache, inference_delay=3,
        )
        uncached = to_model_tensors(
            synthetic_scene, "ego", self._args(), "cpu",
            map_cache=None, inference_delay=3,
        )
        assert int(cached["delay"].item()) == 3
        assert int(uncached["delay"].item()) == 3


class TestDumpStepNPZ:
    """dump_step_npz produces un-normalised per-step arrays in the shape the
    training NPZ loader expects."""

    def _dump(self, scene, future_len=80, predicted_neighbor_num=32):
        cache = MapTensorCache(scene.map_data)
        return dump_step_npz(
            scene, cache,
            future_len=future_len,
            predicted_neighbor_num=predicted_neighbor_num,
        )

    def test_has_all_expected_keys(self, synthetic_scene):
        data = self._dump(synthetic_scene)
        expected = {
            "ego_agent_past", "ego_current_state", "neighbor_agents_past",
            "static_objects", "lanes", "lanes_speed_limit",
            "lanes_has_speed_limit", "polygons", "line_strings",
            "route_lanes", "route_lanes_speed_limit", "route_lanes_has_speed_limit",
            "goal_pose", "ego_shape", "turn_indicators",
            "ego_agent_future", "neighbor_agents_future", "version",
        }
        missing = expected - set(data.keys())
        assert not missing, f"dump missing keys: {missing}"

    def test_shapes_and_dtypes_match_loader(self, synthetic_scene):
        data = self._dump(synthetic_scene, future_len=80, predicted_neighbor_num=32)
        assert data["ego_agent_future"].shape == (80, 3)
        assert data["neighbor_agents_future"].shape == (_MAX_NUM_NEIGHBORS, 80, 4)
        assert data["ego_agent_future"].dtype == np.float32
        assert data["neighbor_agents_future"].dtype == np.float32
        # has_speed_limit fields must be bool (training loader expects it).
        assert data["lanes_has_speed_limit"].dtype == bool
        assert data["route_lanes_has_speed_limit"].dtype == bool
        assert data["version"].dtype == np.int64
        assert int(data["version"]) == 1

    def test_future_placeholders_are_zero(self, synthetic_scene):
        data = self._dump(synthetic_scene)
        # GT-future is always zero-filled — ranked-SFT generates its own.
        assert np.all(data["ego_agent_future"] == 0)
        assert np.all(data["neighbor_agents_future"] == 0)

    def test_ego_past_has_heading_rad_not_cos_sin(self, synthetic_scene):
        """ego_agent_past ends up as (x, y, heading_rad) so the training NPZ
        loader's heading_to_cos_sin expansion does the right thing at load."""
        data = self._dump(synthetic_scene)
        assert data["ego_agent_past"].shape[-1] == 3
        h = data["ego_agent_past"][..., 2]
        assert np.all(np.isfinite(h))
        assert np.all(np.abs(h) <= np.pi + 1e-5)

    def test_goal_pose_has_heading_rad(self, synthetic_scene):
        data = self._dump(synthetic_scene)
        assert data["goal_pose"].shape == (3,)
        assert data["goal_pose"].dtype == np.float32
        assert abs(float(data["goal_pose"][2])) <= np.pi + 1e-5

    def test_custom_future_len_propagates(self, synthetic_scene):
        """Caller-provided future_len flows through to both ego and neighbor futures."""
        data = self._dump(synthetic_scene, future_len=40)
        assert data["ego_agent_future"].shape == (40, 3)
        # Neighbor count is locked at _MAX_NUM_NEIGHBORS (past and future must match)
        from scenario_generation.tensor_converter import _MAX_NUM_NEIGHBORS
        assert data["neighbor_agents_future"].shape == (_MAX_NUM_NEIGHBORS, 40, 4)

    def test_mismatched_neighbor_count_raises(self, synthetic_scene):
        """predicted_neighbor_num must equal _MAX_NUM_NEIGHBORS (past is fixed)."""
        import pytest
        cache = MapTensorCache(synthetic_scene.map_data)
        with pytest.raises(ValueError, match="predicted_neighbor_num"):
            dump_step_npz(
                synthetic_scene, cache,
                future_len=80, predicted_neighbor_num=16,
            )
