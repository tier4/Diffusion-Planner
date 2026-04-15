"""Tests for MapTensorCache in tensor_converter.py."""

from __future__ import annotations

import numpy as np

from scenario_generation.tensor_converter import (
    MapTensorCache,
    _build_lanes,
    _build_line_strings,
    _build_polygons,
    _build_static_objects,
    _NUM_LANES,
    _NUM_LINE_STRINGS,
    _NUM_POLYGONS,
    _NUM_STATIC,
    _POINTS_PER_LANELET,
    _POINTS_PER_LINE_STRING,
    _POINTS_PER_POLYGON,
    _SEGMENT_POINT_DIM,
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
        assert np.all(lanes[0, 10:] == 0.0)

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
