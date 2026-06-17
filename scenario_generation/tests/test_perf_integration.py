"""Integration smoke tests and performance profiling for optimizations.

Profiles tensor conversion (with/without MapTensorCache) and arc-length
computation (loop vs vectorized) to measure speedups.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from scenario_generation.scene_context import Agent, AgentType, MapData, SceneContext
from scenario_generation.simulate import (
    _cat_tensor_dicts,
    _predict_batch,
    _save_and_close,
    advance_scene,
)
from scenario_generation.tensor_converter import (
    _NUM_LANES,
    MapTensorCache,
    _build_lanes,
    _build_line_strings,
    _build_polygons,
    _build_static_objects,
    to_model_tensors,
)
from scenario_generation.transforms import _rotation_matrix


def _make_large_scene(n_agents: int = 8) -> SceneContext:
    """Create a scene with many agents and dense map data for profiling."""
    T_past = 31
    agents = []
    for i in range(n_agents):
        angle = 2 * np.pi * i / n_agents
        x = 20 * np.cos(angle)
        y = 20 * np.sin(angle)
        traj = np.zeros((T_past, 3), dtype=np.float32)
        vels = np.zeros((T_past, 2), dtype=np.float32)
        for t in range(T_past):
            frac = t / (T_past - 1)
            traj[t] = [
                x - (1 - frac) * 3 * np.cos(angle),
                y - (1 - frac) * 3 * np.sin(angle),
                angle,
            ]
            vels[t] = [3 * np.cos(angle), 3 * np.sin(angle)]

        route = np.random.randn(25, 20, 33).astype(np.float32) * 0.1
        agents.append(
            Agent(
                id=f"agent_{i}",
                agent_type=AgentType.VEHICLE,
                length=4.5,
                width=1.8,
                wheelbase=2.9,
                past_trajectory=traj,
                past_velocities=vels,
                goal_pose=np.array(
                    [x + 50 * np.cos(angle), y + 50 * np.sin(angle), angle], dtype=np.float32
                ),
                route_lanes=route,
                route_speed_limit=np.zeros((25, 1), dtype=np.float32),
                route_has_speed_limit=np.zeros((25, 1), dtype=bool),
                turn_indicators=np.zeros(T_past, dtype=np.int32),
            )
        )

    # Dense map: populate all 140 lanes, 10 polygons, 60 line strings
    lanes = np.random.randn(140, 20, 33).astype(np.float32) * 10
    polygons = np.random.randn(10, 40, 3).astype(np.float32) * 10
    line_strings = np.random.randn(60, 20, 4).astype(np.float32) * 10
    static_objects = np.random.randn(5, 10).astype(np.float32) * 5

    return SceneContext(
        agents=agents,
        map_data=MapData(
            lanes=lanes,
            lanes_speed_limit=np.full((140, 1), 8.33, dtype=np.float32),
            lanes_has_speed_limit=np.ones((140, 1), dtype=bool),
            polygons=polygons,
            line_strings=line_strings,
            static_objects=static_objects,
        ),
        ego_agent_id="agent_0",
    )


@pytest.mark.benchmark
class TestMapCacheProfile:
    """Profile MapTensorCache vs uncached conversion."""

    def test_map_cache_speedup(self):
        scene = _make_large_scene(n_agents=8)
        args = MagicMock()
        args.predicted_neighbor_num = 5
        args.future_len = 80
        args.observation_normalizer = lambda x: x

        agent_ids = [a.id for a in scene.agents]
        n_warmup = 2
        n_iter = 20

        # Warmup
        for _ in range(n_warmup):
            for aid in agent_ids:
                to_model_tensors(scene, aid, args, "cpu")

        # Uncached
        t0 = time.perf_counter()
        for _ in range(n_iter):
            for aid in agent_ids:
                to_model_tensors(scene, aid, args, "cpu")
        uncached_time = time.perf_counter() - t0

        # Cached
        for _ in range(n_warmup):
            cache = MapTensorCache(scene.map_data)
            for aid in agent_ids:
                to_model_tensors(scene, aid, args, "cpu", map_cache=cache)

        t0 = time.perf_counter()
        for _ in range(n_iter):
            cache = MapTensorCache(scene.map_data)
            for aid in agent_ids:
                to_model_tensors(scene, aid, args, "cpu", map_cache=cache)
        cached_time = time.perf_counter() - t0

        speedup = uncached_time / max(cached_time, 1e-9)
        print(f"\n[PROFILE] MapTensorCache: {n_iter}x{len(agent_ids)} agents")
        print(f"  Uncached: {uncached_time:.3f}s")
        print(f"  Cached:   {cached_time:.3f}s")
        print(f"  Speedup:  {speedup:.2f}x")

        # Only log; no timing assertion (flaky in CI)


@pytest.mark.benchmark
class TestMapOnlyProfile:
    """Profile just the map tensor transforms (isolated from agent tensors)."""

    def test_map_transforms_speedup(self):
        scene = _make_large_scene(n_agents=8)
        R = _rotation_matrix(0.5)
        ego_xy = np.array([10.0, 5.0], dtype=np.float64)

        n_agents = 8
        n_iter = 50

        # Uncached: rebuild from scratch each time
        t0 = time.perf_counter()
        for _ in range(n_iter):
            for _ in range(n_agents):
                _build_lanes(scene.map_data.lanes, R, ego_xy, _NUM_LANES)
                _build_static_objects(scene.map_data.static_objects, R, ego_xy)
                _build_polygons(scene.map_data.polygons, R, ego_xy)
                _build_line_strings(scene.map_data.line_strings, R, ego_xy)
        uncached_time = time.perf_counter() - t0

        # Cached: create cache once per "step", reuse for all agents
        t0 = time.perf_counter()
        for _ in range(n_iter):
            cache = MapTensorCache(scene.map_data)
            for _ in range(n_agents):
                cache.get_lanes_ego(R, ego_xy)
                cache.get_static_objects_ego(R, ego_xy)
                cache.get_polygons_ego(R, ego_xy)
                cache.get_line_strings_ego(R, ego_xy)
        cached_time = time.perf_counter() - t0

        speedup = uncached_time / max(cached_time, 1e-9)
        print(f"\n[PROFILE] Map-only transforms ({n_agents} agents, {n_iter} iters)")
        print(f"  Uncached: {uncached_time:.3f}s")
        print(f"  Cached:   {cached_time:.3f}s")
        print(f"  Speedup:  {speedup:.2f}x")


@pytest.mark.benchmark
class TestArcLengthProfile:
    """Profile vectorized vs loop arc-length computation."""

    def test_arc_vectorized_speedup(self):
        rng = np.random.default_rng(42)
        pts = np.cumsum(rng.random((500, 2)), axis=0).astype(np.float64)

        n_iter = 1000

        # Loop version
        t0 = time.perf_counter()
        for _ in range(n_iter):
            arc = np.zeros(len(pts))
            for i in range(1, len(pts)):
                arc[i] = arc[i - 1] + np.linalg.norm(pts[i] - pts[i - 1])
        loop_time = time.perf_counter() - t0

        # Vectorized version
        t0 = time.perf_counter()
        for _ in range(n_iter):
            diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            arc_v = np.concatenate([[0.0], np.cumsum(diffs)])
        vec_time = time.perf_counter() - t0

        speedup = loop_time / max(vec_time, 1e-9)
        print(f"\n[PROFILE] Arc-length computation ({len(pts)} points, {n_iter} iters)")
        print(f"  Loop:       {loop_time:.3f}s")
        print(f"  Vectorized: {vec_time:.3f}s")
        print(f"  Speedup:    {speedup:.2f}x")

        np.testing.assert_allclose(arc, arc_v, atol=1e-10)


@pytest.mark.benchmark
class TestBatchInferenceProfile:
    """Profile batched vs sequential tensor dict concatenation."""

    def test_cat_tensor_dicts_scaling(self):
        n_agents = 8
        dicts = []
        for _ in range(n_agents):
            dicts.append(
                {
                    "ego_agent_past": torch.randn(1, 31, 4),
                    "neighbor_agents_past": torch.randn(1, 32, 31, 11),
                    "lanes": torch.randn(1, 140, 20, 33),
                    "route_lanes": torch.randn(1, 25, 20, 33),
                    "polygons": torch.randn(1, 10, 40, 3),
                    "line_strings": torch.randn(1, 60, 20, 4),
                    "static_objects": torch.randn(1, 5, 10),
                }
            )

        n_iter = 100

        t0 = time.perf_counter()
        for _ in range(n_iter):
            _cat_tensor_dicts(dicts)
        cat_time = time.perf_counter() - t0

        print(f"\n[PROFILE] _cat_tensor_dicts ({n_agents} agents, {n_iter} iters)")
        print(f"  Time: {cat_time:.3f}s ({cat_time / n_iter * 1000:.1f}ms/call)")


class TestFullSimulationSmoke:
    """End-to-end smoke test: build scene, run 2-step simulation with mock model."""

    def test_simulation_smoke(self, synthetic_scene, tmp_output_dir):
        def mock_forward(data):
            B = data["ego_agent_past"].shape[0]
            pred = torch.zeros(B, 6, 80, 4)
            # Small forward movement
            for b in range(B):
                pred[b, 0, :, 0] = torch.linspace(0, 5, 80)
                pred[b, 0, :, 2] = 1.0  # cos(0) = 1
            return None, {"prediction": pred}

        model = MagicMock()
        model.decoder = MagicMock()
        model.side_effect = mock_forward

        args = MagicMock()
        args.predicted_neighbor_num = 5
        args.future_len = 80
        args.observation_normalizer = lambda x: x

        from scenario_generation.simulate import run_simulation

        run_simulation(
            model,
            args,
            synthetic_scene,
            n_steps=2,
            output_dir=tmp_output_dir,
            device="cpu",
            per_agent=True,
            mode="closed_loop",
        )

        # Verify output structure
        overview_imgs = list(tmp_output_dir.glob("step_*.png"))
        assert len(overview_imgs) == 2

        for agent in synthetic_scene.agents:
            agent_imgs = list((tmp_output_dir / agent.id).glob("step_*.png"))
            assert len(agent_imgs) == 2, f"Missing images for {agent.id}"

        for img in overview_imgs:
            assert img.stat().st_size > 0
