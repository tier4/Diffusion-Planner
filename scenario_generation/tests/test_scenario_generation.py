"""Tests for the scenario_generation module.

Tests coordinate transforms, NPZ loading, and tensor conversion.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scenario_generation.npz_loader import from_npz
from scenario_generation.scene_context import Agent, AgentType, MapData, SceneContext
from scenario_generation.transforms import (
    _rotation_matrix,
    transform_cos_sin,
    transform_directions,
    transform_headings,
    transform_positions,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"
_SAMPLE_NPZ = _TEST_DATA_DIR / "fixture_scene.npz"


def _make_synthetic_npz(tmp_path: Path) -> Path:
    """Create a minimal synthetic NPZ for testing without real data."""
    T_past = 31
    T_future = 80
    N_nb = 32

    data = {
        "ego_agent_past": np.zeros((T_past, 3), dtype=np.float32),
        "ego_current_state": np.zeros(10, dtype=np.float32),
        "ego_agent_future": np.zeros((T_future, 3), dtype=np.float32),
        "neighbor_agents_past": np.zeros((N_nb, T_past, 11), dtype=np.float32),
        "neighbor_agents_future": np.zeros((N_nb, T_future, 3), dtype=np.float32),
        "static_objects": np.zeros((5, 10), dtype=np.float32),
        "lanes": np.zeros((140, 20, 33), dtype=np.float32),
        "lanes_speed_limit": np.zeros((140, 1), dtype=np.float32),
        "lanes_has_speed_limit": np.zeros((140, 1), dtype=bool),
        "route_lanes": np.zeros((25, 20, 33), dtype=np.float32),
        "route_lanes_speed_limit": np.zeros((25, 1), dtype=np.float32),
        "route_lanes_has_speed_limit": np.zeros((25, 1), dtype=bool),
        "polygons": np.zeros((10, 40, 2), dtype=np.float32),
        "line_strings": np.zeros((10, 20, 2), dtype=np.float32),
        "goal_pose": np.array([50.0, 10.0, 0.5], dtype=np.float32),
        "ego_shape": np.array([2.79, 4.34, 1.70], dtype=np.float32),
        "turn_indicators": np.zeros(T_past, dtype=np.int32),
    }

    # Ego: drove straight ahead from (-10, 0) to (0, 0) with heading=0
    for t in range(T_past):
        x = -10.0 + t * (10.0 / (T_past - 1))
        data["ego_agent_past"][t] = [x, 0.0, 0.0]
    data["ego_current_state"][:4] = [0.0, 0.0, 1.0, 0.0]
    data["ego_current_state"][4] = 3.0  # vx
    data["ego_current_state"][5] = 0.0  # vy

    # Neighbor 0: at (5, 3) heading pi/4, moving
    for t in range(T_past):
        x = 2.0 + t * (3.0 / (T_past - 1))
        y = 1.0 + t * (2.0 / (T_past - 1))
        h = math.pi / 4
        data["neighbor_agents_past"][0, t, :4] = [x, y, math.cos(h), math.sin(h)]
        data["neighbor_agents_past"][0, t, 4:6] = [2.0, 1.0]  # vx, vy
        data["neighbor_agents_past"][0, t, 6:8] = [1.8, 4.5]  # width, length
        data["neighbor_agents_past"][0, t, 8] = 1.0  # is_vehicle

    # Neighbor 1: at (10, -2) heading 0
    for t in range(T_past):
        data["neighbor_agents_past"][1, t, :4] = [10.0, -2.0, 1.0, 0.0]
        data["neighbor_agents_past"][1, t, 4:6] = [5.0, 0.0]
        data["neighbor_agents_past"][1, t, 6:8] = [1.8, 4.5]
        data["neighbor_agents_past"][1, t, 8] = 1.0

    # Add some lane data
    for i in range(3):
        for pt in range(20):
            x = i * 10.0 + pt * 0.5
            data["lanes"][i, pt, 0] = x
            data["lanes"][i, pt, 1] = 0.0
            data["lanes"][i, pt, 2] = 0.5  # dx
            data["lanes"][i, pt, 3] = 0.0  # dy

    npz_path = tmp_path / "test_scene.npz"
    np.savez(str(npz_path), **data)
    return npz_path


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------


class TestTransforms:
    def test_rotation_matrix_identity(self):
        R = _rotation_matrix(0.0)
        np.testing.assert_allclose(R, np.eye(2), atol=1e-10)

    def test_rotation_matrix_90deg(self):
        R = _rotation_matrix(math.pi / 2)
        # Ego heading=pi/2 (north). Point at (1,0) (east) becomes (0,-1) in ego frame (right).
        result = R @ np.array([1.0, 0.0])
        np.testing.assert_allclose(result, [0.0, -1.0], atol=1e-10)

    def test_transform_positions_identity(self):
        pts = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        R = _rotation_matrix(0.0)
        ego_xy = np.array([0.0, 0.0])
        result = transform_positions(pts, R, ego_xy)
        np.testing.assert_allclose(result, pts, atol=1e-6)

    def test_transform_positions_translation(self):
        pts = np.array([[10.0, 5.0]], dtype=np.float32)
        R = _rotation_matrix(0.0)
        ego_xy = np.array([3.0, 2.0])
        result = transform_positions(pts, R, ego_xy)
        np.testing.assert_allclose(result, [[7.0, 3.0]], atol=1e-6)

    def test_transform_positions_rotation(self):
        pts = np.array([[1.0, 0.0]], dtype=np.float32)
        R = _rotation_matrix(math.pi / 2)
        ego_xy = np.array([0.0, 0.0])
        result = transform_positions(pts, R, ego_xy)
        # Ego heading=pi/2 (north). (1,0) in world = (0,-1) in ego frame (right side)
        np.testing.assert_allclose(result, [[0.0, -1.0]], atol=1e-6)

    def test_transform_positions_combined(self):
        pts = np.array([[5.0, 3.0]], dtype=np.float32)
        R = _rotation_matrix(math.pi / 2)
        ego_xy = np.array([2.0, 1.0])
        # Translate: (3, 2). R=[[0,1],[-1,0]] @ [3,2] = [2, -3]
        result = transform_positions(pts, R, ego_xy)
        np.testing.assert_allclose(result, [[2.0, -3.0]], atol=1e-5)

    def test_transform_directions_no_translation(self):
        dirs = np.array([[3.0, 4.0]], dtype=np.float32)
        R = _rotation_matrix(0.0)
        result = transform_directions(dirs, R)
        np.testing.assert_allclose(result, [[3.0, 4.0]], atol=1e-6)

    def test_transform_headings(self):
        h = np.array([math.pi / 4, math.pi / 2])
        result = transform_headings(h, math.pi / 4)
        np.testing.assert_allclose(result, [0.0, math.pi / 4], atol=1e-6)

    def test_transform_cos_sin(self):
        cs = np.array([[1.0, 0.0]], dtype=np.float32)  # heading=0
        R = _rotation_matrix(math.pi / 2)
        result = transform_cos_sin(cs, R)
        # R = [[cos, sin], [-sin, cos]] rotates direction vectors by -heading.
        # With ego heading=pi/2: R @ [1, 0] = [0, -1]
        np.testing.assert_allclose(result, [[0.0, -1.0]], atol=1e-6)

    def test_transform_preserves_batch_shape(self):
        pts = np.random.randn(5, 8, 2).astype(np.float32)
        R = _rotation_matrix(0.3)
        ego_xy = np.array([1.0, 2.0])
        result = transform_positions(pts, R, ego_xy)
        assert result.shape == (5, 8, 2)


# ---------------------------------------------------------------------------
# NPZ loader tests
# ---------------------------------------------------------------------------


class TestNpzLoader:
    def test_synthetic_npz_loads(self, tmp_path):
        npz_path = _make_synthetic_npz(tmp_path)
        scene = from_npz(npz_path)

        assert scene.ego_agent_id == "ego"
        assert scene.ego_agent is not None
        assert scene.ego_agent.agent_type == AgentType.VEHICLE

        # 2 valid neighbors (rest are zeros)
        non_ego = [a for a in scene.agents if a.id != "ego"]
        assert len(non_ego) == 2

    def test_ego_shape_loaded(self, tmp_path):
        npz_path = _make_synthetic_npz(tmp_path)
        scene = from_npz(npz_path)

        ego = scene.ego_agent
        assert abs(ego.wheelbase - 2.79) < 1e-5
        assert abs(ego.length - 4.34) < 1e-5
        assert abs(ego.width - 1.70) < 1e-5

    def test_ego_trajectory_shape(self, tmp_path):
        npz_path = _make_synthetic_npz(tmp_path)
        scene = from_npz(npz_path)

        ego = scene.ego_agent
        assert ego.past_trajectory.shape == (31, 3)  # (T_past, [x, y, heading])

    def test_ego_current_position_at_origin(self, tmp_path):
        npz_path = _make_synthetic_npz(tmp_path)
        scene = from_npz(npz_path)

        pos = scene.ego_agent.current_position
        np.testing.assert_allclose(pos, [0.0, 0.0], atol=1e-5)

    def test_neighbor_heading_converted_to_radians(self, tmp_path):
        npz_path = _make_synthetic_npz(tmp_path)
        scene = from_npz(npz_path)

        nb0 = scene.get_agent("neighbor_0")
        h = nb0.current_heading
        assert abs(h - math.pi / 4) < 1e-5

    def test_neighbor_wheelbase_derived(self, tmp_path):
        npz_path = _make_synthetic_npz(tmp_path)
        scene = from_npz(npz_path)

        nb0 = scene.get_agent("neighbor_0")
        assert abs(nb0.wheelbase - 4.5 * 0.65) < 1e-5

    def test_goal_pose_loaded(self, tmp_path):
        npz_path = _make_synthetic_npz(tmp_path)
        scene = from_npz(npz_path)

        gp = scene.ego_agent.goal_pose
        assert gp is not None
        np.testing.assert_allclose(gp[:2], [50.0, 10.0], atol=1e-5)
        assert abs(gp[2] - 0.5) < 1e-5

    def test_map_data_shapes(self, tmp_path):
        npz_path = _make_synthetic_npz(tmp_path)
        scene = from_npz(npz_path)

        assert scene.map_data.lanes.shape == (140, 20, 33)
        assert scene.map_data.polygons.shape == (10, 40, 2)
        assert scene.map_data.line_strings.shape == (10, 20, 2)
        assert scene.map_data.static_objects.shape == (5, 10)

    def test_turn_indicators_loaded(self, tmp_path):
        npz_path = _make_synthetic_npz(tmp_path)
        scene = from_npz(npz_path)

        assert scene.ego_agent.turn_indicators is not None
        assert scene.ego_agent.turn_indicators.shape == (31,)

    @pytest.mark.skipif(not _SAMPLE_NPZ.exists(), reason="Real NPZ not available")
    def test_real_npz_loads(self):
        scene = from_npz(_SAMPLE_NPZ)

        assert scene.ego_agent is not None
        assert scene.ego_agent.past_trajectory.shape[0] == 31
        assert scene.ego_agent.past_trajectory.shape[1] == 3
        assert len(scene.agents) >= 1

        # Check at least some neighbors were extracted
        non_ego = [a for a in scene.agents if a.id != "ego"]
        assert len(non_ego) > 0

    @pytest.mark.skipif(not _SAMPLE_NPZ.exists(), reason="Real NPZ not available")
    def test_real_npz_future_trajectory(self):
        scene = from_npz(_SAMPLE_NPZ)

        assert scene.ego_agent.future_trajectory is not None
        assert scene.ego_agent.future_trajectory.shape == (80, 3)


# ---------------------------------------------------------------------------
# Tensor converter tests (no model needed for shape/value checks)
# ---------------------------------------------------------------------------


class TestTensorConverter:
    def test_ego_identity_transform(self, tmp_path):
        """When using original ego, the transform should be identity-like."""
        from scenario_generation.tensor_converter import (
            _build_ego_agent_past,
            _build_ego_current_state,
            _heading_to_cos_sin,
        )

        npz_path = _make_synthetic_npz(tmp_path)
        scene = from_npz(npz_path)
        ego = scene.ego_agent

        R = _rotation_matrix(ego.current_heading)
        ego_xy = ego.current_position.astype(np.float64)

        past = _build_ego_agent_past(ego, R, ego_xy, ego.current_heading)
        assert past.shape == (1, 31, 4)

        # Current position (last timestep) should be at origin
        np.testing.assert_allclose(past[0, -1, :2], [0.0, 0.0], atol=1e-5)
        # Current heading=0 -> cos=1, sin=0
        np.testing.assert_allclose(past[0, -1, 2:4], [1.0, 0.0], atol=1e-5)

    def test_ego_current_state_at_origin(self, tmp_path):
        from scenario_generation.tensor_converter import _build_ego_current_state

        npz_path = _make_synthetic_npz(tmp_path)
        scene = from_npz(npz_path)
        ego = scene.ego_agent

        R = _rotation_matrix(ego.current_heading)
        ego_xy = ego.current_position.astype(np.float64)

        state = _build_ego_current_state(ego, R)
        assert state.shape == (1, 10)
        np.testing.assert_allclose(state[0, :4], [0.0, 0.0, 1.0, 0.0], atol=1e-5)

    def test_neighbor_tensor_shape(self, tmp_path):
        from scenario_generation.tensor_converter import _build_neighbor_agents_past

        npz_path = _make_synthetic_npz(tmp_path)
        scene = from_npz(npz_path)
        ego = scene.ego_agent

        R = _rotation_matrix(ego.current_heading)
        ego_xy = ego.current_position.astype(np.float64)

        nb = _build_neighbor_agents_past(scene, "ego", R, ego_xy, ego.current_heading)
        assert nb.shape == (1, 32, 31, 11)

        # Slot 0 should be closest neighbor (neighbor_0 at ~(5,3), distance ~5.83)
        # Slot 1 should be neighbor_1 (at (10,-2), distance ~10.2)
        assert np.any(nb[0, 0] != 0)
        assert np.any(nb[0, 1] != 0)
        # Slot 2+ should be empty
        assert np.all(nb[0, 2] == 0)

    def test_neighbor_sorted_by_distance(self, tmp_path):
        from scenario_generation.tensor_converter import _build_neighbor_agents_past

        npz_path = _make_synthetic_npz(tmp_path)
        scene = from_npz(npz_path)
        ego = scene.ego_agent

        R = _rotation_matrix(ego.current_heading)
        ego_xy = ego.current_position.astype(np.float64)

        nb = _build_neighbor_agents_past(scene, "ego", R, ego_xy, ego.current_heading)

        # Slot 0 (neighbor_0): current pos at (5, 3), distance ~5.83
        # Slot 1 (neighbor_1): current pos at (10, -2), distance ~10.2
        dist_0 = np.sqrt(nb[0, 0, -1, 0] ** 2 + nb[0, 0, -1, 1] ** 2)
        dist_1 = np.sqrt(nb[0, 1, -1, 0] ** 2 + nb[0, 1, -1, 1] ** 2)
        assert dist_0 < dist_1

    def test_ego_switching_moves_origin(self, tmp_path):
        """When a neighbor becomes ego, positions should re-center."""
        from scenario_generation.tensor_converter import (
            _build_ego_agent_past,
            _build_ego_current_state,
        )

        npz_path = _make_synthetic_npz(tmp_path)
        scene = from_npz(npz_path)

        nb0 = scene.get_agent("neighbor_0")
        R = _rotation_matrix(nb0.current_heading)
        ego_xy = nb0.current_position.astype(np.float64)

        past = _build_ego_agent_past(nb0, R, ego_xy, nb0.current_heading)
        assert past.shape == (1, 31, 4)

        # New ego's current position should be at origin
        np.testing.assert_allclose(past[0, -1, :2], [0.0, 0.0], atol=1e-4)
        # Heading should be 0 (cos=1, sin=0)
        np.testing.assert_allclose(past[0, -1, 2:4], [1.0, 0.0], atol=1e-4)

        state = _build_ego_current_state(nb0, R)
        np.testing.assert_allclose(state[0, :4], [0.0, 0.0, 1.0, 0.0], atol=1e-4)

    def test_goal_pose_transform(self, tmp_path):
        from scenario_generation.tensor_converter import _build_goal_pose

        npz_path = _make_synthetic_npz(tmp_path)
        scene = from_npz(npz_path)
        ego = scene.ego_agent

        R = _rotation_matrix(ego.current_heading)
        ego_xy = ego.current_position.astype(np.float64)

        gp = _build_goal_pose(ego, R, ego_xy, ego.current_heading)
        assert gp.shape == (1, 4)
        # Goal at (50, 10) relative to ego at origin heading=0 should stay (50, 10)
        np.testing.assert_allclose(gp[0, :2], [50.0, 10.0], atol=1e-4)

    def test_lane_transform_preserves_mask(self, tmp_path):
        from scenario_generation.tensor_converter import _build_lanes

        npz_path = _make_synthetic_npz(tmp_path)
        scene = from_npz(npz_path)

        R = _rotation_matrix(0.0)
        ego_xy = np.array([0.0, 0.0])

        lanes = _build_lanes(scene.map_data.lanes, R, ego_xy, 140)
        assert lanes.shape == (1, 140, 20, 33)

        # Lanes beyond index 2 should be zeros
        assert np.all(lanes[0, 5:] == 0)
        # First 3 lanes should have non-zero data
        assert np.any(lanes[0, 0] != 0)

    def test_pad_or_truncate(self):
        from scenario_generation.tensor_converter import _pad_or_truncate

        arr = np.arange(10).reshape(10, 1)

        # Truncate: keep last 5
        result = _pad_or_truncate(arr, 5, axis=0)
        assert result.shape == (5, 1)
        np.testing.assert_array_equal(result.flatten(), [5, 6, 7, 8, 9])

        # Pad: zeros at front
        result = _pad_or_truncate(arr, 15, axis=0)
        assert result.shape == (15, 1)
        np.testing.assert_array_equal(result[:5].flatten(), [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(result[5:].flatten(), np.arange(10))

        # No-op
        result = _pad_or_truncate(arr, 10, axis=0)
        assert result.shape == (10, 1)
        np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# SceneContext API tests
# ---------------------------------------------------------------------------


class TestSceneContext:
    def test_get_agent_found(self, tmp_path):
        npz_path = _make_synthetic_npz(tmp_path)
        scene = from_npz(npz_path)

        ego = scene.get_agent("ego")
        assert ego.id == "ego"

    def test_get_agent_not_found(self, tmp_path):
        npz_path = _make_synthetic_npz(tmp_path)
        scene = from_npz(npz_path)

        with pytest.raises(KeyError, match="nonexistent"):
            scene.get_agent("nonexistent")

    def test_agent_current_velocity_from_velocities(self):
        agent = Agent(
            id="test",
            agent_type=AgentType.VEHICLE,
            length=4.0,
            width=2.0,
            wheelbase=2.6,
            past_trajectory=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
            past_velocities=np.array([[0, 0], [10, 0]], dtype=np.float32),
        )
        vel = agent.current_velocity
        np.testing.assert_allclose(vel, [10, 0], atol=1e-5)

    def test_agent_current_velocity_derived(self):
        agent = Agent(
            id="test",
            agent_type=AgentType.VEHICLE,
            length=4.0,
            width=2.0,
            wheelbase=2.6,
            past_trajectory=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        )
        vel = agent.current_velocity
        # (1-0) / 0.1 = 10 m/s in x
        np.testing.assert_allclose(vel, [10, 0], atol=1e-5)

    def test_extensible_agent_fields(self):
        """Verify optional fields for route/goal work correctly."""
        route = np.zeros((5, 20, 33), dtype=np.float32)
        goal = np.array([100.0, 50.0, 1.0], dtype=np.float32)

        agent = Agent(
            id="extended",
            agent_type=AgentType.VEHICLE,
            length=4.0,
            width=2.0,
            wheelbase=2.6,
            past_trajectory=np.array([[0, 0, 0]], dtype=np.float32),
            goal_pose=goal,
            route_lanes=route,
        )
        assert agent.goal_pose is not None
        assert agent.route_lanes.shape == (5, 20, 33)
