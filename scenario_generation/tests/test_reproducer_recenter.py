"""build_input_np must re-center a recorded frame's neighbors onto the live ego
(world_to_ego_frame), so a neighbor at a known recorded-ego-frame position lands at
the correct live-ego-frame coordinate. This is the geometric core of the reproducer.
"""

import json
import math
from types import SimpleNamespace

import numpy as np

from scenario_generation.replay import _ego_border_distance
from scenario_generation.reproducer_rollout import _EgoDyn, build_input_np
from scenario_generation.route_timeline import RouteTimeline


def _one_neighbor_route(tmp_path, p_rec=(5.0, 0.0)):
    """A single recorded frame (ego at world origin, heading 0) with ONE neighbor at
    ``p_rec`` in the recorded-ego frame (heading 0)."""
    nb = np.zeros((320, 31, 11), np.float32)
    nb[0, -1, 0:2] = p_rec  # x, y
    nb[0, -1, 2:4] = (1.0, 0.0)  # cos, sin (heading 0)
    nb[0, -1, 4] = 0.5  # vx -> makes the first-6 cols nonzero (valid, not masked)
    nb[0, -1, 6:8] = (2.0, 4.0)  # width, length
    npz = dict(
        ego_agent_past=np.zeros((31, 3), np.float32),
        ego_current_state=np.zeros((10,), np.float32),
        neighbor_agents_past=nb,
        lanes=np.zeros((1, 20, 33), np.float32),
        lanes_speed_limit=np.zeros((1, 1), np.float32),
        lanes_has_speed_limit=np.zeros((1, 1), bool),
        route_lanes=np.zeros((1, 20, 33), np.float32),
        route_lanes_speed_limit=np.zeros((1, 1), np.float32),
        route_lanes_has_speed_limit=np.zeros((1, 1), bool),
        polygons=np.zeros((1, 40, 3), np.float32),
        line_strings=np.zeros((1, 20, 4), np.float32),
        static_objects=np.zeros((5, 10), np.float32),
        ego_shape=np.array([4.76, 7.24, 2.29], np.float32),
        turn_indicators=np.zeros((31,), np.int64),
        goal_pose=np.array([0.0, 0.0, 0.0], np.float32),
    )
    p = tmp_path / "r_0000000000.npz"
    np.savez_compressed(p, **npz)
    (tmp_path / "r_0000000000.json").write_text(
        json.dumps({"x": 0.0, "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1})
    )
    return RouteTimeline([p])


def test_recenter_pure_translation(tmp_path):
    """Recorded ego at origin, live ego at (2,1): neighbor (5,0) -> (3,-1)."""
    tl = _one_neighbor_route(tmp_path, p_rec=(5.0, 0.0))
    live_pose = np.array([2.0, 1.0, 0.0])
    ego_hist = np.zeros((31, 3))
    ego_hist[:, :2] = live_pose[:2]
    _, neighbors_live = build_input_np(tl, 0, live_pose, ego_hist, _EgoDyn(speed=3.0))
    assert np.allclose(neighbors_live[0, :2], [3.0, -1.0], atol=1e-5)
    assert np.allclose(neighbors_live[0, 2:4], [1.0, 0.0], atol=1e-5)  # heading unchanged


def test_recenter_pure_rotation(tmp_path):
    """Live ego rotated +90deg at origin: neighbor (5,0) -> (0,-5), heading -> -90deg."""
    tl = _one_neighbor_route(tmp_path, p_rec=(5.0, 0.0))
    live_pose = np.array([0.0, 0.0, math.pi / 2])
    ego_hist = np.zeros((31, 3))
    ego_hist[:, 2] = math.pi / 2
    _, neighbors_live = build_input_np(tl, 0, live_pose, ego_hist, _EgoDyn(speed=3.0))
    assert np.allclose(neighbors_live[0, :2], [0.0, -5.0], atol=1e-5)
    # heading (1,0) rotated by R(+90deg)=[[0,1],[-1,0]] -> (0,-1)
    assert np.allclose(neighbors_live[0, 2:4], [0.0, -1.0], atol=1e-5)


def test_ego_border_distance_handles_empty_line_strings():
    ego = SimpleNamespace(
        current_position=np.array([0.0, 0.0], dtype=np.float32),
        current_heading=0.0,
        wheelbase=2.0,
        length=4.0,
        width=2.0,
    )
    map_data = SimpleNamespace(line_strings=np.zeros((0, 20, 4), dtype=np.float32))

    assert _ego_border_distance(ego, map_data) is None


def test_ego_border_distance_keeps_large_real_clearance():
    far_border_y_m = 150.0
    ego = SimpleNamespace(
        current_position=np.array([0.0, 0.0], dtype=np.float32),
        current_heading=0.0,
        wheelbase=2.0,
        length=4.0,
        width=2.0,
    )
    line_strings = np.zeros((1, 20, 4), dtype=np.float32)
    line_strings[0, 0, :2] = [-10.0, far_border_y_m]
    line_strings[0, 1, :2] = [10.0, far_border_y_m]
    line_strings[0, :2, 3] = 1.0
    map_data = SimpleNamespace(line_strings=line_strings)

    rb_info = _ego_border_distance(ego, map_data)

    assert rb_info is not None
    expected_clearance_m = far_border_y_m - ego.width / 2
    assert np.isclose(rb_info[2], expected_clearance_m)
