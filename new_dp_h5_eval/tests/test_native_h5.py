from __future__ import annotations

import numpy as np

from new_dp_h5_eval.open_loop import metric_view
from new_dp_h5_eval.metric_compat import legacy_route_lanes
from new_dp_h5_eval.transforms import recenter_frame_to_pose
from scenario_generation.reproducer_rollout import _ego_state_from_frame


def test_recenter_transforms_lane_offsets_as_vectors_and_preserves_padding():
    lane = np.zeros((2, 2, 6), dtype=np.float32)
    lane[0, 0] = [11, 22, 1, 2, 3, 4]
    frame = {"lanes": lane}
    out = recenter_frame_to_pose(frame, np.array([10, 20]), np.array([0, 1]))["lanes"]
    np.testing.assert_allclose(out[0, 0], [2, -1, 2, -1, 4, -3])
    np.testing.assert_array_equal(out[1], 0)


def test_recenter_pose_keeps_non_spatial_features():
    ego = np.array([[11, 22, 0, 1, 7, .2]], dtype=np.float32)
    out = recenter_frame_to_pose(
        {"ego_agent_past": ego}, np.array([10, 20]), np.array([0, 1])
    )["ego_agent_past"]
    np.testing.assert_allclose(out[0], [2, -1, 1, 0, 7, .2])


def test_metric_view_uses_native_shape_and_label_without_broadcast_guessing():
    frame = {
        "ego_agent_past": np.zeros((31, 6), np.float32),
        "ego_agent_future": np.zeros((80, 6), np.float32),
        "route_lanes": np.zeros((25, 20, 6), np.float32),
        "lanes": np.zeros((140, 20, 6), np.float32),
        "neighbor_agents_future": np.zeros((320, 80, 4), np.float32),
        "neighbor_agents_past": np.zeros((320, 31, 4), np.float32),
        "agent_shape": np.arange(640, dtype=np.float32).reshape(320, 2),
        "agent_label": np.arange(960, dtype=np.float32).reshape(320, 3),
        "ego_shape": np.array([2.7, 4.8, 1.9], np.float32),
    }
    view = metric_view(frame)["neighbor_agents_past"].numpy()
    np.testing.assert_array_equal(view[:, 0, 6:8], frame["agent_shape"])
    np.testing.assert_array_equal(view[:, -1, 8:11], frame["agent_label"])
    np.testing.assert_array_equal(view[..., 4:6], 0)


def test_legacy_route_metric_view_derives_tangent_and_maps_red():
    route = np.zeros((1, 3, 6), np.float32)
    route[0, :, :2] = [[1, 1], [2, 1], [3, 1]]
    route[0, :, 2:6] = [0, 2, 0, -2]
    tl = np.zeros((1, 31, 6), np.float32); tl[0, -1, 2] = 1
    out = legacy_route_lanes({"route_lanes": route, "route_traffic_light_past": tl})
    np.testing.assert_allclose(out[0, :, 2:4], [[1, 0], [1, 0], [1, 0]])
    np.testing.assert_array_equal(out[0, :, 4:8], [[0, 2, 0, -2]] * 3)
    np.testing.assert_array_equal(out[0, :, 10], 1)


def test_native_seed_decodes_cos_sin_and_preserves_speed_yaw_rate():
    past = np.zeros((31, 6), np.float32)
    past[:, 2:4] = [0, 1]
    past[:, 4:6] = [4.5, .3]

    class Timeline:
        native_h5 = True
        poses = np.array([[10.0, 20.0, 0.25]])

        def npz(self, _index):
            return {"ego_agent_past": past}

    _, history, dynamics = _ego_state_from_frame(Timeline(), 0)
    np.testing.assert_allclose(history[:, 2], np.pi / 2 + .25)
    np.testing.assert_allclose(history[:, 3:5], np.tile([4.5, .3], (31, 1)))
    assert dynamics.speed == 4.5 and np.isclose(dynamics.yaw_rate, .3)
