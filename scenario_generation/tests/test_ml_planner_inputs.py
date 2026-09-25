import math

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from new_dp_h5_eval.schema import MODEL_INPUT_NAMES
from scenario_generation.ml_planner_inputs import (
    INPUT_BUILDERS,
    MlPlannerObservation,
    MlPlannerOnnx,
    infer_traffic_light_future,
)

SHAPES = {
    "initial_noise": (3, 8, 4),
    "ego_agent_past": (4, 6),
    "neighbor_agents_past": (2, 4, 4),
    "agent_shape": (2, 2),
    "agent_label": (2, 3),
    "lanes": (3, 5, 6),
    "lane_types": (3, 20),
    "lanes_speed_limit": (3, 1),
    "lane_traffic_light_past": (3, 4, 6),
    "lane_traffic_light_future": (3, 8, 6),
    "route_lanes": (2, 5, 6),
    "route_lane_types": (2, 20),
    "route_lanes_speed_limit": (2, 1),
    "route_traffic_light_past": (2, 4, 6),
    "route_traffic_light_future": (2, 8, 6),
    "intersection_area": (2, 6, 2),
    "stop_lines": (2, 2, 2),
    "road_borders": (2, 5, 2),
    "goal_pose": (4,),
    "ego_shape": (3,),
    "turn_indicators": (4,),
}
GREEN, AMBER = 3, 2


class _Model:
    shapes = SHAPES


def _lanes(n, tl_ids):
    return {
        "lanelet_id": np.arange(n, dtype=np.int64),
        "center": np.ones((n, 5, 2), np.float32),
        "left": np.full((n, 5, 2), 2.0, np.float32),
        "right": np.zeros((n, 5, 2), np.float32),
        "boundary_type": np.array([[4, 8]] * n, dtype=np.int8),
        "speed_limit_mps": np.array([10.0] + [np.nan] * (n - 1), np.float32),
        "turn_direction": np.full(n, -1, np.int8),
        "traffic_light_id": np.array(tl_ids, dtype=np.int64),
    }


class _Features:
    def select(self, ego_pose, route_ids, **_):
        return {
            "lanes": _lanes(2, [7, -1]),
            "route_lanes": _lanes(1, [7]),
            "intersection_areas": np.zeros((1, 6, 2), np.float32),
            "stop_lines": np.zeros((0, 2, 2), np.float32),
            "road_borders": np.ones((1, 5, 2), np.float32),
        }


def _state(x, y, yaw, *, type_=1, subtype=1, v=0.0):
    return {
        "type": type_,
        "subtype": subtype,
        "pose": {"x": x, "y": y, "z": 0.0, "yaw": yaw},
        "twist": {"linear_x": v, "linear_y": 0.0, "angular_z": 0.1},
        "bounding_box": {"center": {"x": 1.5, "y": 0.0}, "dimensions": {"x": 4.8, "y": 1.9}},
    }


def _observation():
    return MlPlannerObservation(_Model(), _Features(), [1, 2], np.array([10.0, 0.0, 0.0]))


def test_every_input_is_built_at_the_shape_the_graph_declares():
    obs = _observation()
    obs.record({"ego": _state(0, 0, 0, type_=0, v=1.0)}, "ego", {}, 1)
    inputs = obs.build()

    assert set(inputs) == set(SHAPES) - {"initial_noise"}
    for name, value in inputs.items():
        assert value.shape == SHAPES[name], name


def test_lanes_carry_boundary_offsets_types_and_unknown_speed_as_zero():
    obs = _observation()
    obs.record({"ego": _state(0, 0, 0, type_=0)}, "ego", {}, 1)
    inputs = obs.build()

    np.testing.assert_array_equal(inputs["lanes"][0, 0], [1, 1, 1, 1, -1, -1])
    assert inputs["lane_types"][0, 4] == 1 and inputs["lane_types"][0, 10 + 8] == 1
    np.testing.assert_array_equal(inputs["lanes_speed_limit"][:, 0], [10.0, 0.0, 0.0])
    np.testing.assert_allclose(inputs["ego_shape"], [1.5 + 2.4, 4.8, 1.9])


def test_a_light_is_unknown_until_heard_and_again_once_it_goes_silent():
    obs = _observation()
    lights = [{}, {7: [(GREEN, 1, 2, 1.0)]}, {}, {}, {}]
    for groups in lights:
        obs.record({"ego": _state(0, 0, 0, type_=0)}, "ego", groups, 1)
    past = obs.build()["lane_traffic_light_past"]

    # The window is the last four ticks: green heard at tick 1, held for 0.2 s, then unknown.
    assert past[0, :, 0].tolist() == [1, 1, 1, 0]
    assert past[0, :, 3].tolist() == [0, 0, 0, 1]
    # A lane without a light reads "no light" at every step, not "unknown".
    assert past[1, :, 4].tolist() == [1, 1, 1, 1]
    assert not past[2].any()


def test_amber_turns_red_thirty_steps_after_it_came_on():
    past = np.zeros((1, 31, 6), np.float32)
    past[0, -10:, 1] = 1.0
    future = infer_traffic_light_future(past, 80)

    assert future[0, :20, 1].all() and not future[0, 20:, 1].any()
    assert future[0, 20:, 2].all()


def test_a_neighbor_has_no_past_before_it_was_first_seen():
    obs = _observation()
    for tick in range(4):
        states = {"ego": _state(0, 0, 0, type_=0)}
        if tick >= 2:
            states["npc"] = _state(5.0 + tick, 0, 0)
        obs.record(states, "ego", {}, 1)
    inputs = obs.build()

    past = inputs["neighbor_agents_past"][0]
    assert not past[:2].any()
    # Box centre: 1.5 m ahead of the reported base_link.
    assert past[2, 0] == pytest.approx(8.5) and past[3, 0] == pytest.approx(9.5)
    np.testing.assert_array_equal(inputs["agent_label"][0], [1, 0, 0])
    np.testing.assert_allclose(inputs["agent_shape"][0], [1.9, 4.8])


def test_the_ego_history_is_in_the_current_frame_and_holds_its_first_sample():
    obs = _observation()
    for tick in range(2):
        obs.record({"ego": _state(tick, 0, math.pi / 2, type_=0, v=0.1)}, "ego", {}, 2)
    inputs = obs.build()

    ego = inputs["ego_agent_past"]
    np.testing.assert_allclose(ego[-1, :4], [0, 0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(ego[0, :2], [0, 1], atol=1e-6)
    # Below the moving threshold the yaw rate is reported as zero.
    assert not ego[:, 5].any()
    np.testing.assert_array_equal(inputs["turn_indicators"], [2, 2, 2, 2])


def _graph_with_inputs(tmp_path, names):
    inputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, ["batch", 1]) for n in names]
    out = helper.make_tensor_value_info("trajectory", TensorProto.FLOAT, ["batch", 1])
    node = helper.make_node("Identity", [names[0]], ["trajectory"])
    path = tmp_path / "model.onnx"
    model = helper.make_model(
        helper.make_graph([node], "g", inputs, [out]), opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    onnx.save(model, path)
    return str(path)


def test_a_model_asking_for_an_unknown_input_is_refused_at_load(tmp_path):
    path = _graph_with_inputs(tmp_path, ["initial_noise", "ego_agent_past", "ego_current_state"])

    with pytest.raises(ValueError, match="ego_current_state"):
        MlPlannerOnnx(path, device="cpu")


def test_the_current_contract_is_fully_buildable():
    assert set(MODEL_INPUT_NAMES) <= set(INPUT_BUILDERS)
