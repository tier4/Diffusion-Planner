"""H5 format 6 frames and format-6 (steering-input) models."""

import math
from types import SimpleNamespace

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_dp_h5_eval.dataset import H5FrameIndex
from new_dp_h5_eval.ego_layout import (
    STEERING_PAST_KEY,
    WHEEL_BASE_KEY,
    model_takes_steering,
    to_format5_layout,
    to_model_feed,
)
from new_dp_h5_eval.model import NewDpOnnxRunner
from new_dp_h5_eval.schema import H5_FORMAT, MODEL_INPUT_NAMES
from new_dp_h5_eval.vehicle import VehicleDimensions, sim_ego_shape

ERGA = VehicleDimensions(wheel_base=4.99, front_overhang=2.57, length=10.88, width=2.5)
NATIVE6_SHAPE = np.array([ERGA.base_link_to_front, ERGA.length, ERGA.width, ERGA.wheel_base])


def _format6_frame(past_steps: int = 31) -> dict[str, np.ndarray]:
    """A format-6 frame whose ego columns are numbered so a dropped/moved column shows."""
    past = np.zeros((past_steps, 7), np.float32)
    past[:] = np.arange(7, dtype=np.float32)
    past[:, 5] = np.linspace(-0.1, 0.2, past_steps)  # steering_tire_angle
    future = np.tile(np.arange(7, dtype=np.float32) + 100, (80, 1))
    return {
        "ego_agent_past": past,
        "ego_agent_future": future,
        "ego_shape": NATIVE6_SHAPE.astype(np.float32),
        "lanes": np.ones((2, 3), np.float32),
    }


def test_format6_frame_reads_in_the_format5_layout_with_its_extras_kept():
    frame = _format6_frame()
    out = to_format5_layout(frame)

    np.testing.assert_array_equal(
        out["ego_agent_past"], frame["ego_agent_past"][:, [0, 1, 2, 3, 4, 6]]
    )
    np.testing.assert_array_equal(
        out["ego_agent_future"], frame["ego_agent_future"][:, [0, 1, 2, 3, 4, 6]]
    )
    np.testing.assert_array_equal(out["ego_shape"], NATIVE6_SHAPE[:3].astype(np.float32))
    np.testing.assert_array_equal(out[STEERING_PAST_KEY], frame["ego_agent_past"][:, 5])
    assert out[WHEEL_BASE_KEY] == np.float32(ERGA.wheel_base)
    assert out["lanes"] is frame["lanes"]
    assert frame["ego_agent_past"].shape == (31, 7)  # the input is not modified


def test_format6_stack_of_frames_converts_along_the_last_axis():
    frames = {key: np.stack([value] * 3) for key, value in _format6_frame().items()}
    out = to_format5_layout(frames)
    assert out["ego_agent_past"].shape == (3, 31, 6)
    assert out[STEERING_PAST_KEY].shape == (3, 31)
    assert out[WHEEL_BASE_KEY].shape == (3,)


def test_format5_frame_is_returned_unchanged():
    frame = {"ego_agent_past": np.zeros((31, 6), np.float32), "ego_shape": np.zeros(3)}
    assert to_format5_layout(frame) is frame


def test_format5_model_feed_is_unchanged():
    feed = {"ego_agent_past": np.zeros((2, 31, 6)), "ego_shape": np.zeros((2, 3))}
    assert to_model_feed(feed, None, takes_steering=False) is feed


def test_format6_model_feed_rebuilds_the_steering_column_and_normalized_wheel_base():
    frame = to_format5_layout(_format6_frame())
    feed = {
        "ego_agent_past": frame["ego_agent_past"][None],
        "ego_shape": frame["ego_shape"][None] / 10.0,
    }
    extras = {key: frame[key][None] for key in (STEERING_PAST_KEY, WHEEL_BASE_KEY)}
    out = to_model_feed(feed, extras, takes_steering=True)

    np.testing.assert_array_equal(out["ego_agent_past"][0], _format6_frame()["ego_agent_past"])
    np.testing.assert_allclose(out["ego_shape"][0], NATIVE6_SHAPE / 10.0, rtol=1e-6)


def test_format6_model_refuses_format5_data():
    feed = {"ego_agent_past": np.zeros((1, 31, 6)), "ego_shape": np.zeros((1, 3))}
    with pytest.raises(ValueError, match="format-5 data does not record"):
        to_model_feed(feed, None, takes_steering=True)


def _session(past_width, shape_width, seen=None):
    def run(_outputs, feed):
        seen.append(feed)
        return np.zeros((1, 321, 80, 4), np.float32), np.zeros((1, 3), np.float32)

    inputs = [
        SimpleNamespace(name=name, shape=["batch", 31, past_width])
        if name == "ego_agent_past"
        else SimpleNamespace(name=name, shape=["batch", shape_width])
        if name == "ego_shape"
        else SimpleNamespace(name=name, shape=["batch"])
        for name in (*MODEL_INPUT_NAMES, "initial_noise")
    ]
    return SimpleNamespace(get_inputs=lambda: inputs, run=run)


def test_model_layout_is_read_from_the_onnx_ego_input_widths():
    assert model_takes_steering(_session(6, 3)) is False
    assert model_takes_steering(_session(7, 4)) is True
    with pytest.raises(ValueError, match="neither"):
        model_takes_steering(_session(7, 3))


def _runner(past_width, shape_width, seen):
    runner = NewDpOnnxRunner.__new__(NewDpOnnxRunner)
    runner.session = _session(past_width, shape_width, seen)
    runner.takes_steering = model_takes_steering(runner.session)
    return runner


def _model_frame(frame):
    out = {name: np.zeros(1, np.float32) for name in MODEL_INPUT_NAMES}
    out.update(frame)
    return out


@pytest.mark.parametrize("model", ["format5", "format6"])
def test_open_loop_runner_feeds_format6_data_to_either_model(model):
    seen = []
    runner = _runner(*((6, 3) if model == "format5" else (7, 4)), seen)
    frame = _model_frame(to_format5_layout(_format6_frame()))
    runner.predict([frame], [0])

    fed = seen[0]
    if model == "format5":
        assert fed["ego_agent_past"].shape == (1, 31, 6)
        np.testing.assert_allclose(fed["ego_shape"][0], NATIVE6_SHAPE[:3] / 10.0, rtol=1e-6)
    else:
        assert fed["ego_agent_past"].shape == (1, 31, 7)
        np.testing.assert_allclose(
            fed["ego_agent_past"][0, :, 5], _format6_frame()["ego_agent_past"][:, 5]
        )
        np.testing.assert_allclose(fed["ego_shape"][0], NATIVE6_SHAPE / 10.0, rtol=1e-6)


def test_open_loop_runner_refuses_format5_data_for_a_format6_model():
    runner = _runner(7, 4, [])
    frame = _model_frame({"ego_agent_past": np.zeros((31, 6), np.float32)})
    with pytest.raises(ValueError, match="format-5 data does not record"):
        runner.predict([frame], [0])


def test_scoring_shape_checks_the_format6_wheel_base_too():
    vehicles = {"erga": ERGA}
    rows = np.tile(NATIVE6_SHAPE, (4, 1))
    np.testing.assert_allclose(
        sim_ego_shape(rows, "erga", vehicles, source="s"),
        [ERGA.wheel_base, ERGA.length, ERGA.width],
    )
    rows[2, 3] += 0.5
    with pytest.raises(ValueError, match="wheel_base"):
        sim_ego_shape(rows, "erga", vehicles, source="s")


def test_frame_index_reads_a_format6_shard(tmp_path):
    shard = tmp_path / "frames.h5"
    shapes = {
        "ego_agent_past": (31, 7),
        "ego_agent_future": (80, 7),
        "ego_shape": (4,),
        "neighbor_agents_past": (320, 31, 4),
        "agent_shape": (320, 2),
        "agent_label": (320, 3),
    }
    native = _format6_frame()
    with h5py.File(shard, "w") as file:
        file.attrs["format"] = H5_FORMAT
        file.attrs["format_version"] = 6
        file.attrs["num_frames"] = 2
        frames = file.create_group("frames")
        for name in {*MODEL_INPUT_NAMES, "ego_agent_future"}:
            value = native[name] if name in native and name != "lanes" else None
            shape = shapes.get(name, (1,))
            data = np.zeros((2, *shape), np.float32) if value is None else np.stack([value] * 2)
            frames.create_dataset(name, data=data)
    index = tmp_path / "index.parquet"
    pq.write_table(
        pa.table({"h5_path": [shard.name], "frame_index": [1], "frame_time_ns": [100]}), index
    )
    with H5FrameIndex(index) as dataset:
        frame = dataset.frame(0)
        assert dataset.ego_shapes(shard).shape == (2, 4)
    assert frame["ego_agent_past"].shape == (31, 6)
    np.testing.assert_array_equal(frame["ego_agent_past"][:, 5], native["ego_agent_past"][:, 6])
    np.testing.assert_array_equal(frame[STEERING_PAST_KEY], native["ego_agent_past"][:, 5])


# ---- per-segment closed loop ----


def _native_frame(steering: bool) -> dict[str, np.ndarray]:
    from scenario_generation.reproducer_rollout import PAST

    frame = {
        "ego_agent_past": np.zeros((PAST, 6), np.float32),
        "neighbor_agents_past": np.zeros((320, PAST, 4), np.float32),
        "agent_shape": np.zeros((320, 2), np.float32),
        "agent_label": np.zeros((320, 3), np.float32),
        "lanes": np.zeros((140, 20, 6), np.float32),
        "route_lanes": np.zeros((25, 20, 6), np.float32),
        "route_traffic_light_past": np.zeros((25, 20, 5), np.float32),
        "road_borders": np.zeros((100, 20, 2), np.float32),
    }
    frame["ego_agent_past"][:, 2] = 1.0
    frame["ego_agent_past"][:, 4] = 5.0
    if steering:
        frame[STEERING_PAST_KEY] = np.linspace(0.0, 0.3, PAST, dtype=np.float32)
        frame[WHEEL_BASE_KEY] = np.float32(ERGA.wheel_base)
    return frame


class _Timeline:
    native_h5 = True
    poses = np.zeros((3, 3))
    sim_ego_shape = np.array([ERGA.wheel_base, ERGA.length, ERGA.width], np.float32)

    def __init__(self, steering):
        self.frame = _native_frame(steering)

    def npz(self, _index):
        return self.frame


@pytest.mark.parametrize("steering", [False, True])
def test_seeded_ego_history_carries_the_recorded_steering_into_the_model_input(steering):
    from scenario_generation.reproducer_rollout import _ego_state_from_frame, build_input_np

    tl = _Timeline(steering)
    pose, ego_hist, dyn = _ego_state_from_frame(tl, 0)
    recen, _ = build_input_np(tl, 0, pose, ego_hist, dyn)

    assert ego_hist.shape[1] == (6 if steering else 5)
    assert recen["ego_agent_past"].shape == (1, 31, 6)
    if steering:
        np.testing.assert_allclose(recen[STEERING_PAST_KEY][0], tl.frame[STEERING_PAST_KEY])
        assert np.isclose(dyn.steering, 0.3)
        assert recen[WHEEL_BASE_KEY][0] == np.float32(ERGA.wheel_base)
    else:
        assert STEERING_PAST_KEY not in recen and WHEEL_BASE_KEY not in recen
        assert dyn.steering == 0.0


def _placed_state(speed):
    from scenario_generation.reproducer_rollout import DT, PAST

    hist = np.zeros((PAST, 6))
    hist[:, 0] = speed * DT * np.arange(-PAST + 1, 1)
    hist[:, 3] = speed
    hist[:, 5] = 0.05
    return SimpleNamespace(
        k=0,
        warmup_steps=0,
        tracker=None,
        live_pose=hist[-1, :3].copy(),
        dyn=SimpleNamespace(speed=speed),
        ego_hist=hist,
        sim_time=0.0,
        accels=np.zeros(200, dtype=np.float32),
        unstick_after=0,
        tl=_Timeline(steering=True),
    )


def test_placed_ego_appends_the_kinematic_steering_of_its_motion():
    from scenario_generation.perf_timer import Timers
    from scenario_generation.reproducer_rollout import DT, _advance_step

    speed, yaw_rate = 5.0, 0.2
    s = _placed_state(speed)
    t = np.arange(1, 81) * DT
    poses = np.stack([speed * t, np.zeros_like(t), yaw_rate * t], axis=-1)
    override = (poses[0], poses[1])
    _advance_step(s, np.zeros((80, 4)), 0, "cpu", Timers(), override=override)

    expected = math.atan(ERGA.wheel_base * s.dyn.yaw_rate / s.dyn.speed)
    assert np.isclose(s.dyn.yaw_rate, yaw_rate)
    assert np.isclose(s.ego_hist[-1, 5], expected) and np.isclose(s.dyn.steering, expected)


def test_placed_ego_below_the_hold_speed_keeps_its_steering():
    from scenario_generation.perf_timer import Timers
    from scenario_generation.reproducer_rollout import _advance_step

    s = _placed_state(0.0)
    pose = s.live_pose.copy()
    _advance_step(s, np.zeros((80, 4)), 0, "cpu", Timers(), override=(pose, pose))
    assert s.ego_hist[-1, 5] == 0.05


def test_batched_backend_refuses_format6_routes():
    from new_dp_h5_eval.batched_closed_loop import BatchedRollout

    segment = SimpleNamespace(timeline=SimpleNamespace(format_version=6, h5_path="a/frames.h5"))
    with pytest.raises(ValueError, match="per-segment backend for format 6"):
        BatchedRollout(None, [segment], None, None, None)
