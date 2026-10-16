"""Steering in ego_agent_past, wheel base in ego_shape, and packed frame rows."""

import json

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_dp_h5_eval.closed_loop import NativeH5RouteTimeline
from new_dp_h5_eval.dataset import H5FrameIndex
from new_dp_h5_eval.model import NewDpOnnxRunner
from new_dp_h5_eval.open_loop import metric_view
from new_dp_h5_eval.render import render_view
from new_dp_h5_eval.schema import H5_FORMAT, MODEL_INPUT_NAMES
from new_dp_h5_eval.vehicle import VehicleDimensions, sim_ego_shape
from scenario_generation.reproducer_rollout import (
    PAST,
    _ego_state_from_frame,
    _EgoDyn,
    build_input_np,
)

BUS = VehicleDimensions(wheel_base=4.76012, front_overhang=0.95099, length=7.2369, width=2.42741)
BUS_SHAPE = [BUS.base_link_to_front, BUS.length, BUS.width, BUS.wheel_base]
COUNT = 3


def _frames():
    rng = np.random.default_rng(0)
    frames = {}
    for name in MODEL_INPUT_NAMES:
        shape = {
            "ego_agent_past": (PAST, 7),
            "neighbor_agents_past": (320, PAST, 4),
            "agent_shape": (320, 2),
            "agent_label": (320, 3),
            "ego_shape": (4,),
        }.get(name, (2,))
        frames[name] = rng.normal(size=(COUNT, *shape)).astype(np.float32)
    frames["ego_shape"][:] = BUS_SHAPE
    return frames


def _write_shard(path, frames, *, packed):
    with h5py.File(path, "w") as file:
        file.attrs["format"] = H5_FORMAT
        file.attrs["format_version"] = 5
        file.attrs["num_frames"] = COUNT
        file.attrs["frame_interval_s"] = 0.1
        file.attrs["pose_frame_id"] = "map"
        file.attrs["project_id"] = "bus"
        if packed:
            # ML-Planner's h5_layout.PackedLayout: keys sorted, tensors flattened in order.
            keys = sorted(frames)
            file.attrs["packed_frame_layout"] = json.dumps(
                {"keys": keys, "shapes": [list(frames[k].shape[1:]) for k in keys]}
            )
            file.create_dataset(
                "frames_packed",
                data=np.concatenate([frames[k].reshape(COUNT, -1) for k in keys], axis=1),
                chunks=True,
            )
        else:
            group = file.create_group("frames")
            for key, value in frames.items():
                group.create_dataset(key, data=value)
        metadata = file.create_group("metadata")
        metadata.create_dataset(
            "frame_time_ns", data=[1_000_000_000 + i * 100_000_000 for i in range(COUNT)]
        )
        for name, value in {"x": np.arange(COUNT, dtype=np.float64), "qw": np.ones(COUNT)}.items():
            metadata.create_dataset(name, data=value)
        for name in ("y", "z", "qx", "qy", "qz"):
            metadata.create_dataset(name, data=np.zeros(COUNT))


@pytest.mark.parametrize("packed", [False, True])
def test_both_layouts_read_the_same_tensors(tmp_path, packed):
    frames = _frames()
    shard = tmp_path / "frames.h5"
    _write_shard(shard, frames, packed=packed)
    timeline = NativeH5RouteTimeline(shard, vehicle_params={"bus": BUS})
    try:
        for i in range(COUNT):
            got = timeline.npz(i)
            assert set(got) == set(frames)
            for key in frames:
                np.testing.assert_array_equal(got[key], frames[key][i])
        np.testing.assert_allclose(timeline.sim_ego_shape, [BUS.wheel_base, BUS.length, BUS.width])
    finally:
        timeline.close()

    index = tmp_path / "index.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"h5_path": shard.name, "frame_index": 1, "frame_time_ns": 1_100_000_000}]
        ),
        index,
    )
    with H5FrameIndex(index) as dataset:
        np.testing.assert_array_equal(
            dataset.frame(0)["ego_agent_past"], frames["ego_agent_past"][1]
        )
        shapes = dataset.ego_shapes(dataset.shard_path(0))
        # A packed shard reads only the indexed frames; a frames/ shard reads them all.
        assert len(shapes) == (1 if packed else COUNT)
        np.testing.assert_allclose(shapes, np.tile(BUS_SHAPE, (len(shapes), 1)), rtol=0, atol=1e-5)


def test_packed_layout_that_does_not_cover_the_row_is_rejected(tmp_path):
    frames = _frames()
    shard = tmp_path / "frames.h5"
    _write_shard(shard, frames, packed=True)
    with h5py.File(shard, "r+") as file:
        layout = json.loads(file.attrs["packed_frame_layout"])
        layout["shapes"][0] = [1]
        file.attrs["packed_frame_layout"] = json.dumps(layout)
    with pytest.raises(ValueError, match="packed_frame_layout"):
        NativeH5RouteTimeline(shard)


def test_four_column_ego_shape_must_carry_the_project_wheel_base():
    np.testing.assert_allclose(
        sim_ego_shape(np.array([BUS_SHAPE]), "bus", {"bus": BUS}, source="t"),
        [BUS.wheel_base, BUS.length, BUS.width],
    )
    wrong = [*BUS_SHAPE[:3], 3.1]
    with pytest.raises(ValueError, match="disagree"):
        sim_ego_shape(np.array([wrong]), "bus", {"bus": BUS}, source="t")


def test_open_loop_metric_view_reads_yaw_rate_from_its_steering_column():
    frame = {
        "ego_agent_past": np.zeros((PAST, 7), np.float32),
        "ego_agent_future": np.zeros((80, 7), np.float32),
        "neighbor_agents_past": np.zeros((320, PAST, 4), np.float32),
        "neighbor_agents_future": np.zeros((320, 80, 4), np.float32),
        "agent_shape": np.zeros((320, 2), np.float32),
        "agent_label": np.zeros((320, 3), np.float32),
        "route_lanes": np.zeros((25, 20, 6), np.float32),
        "lanes": np.zeros((140, 20, 6), np.float32),
        "lane_traffic_light_past": np.zeros((140, PAST, 6), np.float32),
    }
    frame["ego_agent_past"][-1, 4:7] = [5.0, 0.1, 0.3]
    current = metric_view(frame, np.zeros(3, np.float32))["ego_current_state"].numpy()
    assert current[4] == 5.0 and np.isclose(current[8], 0.1) and np.isclose(current[9], 0.3)


def test_open_loop_drawing_reads_yaw_rate_from_its_steering_column():
    frame = {
        "ego_agent_past": np.zeros((PAST, 7), np.float32),
        "road_borders": np.zeros((30, 20, 2), np.float32),
        "lanes": np.zeros((140, 20, 6), np.float32),
        "route_lanes": np.zeros((25, 20, 6), np.float32),
        "route_traffic_light_past": np.zeros((25, PAST, 6), np.float32),
    }
    frame["ego_agent_past"][-1, 4:7] = [5.0, 0.1, 0.3]
    current = render_view(frame, np.zeros(3, np.float32))["ego_current_state"][0]
    assert current[4] == 5.0 and np.isclose(current[8], 0.1) and np.isclose(current[9], 0.3)


def test_closed_loop_seed_and_input_keep_the_steering_columns():
    past = np.zeros((PAST, 7), np.float32)
    past[:, 2] = 1.0
    past[:, 4:7] = [4.5, 0.08, 0.3]
    frame = {
        "ego_agent_past": past,
        "neighbor_agents_past": np.zeros((320, PAST, 4), np.float32),
        "agent_shape": np.zeros((320, 2), np.float32),
        "agent_label": np.zeros((320, 3), np.float32),
        "lanes": np.zeros((140, 20, 6), np.float32),
        "route_lanes": np.zeros((25, 20, 6), np.float32),
        "route_traffic_light_past": np.zeros((25, 20, 5), np.float32),
        "road_borders": np.zeros((30, 20, 2), np.float32),
    }

    class Timeline:
        native_h5 = True
        poses = np.array([[0.0, 0.0, 0.0]])
        sim_ego_shape = np.array([BUS.wheel_base, BUS.length, BUS.width], np.float32)

        def npz(self, _index):
            return frame

    pose, history, dyn = _ego_state_from_frame(Timeline(), 0)
    np.testing.assert_allclose(history[:, 3:6], np.tile([4.5, 0.3, 0.08], (PAST, 1)), rtol=1e-6)
    assert np.isclose(dyn.yaw_rate, 0.3)

    recen, _ = build_input_np(Timeline(), 0, pose, history, _EgoDyn(speed=5.0, yaw_rate=0.25))
    ego = recen["ego_agent_past"][0]
    assert ego.shape == (PAST, 7)
    np.testing.assert_allclose(ego[:-1, 4:7], np.tile([4.5, 0.08, 0.3], (PAST - 1, 1)), rtol=1e-6)
    np.testing.assert_allclose(ego[-1, 4:7], [5.0, 0.08, 0.25], rtol=1e-6)


def test_runner_names_the_format_a_model_needs():
    runner = NewDpOnnxRunner.__new__(NewDpOnnxRunner)
    runner.feature_widths = {"ego_agent_past": 7, "ego_shape": 4}
    runner.check_frame({"ego_agent_past": np.zeros((PAST, 7)), "ego_shape": np.zeros(4)})
    with pytest.raises(ValueError, match="7/4"):
        runner.check_frame({"ego_agent_past": np.zeros((PAST, 6)), "ego_shape": np.zeros(3)})
