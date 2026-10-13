"""Native H5 is simulated and scored with the real wheelbase, not base_link_to_front.

Covers the closed-loop reproducer.
"""

import json
import sys

import h5py
import numpy as np
import pytest

from new_dp_h5_eval import run_all_groups_closed_loop
from new_dp_h5_eval.closed_loop import NativeH5RouteTimeline
from new_dp_h5_eval.schema import H5_FORMAT, MODEL_INPUT_NAMES
from new_dp_h5_eval.vehicle import (
    DEFAULT_VEHICLE_PARAMS,
    VehicleDimensions,
    load_vehicle_params,
    sim_ego_shape,
)
from scenario_generation.metrics.road_border import score_road_border_step
from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import _seed_state, _sim_ego_shape

# Erga as in dataset/data_converter_param.json of the meta repository.
ERGA = {
    "ego_wheel_base": 4.9901,
    "ego_front_overhang": 2.5667,
    "ego_length": 10.8812,
    "ego_width": 2.8736,
}
ERGA_NATIVE = [7.5568, 10.8812, 2.8736]  # what the ML-Planner converter writes
ERGA_SIM = [4.9901, 10.8812, 2.8736]


_FIELD_SHAPES = {
    "ego_agent_past": (31, 6),
    "neighbor_agents_past": (320, 31, 4),
    "agent_shape": (320, 2),
    "agent_label": (320, 3),
    "lanes": (140, 20, 6),
    "route_lanes": (25, 20, 6),
}


def _write_params(path, entries):
    path.write_text(json.dumps(entries))
    return path


def _write_route(path, *, ego_shape, project_id="erga", count=3):
    with h5py.File(path, "w") as file:
        file.attrs["format"] = H5_FORMAT
        file.attrs["format_version"] = 5
        file.attrs["num_frames"] = count
        file.attrs["frame_interval_s"] = 0.1
        file.attrs["pose_frame_id"] = "map"
        if project_id is not None:
            file.attrs["project_id"] = project_id
        frames = file.create_group("frames")
        for name in MODEL_INPUT_NAMES:
            if name == "ego_shape":
                data = np.tile(np.asarray(ego_shape, np.float32), (count, 1))
            else:
                data = np.zeros(_FIELD_SHAPES.get(name, (1,)), np.float32)[None].repeat(count, 0)
            frames.create_dataset(name, data=data)
        metadata = file.create_group("metadata")
        metadata.create_dataset(
            "frame_time_ns", data=[1_000_000_000 + i * 100_000_000 for i in range(count)]
        )
        for name, values in {
            "x": np.arange(count, dtype=np.float64),
            "y": np.zeros(count),
            "z": np.zeros(count),
            "qx": np.zeros(count),
            "qy": np.zeros(count),
            "qz": np.zeros(count),
            "qw": np.ones(count),
        }.items():
            metadata.create_dataset(name, data=values)
    return path


def test_load_vehicle_params_skips_entries_without_ego_dimensions(tmp_path):
    params = _write_params(
        tmp_path / "params.json",
        {"default": {"min_distance": 50.0}, "erga": {**ERGA, "min_distance": 50.0}},
    )
    vehicles = load_vehicle_params(params)
    assert list(vehicles) == ["erga"]
    assert vehicles["erga"] == VehicleDimensions(4.9901, 2.5667, 10.8812, 2.8736)
    assert vehicles["erga"].base_link_to_front == pytest.approx(7.5568)


def test_load_vehicle_params_rejects_a_table_without_any_vehicle(tmp_path):
    params = _write_params(tmp_path / "params.json", {"default": {"min_distance": 50.0}})
    with pytest.raises(ValueError, match="no project entry"):
        load_vehicle_params(params)


@pytest.mark.parametrize(
    "entries, message",
    [([ERGA], "json object"), ({"erga": {**ERGA, "ego_wheel_base": None}}, "non-numeric")],
)
def test_load_vehicle_params_reports_a_malformed_table(tmp_path, entries, message):
    with pytest.raises(ValueError, match=message):
        load_vehicle_params(_write_params(tmp_path / "params.json", entries))


def test_sim_ego_shape_replaces_base_link_to_front_with_the_wheelbase():
    vehicles = {"erga": VehicleDimensions(4.9901, 2.5667, 10.8812, 2.8736)}
    shape = sim_ego_shape(np.array([ERGA_NATIVE] * 2), "erga", vehicles, source="shard")
    assert shape.dtype == np.float32
    np.testing.assert_allclose(shape, ERGA_SIM, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "native, project, message",
    [
        ([4.9901, 10.8812, 2.8736], "erga", "disagree"),  # already a wheelbase
        ([7.5568, 10.0, 2.8736], "erga", "disagree"),  # converted with another length
        (ERGA_NATIVE, "x2_dev", "no ego dimensions"),
    ],
)
def test_sim_ego_shape_rejects_a_shard_that_does_not_match_the_table(native, project, message):
    vehicles = {"erga": VehicleDimensions(4.9901, 2.5667, 10.8812, 2.8736)}
    with pytest.raises(ValueError, match=message):
        sim_ego_shape(np.array([native]), project, vehicles, source="shard")


def test_native_timeline_exposes_the_simulated_shape(tmp_path):
    shard = _write_route(tmp_path / "frames.h5", ego_shape=ERGA_NATIVE)
    vehicles = load_vehicle_params(_write_params(tmp_path / "params.json", {"erga": ERGA}))
    timeline = NativeH5RouteTimeline(shard, vehicle_params=vehicles)
    try:
        np.testing.assert_allclose(timeline.sim_ego_shape, ERGA_SIM, rtol=0, atol=1e-6)
        np.testing.assert_allclose(_sim_ego_shape(timeline, 0), ERGA_SIM, rtol=0, atol=1e-6)
        # The model input still carries the native value.
        np.testing.assert_allclose(timeline.npz(0)["ego_shape"], ERGA_NATIVE, rtol=0, atol=1e-6)
    finally:
        timeline.close()


def test_native_timeline_without_vehicle_params_cannot_be_rolled_out(tmp_path):
    shard = _write_route(tmp_path / "frames.h5", ego_shape=ERGA_NATIVE)
    timeline = NativeH5RouteTimeline(shard)
    try:
        assert timeline.sim_ego_shape is None
        with pytest.raises(ValueError, match="vehicle_params"):
            _sim_ego_shape(timeline, 0)
    finally:
        timeline.close()


def test_native_timeline_requires_a_project_id_with_vehicle_params(tmp_path):
    shard = _write_route(tmp_path / "frames.h5", ego_shape=ERGA_NATIVE, project_id=None)
    vehicles = {"erga": VehicleDimensions(4.9901, 2.5667, 10.8812, 2.8736)}
    with pytest.raises(ValueError, match="project_id"):
        NativeH5RouteTimeline(shard, vehicle_params=vehicles)


def test_seeded_segment_simulates_and_scores_with_the_real_wheelbase(tmp_path):
    shard = _write_route(tmp_path / "frames.h5", ego_shape=ERGA_NATIVE)
    vehicles = load_vehicle_params(_write_params(tmp_path / "params.json", {"erga": ERGA}))
    timeline = NativeH5RouteTimeline(shard, vehicle_params=vehicles)
    try:
        state = _seed_state(
            timeline,
            0,
            3,
            search_radius=5.0,
            warmup_steps=0,
            near_miss_thresh=0.5,
            goal_reach_m=2.0,
            max_stuck_steps=10,
            timers=Timers(),
        )
        assert state.tracker.wheelbase == pytest.approx(4.9901, abs=1e-5)
        np.testing.assert_allclose(state.ego_shape, ERGA_SIM, rtol=0, atol=1e-6)
    finally:
        timeline.close()


def test_road_border_is_scored_with_the_metric_shape():
    # A border across the path 8.5 m ahead of the rear axle: inside the box the native value
    # gives (front at (10.8812 + 7.5568) / 2 = 9.22 m) but 0.56 m beyond the simulated box
    # (front at (10.8812 + 4.9901) / 2 = 7.94 m, the reproducer's symmetric overhangs).
    line_strings = np.zeros((1, 1, 20, 4), np.float32)
    line_strings[0, 0, :2, :2] = [[8.5, -5.0], [8.5, 5.0]]
    line_strings[0, 0, :2, 3] = 1.0
    native_only = {"line_strings": line_strings, "ego_shape": np.array([ERGA_NATIVE], np.float32)}
    with_metric = {**native_only, "metric_ego_shape": np.array([ERGA_SIM], np.float32)}
    assert score_road_border_step(native_only, device="cpu")["rb_dist_m"] == 0.0
    assert score_road_border_step(with_metric, device="cpu")["rb_dist_m"] == pytest.approx(
        8.5 - (10.8812 + 4.9901) / 2, abs=0.05
    )


def test_npz_timelines_keep_their_ego_shape():
    class NpzTimeline:
        def npz(self, idx):
            return {"ego_shape": np.array([[4.76012, 7.2369, 2.42741]])}

    np.testing.assert_allclose(
        _sim_ego_shape(NpzTimeline(), 0), [4.76012, 7.2369, 2.42741], rtol=0, atol=1e-6
    )


def test_default_vehicle_params_is_the_meta_repository_table():
    assert DEFAULT_VEHICLE_PARAMS.parts[-2:] == ("dataset", "data_converter_param.json")


@pytest.mark.parametrize(
    "main, argv",
    [
        (
            run_all_groups_closed_loop.main,
            ["--closed_loop_h5_root", "manifest.json", "--model_path", "m.onnx", "--out_root", "o"],
        ),
    ],
)
def test_runners_stop_when_the_vehicle_params_are_missing(
    tmp_path, monkeypatch, capsys, main, argv
):
    missing = str(tmp_path / "missing.json")
    monkeypatch.setattr(sys, "argv", ["runner", *argv, "--vehicle_params", missing])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "--vehicle_params" in capsys.readouterr().err
