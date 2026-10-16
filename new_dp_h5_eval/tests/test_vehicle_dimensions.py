"""Native H5 is simulated and scored with the real wheelbase, not base_link_to_front.

Covers the closed-loop reproducer and the open-loop object-avoidance scorer.
"""

import json
import sys
import types
from types import SimpleNamespace

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from new_dp_h5_eval import closed_loop as native
from new_dp_h5_eval import open_loop, run_all_groups_closed_loop
from new_dp_h5_eval.closed_loop import NativeH5RouteTimeline, plan_optimizer_for_project
from new_dp_h5_eval.dataset import H5FrameIndex
from new_dp_h5_eval.open_loop import _scoring_ego_shapes, metric_view
from new_dp_h5_eval.schema import H5_FORMAT, MODEL_INPUT_NAMES
from new_dp_h5_eval.vehicle import (
    DEFAULT_VEHICLE_PARAMS,
    VehicleDimensions,
    load_vehicle_params,
    sim_ego_shape,
)
from scenario_generation.closed_loop_evaluation import ClosedLoopEvalConfig, RolloutParams
from scenario_generation.metrics.road_border import score_road_border_step
from scenario_generation.perf_timer import Timers
from scenario_generation.plan_optimizer import PlanOptimizerConfig
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
        # Ground truth the open-loop scorers read; not model inputs.
        frames.create_dataset("ego_agent_future", data=np.zeros((count, 80, 6), np.float32))
        frames.create_dataset(
            "neighbor_agents_future", data=np.zeros((count, 320, 80, 4), np.float32)
        )
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


def _write_index(path, shards):
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"h5_path": shard.name, "frame_index": 0, "frame_time_ns": 1_000_000_000}
                for shard in shards
            ]
        ),
        path,
    )
    return path


def test_open_loop_shapes_are_checked_per_shard(tmp_path):
    shard = _write_route(tmp_path / "frames.h5", ego_shape=ERGA_NATIVE)
    index = _write_index(tmp_path / "index.parquet", [shard])
    vehicles = load_vehicle_params(_write_params(tmp_path / "params.json", {"erga": ERGA}))
    with H5FrameIndex(index) as dataset:
        assert dataset.project_id(dataset.shard_path(0)) == "erga"
        shapes = _scoring_ego_shapes(dataset, {"object_avoidance": [({}, 0)]}, vehicles)
    np.testing.assert_allclose(shapes[0], ERGA_SIM, rtol=0, atol=1e-6)


class _FakeRunner:
    calls = 0

    def __init__(self, *_args, **_kwargs):
        pass

    def predict(self, frames, _seeds):
        _FakeRunner.calls += 1
        return np.zeros((len(frames), 1, 80, 4), np.float32), np.zeros(len(frames))


def _patch_open_loop(monkeypatch, scored_shapes):
    def fake_metric(prediction, data, _params):
        scored_shapes.append(np.asarray(data["ego_shape"]))
        return SimpleNamespace(
            scores={"success_rate_percent": torch.zeros(len(prediction))}, details={}
        )

    _FakeRunner.calls = 0
    monkeypatch.setattr(open_loop, "NewDpOnnxRunner", _FakeRunner)
    monkeypatch.setattr(
        open_loop, "METRICS", {"object_avoidance": fake_metric, "arrival": fake_metric}
    )


def test_open_loop_run_scores_object_avoidance_with_the_real_wheelbase(tmp_path, monkeypatch):
    shard = _write_route(tmp_path / "frames.h5", ego_shape=ERGA_NATIVE)
    index = _write_index(tmp_path / "index.parquet", [shard])
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({"object_avoidance": [{"h5_path": shard.name, "frame_index": 0}]}))
    vehicles = load_vehicle_params(_write_params(tmp_path / "params.json", {"erga": ERGA}))
    scored = []
    _patch_open_loop(monkeypatch, scored)
    open_loop.run(
        matrix,
        index,
        tmp_path / "m.onnx",
        tmp_path / "out",
        visualize=False,
        vehicle_params=vehicles,
    )
    (shapes,) = scored
    np.testing.assert_allclose(shapes, [ERGA_SIM], rtol=0, atol=1e-6)


def test_open_loop_run_rejects_a_bad_shard_before_any_inference(tmp_path, monkeypatch):
    good = _write_route(tmp_path / "good.h5", ego_shape=ERGA_NATIVE)
    # Converted with another layout; listed in a later metric than the good shard.
    bad = _write_route(tmp_path / "bad.h5", ego_shape=ERGA_SIM)
    index = _write_index(tmp_path / "index.parquet", [good, bad])
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "arrival": [{"h5_path": good.name, "frame_index": 0}],
                "object_avoidance": [{"h5_path": bad.name, "frame_index": 0}],
            }
        )
    )
    vehicles = load_vehicle_params(_write_params(tmp_path / "params.json", {"erga": ERGA}))
    _patch_open_loop(monkeypatch, [])
    with pytest.raises(ValueError, match="bad.h5"):
        open_loop.run(
            matrix,
            index,
            tmp_path / "m.onnx",
            tmp_path / "out",
            visualize=False,
            vehicle_params=vehicles,
        )
    assert _FakeRunner.calls == 0
    assert not (tmp_path / "out" / "details").exists()


def test_metric_view_takes_the_scoring_shape():
    frame = {
        "ego_agent_past": np.zeros((31, 6), np.float32),
        "ego_agent_future": np.zeros((80, 6), np.float32),
        "route_lanes": np.zeros((25, 20, 6), np.float32),
        "lanes": np.zeros((140, 20, 6), np.float32),
        "neighbor_agents_future": np.zeros((320, 80, 4), np.float32),
        "neighbor_agents_past": np.zeros((320, 31, 4), np.float32),
        "agent_shape": np.zeros((320, 2), np.float32),
        "agent_label": np.zeros((320, 3), np.float32),
        "ego_shape": np.array(ERGA_NATIVE, np.float32),
    }
    view = metric_view(frame, np.array(ERGA_SIM, np.float32))
    np.testing.assert_allclose(view["ego_shape"], ERGA_SIM, rtol=0, atol=1e-6)


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
        (open_loop.main, ["matrix.json", "index.parquet", "m.onnx", "out"]),
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


VEHICLE_A = {
    "ego_wheel_base": 3.0,
    "ego_front_overhang": 1.0,
    "ego_length": 5.0,
    "ego_width": 2.0,
}
VEHICLE_A_NATIVE = [4.0, 5.0, 2.0]
VEHICLE_B = {
    "ego_wheel_base": 2.0,
    "ego_front_overhang": 0.5,
    "ego_length": 3.0,
    "ego_width": 1.5,
}
VEHICLE_B_NATIVE = [2.5, 3.0, 1.5]
PROJECT_VEHICLES = {"project_a": "vehicle_a", "project_b": "vehicle_b"}


def _optimizer_config(vehicle: str, wheel_base: float) -> PlanOptimizerConfig:
    return PlanOptimizerConfig(
        param_yaml=f"{vehicle}/ml_planner.param.yaml",
        vehicle_yaml=f"{vehicle}/vehicle_info.param.yaml",
        vehicle={"wheel_base": wheel_base, "wheel_tread": 2.0},
    )


def _per_vehicle_optimizers() -> dict[str, PlanOptimizerConfig]:
    return {"vehicle_a": _optimizer_config("a", 3.0), "vehicle_b": _optimizer_config("b", 2.0)}


def _two_vehicles(tmp_path):
    return load_vehicle_params(
        _write_params(tmp_path / "params.json", {"project_a": VEHICLE_A, "project_b": VEHICLE_B})
    )


@pytest.mark.parametrize(
    "project, optimizers, message",
    [
        ("unknown", _per_vehicle_optimizers(), "not in the project vehicle map"),
        # A config placed under the wrong vehicle.
        ("project_a", {"vehicle_a": _optimizer_config("b", 2.0)}, "wheel_base 2.0"),
    ],
)
def test_plan_optimizer_for_project_rejects(tmp_path, project, optimizers, message):
    with pytest.raises(ValueError, match=message):
        plan_optimizer_for_project(
            project, PROJECT_VEHICLES, optimizers, _two_vehicles(tmp_path), source="s"
        )


def test_each_route_rolls_out_with_its_vehicles_optimizer(tmp_path, monkeypatch):
    from diffusion_planner.config.closed_loop_config import ClosedLoopConfig

    routes = [
        {
            "route_id": "route_a",
            "h5_path": str(
                _write_route(tmp_path / "a.h5", ego_shape=VEHICLE_A_NATIVE, project_id="project_a")
            ),
        },
        {
            "route_id": "route_b",
            "h5_path": str(
                _write_route(tmp_path / "b.h5", ego_shape=VEHICLE_B_NATIVE, project_id="project_b")
            ),
        },
    ]
    seen = {}

    def fake_render_segment(model, model_args, tl, start, end, png_dir, **kwargs):
        seen[tl.project_id] = kwargs["plan_optimizer"]
        return {}

    monkeypatch.setattr(native, "render_segment", fake_render_segment)
    params = native.rollout_params_from_closed_loop_config(
        ClosedLoopConfig(device="cpu"), drop_objects=False, draw_every=None, plan_optimizer=None
    )
    evaluator = native.NativeH5FullRouteClosedLoopEvaluation(
        model=None,
        model_args=None,
        config=ClosedLoopEvalConfig(
            out_dir=tmp_path / "out", params=params, fps=10.0, verbose=False, profile=False
        ),
        routes=routes,
        vehicle_params=_two_vehicles(tmp_path),
        plan_optimizers=_per_vehicle_optimizers(),
        project_vehicles=PROJECT_VEHICLES,
        seg_len=100,
        ddp_rank=0,
        ddp_world_size=1,
    )
    for job in evaluator.discover_jobs():
        evaluator.run_job(job, frames_root=tmp_path / "frames")
    assert seen["project_a"].vehicle["wheel_base"] == 3.0
    assert seen["project_b"].vehicle["wheel_base"] == 2.0
    assert evaluator.config.params.plan_optimizer is None


def test_closed_loop_runner_stops_on_a_route_without_its_vehicles_config(
    tmp_path, monkeypatch, capsys
):
    module = types.ModuleType("ml_planner_optimizer")
    module.__file__ = str(tmp_path / "install/lib/python3.10/site-packages/ml_planner_optimizer.so")
    monkeypatch.setitem(sys.modules, "ml_planner_optimizer", module)
    root = tmp_path / "configs" / "vehicle_a"
    root.mkdir(parents=True)
    (root / "ml_planner.param.yaml").write_text("")
    (root / "vehicle_info.param.yaml").write_text(
        "/**:\n  ros__parameters:\n    wheel_base: 3.0\n    wheel_tread: 1.6\n"
    )
    shard = _write_route(tmp_path / "b.h5", ego_shape=VEHICLE_B_NATIVE, project_id="project_b")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"site": [str(shard)]}))
    vehicle_map = tmp_path / "project_vehicle_map.json"
    vehicle_map.write_text(json.dumps(PROJECT_VEHICLES))
    _two_vehicles(tmp_path)
    argv = [
        "runner",
        "--closed_loop_h5_root", str(manifest),
        "--model_path", "m.onnx",
        "--out_root", str(tmp_path / "out"),
        "--vehicle_params", str(tmp_path / "params.json"),
        "--project_vehicle_map", str(vehicle_map),
        "--plan_optimizer",
        "--plan_optimizer_config_dir", str(tmp_path / "configs"),
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        run_all_groups_closed_loop.main()
    assert exc.value.code == 2
    assert "no plan optimizer config for vehicle 'vehicle_b'" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
