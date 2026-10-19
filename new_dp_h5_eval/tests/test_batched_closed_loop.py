"""The batched closed-loop backend reproduces the reproducer's rollout, scoring and trace on a recorded route."""

import json
from dataclasses import replace

import h5py
import numpy as np
import pytest
import torch
from diffusion_planner.config.closed_loop_config import ClosedLoopConfig

from new_dp_h5_eval.batched_closed_loop import (
    VIDEO_NAME,
    BatchedNativeH5Evaluation,
    run_batched,
    simulator_config,
)
from new_dp_h5_eval.closed_loop import (
    NativeH5RouteTimeline,
    model_args,
    rollout_params_from_closed_loop_config,
)
from new_dp_h5_eval.schema import H5_FORMAT, MODEL_INPUT_NAMES
from new_dp_h5_eval.vehicle import load_vehicle_params
from scenario_generation.closed_loop_evaluation import ClosedLoopEvalConfig, FullRouteRouteJob
from scenario_generation.render_pool import render_pool
from scenario_generation.reproducer_rollout import render_segment

FRAMES = 80
SPACING_M = 0.5
SHAPES = {
    "ego_agent_past": (31, 6),
    "neighbor_agents_past": (320, 31, 4),
    "agent_shape": (320, 2),
    "agent_label": (320, 3),
    "lanes": (140, 20, 6),
    "lane_types": (140, 20),
    "lanes_speed_limit": (140, 1),
    "lane_traffic_light_past": (140, 31, 6),
    "lane_traffic_light_future": (140, 80, 6),
    "route_lanes": (25, 20, 6),
    "route_lane_types": (25, 20),
    "route_lanes_speed_limit": (25, 1),
    "route_traffic_light_past": (25, 31, 6),
    "route_traffic_light_future": (25, 80, 6),
    "intersection_area": (10, 40, 2),
    "stop_lines": (30, 2, 2),
    "road_borders": (30, 20, 2),
    "goal_pose": (4,),
    "ego_shape": (3,),
    "turn_indicators": (31,),
}
ERGA = {
    "ego_wheel_base": 4.9901,
    "ego_front_overhang": 2.5667,
    "ego_length": 10.8812,
    "ego_width": 2.8736,
}
CAR_X, CAR_Y = 25.0, 3.6


def _frames() -> dict[str, np.ndarray]:
    """Frames of an ego driving 5 m/s along a straight two-lane road past a parked car in the left lane, each in its own ego frame."""
    data = {name: np.zeros((FRAMES, *shape), np.float32) for name, shape in SHAPES.items()}
    x = np.arange(FRAMES) * SPACING_M
    steps = np.arange(-30, 1) * SPACING_M
    data["ego_agent_past"][..., 0] = steps
    data["ego_agent_past"][..., 2] = 1.0
    data["ego_agent_past"][..., 4] = 5.0
    ahead = np.linspace(-10.0, 50.0, 20)
    for lanes, key in ((data["lanes"], "lanes"), (data["route_lanes"], "route_lanes")):
        lanes[:, 0, :, 0] = ahead
        lanes[:, 0, :, 3], lanes[:, 0, :, 5] = 1.75, -1.75
        data[f"{key}_speed_limit"][:, 0] = 10.0
    data["lanes"][:, 1, :, 0] = ahead
    data["lanes"][:, 1, :, 1] = 3.5
    data["lanes"][:, 1, :, 3], data["lanes"][:, 1, :, 5] = 1.75, -1.75
    data["road_borders"][:, 0, :, 0] = ahead
    data["road_borders"][:, 0, :, 1] = -1.75
    data["road_borders"][:, 1, :, 0] = ahead
    data["road_borders"][:, 1, :, 1] = 5.25
    data["neighbor_agents_past"][:, 0, :, 0] = (CAR_X - x)[:, None]
    data["neighbor_agents_past"][:, 0, :, 1] = CAR_Y
    data["neighbor_agents_past"][:, 0, :, 2] = 1.0
    data["agent_shape"][:, 0] = (1.8, 4.5)
    data["agent_label"][:, 0, 0] = 1.0
    data["goal_pose"][:, 0] = x[-1] - x
    data["goal_pose"][:, 2] = 1.0
    data["ego_shape"][:] = (7.5568, 10.8812, 2.8736)
    data["turn_indicators"][:] = 1.0
    return data


def _write_route(path) -> None:
    """Write the frames of ``_frames`` as a native closed-loop H5 shard."""
    with h5py.File(path, "w") as file:
        file.attrs.update(
            format=H5_FORMAT,
            format_version=5,
            num_frames=FRAMES,
            frame_interval_s=0.1,
            pose_frame_id="map",
            project_id="erga",
        )
        frames = file.create_group("frames")
        for name, values in _frames().items():
            frames.create_dataset(name, data=values)
        metadata = file.create_group("metadata")
        metadata.create_dataset(
            "frame_time_ns", data=1_000_000_000 + np.arange(FRAMES) * 100_000_000
        )
        poses = {"x": np.arange(FRAMES) * SPACING_M, "qw": np.ones(FRAMES)}
        for name in ("x", "y", "z", "qx", "qy", "qz", "qw"):
            metadata.create_dataset(name, data=poses.get(name, np.zeros(FRAMES)))


class StraightPlanner:
    """Plans ahead at ``speed``, drifting sideways at ``drift`` m/s, and signals DISABLE, in the reproducer's model protocol."""

    def __init__(self, speed: float, drift: float = 0.0) -> None:
        self.speed = speed
        self.drift = drift
        self.seed = 0

    def __call__(self, data, seeds=None):
        batch = data["ego_agent_past"].shape[0]
        times = torch.arange(1, 81) * 0.1
        heading = torch.atan2(torch.tensor(self.drift), torch.tensor(self.speed))
        plan = torch.zeros(batch, 1, 80, 4)
        plan[..., 0] = times * self.speed
        plan[..., 1] = times * self.drift
        plan[..., 2] = torch.cos(heading)
        plan[..., 3] = torch.sin(heading)
        logits = torch.full((batch, 5), -1e9)
        logits[:, 1] = 1.0
        return None, {"prediction": plan, "turn_indicator_logit": logits}


@pytest.fixture
def route(tmp_path):
    """A recorded route, its vehicle table and the evaluation settings."""
    path = tmp_path / "frames.h5"
    _write_route(path)
    (tmp_path / "params.json").write_text(json.dumps({"erga": ERGA}))
    vehicles = load_vehicle_params(tmp_path / "params.json")
    params = rollout_params_from_closed_loop_config(
        ClosedLoopConfig(device="cpu"), drop_objects=False, draw_every=None
    )
    return path, vehicles, params


def _canonical(path, vehicles, params, model, out_dir, stop=FRAMES):
    """Metrics and trace of the reproducer's own rollout of frames ``0:stop``."""
    timeline = NativeH5RouteTimeline(path, 0, stop, vehicle_params=vehicles)
    try:
        metrics = render_segment(
            model,
            model_args(),
            timeline,
            0,
            stop,
            out_dir,
            **params.render_kwargs(),
            draw_pool=None,
        )
    finally:
        timeline.close()
    return metrics, (out_dir / "rollout.jsonl").read_text().splitlines()


def _batched(path, vehicles, params, model, frames_root, stop=FRAMES):
    """Metrics and trace of the batched backend on frames ``0:stop``."""
    route = {"h5_path": str(path), "route_id": "group__route", "frame_stop": stop}
    evaluator = BatchedNativeH5Evaluation(
        model,
        model_args(),
        config=ClosedLoopEvalConfig(
            out_dir=frames_root / "out", params=params, fps=10.0, verbose=False, profile=False
        ),
        routes=[route],
        vehicle_params=vehicles,
        seg_len=100000,
        ddp_rank=0,
        ddp_world_size=1,
    )
    job = FullRouteRouteJob(job_id="group__route", route_key="group__route", seg_len=100000)
    run_batched(model, {"group": [job]}, {"group": evaluator}, frames_root, None)
    ((_, png_dir, metrics),) = evaluator.rolled_out["group__route"]
    return metrics, (png_dir / "rollout.jsonl").read_text().splitlines()


@pytest.mark.parametrize(
    ("speed", "drift", "abort_m", "drop_objects", "stop"),
    [
        (5.0, 0.0, 50.0, False, FRAMES),
        (7.0, 0.0, 50.0, False, FRAMES),
        (5.0, 1.5, 2.0, False, FRAMES),
        (7.0, 0.0, 50.0, True, FRAMES),
        (5.0, 0.0, 50.0, False, 6),
    ],
)
def test_batched_rollout_reproduces_the_reproducer(
    route, tmp_path, speed, drift, abort_m, drop_objects, stop
):
    """Steps, termination (a drive that leaves the recording ends as diverged, a route shorter than the goal radius ends at once), every event block and every trace line agree with the reproducer, with and without objects."""
    path, vehicles, params = route
    params = replace(params, abort_deviation_m=abort_m, abort_after=5, drop_objects=drop_objects)
    planner = StraightPlanner(speed, drift)
    expected, expected_trace = _canonical(path, vehicles, params, planner, tmp_path / "c", stop)
    actual, actual_trace = _batched(path, vehicles, params, planner, tmp_path / "b", stop)
    if drift:
        assert expected["terminated"] == "diverged"
    for key in (
        "n_steps_run",
        "terminated",
        "object_rear",
        "red_light_violation",
        "strong_brake",
        "reproducer",
        "turn_indicator",
    ):
        assert actual[key] == expected[key], key
    for key in ("route_completion", "mean_gt_deviation_m", "progress_m"):
        assert actual[key] == pytest.approx(expected[key], abs=1e-3), key
    for block in ("object", "road_border"):
        for key, value in expected[block].items():
            if key.startswith("_"):
                continue
            assert actual[block][key] == pytest.approx(value, abs=1e-3), (block, key)
    assert len(actual_trace) == len(expected_trace)
    for line_a, line_e in zip(actual_trace, expected_trace):
        a, e = json.loads(line_a), json.loads(line_e)
        assert a.keys() == e.keys()
        for key, value in e.items():
            assert a[key] == (
                pytest.approx(value, abs=2e-3) if isinstance(value, (float, list)) else value
            ), (e.get("k"), key)


def test_drawn_segments_leave_one_video_and_no_frames(route, tmp_path):
    """With drawing on, each segment's frames are encoded into its video as soon as it ends."""
    path, vehicles, params = route
    params = replace(params, draw_every=10)
    with render_pool(1) as pool:
        route_entry = {"h5_path": str(path), "route_id": "group__route"}
        evaluator = BatchedNativeH5Evaluation(
            StraightPlanner(5.0),
            model_args(),
            config=ClosedLoopEvalConfig(
                out_dir=tmp_path / "out", params=params, fps=10.0, verbose=False, profile=False
            ),
            routes=[route_entry],
            vehicle_params=vehicles,
            seg_len=100000,
            ddp_rank=0,
            ddp_world_size=1,
        )
        job = FullRouteRouteJob(job_id="group__route", route_key="group__route", seg_len=100000)
        run_batched(StraightPlanner(5.0), {"group": [job]}, {"group": evaluator}, tmp_path, pool)
    ((_, png_dir, _),) = evaluator.rolled_out["group__route"]
    assert (png_dir / VIDEO_NAME).stat().st_size > 0
    assert not list(png_dir.glob("*.png"))


def test_simulator_settings_follow_the_rollout_parameters(route):
    """The replan period, unstick, divergence and heading gate come from the rollout parameters."""
    _, _, params = route
    config = simulator_config(
        replace(params, replan_interval=4, abort_deviation_m=0.0, yaw_gate=False)
    )
    assert config.replan_steps == 4
    assert config.diverge_m is None
    assert config.frames.heading_gate_rad == float("inf")
    assert config.unstick_after == params.unstick_after


@pytest.mark.parametrize(
    "change",
    [
        {"tracker_mode": "mpc"},
        {"neighbor_history_mode": "sim"},
        {"goal_mode": "route"},
        {"warmup_steps": 3},
    ],
)
def test_settings_the_simulator_does_not_reproduce_are_rejected(route, change):
    """A rollout setting the devkit simulator does not reproduce stops the run before it starts."""
    _, _, params = route
    with pytest.raises(ValueError, match="reproduces only"):
        simulator_config(replace(params, **change))
