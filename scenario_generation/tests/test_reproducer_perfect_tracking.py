"""Perfect tracking in the reproducer: the ego lands exactly on the predicted first point, heading
along the path, through the one shared implementation (mpc_tracker.place_on_trajectory)."""

import json
import math

import numpy as np
import pytest

from scenario_generation.mpc_tracker import place_on_trajectory
from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import _advance_step, _ego_pred_to_world, _seed_state
from scenario_generation.route_timeline import RouteTimeline

EGO_SHAPE = np.array([4.76, 7.24, 2.29], dtype=np.float32)


def _make_route(tmp_path, n=30):
    paths = []
    for i in range(n):
        p = tmp_path / f"route_{i:010d}.npz"
        np.savez_compressed(
            p,
            ego_agent_past=np.zeros((31, 3), dtype=np.float32),
            ego_shape=EGO_SHAPE,
            turn_indicators=np.zeros(31, dtype=np.int64),
        )
        (tmp_path / f"route_{i:010d}.json").write_text(
            json.dumps(
                {
                    "timestamp": float(i),
                    "x": float(i),
                    "y": 0.0,
                    "z": 0.0,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                }
            )
        )
        paths.append(p)
    return RouteTimeline(paths)


def _curved_prediction(bad_heading: float) -> np.ndarray:
    """An 80-step left curve (radius 20 m) in the ego frame, with a heading channel that
    disagrees with the path."""
    ang = np.linspace(0.05, 1.5, 80)
    r = 20.0
    pred = np.zeros((80, 4), dtype=np.float32)
    pred[:, 0] = r * np.sin(ang)
    pred[:, 1] = r - r * np.cos(ang)
    pred[:, 2] = math.cos(bad_heading)
    pred[:, 3] = math.sin(bad_heading)
    return pred


def test_perfect_mode_lands_on_the_predicted_point_heading_along_the_path(tmp_path):
    tl = _make_route(tmp_path)
    timers = Timers()
    s = _seed_state(
        tl,
        0,
        len(tl),
        search_radius=1.5,
        warmup_steps=0,
        near_miss_thresh=0.5,
        goal_reach_m=0.0,
        max_stuck_steps=0,
        timers=timers,
        tracker_mode="perfect",
    )
    start = s.live_pose.copy()
    pred = _curved_prediction(bad_heading=1.0)
    _advance_step(s, pred, idx=0, device="cpu", timers=timers)

    wxy, _ = _ego_pred_to_world(pred[:, :2], pred[:, 2:4], *start)
    expected, _ = place_on_trajectory(start, wxy, 0.1)
    assert s.live_pose[0] == pytest.approx(wxy[0, 0], abs=1e-4)
    assert s.live_pose[1] == pytest.approx(wxy[0, 1], abs=1e-4)
    # heading follows the path, not the prediction's heading channel (1.0 rad)
    assert s.live_pose[2] == pytest.approx(expected[2], abs=1e-4)
    assert abs(s.live_pose[2] - (start[2] + 1.0)) > 0.5


def test_cached_plan_steps_use_the_same_placement():
    # render_segment places the ego on a cached plan with place_on_trajectory; the tracker
    # does the same on a fresh plan, so both give the identical pose for the identical reference.
    from scenario_generation.mpc_tracker import PerfectTracker

    ang = np.linspace(0.05, 1.5, 40)
    ref = np.column_stack([20 * np.sin(ang), 20 - 20 * np.cos(ang)])
    x0 = np.array([0.0, 0.0, 0.0])
    pose_cached, v_cached = place_on_trajectory(x0, ref, 0.1)
    pose_tracker, v_tracker = PerfectTracker(dt=0.1).track(
        np.r_[x0, 0.0], np.column_stack([ref, np.zeros(40)])
    )
    np.testing.assert_allclose(pose_cached, pose_tracker, atol=1e-5)
    assert v_cached == pytest.approx(v_tracker)
