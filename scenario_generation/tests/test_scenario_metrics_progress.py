import numpy as np
import pytest

from scenario_generation.scenario_metrics import registry
from scenario_generation.scenario_metrics.progress import departure_progress, yield_progress
from scenario_generation.scenario_metrics.testing import (
    make_input,
    speed_profile_path,
    straight_path,
)

ANCHOR = 10


def _ego_with_speeds(speeds: np.ndarray):
    """Ego standing still until ANCHOR, then following ``speeds`` along +x."""
    return speed_profile_path(np.concatenate([np.zeros(ANCHOR), speeds]))


def _input(label: str, ego_xy, ego_yaw, *, terminated: str = "max_steps", n_frames: int = 80):
    rec_xy, rec_yaw = straight_path(n_frames, 2.0)
    return make_input(
        label=label,
        ego_xy=ego_xy,
        ego_yaw=ego_yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=ANCHOR,
        terminated=terminated,
    )


@pytest.mark.parametrize(
    "label", ["departure", "traffic_light_go", "pedestrian_yield", "vehicle_yield", "temporal_stop"]
)
def test_labels_are_registered(label):
    assert registry.METRICS[label] in (departure_progress, yield_progress)


def test_departure_pass_and_fail():
    ok = departure_progress(_input("departure", *_ego_with_speeds(np.full(40, 1.0))))
    assert ok.passed is True
    assert ok.values["progress_m"] == pytest.approx(3.0)
    assert ok.details["time_to_threshold_s"] == pytest.approx(2.0)
    slow = departure_progress(_input("traffic_light_go", *_ego_with_speeds(np.full(40, 0.5))))
    assert slow.passed is False
    assert slow.values["progress_m"] == pytest.approx(1.5)
    assert slow.details["time_to_threshold_s"] is None


def test_yield_pass_and_fail():
    held = yield_progress(_input("pedestrian_yield", *_ego_with_speeds(np.full(40, 0.1))))
    assert held.passed is True
    assert held.values["progress_m"] == pytest.approx(0.3)
    crept = yield_progress(_input("vehicle_yield", *_ego_with_speeds(np.full(40, 1.0))))
    assert crept.passed is False
    assert crept.details["time_exceeded_s"] == pytest.approx(0.6)


def test_anchor_never_reached_is_not_applicable():
    xy, yaw = straight_path(ANCHOR - 2, 0.0)
    for fn, label in [(departure_progress, "departure"), (yield_progress, "temporal_stop")]:
        r = fn(_input(label, xy, yaw, terminated="max_steps"))
        assert r.passed is None and r.reason


def test_early_goal_passes_departure():
    # Trace ends 1 s after the anchor, having moved only 0.3 m.
    xy, yaw = _ego_with_speeds(np.full(10, 0.3))
    dep = departure_progress(_input("departure", xy, yaw, terminated="goal"))
    assert dep.passed is True and dep.details["horizon_truncated"]
    xy, yaw = straight_path(ANCHOR - 2, 3.0)
    assert departure_progress(_input("departure", xy, yaw, terminated="goal")).passed is None


def test_early_goal_yield_fails_only_past_the_waiting_point():
    # The human (2 m/s) is at arc 2.0 m at the anchor; the ego stands at 0 until then.
    xy, yaw = _ego_with_speeds(np.full(10, 0.3))
    short = yield_progress(_input("vehicle_yield", xy, yaw, terminated="goal"))
    assert short.passed is None and "stopped short" in short.reason
    assert short.values["ego_max_arc_m"] == pytest.approx(0.27)
    assert short.values["human_anchor_arc_m"] == pytest.approx(2.0)
    # Ego already 2.6 m along the path at the anchor, then creeping (< 0.5 m progress
    # since anchor_step): it got past the human's waiting point all the same.
    xy, yaw = speed_profile_path(np.concatenate([np.full(ANCHOR, 2.6), np.full(10, 0.03)]))
    past = yield_progress(_input("vehicle_yield", xy, yaw, terminated="goal"))
    assert past.passed is False and past.values["ego_max_arc_m"] >= 2.5
    # Goal before the anchor was ever replayed: same rule.
    rec_idx = np.zeros(ANCHOR + 5, dtype=int)
    xy, yaw = straight_path(ANCHOR + 5, 3.0)
    rec_xy, rec_yaw = straight_path(80, 2.0)
    far = make_input(
        label="vehicle_yield",
        ego_xy=xy,
        ego_yaw=yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=ANCHOR,
        rec_idx=rec_idx,
        terminated="goal",
    )
    assert far.anchor_step is None
    assert yield_progress(far).passed is False
    xy, yaw = straight_path(ANCHOR - 2, 1.0)
    assert yield_progress(_input("vehicle_yield", xy, yaw, terminated="goal")).passed is None


def test_truncated_horizon_without_goal_scores_available_steps():
    xy, yaw = _ego_with_speeds(np.full(10, 0.3))
    r = yield_progress(_input("pedestrian_yield", xy, yaw, terminated="max_steps"))
    assert r.passed is True and r.details["horizon_truncated"] and r.details["available_steps"] == 9
    assert "before the horizon" in r.reason
    # A verdict the available steps already decide stands, whatever the termination.
    xy, yaw = _ego_with_speeds(np.full(10, 3.0))
    assert (
        yield_progress(_input("pedestrian_yield", xy, yaw, terminated="max_steps")).passed is False
    )
    assert departure_progress(_input("departure", xy, yaw, terminated="max_steps")).passed is True
    # Nothing after the anchor at all: not applicable.
    xy, yaw = _ego_with_speeds(np.zeros(1))
    assert yield_progress(_input("pedestrian_yield", xy, yaw)).passed is None


def test_progress_is_measured_along_the_path_not_euclidean():
    # Stationary on the path, then a 3 m sideways swerve: large displacement, no progress.
    xy, yaw = _ego_with_speeds(np.zeros(40))
    xy[ANCHOR + 1 :, 1] = np.minimum(np.arange(len(xy) - ANCHOR - 1) * 0.3, 3.0)
    assert np.linalg.norm(xy[-1] - xy[ANCHOR]) > 2.0
    dep = departure_progress(_input("departure", xy, yaw))
    assert dep.passed is False and dep.values["progress_m"] == pytest.approx(0.0)
    assert yield_progress(_input("pedestrian_yield", xy, yaw)).passed is True


def test_reference_progress_is_the_humans():
    r = departure_progress(_input("departure", *_ego_with_speeds(np.full(40, 1.0))))
    assert r.values["reference_progress_m"] == pytest.approx(2.0 * 3.0)


def test_open_loop_reference_counts_a_swerve_as_departure():
    # The swerve above: open loop's Euclidean displacement departs, arc progress does not.
    xy, yaw = _ego_with_speeds(np.zeros(40))
    xy[ANCHOR + 1 :, 1] = np.minimum(np.arange(len(xy) - ANCHOR - 1) * 0.3, 3.0)
    dep = departure_progress(_input("departure", xy, yaw))
    assert dep.passed is False and dep.values["progress_m"] == pytest.approx(0.0)
    assert dep.values["ol_max_displacement_m"] == pytest.approx(3.0)
    assert dep.values["ol_passed"] == 1.0


def test_open_loop_reference_forward_is_along_the_anchor_heading():
    # Ego heading 30 deg off the recorded path, creeping 0.55 m along its own heading:
    # 0.48 m of arc progress (yielded), 0.55 m along its +x (open loop: not yielded).
    heading = np.radians(30.0)
    speeds = np.r_[np.zeros(ANCHOR), np.full(11, 0.5), np.zeros(29)]
    xy, yaw = speed_profile_path(speeds, heading=heading)
    r = yield_progress(_input("pedestrian_yield", xy, yaw))
    assert r.passed is True
    assert r.values["progress_m"] == pytest.approx(0.55 * np.cos(heading))
    assert r.values["ol_max_forward_progress_m"] == pytest.approx(0.55)
    assert r.values["ol_passed"] == 0.0


def test_open_loop_reference_leaves_verdicts_and_values_unchanged():
    existing = {"progress_m", "threshold_m", "horizon_s", "reference_progress_m"}
    dep = departure_progress(_input("departure", *_ego_with_speeds(np.full(40, 1.0))))
    assert dep.passed is True and dep.values["progress_m"] == pytest.approx(3.0)
    assert set(dep.values) == existing | {"ol_max_displacement_m", "ol_passed"}
    # Straight along the path, the two definitions agree.
    assert dep.values["ol_max_displacement_m"] == pytest.approx(3.0)
    held = yield_progress(_input("pedestrian_yield", *_ego_with_speeds(np.full(40, 0.1))))
    assert held.passed is True and held.values["progress_m"] == pytest.approx(0.3)
    assert set(held.values) == existing | {"ol_max_forward_progress_m", "ol_passed"}
    assert held.values["ol_max_forward_progress_m"] == pytest.approx(0.3)
    assert held.values["ol_passed"] == 1.0
    # Nothing after the anchor: no reference values either.
    xy, yaw = _ego_with_speeds(np.zeros(1))
    r = yield_progress(_input("pedestrian_yield", xy, yaw))
    assert r.passed is None and not any(k.startswith("ol_") for k in r.values)
