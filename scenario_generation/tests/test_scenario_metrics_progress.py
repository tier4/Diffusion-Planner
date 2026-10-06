import numpy as np
import pytest

from scenario_generation.scenario_metrics import registry
from scenario_generation.scenario_metrics.progress import (
    CLOSED_LOOP_DEPARTURE_HORIZON_S,
    TRAFFIC_LIGHT_GO_STOP_LINE_TOLERANCE_M,
    WAIT_SPAN_LABELS,
    YIELD_LABELS,
    YIELD_MAX_EXCESS_PROGRESS_M,
    departure_progress,
    yield_progress,
    yield_wait,
)
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
    assert label in registry.METRICS


def test_departure_pass_and_fail():
    ok = departure_progress(_input("traffic_light_go", *_ego_with_speeds(np.full(40, 1.0))))
    assert ok.passed is True
    assert ok.values["progress_m"] == pytest.approx(3.0)
    assert ok.details["time_to_threshold_s"] == pytest.approx(2.0)
    slow = departure_progress(_input("traffic_light_go", *_ego_with_speeds(np.full(40, 0.5))))
    assert slow.passed is False
    assert slow.values["progress_m"] == pytest.approx(1.5)
    assert slow.details["time_to_threshold_s"] is None


def test_departure_gets_the_closed_loop_horizon():
    # 0.5 m/s gains 1.5 m in 3 s but 2.5 m in 5 s.
    xy, yaw = _ego_with_speeds(np.full(60, 0.5))
    departure = departure_progress(_input("departure", xy, yaw))
    assert departure.passed is True
    assert departure.values["horizon_s"] == CLOSED_LOOP_DEPARTURE_HORIZON_S
    assert departure_progress(_input("traffic_light_go", xy, yaw)).passed is False


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
    r = departure_progress(_input("traffic_light_go", *_ego_with_speeds(np.full(40, 1.0))))
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
    dep = departure_progress(_input("traffic_light_go", *_ego_with_speeds(np.full(40, 1.0))))
    assert dep.passed is True and dep.values["progress_m"] == pytest.approx(3.0)
    assert set(dep.values) == existing | {
        "replay_ahead_s",
        "ol_max_displacement_m",
        "ol_passed",
    }
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


RED = (50, 100)  # the human waits at x = 25 m over these frames; the light turns green at 100
LINE_X = 28.5  # stop line, world x: 0.5 m ahead of the human's front (axle + 3 m)


def _red_input(ego_speeds, label="traffic_light_go", rec_idx=None):
    """Human: 5 m/s to x = 25 m, waits at the red over ``RED``, departs at the anchor."""
    rec_speeds = np.r_[np.full(RED[0], 5.0), np.zeros(RED[1] - RED[0]), np.full(60, 2.0)]
    rec_xy, rec_yaw = speed_profile_path(rec_speeds)
    ego_xy, ego_yaw = speed_profile_path(ego_speeds)

    def frame(i):
        line = np.array([[[LINE_X - rec_xy[i, 0], -2.0], [LINE_X - rec_xy[i, 0], 2.0]]])
        return {"stop_lines": line, "ego_shape": np.array([2.0, 4.0, 2.0])}

    return make_input(
        label=label,
        ego_xy=ego_xy,
        ego_yaw=ego_yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=RED[1],
        rec_idx=rec_idx,
        frames=frame,
        terminated="max_steps",
    )


def _ego_speeds(creep_m: float = 0.0, stop_offset_m: float = 0.0):
    """Like the human, stopping ``stop_offset_m`` further, creeping ``creep_m`` on the red."""
    speeds = np.r_[np.full(RED[0], 5.0), np.zeros(RED[1] - RED[0]), np.full(60, 2.0)]
    speeds[RED[0] - 1] += stop_offset_m / 0.1
    speeds[RED[0] + 10 : RED[0] + 30] = creep_m / 2.0  # 2 s of creep
    return speeds


def test_traffic_light_go_holding_behind_the_line_passes():
    r = departure_progress(_red_input(_ego_speeds()))
    assert r.passed is True
    assert r.values["stop_line_found"] == 1.0
    assert r.values["furthest_past_line_m"] == pytest.approx(-0.5, abs=0.05)
    assert r.values["first_stop_past_line_m"] == pytest.approx(-0.5, abs=0.05)
    assert r.values["hold_creep_m"] == pytest.approx(0.0, abs=0.05)
    assert r.values["stop_line_tolerance_m"] == TRAFFIC_LIGHT_GO_STOP_LINE_TOLERANCE_M
    # Stopping closer to the line, still short of it, also passes.
    assert departure_progress(_red_input(_ego_speeds(stop_offset_m=0.4))).passed is True


def test_traffic_light_go_creeping_past_the_line_on_red_fails():
    r = departure_progress(_red_input(_ego_speeds(creep_m=1.5)))
    assert r.passed is False and r.reason == "ego passed the stop line on red"
    assert r.values["first_stop_past_line_m"] == pytest.approx(-0.5, abs=0.05)
    assert r.values["hold_creep_m"] == pytest.approx(1.5, abs=0.05)
    assert r.values["furthest_past_line_m"] == pytest.approx(1.0, abs=0.05)
    # The departure after the green is not judged by the line.
    assert r.values["progress_m"] >= r.values["threshold_m"]


def test_traffic_light_go_edging_onto_the_line_passes():
    # 0.4 m past the line is within the tolerance.
    r = departure_progress(_red_input(_ego_speeds(creep_m=0.9)))
    assert r.values["furthest_past_line_m"] == pytest.approx(0.4, abs=0.05)
    assert TRAFFIC_LIGHT_GO_STOP_LINE_TOLERANCE_M == 0.5 and r.passed is True


def test_departure_is_not_judged_by_the_red():
    r = departure_progress(_red_input(_ego_speeds(creep_m=1.0), label="departure"))
    assert r.passed is True and "furthest_past_line_m" not in r.values


def test_replay_ahead_is_reported():
    inp = _red_input(_ego_speeds(), rec_idx=np.minimum(np.arange(210) + 20, 209))
    r = departure_progress(inp)
    assert r.values["replay_ahead_s"] == pytest.approx(2.0)


def test_traffic_light_go_anchor_never_replayed_stays_not_applicable():
    xy, yaw = straight_path(ANCHOR - 2, 0.0)
    r = departure_progress(_input("traffic_light_go", xy, yaw))
    assert r.passed is None and "replay_ahead_s" not in r.values


# --- yield_wait: the ego's progress while the replay is inside the human's wait ---

SPAN = (ANCHOR, ANCHOR + 29)  # a 3 s wait


def _wait_input(rec_idx, *, label="pedestrian_yield", span=SPAN, terminated="goal", push_m=0.0):
    """Recording that waits at x = 10 m over ``span``; the ego holds there (or pushes on
    ``push_m`` past it over the span's steps) while the cursor replays ``rec_idx`` (one
    entry per sim step)."""
    speeds = np.full(80, 2.0)
    speeds[span[0] - 1 : span[1]] = 0.0
    rec_xy, rec_yaw = speed_profile_path(speeds)
    k = len(rec_idx)
    ego_xy = np.repeat(rec_xy[[span[0]]], k, axis=0)
    rec_idx = np.asarray(rec_idx)
    in_span = (rec_idx >= span[0]) & (rec_idx <= span[1])
    ego_xy[in_span, 0] += np.linspace(0.0, push_m, int(in_span.sum()))
    ego_xy[rec_idx > span[1], 0] += push_m
    return make_input(
        label=label,
        ego_xy=ego_xy,
        ego_yaw=np.zeros(k),
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=ANCHOR,
        rec_idx=np.asarray(rec_idx),
        terminated=terminated,
        span_frames=span,
    )


def test_span_anchors_are_scored_by_yield_conflict_and_others_by_the_fixed_horizon():
    for label in YIELD_LABELS:
        with_span = _wait_input(np.arange(60), label=label)
        expected = "yield_conflict" if label in WAIT_SPAN_LABELS else "yield_progress"
        assert registry.score(with_span).metric == expected
        without = _input(label, *_ego_with_speeds(np.full(40, 0.1)))
        assert registry.score(without).metric == "yield_progress"


def test_waiting_as_long_as_the_human_passes():
    # The cursor replays the span one frame per step: the ego stayed the human's 3 s.
    r = yield_wait(_wait_input(np.arange(60)))
    assert r.passed is True
    assert r.values["wait_ratio"] == pytest.approx(1.0)
    assert r.values["human_wait_s"] == pytest.approx(3.0)
    assert r.details["max_replay_jump_frames"] == 1


def test_leaving_the_span_early_without_pushing_on_passes():
    # Five frames per step: the replay leaves the 3 s span after 0.6 s, but the ego
    # stays where the human waited.
    rec_idx = np.r_[np.arange(ANCHOR), np.arange(ANCHOR, 60, 5)]
    r = yield_wait(_wait_input(rec_idx))
    assert r.passed is True
    assert r.values["wait_ratio"] == pytest.approx(0.2)
    assert r.details["max_replay_jump_frames"] == 5


def test_pushing_on_past_the_human_fails():
    r = yield_wait(_wait_input(np.arange(60), push_m=YIELD_MAX_EXCESS_PROGRESS_M + 0.5))
    assert r.passed is False
    assert r.values["excess_progress_m"] == pytest.approx(YIELD_MAX_EXCESS_PROGRESS_M + 0.5)
    assert r.reason == "ego pushed on past the human while yielding"
    edged = yield_wait(_wait_input(np.arange(60), push_m=YIELD_MAX_EXCESS_PROGRESS_M - 0.5))
    assert edged.passed is True


def test_trace_ending_inside_the_span_is_scored_only_once_pushed_on():
    early = yield_wait(_wait_input(np.arange(ANCHOR + 10), terminated="max_steps"))
    assert early.passed is None
    assert early.reason == "trace ended (max_steps) inside the span"
    pushed = yield_wait(_wait_input(np.arange(ANCHOR + 10), terminated="max_steps", push_m=4.0))
    assert pushed.passed is False


def test_span_never_replayed_is_not_applicable():
    r = yield_wait(_wait_input(np.arange(ANCHOR - 2)))
    assert r.passed is None
    assert r.reason == "span never replayed"


def test_yield_wait_keeps_the_fixed_horizon_values_for_reference():
    inp = _wait_input(np.arange(60))
    r = yield_wait(inp)
    fixed = yield_progress(inp)
    for key, value in fixed.values.items():
        assert r.values[key] == pytest.approx(value)
    assert r.values["fixed_horizon_passed"] == float(fixed.passed)
