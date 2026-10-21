"""Span-aware scenario metrics: an anchor's event span sets the scored interval."""

import argparse
import dataclasses
import json

import numpy as np
import pytest

from scenario_generation.scenario_metrics import geometry  # noqa: F401  (registers the scorers)
from scenario_generation.scenario_metrics.registry import score
from scenario_generation.scenario_metrics.testing import make_input, straight_path

DT = 0.1
ANCHOR = 10


def _turn_path(n, speed=5.0, radius=15.0, straight=20, turn_frames=None):
    """Straight for ``straight`` frames, a left arc of ``radius`` (for ``turn_frames``), straight again."""
    s = np.arange(n) * speed * DT
    s0 = straight * speed * DT
    arc = np.clip(s - s0, 0.0, None if turn_frames is None else turn_frames * speed * DT)
    theta = arc / radius
    x = np.where(s < s0, s, s0 + radius * np.sin(theta))
    y = np.where(s < s0, 0.0, radius * (1.0 - np.cos(theta)))
    extra = s - s0 - arc  # straight run after the arc
    x, y = (
        x + np.where(extra > 0, extra * np.cos(theta), 0.0),
        y + np.where(extra > 0, extra * np.sin(theta), 0.0),
    )
    return np.stack([x, y], axis=1), theta


def _lane(y, x0=-50.0, x1=2000.0, points=400, half_width=1.75):
    lane = np.zeros((points, 8))
    lane[:, 0] = np.linspace(x0, x1, points)
    lane[:, 1] = y
    lane[:, 2] = 1.0
    lane[:, 5] = half_width
    lane[:, 7] = -half_width
    return lane


def _with_span(inp, first, last):
    return dataclasses.replace(inp, span_frames=(first, last))


# ------------------------------------------------------------ metric: span interval


def test_simple_turn_span_covers_the_turn_the_fixed_horizon_cuts_short():
    # Straight for 8 s after the anchor, then the turn: the fixed 8 s sees none of it.
    rec_xy, rec_yaw = _turn_path(300, straight=ANCHOR + 80)
    ego_xy, ego_yaw = straight_path(300, 5.0)  # never turns
    inp = make_input(
        label="simple_turn",
        ego_xy=ego_xy,
        ego_yaw=ego_yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=ANCHOR,
    )
    fixed = score(inp)
    assert fixed.passed is True and fixed.values["span_used"] == 0.0
    spanned = score(_with_span(inp, ANCHOR, ANCHOR + 150))
    assert spanned.passed is False and spanned.values["span_used"] == 1.0
    assert spanned.values["eval_interval_s"] == pytest.approx(15.0)
    assert spanned.details["end_frame"] == ANCHOR + 150


def test_no_span_or_empty_span_keeps_the_fixed_horizon():
    rec_xy, rec_yaw = _turn_path(200)
    inp = make_input(
        label="simple_turn",
        ego_xy=rec_xy,
        ego_yaw=rec_yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=ANCHOR,
    )
    base = score(inp)
    empty = score(_with_span(inp, ANCHOR, ANCHOR))
    assert base.to_json() == empty.to_json()
    assert base.details["end_frame"] == ANCHOR + 80


def test_span_start_before_the_anchor_starts_the_stretch_there():
    rec_xy, rec_yaw = _turn_path(200)
    inp = make_input(
        label="simple_turn",
        ego_xy=rec_xy,
        ego_yaw=rec_yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=ANCHOR,
    )
    r = score(_with_span(inp, ANCHOR - 5, ANCHOR + 40))
    assert r.details["start_frame"] == ANCHOR - 5 and r.details["start_step"] == ANCHOR - 5


def test_lane_change_finishing_after_8_s_is_scored_only_with_its_span():
    n = 300
    rec_xy, rec_yaw = straight_path(n, 5.0)
    # Lateral move of 3.5 m from 9 s to 12 s after the anchor.
    t = (np.arange(n) - ANCHOR) * DT
    rec_xy[:, 1] = 3.5 * np.clip((t - 9.0) / 3.0, 0.0, 1.0)
    frames = {
        ANCHOR: {"lanes": np.stack([_lane(0.0), _lane(3.5)]), "route_lanes": _lane(0.0)[None]}
    }
    inp = make_input(
        label="lane_change",
        ego_xy=rec_xy,
        ego_yaw=rec_yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=ANCHOR,
        frames=frames,
    )
    fixed = score(inp)
    assert fixed.passed is None and "no lane change" in fixed.reason
    spanned = score(_with_span(inp, ANCHOR, ANCHOR + 140))
    assert spanned.passed is True
    assert spanned.values["gt_lateral_shift_m"] == pytest.approx(3.5, abs=1e-6)


@pytest.mark.parametrize(("delay_s", "passed"), [(1.0, True), (4.0, False)])
def test_lane_change_gets_a_grace_past_the_humans_settling(delay_s, passed):
    n = 300
    rec_xy, rec_yaw = straight_path(n, 5.0)
    ego_xy = rec_xy.copy()
    # The human moves 3.5 m sideways from 9 s to 12 s after the anchor; the ego the same,
    # ``delay_s`` later.
    t = (np.arange(n) - ANCHOR) * DT
    rec_xy[:, 1] = 3.5 * np.clip((t - 9.0) / 3.0, 0.0, 1.0)
    ego_xy[:, 1] = 3.5 * np.clip((t - 9.0 - delay_s) / 3.0, 0.0, 1.0)
    frames = {
        ANCHOR: {"lanes": np.stack([_lane(0.0), _lane(3.5)]), "route_lanes": _lane(0.0)[None]}
    }
    inp = make_input(
        label="lane_change",
        ego_xy=ego_xy,
        ego_yaw=rec_yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=ANCHOR,
        frames=frames,
    )
    r = score(_with_span(inp, ANCHOR, ANCHOR + 120))
    assert r.passed is passed
    assert r.details["end_frame"] == ANCHOR + 120 + round(geometry.LANE_CHANGE_GRACE_S / DT)


def test_object_avoidance_collision_after_8_s_counts_inside_the_span():
    n = 300
    rec_xy, rec_yaw = straight_path(n, 5.0)
    clearance = np.full(n, 3.0)
    clearance[ANCHOR + 100] = -0.1
    inp = make_input(
        label="object_avoidance",
        ego_xy=rec_xy,
        ego_yaw=rec_yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=ANCHOR,
        clearance_m=clearance,
    )
    assert score(inp).passed is True
    r = score(_with_span(inp, ANCHOR, ANCHOR + 120))
    assert r.passed is False and r.details["first_collision_step"] == ANCHOR + 100


# ------------------------------------------------------------ metric: lane_follow


def _lane_follow(offset=0.0, n=250, stall_after=None, road_border=None, span=(0, 200)):
    rec_xy, rec_yaw = straight_path(n, 8.0)
    ego_xy = rec_xy.copy()
    ego_xy[:, 1] += offset
    if stall_after is not None:
        ego_xy[stall_after:] = ego_xy[stall_after]
    lanes = {"route_lanes": _lane(0.0)[None], "lanes": _lane(0.0)[None]}
    loaded = []

    def frames(i):
        loaded.append(i)
        return lanes

    inp = make_input(
        label="lane_follow",
        ego_xy=ego_xy,
        ego_yaw=rec_yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=0,
        frames=frames,
        # A stalled ego never reaches the window's goal.
        terminated="goal" if stall_after is None else "max_steps",
    )
    rb = None if road_border is None else np.asarray(road_border, dtype=np.float64)
    return dataclasses.replace(inp, span_frames=span, road_border_m=rb), loaded


def test_lane_follow_on_the_centerline_passes_and_reads_lanes_along_the_span():
    inp, loaded = _lane_follow(offset=0.3)
    r = score(inp)
    assert r.passed is True
    assert r.values["max_lateral_error_m"] == pytest.approx(0.3, abs=1e-6)
    assert r.values["p95_lateral_error_m"] == pytest.approx(0.3, abs=1e-6)
    assert r.values["rec_max_lateral_error_m"] == pytest.approx(0.0, abs=1e-6)
    assert r.values["eval_interval_s"] == pytest.approx(20.0)
    assert sorted(set(loaded)) == list(range(0, 201, geometry.LANE_FOLLOW_LANE_FRAME_STEP))


def test_lane_follow_off_centre_fails():
    r = score(_lane_follow(offset=0.7)[0])
    assert r.passed is False and "lateral error" in r.reason


def test_lane_follow_stall_fails_progress():
    r = score(_lane_follow(stall_after=100)[0])  # half the span
    assert r.passed is False and "cover" in r.reason
    assert r.values["progress_ratio"] < geometry.LANE_FOLLOW_MIN_PROGRESS_RATIO


def test_lane_follow_road_border_contact_fails():
    rb = np.full(250, 1.0)
    rb[50] = 0.05
    r = score(_lane_follow(road_border=rb)[0])
    assert r.passed is False and "road-border" in r.reason
    assert r.values["min_road_border_m"] == pytest.approx(0.05)
