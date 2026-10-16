import numpy as np
import pytest

from scenario_generation.scenario_metrics import registry
from scenario_generation.scenario_metrics.conflict import agent_tracks, conflicts
from scenario_generation.scenario_metrics.progress import YIELD_MIN_PET_S, yield_conflict
from scenario_generation.scenario_metrics.testing import make_input, speed_profile_path

N = 200
ANCHOR = 50
SPAN = (50, 119)  # the human waits at x = 10 m from 5 s to 12 s
EGO_SHAPE = np.array([2.8, 4.5, 1.8])  # wheelbase, length, width: front 3.65 m, rear 0.85 m


def _human():
    speeds = np.full(N, 2.0)
    speeds[SPAN[0] - 1 : SPAN[1]] = 0.0
    return speed_profile_path(speeds)


def _pedestrian(t):
    """Crosses the path at x = 15 m from y = -6 m, 1.2 m/s, starting at 3 s."""
    return np.array([15.0, -6.0 + 1.2 * max(t - 30, 0) * 0.1])


def _merging_vehicle(t):
    """Drives at 5 m/s from (12, 3), closing in at 10 deg until on the path at y = 0."""
    x = 12.0 + 5.0 * t * 0.1
    y = max(3.0 - np.tan(np.radians(10.0)) * (x - 12.0), 0.0)
    return np.array([x, y])


def _frames(rec_xy, rec_yaw, agents):
    """``load_frame`` with each agent's last 2 s, ego-centric at frame i."""

    def load(i):
        c, s = np.cos(rec_yaw[i]), np.sin(rec_yaw[i])
        past = np.zeros((max(len(agents), 1), 21, 4))
        for a, (position, _, _) in enumerate(agents):
            for j, t in enumerate(range(i - 20, i + 1)):
                if t < 0:
                    continue
                d = position(t) - rec_xy[i]
                past[a, j, :2] = [c * d[0] + s * d[1], -s * d[0] + c * d[1]]
                past[a, j, 2] = 1.0
        label = np.array([t for _, t, _ in agents] or [[0, 0, 0]], dtype=float)
        shape = np.array([w for _, _, w in agents] or [[0, 0]], dtype=float)
        return {
            "neighbor_agents_past": past,
            "agent_label": label,
            "agent_shape": shape,
            "ego_shape": EGO_SHAPE,
        }

    return load


PEDESTRIAN = (_pedestrian, [0, 1, 0], [0.6, 0.6])
MERGING = (_merging_vehicle, [1, 0, 0], [1.8, 4.5])


def _input(ego_speeds=None, agents=(PEDESTRIAN,)):
    rec_xy, rec_yaw = _human()
    if ego_speeds is None:
        ego_xy, ego_yaw = rec_xy, rec_yaw
    else:
        ego_xy, ego_yaw = speed_profile_path(ego_speeds)
    return make_input(
        label="pedestrian_yield",
        ego_xy=ego_xy,
        ego_yaw=ego_yaw,
        rec_xy=rec_xy,
        rec_yaw=rec_yaw,
        anchor_frame=ANCHOR,
        frames=_frames(rec_xy, rec_yaw, list(agents)),
        span_frames=SPAN,
    )


def test_tracks_chain_one_agent_across_frames():
    tracks = agent_tracks(_input())
    assert len(tracks) == 1 and tracks[0]["type"] == "pedestrian"
    assert min(tracks[0]["pos"]) == 0 and max(tracks[0]["pos"]) == N - 1


def test_the_crossing_pedestrian_is_found():
    inp = _input()
    (c,) = conflicts(agent_tracks(inp), inp.rec_xy, float(inp.rec_xy[-1, 0]), 1.8)
    assert c.kind == "cross" and c.agent_type == "pedestrian"
    assert c.s_m == pytest.approx(15.0, abs=0.1)
    # In the band |y| <= 0.9 + 0.3 + 2.0 m: y from -3.2 to 3.2 m.
    assert c.first_frame == pytest.approx(30 + 2.8 / 0.12, abs=1)
    assert c.last_frame == pytest.approx(30 + 9.2 / 0.12, abs=1)


def test_waiting_like_the_human_passes():
    r = registry.score(_input())
    assert r.metric == "yield_conflict" and r.passed is True
    assert r.details["target_type"] == "pedestrian" and r.details["order"] == "agent_first"
    assert r.values["pet_s"] == pytest.approx(r.values["human_pet_s"], abs=0.11)


def test_rolling_through_the_crossing_fails():
    r = yield_conflict(_input(ego_speeds=np.full(N, 2.0)))
    assert r.passed is False and r.details["order"] == "overlap"
    assert r.reason == f"post-encroachment time under {YIELD_MIN_PET_S} s"


def test_going_before_the_pedestrian_fails():
    r = yield_conflict(_input(ego_speeds=np.full(N, 6.0)))
    assert r.passed is False and r.details["order"] == "ego_first"
    assert r.values["pet_s"] < 0


def test_no_agent_near_the_path_is_not_applicable():
    r = yield_conflict(_input(agents=()))
    assert r.passed is None and "likely not a yield" in r.reason


def test_a_merging_vehicle_is_judged_at_its_entry():
    inp = _input(agents=(MERGING,))
    (c,) = conflicts(agent_tracks(inp), inp.rec_xy, float(inp.rec_xy[-1, 0]), 1.8)
    assert c.kind == "merge" and c.agent_type == "vehicle"
    # Enters the band |y| <= 0.9 + 0.9 + 0.3 m at x = 12 + (3 - 2.1) / tan(10 deg).
    assert c.s_m == pytest.approx(12.0 + 0.9 / np.tan(np.radians(10.0)), abs=0.6)
