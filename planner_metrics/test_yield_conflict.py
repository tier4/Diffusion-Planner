import numpy as np
import pytest
import torch

from planner_metrics.yield_conflict import (
    ORDERS,
    PEDESTRIAN,
    VEHICLE,
    evaluate_yield_conflict_with_details,
    extend_path,
    find_conflict,
)

_PARAMETERS = {"minimum_pet_seconds": 0.5, "maximum_conflict_distance_m": 20.0}
_STEPS = 80
_EGO_SHAPE = (2.8, 4.5, 1.8)  # wheelbase, length, width: front 3.65 m, rear 0.85 m
_T = np.arange(1, _STEPS + 1) * 0.1  # prediction / future timestamps


def _ego(start_s: float, speed: float) -> np.ndarray:
    """Along +x: waits at the origin until ``start_s``, then drives at ``speed``."""
    x = np.clip(_T - start_s, 0.0, None) * speed
    traj = np.zeros((_STEPS, 4))
    traj[:, 0], traj[:, 2] = x, 1.0
    return traj


def _crossing(y0: float, x: float = 8.0, speed: float = 1.2) -> np.ndarray:
    """Positions at t = 0 .. 8 s of an agent crossing the path at ``x``, leftwards."""
    t = np.r_[0.0, _T]
    return np.stack([np.full_like(t, x), y0 + speed * t], axis=1)


def _merging(x0: float = 4.0, y0: float = 3.0, speed: float = 5.0) -> np.ndarray:
    """Closing in at 10 deg from the left until on the path, then along it."""
    t = np.r_[0.0, _T]
    x = x0 + speed * t
    return np.stack([x, np.maximum(y0 - np.tan(np.radians(10.0)) * (x - x0), 0.0)], axis=1)


def _scene(agents, human, kind=PEDESTRIAN, width=0.6):
    past = np.zeros((max(len(agents), 1), 21, 11))
    future = np.zeros((max(len(agents), 1), _STEPS, 3))
    for i, xy in enumerate(agents):
        past[i, -1, :2] = xy[0]
        past[i, -1, 6] = width
        past[i, -1, 8 + kind] = 1.0
        future[i, :, :2] = xy[1:]
    return {
        "ego_agent_future": torch.tensor(human[None], dtype=torch.float32),
        "neighbor_agents_past": torch.tensor(past[None], dtype=torch.float32),
        "neighbor_agents_future": torch.tensor(future[None], dtype=torch.float32),
        "ego_shape": torch.tensor([_EGO_SHAPE], dtype=torch.float32),
    }


def _evaluate(pred, agents, human, **kw):
    r = evaluate_yield_conflict_with_details(
        torch.tensor(pred[None], dtype=torch.float32), _scene(agents, human, **kw), _PARAMETERS
    )
    return r, {k: v[0].item() for k, v in r.details["yield_conflict"].items()}


# The human waits until 4.5 s for a pedestrian who leaves the band (|y| <= 0.9 + 0.3 + 2.0 m)
# at about 5.2 s, then drives at 2 m/s and reaches x = 8 - 0.3 - 3.65 m at about 6.5 s.
_HUMAN = _ego(4.5, 2.0)
_PEDESTRIAN = _crossing(-3.0)


def test_the_crossing_pedestrian_is_found():
    xy = _PEDESTRIAN
    path = extend_path(np.vstack([[0.0, 0.0], _HUMAN[:, :2]]))
    c = find_conflict(xy, np.ones(len(xy), bool), PEDESTRIAN, 0.3, path, 1.8)
    assert c is not None and c.kind == "cross"
    assert c.s_m == pytest.approx(8.0, abs=0.05)
    assert c.first == 0 and c.last == pytest.approx(6.2 / 1.2 * 10, abs=1)


def test_waiting_like_the_human_passes_with_the_humans_pet():
    r, d = _evaluate(_HUMAN, [_PEDESTRIAN], _HUMAN)
    assert r.scores["success_rate_percent"].tolist() == [100.0]
    assert d["target_found"] and ORDERS[int(d["order"])] == "agent_first"
    assert d["pet_seconds"] == pytest.approx(d["human_pet_seconds"]) and d["pet_seconds"] > 0.5


def test_driving_into_the_crossing_fails():
    r, d = _evaluate(_ego(0.0, 2.0), [_PEDESTRIAN], _HUMAN)
    assert r.scores["success_rate_percent"].tolist() == [0.0]
    assert ORDERS[int(d["order"])] == "overlap"


def test_going_before_the_pedestrian_arrives_fails():
    late = _crossing(-9.0)  # enters the band at about 4.8 s
    r, d = _evaluate(_ego(0.0, 6.0), [late], _ego(6.5, 2.0))
    assert r.scores["success_rate_percent"].tolist() == [0.0]
    assert ORDERS[int(d["order"])] == "ego_first" and d["pet_seconds"] < 0


def test_following_the_pedestrian_too_closely_fails():
    # Reaches the conflict point 0.2 s after the pedestrian leaves it.
    leave = 6.2 / 1.2
    pred = _ego(leave + 0.2 - 4.05 / 2.0, 2.0)
    r, d = _evaluate(pred, [_PEDESTRIAN], _HUMAN)
    assert ORDERS[int(d["order"])] == "agent_first" and d["pet_seconds"] < 0.5
    assert r.scores["success_rate_percent"].tolist() == [0.0]


def test_staying_put_passes():
    r, d = _evaluate(np.zeros((_STEPS, 4)), [_PEDESTRIAN], _HUMAN)
    assert r.scores["success_rate_percent"].tolist() == [100.0]
    assert ORDERS[int(d["order"])] == "ego_never_reached"


def test_no_agent_passes_and_is_counted_as_not_found():
    r, d = _evaluate(_HUMAN, [], _HUMAN)
    assert r.scores["success_rate_percent"].tolist() == [100.0]
    assert r.scores["target_found_rate_percent"].tolist() == [0.0]
    assert not d["target_found"] and ORDERS[int(d["order"])] == "no_target"


def test_a_merging_vehicle_is_judged_at_its_entry():
    xy = _merging()
    path = extend_path(np.vstack([[0.0, 0.0], _ego(0.0, 4.0)[:, :2]]))
    c = find_conflict(xy, np.ones(len(xy), bool), VEHICLE, 0.9, path, 1.8)
    assert c is not None and c.kind == "merge"
    # Enters the band |y| <= 0.9 + 0.9 + 0.3 m at x = 4 + (3 - 2.1) / tan(10 deg).
    assert c.s_m == pytest.approx(4.0 + 0.9 / np.tan(np.radians(10.0)), abs=0.6)


def test_a_far_conflict_is_not_a_target():
    far = _crossing(-3.0, x=30.0)
    _, d = _evaluate(_ego(0.0, 5.0), [far], _ego(0.0, 5.0))
    assert not d["target_found"]
