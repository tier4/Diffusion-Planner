"""Golden-output regression tests for ``rlvr.reward.compute_reward_batch``.

This pins the *full* ``RewardBreakdown`` output of ``compute_reward_batch`` on a
fixed set of synthetic scenarios. The golden values live in
``rlvr/reward_golden.json`` and were recorded from the current implementation.
Any later change that alters a subscore, a gate, or the aggregate value will
make these tests fail.

This is the PR0 safety net for issue #130 ("carve reward subscores into a shared
metrics library"): it lets the subsequent

  * PR1 — pure *move* of the subscore / geometry / config code into
    ``diffusion_planner.metrics`` (``reward.py`` re-exports), and
  * PR2 — behavior-preserving *split* of ``compute_reward_batch`` into
    ``compute_subscores_batch`` (raw) + ``_shape_reward`` (weighting / gates)

prove "no behavior change" mechanically rather than by eyeballing diffs.

The scenarios are deterministic (fixed synthetic tensors, CPU, no RNG) and are
chosen to exercise every branch of the aggregate and every config-gated penalty:
clean on-road driving, neighbor collision, road-border crossing (both ``gate``
and ``survival`` aggregation), red-light violation, static-collision clearance
(near-miss and overlap), lane departure (gate off and gate on), a
kinematic-feasibility violation, the longitudinal-acceleration feasibility
penalty, GT-normalized over-progress, baseline-anchored under-progress, and a
stationary-vs-progress contrast.

Regenerate the golden file ONLY after an intentional behavior change::

    REGEN_REWARD_GOLDEN=1 python -m pytest rlvr/test_reward_golden.py

then review the diff to ``rlvr/reward_golden.json`` carefully before committing.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from rlvr.reward import RewardConfig, compute_reward_batch  # noqa: E402

GOLDEN_PATH = Path(__file__).with_name("reward_golden.json")

T = 80
DT = 0.1
EGO_SHAPE_ROW = [2.79, 4.34, 1.70]  # wheel_base, length, width

# Comparison tolerances for float fields. A pure move (PR1) should reproduce
# values bit-for-bit; a behavior-preserving refactor (PR2) may reassociate a
# few float ops, so allow a tiny slack.
ABS_TOL = 1e-6
REL_TOL = 1e-6


# ---------------------------------------------------------------------------
# Trajectory builders
# ---------------------------------------------------------------------------


def _straight_line(speed: float = 0.5, y0: float = 0.0) -> torch.Tensor:
    """(T, 4) ego going straight along +x at ``speed`` m/step, heading 0."""
    t = torch.arange(T, dtype=torch.float32)
    x = t * speed
    y = torch.full((T,), float(y0))
    return torch.stack([x, y, torch.ones(T), torch.zeros(T)], dim=-1)


def _drift_line(speed: float = 0.5, y_end: float = 4.0) -> torch.Tensor:
    """(T, 4) ego drifting laterally from y=0 to y=``y_end`` while moving +x."""
    t = torch.arange(T, dtype=torch.float32)
    x = t * speed
    y = torch.linspace(0.0, float(y_end), T)
    return torch.stack([x, y, torch.ones(T), torch.zeros(T)], dim=-1)


def _pivot_in_place() -> torch.Tensor:
    """(T, 4) ego spinning in place at 3 rad/s — a yaw-rate / curvature violation."""
    theta = torch.arange(T, dtype=torch.float32) * DT * 3.0
    return torch.stack([torch.zeros(T), torch.zeros(T), torch.cos(theta), torch.sin(theta)], dim=-1)


def _hard_accel_line(k: float = 0.05) -> torch.Tensor:
    """(T, 4) ego accelerating along +x as x=k*t^2. With k=0.05, dt=0.1 the
    longitudinal acceleration is ~10 m/s^2 (> max_accel=8), so the feasibility
    acceleration penalty fires."""
    t = torch.arange(T, dtype=torch.float32)
    x = k * t * t
    return torch.stack([x, torch.zeros(T), torch.ones(T), torch.zeros(T)], dim=-1)


def _slow_gt_future(reach: float = 15.0) -> torch.Tensor:
    """(T, 4) ground-truth ego future moving slowly to ``reach`` metres. All
    points are > 0.1 m from the origin so they count as valid GT (>=10 needed to
    activate the overprogress / GT-normalized progress branch)."""
    x = torch.linspace(0.5, float(reach), T)
    return torch.stack([x, torch.zeros(T), torch.ones(T), torch.zeros(T)], dim=-1)


def _npc_straight(offset_y: float, speed: float = 0.5) -> torch.Tensor:
    """(T, 4) neighbor going straight at constant lateral offset."""
    t = torch.arange(T, dtype=torch.float32)
    x = t * speed
    y = torch.full((T,), float(offset_y))
    return torch.stack([x, y, torch.ones(T), torch.zeros(T)], dim=-1)


# ---------------------------------------------------------------------------
# Scene / map builders (channel layouts copied from rlvr/test_reward.py)
# ---------------------------------------------------------------------------


def _lanes_tensor(center_y: float = 0.0, width: float = 3.5) -> torch.Tensor:
    """(1, 140, 20, 33) straight lane along +x. Channels: 0,1 center x/y;
    2,3 direction; 4,5 left boundary x/y; 6,7 right boundary x/y."""
    lanes = torch.zeros(1, 140, 20, 33)
    half_w = width / 2
    for seg in range(10):
        for pt in range(20):
            x = (seg * 20 + pt) * 1.0
            lanes[0, seg, pt, 0] = x
            lanes[0, seg, pt, 1] = center_y
            lanes[0, seg, pt, 2] = 1.0
            lanes[0, seg, pt, 3] = 0.0
            lanes[0, seg, pt, 4] = x
            lanes[0, seg, pt, 5] = center_y + half_w
            lanes[0, seg, pt, 6] = x
            lanes[0, seg, pt, 7] = center_y - half_w
    return lanes


def _make_lane_data() -> dict:
    return {
        "lanes": _lanes_tensor(),
        "ego_shape": torch.tensor([EGO_SHAPE_ROW]),
    }


def _make_road_border_data(border_y_left: float = 3.0, border_y_right: float = -3.0) -> dict:
    """``line_strings`` (1, 60, 20, 4): dim 3 channel is the road-border flag."""
    num_ls = 60
    pts = 20
    ls = torch.zeros(1, num_ls, pts, 4)
    for seg in range(30):
        for pt in range(pts):
            x = seg * (pts * 0.05) + pt * 0.05
            ls[0, seg, pt, 0] = x
            ls[0, seg, pt, 1] = border_y_left
            ls[0, seg, pt, 3] = 1.0  # road-border flag
            ls[0, 30 + seg, pt, 0] = x
            ls[0, 30 + seg, pt, 1] = border_y_right
            ls[0, 30 + seg, pt, 3] = 1.0
    data = _make_lane_data()
    data["line_strings"] = ls
    return data


def _red_light_route_lanes() -> torch.Tensor:
    """(1, 25, 20, 33) straight route lane with a RED light (channel 10) on the
    stretch x in [10, 24] directly on the ego path."""
    rl = torch.zeros(1, 25, 20, 33)
    half_w = 1.75
    for seg in range(10):
        for pt in range(20):
            x = (seg * 20 + pt) * 1.0
            rl[0, seg, pt, 0] = x
            rl[0, seg, pt, 1] = 0.0
            rl[0, seg, pt, 2] = 1.0
            rl[0, seg, pt, 3] = 0.0
            rl[0, seg, pt, 4] = x
            rl[0, seg, pt, 5] = half_w
            rl[0, seg, pt, 6] = x
            rl[0, seg, pt, 7] = -half_w
            if 10.0 <= x <= 24.0:
                rl[0, seg, pt, 10] = 1.0  # RED
    return rl


def _stopped_neighbor_data(center_y: float, width: float = 2.0, length: float = 4.5) -> dict:
    """One stopped NPC parked at (20, center_y). Returns the neighbor pieces of a
    ``data`` dict (future + past with width/length)."""
    x = torch.full((T,), 20.0)
    y = torch.full((T,), float(center_y))
    fut = torch.stack([x, y, torch.ones(T), torch.zeros(T)], dim=-1).unsqueeze(0)  # (1, T, 4)
    nap = torch.zeros(1, 1, 21, 11)
    nap[0, 0, -1, 6] = width
    nap[0, 0, -1, 7] = length
    return {"neighbor_agents_future": fut, "neighbor_agents_past": nap}


def _colliding_neighbor_data(width: float = 2.0, length: float = 4.5) -> dict:
    """One NPC sharing the ego's path (y=0) at the same speed — guaranteed overlap."""
    fut = _npc_straight(offset_y=0.0, speed=0.5).unsqueeze(0)  # (1, T, 4)
    nap = torch.zeros(1, 1, 21, 11)
    nap[0, 0, -1, 6] = width
    nap[0, 0, -1, 7] = length
    return {"neighbor_agents_future": fut, "neighbor_agents_past": nap}


# ---------------------------------------------------------------------------
# Scenarios: name -> (ego_trajs (N, T, 4), data, config)
# ---------------------------------------------------------------------------


def _scenarios() -> dict:
    scen: dict = {}

    # 1. Clean on-road driving at three speeds (no gate fires).
    data = _make_road_border_data()
    data["goal_pose"] = torch.tensor([[100.0, 0.0, 1.0, 0.0]])
    scen["safe_onroad_multi"] = (
        torch.stack([_straight_line(0.5), _straight_line(0.3), _straight_line(0.1)]),
        data,
        RewardConfig(),
    )

    # 2. Neighbor collision (safety gate fires; default 'gate' aggregation).
    data = _make_lane_data()
    data.update(_colliding_neighbor_data())
    data["goal_pose"] = torch.tensor([[100.0, 0.0, 1.0, 0.0]])
    scen["neighbor_collision"] = (
        _straight_line(0.5).unsqueeze(0),
        data,
        RewardConfig(),
    )

    # 3. Road-border crossing (rb gate fires; off_road_fraction / rb_min_dist).
    data = _make_road_border_data()
    data["goal_pose"] = torch.tensor([[40.0, 0.0, 1.0, 0.0]])
    scen["offroad_crossing"] = (
        _drift_line(0.5, y_end=4.0).unsqueeze(0),
        data,
        RewardConfig(),
    )

    # 4. Red-light violation (red_light penalty fires).
    data = _make_lane_data()
    data["route_lanes"] = _red_light_route_lanes()
    data["goal_pose"] = torch.tensor([[40.0, 0.0, 1.0, 0.0]])
    scen["red_light_violation"] = (
        _straight_line(0.5).unsqueeze(0),
        data,
        RewardConfig(),
    )

    # 5. Static-collision near-miss (sc_* fields populate; no overlap).
    data = _make_lane_data()
    data.update(_stopped_neighbor_data(center_y=2.15))
    scen["static_collision_near"] = (
        _straight_line(0.5).unsqueeze(0),
        data,
        RewardConfig(
            static_collision_enabled=True,
            sc_gate_enabled=True,
            sc_near_scale=1.0,
            sc_wide_scale=1.0,
            sc_cont_scale=1.0,
        ),
    )

    # 6. Lane departure with the gate OFF (capture lane_near_frac / lane_wide_frac).
    data = _make_lane_data()
    data["goal_pose"] = torch.tensor([[40.0, 0.0, 1.0, 0.0]])
    scen["lane_departure_nogate"] = (
        _drift_line(0.5, y_end=2.5).unsqueeze(0),
        data,
        RewardConfig(
            enable_lane_departure=True,
            lane_gate_enabled=False,
            lane_near_scale=3.0,
            lane_wide_scale=0.2,
            rb_gate_enabled=False,
        ),
    )

    # 7. Kinematic-feasibility violation (kinematic_violated -> floored total).
    data = _make_lane_data()
    scen["kinematic_pivot"] = (
        _pivot_in_place().unsqueeze(0),
        data,
        RewardConfig(),
    )

    # 8. Collision under 'survival' aggregation mode.
    data = _make_lane_data()
    data.update(_colliding_neighbor_data())
    data["goal_pose"] = torch.tensor([[100.0, 0.0, 1.0, 0.0]])
    scen["survival_mode_collision"] = (
        _straight_line(0.5).unsqueeze(0),
        data,
        RewardConfig(reward_mode="survival"),
    )

    # 9. Stationary vs progressing ego (progress / stopped penalty contrast).
    data = _make_lane_data()
    data["goal_pose"] = torch.tensor([[50.0, 0.0, 1.0, 0.0]])
    scen["stationary_vs_progress"] = (
        torch.stack([_straight_line(0.5), _straight_line(0.0)]),
        data,
        RewardConfig(),
    )

    # 10. Feasibility acceleration penalty (longitudinal accel > max_accel).
    data = _make_lane_data()
    data["goal_pose"] = torch.tensor([[300.0, 0.0, 1.0, 0.0]])
    scen["feasibility_hard_accel"] = (
        _hard_accel_line().unsqueeze(0),
        data,
        RewardConfig(),
    )

    # 11. Overprogress: GT-normalized progress + excess-path penalty.
    data = _make_lane_data()
    data["goal_pose"] = torch.tensor([[100.0, 0.0, 1.0, 0.0]])
    data["ego_agent_future"] = _slow_gt_future(reach=15.0)
    scen["overprogress"] = (
        _straight_line(0.5).unsqueeze(0),
        data,
        RewardConfig(enable_overprogress=True, overprogress_margin=1.1, overprogress_penalty=0.3),
    )

    # 12. Underprogress against a frozen baseline path-length anchor.
    data = _make_lane_data()
    data["goal_pose"] = torch.tensor([[100.0, 0.0, 1.0, 0.0]])
    data["baseline_path_len"] = torch.tensor(40.0)
    scen["underprogress_baseline"] = (
        torch.stack([_straight_line(0.0625), _straight_line(0.5)]),  # ~5m vs ~40m path
        data,
        RewardConfig(
            underprogress_penalty=10.0,
            underprogress_threshold=0.7,
            underprogress_reference="baseline",
        ),
    )

    # 13. Road-border crossing under 'survival' aggregation (proportional credit).
    data = _make_road_border_data()
    data["goal_pose"] = torch.tensor([[40.0, 0.0, 1.0, 0.0]])
    scen["offroad_crossing_survival"] = (
        _drift_line(0.5, y_end=4.0).unsqueeze(0),
        data,
        RewardConfig(reward_mode="survival"),
    )

    # 14. Lane departure WITH the hard gate (lane crossing floors the total).
    data = _make_lane_data()
    data["goal_pose"] = torch.tensor([[40.0, 0.0, 1.0, 0.0]])
    scen["lane_departure_gated"] = (
        _drift_line(0.5, y_end=2.5).unsqueeze(0),
        data,
        RewardConfig(
            enable_lane_departure=True,
            lane_gate_enabled=True,
            lane_near_scale=3.0,
            lane_wide_scale=0.2,
            rb_gate_enabled=False,
        ),
    )

    # 15. Static-collision overlap (static_crossing=True; sc gate fires).
    data = _make_lane_data()
    data.update(_stopped_neighbor_data(center_y=0.0))
    scen["static_collision_overlap"] = (
        _straight_line(0.5).unsqueeze(0),
        data,
        RewardConfig(
            static_collision_enabled=True,
            sc_gate_enabled=True,
            sc_near_scale=1.0,
            sc_wide_scale=1.0,
            sc_cont_scale=1.0,
        ),
    )

    return scen


# ---------------------------------------------------------------------------
# Golden machinery
# ---------------------------------------------------------------------------

_SCENARIOS = _scenarios()


def _breakdown_to_dict(bd) -> dict:
    return dataclasses.asdict(bd)


def _compute(name: str) -> list:
    ego, data, cfg = _SCENARIOS[name]
    return [_breakdown_to_dict(b) for b in compute_reward_batch(ego, data, cfg)]


def _compute_all() -> dict:
    return {name: _compute(name) for name in _SCENARIOS}


def _maybe_regenerate() -> bool:
    if not os.environ.get("REGEN_REWARD_GOLDEN"):
        return False
    GOLDEN_PATH.write_text(json.dumps(_compute_all(), indent=2, sort_keys=True) + "\n")
    return True


# Regenerate once at import time when requested, so a plain pytest run produces
# the file too.
_REGENERATED = _maybe_regenerate()


def _load_golden() -> dict:
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"Golden file {GOLDEN_PATH} is missing. Generate it with "
            "`REGEN_REWARD_GOLDEN=1 python -m pytest rlvr/test_reward_golden.py`."
        )
    return json.loads(GOLDEN_PATH.read_text())


def _assert_field_equal(name: str, idx: int, key: str, computed, expected) -> None:
    ctx = f"{name}[{idx}].{key}"
    if expected is None or isinstance(expected, bool):
        assert computed == expected, f"{ctx}: {computed!r} != {expected!r}"
    elif isinstance(expected, int):
        assert computed == expected, f"{ctx}: {computed!r} != {expected!r}"
    else:
        assert computed == pytest.approx(expected, abs=ABS_TOL, rel=REL_TOL), (
            f"{ctx}: {computed!r} != {expected!r}"
        )


@pytest.mark.skipif(_REGENERATED, reason="golden file was just regenerated")
@pytest.mark.parametrize("name", sorted(_SCENARIOS))
def test_reward_golden(name: str) -> None:
    golden = _load_golden()
    assert name in golden, f"scenario {name!r} missing from golden file (regenerate)"
    computed = _compute(name)
    expected = golden[name]
    assert len(computed) == len(expected), (
        f"{name}: got {len(computed)} breakdowns, golden has {len(expected)}"
    )
    for idx, (c, g) in enumerate(zip(computed, expected)):
        assert set(c) == set(g), f"{name}[{idx}]: field set mismatch {set(c) ^ set(g)}"
        for key in g:
            _assert_field_equal(name, idx, key, c[key], g[key])


@pytest.mark.skipif(_REGENERATED, reason="golden file was just regenerated")
def test_golden_covers_all_scenarios() -> None:
    """The golden file and the scenario set must stay in lock-step."""
    golden = _load_golden()
    assert set(golden) == set(_SCENARIOS), (
        "golden file is out of sync with _scenarios(); regenerate it. "
        f"symmetric diff: {set(golden) ^ set(_SCENARIOS)}"
    )


if __name__ == "__main__":
    # Allow `python rlvr/test_reward_golden.py` to (re)generate or verify.
    if os.environ.get("REGEN_REWARD_GOLDEN"):
        GOLDEN_PATH.write_text(json.dumps(_compute_all(), indent=2, sort_keys=True) + "\n")
        print(f"Wrote golden file: {GOLDEN_PATH}")
    else:
        golden = json.loads(GOLDEN_PATH.read_text())
        for name in sorted(_SCENARIOS):
            computed = _compute(name)
            expected = golden[name]
            for idx, (c, g) in enumerate(zip(computed, expected)):
                for key in g:
                    _assert_field_equal(name, idx, key, c[key], g[key])
            print(f"  PASS  {name} ({len(computed)} breakdown(s))")
