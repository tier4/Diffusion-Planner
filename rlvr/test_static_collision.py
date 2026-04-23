"""Unit tests for rlvr.reward.compute_static_collision_penalty.

Synthetic OBB cases — no model needed.
Run: python -m pytest rlvr/test_static_collision.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytest
import torch

from rlvr.reward import (
    RewardConfig,
    compute_reward_batch,
    compute_static_collision_penalty,
    _closest_points_between_rects,
)

T = 80
DT = 0.1


def _ego_shape() -> torch.Tensor:
    # wheel_base, length, width — j6-ish but irrelevant to SAT (only shapes matter)
    return torch.tensor([2.79, 4.34, 1.70])


def _straight_ego(speed: float = 5.0, heading: float = 0.0, start: tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
    """(1, T, 4) ego traj moving at constant speed along `heading` (rad)."""
    t = torch.arange(T, dtype=torch.float32) * DT
    cos_h = float(torch.cos(torch.tensor(heading)))
    sin_h = float(torch.sin(torch.tensor(heading)))
    x = start[0] + speed * t * cos_h
    y = start[1] + speed * t * sin_h
    cos = torch.full((T,), cos_h)
    sin = torch.full((T,), sin_h)
    return torch.stack([x, y, cos, sin], dim=-1).unsqueeze(0)


def _stopped_neighbor(center: tuple[float, float], heading: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single stopped NPC parked at `center` with heading (rad).

    Returns (future (1,T,4), shapes (1,2) [w,l], valid (1,T))."""
    x = torch.full((T,), center[0])
    y = torch.full((T,), center[1])
    cos = torch.full((T,), float(torch.cos(torch.tensor(heading))))
    sin = torch.full((T,), float(torch.sin(torch.tensor(heading))))
    fut = torch.stack([x, y, cos, sin], dim=-1).unsqueeze(0)  # (1, T, 4)
    shapes = torch.tensor([[2.0, 4.5]])  # [width, length]
    valid = torch.ones(1, T, dtype=torch.bool)
    return fut, shapes, valid


def _moving_neighbor(start: tuple[float, float], speed: float = 10.0, heading: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.arange(T, dtype=torch.float32) * DT
    cos_h = float(torch.cos(torch.tensor(heading)))
    sin_h = float(torch.sin(torch.tensor(heading)))
    x = start[0] + speed * t * cos_h
    y = start[1] + speed * t * sin_h
    cos = torch.full((T,), cos_h)
    sin = torch.full((T,), sin_h)
    fut = torch.stack([x, y, cos, sin], dim=-1).unsqueeze(0)
    shapes = torch.tensor([[2.0, 4.5]])
    valid = torch.ones(1, T, dtype=torch.bool)
    return fut, shapes, valid


def _cfg(**kwargs) -> RewardConfig:
    base = dict(
        static_collision_enabled=True,
        sc_gate_enabled=True,
        sc_near_scale=1.0,
        sc_wide_scale=1.0,
        sc_cont_scale=1.0,
    )
    base.update(kwargs)
    return RewardConfig(**base)


# ---------------------------------------------------------------------------
# Closest-point helper
# ---------------------------------------------------------------------------


def test_closest_points_non_overlapping():
    # Two axis-aligned unit squares separated by 3m on x.
    r1 = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    r2 = torch.tensor([[[4.0, 0.0], [5.0, 0.0], [5.0, 1.0], [4.0, 1.0]]])
    p1, p2 = _closest_points_between_rects(r1, r2)
    # Closest points should be at some y in [0,1], x=1 on r1 and x=4 on r2.
    assert abs(float(p1[0, 0]) - 1.0) < 1e-4
    assert abs(float(p2[0, 0]) - 4.0) < 1e-4
    assert (p2 - p1).norm().item() == pytest.approx(3.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Stopped-neighbor filter
# ---------------------------------------------------------------------------


def test_moving_neighbor_is_excluded():
    """A neighbor moving at 10 m/s should not count as stopped even if ego passes close."""
    ego = _straight_ego(speed=5.0)  # along +x
    # Neighbor moving across ego's path — close at some t but MOVING
    nf, ns, nv = _moving_neighbor(start=(10.0, -5.0), speed=10.0, heading=1.5708)
    out = compute_static_collision_penalty(ego, _ego_shape(), nf, ns, nv, _cfg())
    assert out["stopped_mask"].sum().item() == 0
    assert float(out["crossing_gate"][0]) == 1.0  # no-op, no stopped nb
    assert out["first_crossing_steps"] == [None]


def test_stopped_neighbor_detected():
    ego = _straight_ego(speed=5.0)
    nf, ns, nv = _stopped_neighbor(center=(20.0, 5.0))
    out = compute_static_collision_penalty(ego, _ego_shape(), nf, ns, nv, _cfg())
    assert bool(out["stopped_mask"][0]) is True


# ---------------------------------------------------------------------------
# Overlap / clearance behaviour
# ---------------------------------------------------------------------------


def test_clear_pass_no_penalty():
    """Ego passing stopped NPC with >>1m clearance: no crossing, no penalty."""
    ego = _straight_ego(speed=5.0)  # x=0..40, y=0
    # Park NPC 5m to the side — clearance ≈ 5 - (1.70/2 + 2.0/2) = 5 - 1.85 = 3.15m
    nf, ns, nv = _stopped_neighbor(center=(20.0, 5.0))
    out = compute_static_collision_penalty(ego, _ego_shape(), nf, ns, nv, _cfg())
    assert float(out["crossing_gate"][0]) == 1.0
    assert out["first_crossing_steps"][0] is None
    assert float(out["near_penalty"][0]) == 0.0
    assert float(out["wide_penalty"][0]) == 0.0
    # cont_penalty should be tiny/zero — all clearances > 1m
    assert float(out["cont_penalty"][0]) == 0.0


def test_overlap_flags_crossing():
    """Ego drives STRAIGHT THROUGH a parked car — gate should fire."""
    ego = _straight_ego(speed=5.0)  # passes x=20 around step 40
    nf, ns, nv = _stopped_neighbor(center=(20.0, 0.0))  # sitting on the path
    out = compute_static_collision_penalty(ego, _ego_shape(), nf, ns, nv, _cfg())
    assert float(out["crossing_gate"][0]) == 0.0
    first = out["first_crossing_steps"][0]
    assert first is not None
    # First overlap step should be in the approach phase (before the ego
    # passes the NPC). Loose bound: anywhere in the first 3/4 of the horizon.
    assert 10 <= first <= 60, f"unexpected first_crossing_steps: {first}"


def test_near_zone_no_crossing():
    """Ego grazes NPC at ~0.3m clearance — above default cross_thresh=0.2, below near_thresh=0.4."""
    # Ego box edge at y=+0.85, NPC box edge at y_center - 1.0. For clearance ~0.3m:
    # y_center = 0.85 + 1.0 + 0.3 = 2.15m.
    ego = _straight_ego(speed=5.0)
    nf, ns, nv = _stopped_neighbor(center=(20.0, 2.15))
    out = compute_static_collision_penalty(ego, _ego_shape(), nf, ns, nv, _cfg())
    assert float(out["crossing_gate"][0]) == 1.0  # above cross_thresh → no crossing
    # Non-zero near penalty (clearance is in [cross_thresh, near_thresh)).
    assert float(out["near_penalty"][0]) > 0.0
    min_d = out["per_timestep_min"][0, 1:].min().item()
    assert 0.2 <= min_d < 0.4, f"min_d={min_d}"


# ---------------------------------------------------------------------------
# Ego-moving suppression
# ---------------------------------------------------------------------------


def test_stationary_ego_behind_parked_car_not_flagged():
    """Ego idles 0.5m behind a parked car: no crossing because ego speed < threshold."""
    # Ego sitting at origin, not moving (all timesteps same)
    ego = torch.zeros(1, T, 4)
    ego[..., 0] = 0.0
    ego[..., 1] = 0.0
    ego[..., 2] = 1.0  # cos_h=1, sin_h=0 (facing +x)
    # Car right in front of ego (ego length=4.34, so ego front at x=4.34-wb_half)
    # Place NPC so it almost touches: ego front edge is at x = (4.34-2.79)/2 + 4.34/2 ≈ 2.95
    # NPC center at x=5.0 with length 4.5 → rear at x=2.75 → gap ≈ 0.2m
    nf, ns, nv = _stopped_neighbor(center=(5.0, 0.0))
    out = compute_static_collision_penalty(ego, _ego_shape(), nf, ns, nv, _cfg())
    # Ego never moves faster than 1 m/s → no gate fire
    assert float(out["crossing_gate"][0]) == 1.0
    assert out["first_crossing_steps"][0] is None


# ---------------------------------------------------------------------------
# Disabled flag is a true no-op
# ---------------------------------------------------------------------------


def test_disabled_flag_no_effect_on_reward_totals():
    """Enabling the flag should add sc_penalty on top of otherwise-identical
    reward math. Use a near-miss scene (not an overlap) so the existing
    compute_safety_score_batch collision gate does NOT already fire — otherwise
    both totals would be floored to -50 regardless of the new flag."""
    ego = _straight_ego(speed=5.0)
    # NPC at y=2.15 → ~0.3m clearance (above cross_thresh=0.2, in near zone).
    nf, ns, nv = _stopped_neighbor(center=(20.0, 2.15))
    data = {
        "neighbor_agents_future": nf,
        "neighbor_agents_past": torch.zeros(1, 1, 21, 11),
        "ego_shape": _ego_shape().unsqueeze(0),
    }
    data["neighbor_agents_past"][0, 0, -1, 6] = ns[0, 0]  # width
    data["neighbor_agents_past"][0, 0, -1, 7] = ns[0, 1]  # length

    cfg_off = RewardConfig(static_collision_enabled=False)
    cfg_on = _cfg()

    out_off = compute_reward_batch(ego, data, cfg_off)
    out_on = compute_reward_batch(ego, data, cfg_on)

    # With static_collision_enabled=False, sc_* breakdown fields are default.
    assert out_off[0].static_crossing is False
    assert out_off[0].sc_min_dist == 99.0
    assert out_off[0].sc_n_stopped == 0

    # With the flag on, near penalty should populate and total should drop.
    assert out_on[0].static_crossing is False  # near miss, not overlap
    assert out_on[0].sc_near_penalty > 0.0
    assert out_on[0].sc_n_stopped == 1
    assert out_on[0].total < out_off[0].total


# ---------------------------------------------------------------------------
# Integration: survival-mode first-terminal + kinematic_gate exposure
# ---------------------------------------------------------------------------


def test_survival_mode_sc_crossing_contributes_to_first_terminal():
    """In reward_mode='survival', an sc crossing at timestep N must cap
    survival_frac to N/T — same mechanism as rb_crossing / lane_crossing."""
    # Ego drives straight into a stopped car centered on its path. The
    # overlap will fire sometime in the first 1-2 seconds (steps ~10-40
    # depending on approach geometry) since ego is at 5 m/s and the NPC
    # is at (8m, 0).
    ego = _straight_ego(speed=5.0)
    nf, ns, nv = _stopped_neighbor(center=(8.0, 0.0))
    data = {
        "neighbor_agents_future": nf,
        "neighbor_agents_past": torch.zeros(1, 1, 21, 11),
        "ego_shape": _ego_shape().unsqueeze(0),
    }
    data["neighbor_agents_past"][0, 0, -1, 6] = ns[0, 0]  # width
    data["neighbor_agents_past"][0, 0, -1, 7] = ns[0, 1]  # length

    cfg = RewardConfig(
        static_collision_enabled=True,
        sc_gate_enabled=True,
        sc_near_scale=1.0,
        reward_mode="survival",
    )
    out = compute_reward_batch(ego, data, cfg)

    # Survival-mode should floor the total because the prediction crashes
    # early — can't be the full quality score.
    _OFFROAD_FLOOR = -50.0
    assert out[0].static_crossing is True
    assert out[0].total < 0  # pulled toward the floor by survival_frac blend
    # And it should be strictly above the offroad floor (some survival frac).
    assert out[0].total > _OFFROAD_FLOOR - 1e-3


def test_kinematic_gate_field_exposed():
    """Adding kinematic_gate to RewardBreakdown should never break default paths."""
    ego = _straight_ego(speed=5.0)
    nf, ns, nv = _stopped_neighbor(center=(40.0, 10.0))  # far away, no interaction
    data = {
        "neighbor_agents_future": nf,
        "neighbor_agents_past": torch.zeros(1, 1, 21, 11),
        "ego_shape": _ego_shape().unsqueeze(0),
    }
    data["neighbor_agents_past"][0, 0, -1, 6] = ns[0, 0]
    data["neighbor_agents_past"][0, 0, -1, 7] = ns[0, 1]

    out = compute_reward_batch(ego, data, RewardConfig())
    # Straight-line ego at constant speed is kinematically feasible.
    assert out[0].kinematic_gate is True


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
