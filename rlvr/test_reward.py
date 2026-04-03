"""Unit tests for rlvr.reward -- rule-based trajectory reward.

Uses synthetic tensors (no model needed).
Run: python3 rlvr/test_reward.py
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch

from rlvr.reward import (
    RewardBreakdown,
    RewardConfig,
    compute_feasibility_score_batch,
    compute_group_advantages,
    compute_progress_score_batch,
    compute_red_light_score_batch,
    compute_reward_batch,
    compute_road_border_penalty,
    compute_safety_score_batch,
    compute_smoothness_score_batch,
)


T = 80
CONFIG = RewardConfig()


def _straight_line(speed_m_per_step: float = 0.5) -> torch.Tensor:
    t = torch.arange(T, dtype=torch.float32)
    x = t * speed_m_per_step
    y = torch.zeros(T)
    return torch.stack([x, y, torch.ones(T), torch.zeros(T)], dim=-1)


def _npc_straight(offset_y: float, speed: float = 0.5) -> torch.Tensor:
    t = torch.arange(T, dtype=torch.float32)
    x = t * speed
    y = torch.full((T,), offset_y)
    return torch.stack([x, y, torch.ones(T), torch.zeros(T)], dim=-1)


def _default_ego_shape():
    return torch.tensor([2.79, 4.34, 1.70])


def _default_neighbor_shapes(n: int):
    """(N, 2) = width, length per neighbor."""
    return torch.tensor([[2.0, 4.5]] * n, dtype=torch.float32)


def _make_lane_data(center_y: float = 0.0, width: float = 3.5) -> dict:
    lanes = torch.zeros(1, 140, 20, 33)
    for seg in range(10):
        for pt in range(20):
            x = (seg * 20 + pt) * 1.0
            half_w = width / 2
            lanes[0, seg, pt, 0] = x          # center X
            lanes[0, seg, pt, 1] = center_y   # center Y
            lanes[0, seg, pt, 2] = 1.0        # direction dX
            lanes[0, seg, pt, 3] = 0.0        # direction dY
            lanes[0, seg, pt, 4] = x          # left boundary X
            lanes[0, seg, pt, 5] = center_y + half_w  # left boundary Y
            lanes[0, seg, pt, 6] = x          # right boundary X
            lanes[0, seg, pt, 7] = center_y - half_w  # right boundary Y
    return {"lanes": lanes, "ego_shape": torch.tensor([[2.79, 4.34, 1.70]])}


def _make_road_border_data(border_y_left: float = 3.0, border_y_right: float = -3.0) -> dict:
    """Create line_strings data with road borders at specified y-offsets.

    line_strings shape: (1, num_ls, pts, 4) where dim 3 is [x, y, ?, road_border_flag].
    Road border flag > 0.5 marks road border points.

    Uses multiple line_strings with dense point spacing (0.05m) to ensure
    crossing/proximity detection works reliably — the reward function checks
    minimum distance to the nearest border point.
    """
    num_ls = 60
    pts = 20
    # Dense coverage: 30 line_strings per side × 20 pts × 0.05m = 30m per side
    # Total X coverage = 30m, enough for test trajectories
    ls = torch.zeros(1, num_ls, pts, 4)
    # Left road border (line_strings 0-29)
    for seg in range(30):
        for pt in range(pts):
            x = seg * (pts * 0.05) + pt * 0.05
            ls[0, seg, pt, 0] = x
            ls[0, seg, pt, 1] = border_y_left
            ls[0, seg, pt, 2] = 0.0
            ls[0, seg, pt, 3] = 1.0  # road border flag
    # Right road border (line_strings 30-59)
    for seg in range(30):
        for pt in range(pts):
            x = seg * (pts * 0.05) + pt * 0.05
            ls[0, 30 + seg, pt, 0] = x
            ls[0, 30 + seg, pt, 1] = border_y_right
            ls[0, 30 + seg, pt, 2] = 0.0
            ls[0, 30 + seg, pt, 3] = 1.0  # road border flag
    data = _make_lane_data(center_y=0.0)
    data["line_strings"] = ls
    return data


# -------------------------------------------------------------------------
# Safety score tests
# -------------------------------------------------------------------------

def test_no_collision_straight_line():
    ego = _straight_line().unsqueeze(0)
    npc = _npc_straight(offset_y=20.0).unsqueeze(0)
    npc_valid = torch.ones(1, T, dtype=torch.bool)
    scores, steps = compute_safety_score_batch(
        ego, _default_ego_shape(), npc, _default_neighbor_shapes(1), npc_valid, CONFIG
    )
    assert scores[0].item() == 0.0, f"Expected 0.0, got {scores[0]}"
    assert steps[0] is None
    print("  PASS  no_collision_straight_line")


def test_collision_detected():
    ego = _straight_line(speed_m_per_step=0.5).unsqueeze(0)
    npc_traj = torch.zeros(T, 4)
    for t in range(T):
        npc_traj[t, 0] = t * 0.5
        npc_traj[t, 1] = 10.0 - t * 0.25
        npc_traj[t, 2] = 1.0
    npc = npc_traj.unsqueeze(0)
    npc_valid = torch.ones(1, T, dtype=torch.bool)
    scores, steps = compute_safety_score_batch(
        ego, _default_ego_shape(), npc, _default_neighbor_shapes(1), npc_valid, CONFIG
    )
    assert scores[0].item() <= CONFIG.collision_penalty, f"Expected <= {CONFIG.collision_penalty}, got {scores[0]}"
    assert steps[0] is not None and 0 <= steps[0] < T
    print(f"  PASS  collision_detected (step={steps[0]}, score={scores[0]:.1f})")


def test_no_neighbors():
    ego = _straight_line().unsqueeze(0)
    empty_npc = torch.zeros(0, T, 4)
    empty_shapes = torch.zeros(0, 2)
    empty_valid = torch.zeros(0, T, dtype=torch.bool)
    scores, steps = compute_safety_score_batch(
        ego, _default_ego_shape(), empty_npc, empty_shapes, empty_valid, CONFIG
    )
    assert scores[0].item() == 0.0
    assert steps[0] is None
    print("  PASS  no_neighbors")


def test_multiple_neighbors_one_collision():
    ego = _straight_line(speed_m_per_step=0.5).unsqueeze(0)
    npcs = []
    for i in range(5):
        if i == 2:
            npc = _straight_line(speed_m_per_step=0.5)
            npc[:, 1] = 0.0
        else:
            npc = _npc_straight(offset_y=20.0 + i * 5.0)
        npcs.append(npc)
    npc_tensor = torch.stack(npcs)
    shapes = _default_neighbor_shapes(5)
    npc_valid = torch.ones(5, T, dtype=torch.bool)
    scores, steps = compute_safety_score_batch(
        ego, _default_ego_shape(), npc_tensor, shapes, npc_valid, CONFIG
    )
    assert scores[0].item() <= CONFIG.collision_penalty
    print(f"  PASS  multiple_neighbors_one_collision (step={steps[0]}, score={scores[0]:.1f})")


def test_near_miss_no_penalty():
    ego = _straight_line().unsqueeze(0)
    npc = _npc_straight(offset_y=5.0).unsqueeze(0)
    npc_valid = torch.ones(1, T, dtype=torch.bool)
    scores, steps = compute_safety_score_batch(
        ego, _default_ego_shape(), npc, _default_neighbor_shapes(1), npc_valid, CONFIG
    )
    assert scores[0].item() == 0.0
    assert steps[0] is None
    print("  PASS  near_miss_no_penalty")


def test_proximity_penalty():
    """NPC driving parallel at ~2.5m offset -- within proximity margin."""
    ego = _straight_line(speed_m_per_step=0.5).unsqueeze(0)
    # NPC at y=2.5 -- close enough for proximity penalty (gap < 1m after bbox sizes)
    npc = _npc_straight(offset_y=2.5, speed=0.5).unsqueeze(0)
    npc_valid = torch.ones(1, T, dtype=torch.bool)
    scores, steps = compute_safety_score_batch(
        ego, _default_ego_shape(), npc, _default_neighbor_shapes(1), npc_valid, CONFIG
    )
    assert steps[0] is None, "Should not be a collision"
    assert scores[0].item() < 0, f"Should have proximity penalty, got {scores[0]}"
    print(f"  PASS  proximity_penalty: score={scores[0]:.3f}")


# -------------------------------------------------------------------------
# Progress score tests
# -------------------------------------------------------------------------

def test_progress_toward_goal():
    ego = _straight_line(speed_m_per_step=0.5).unsqueeze(0)
    goal = torch.tensor([100.0, 0.0, 1.0, 0.0])
    scores = compute_progress_score_batch(ego, goal)
    assert scores[0].item() > 0, f"Expected positive, got {scores[0]}"
    print(f"  PASS  progress_toward_goal: {scores[0]:.2f}")


def test_progress_away_from_goal():
    ego = _straight_line(speed_m_per_step=0.5).unsqueeze(0)
    goal = torch.tensor([-50.0, 0.0, 1.0, 0.0])
    scores = compute_progress_score_batch(ego, goal)
    assert scores[0].item() < 0, f"Expected negative, got {scores[0]}"
    print(f"  PASS  progress_away_from_goal: {scores[0]:.2f}")


def test_no_goal_fallback_path_length():
    ego = _straight_line(speed_m_per_step=0.5).unsqueeze(0)
    goal = torch.zeros(4)
    scores = compute_progress_score_batch(ego, goal)
    assert scores[0].item() > 0, f"Expected positive path length, got {scores[0]}"
    print(f"  PASS  no_goal_fallback_path_length: {scores[0]:.2f}")


def test_stationary_trajectory():
    ego = torch.zeros(1, T, 4)
    ego[:, :, 2] = 1.0
    goal = torch.tensor([50.0, 0.0, 1.0, 0.0])
    scores = compute_progress_score_batch(ego, goal)
    assert abs(scores[0].item()) < 1e-5, f"Expected ~0, got {scores[0]}"
    print(f"  PASS  stationary_trajectory: {scores[0]:.6f}")


# -------------------------------------------------------------------------
# Smoothness score tests
# -------------------------------------------------------------------------

def test_straight_line_smooth():
    ego = _straight_line(speed_m_per_step=0.5).unsqueeze(0)
    scores = compute_smoothness_score_batch(ego, CONFIG)
    assert abs(scores[0].item()) < 0.01, f"Expected ~0, got {scores[0]}"
    print(f"  PASS  straight_line_smooth: {scores[0]:.6f}")


def test_zigzag_high_jerk():
    import math
    ego = _straight_line()
    # Sinusoidal lateral oscillation at ~1Hz (period=10 steps at 10Hz)
    # This is a physically plausible swerving trajectory with real jerk
    for t in range(T):
        ego[t, 1] = 1.0 * math.sin(2 * math.pi * t / 10)
    scores = compute_smoothness_score_batch(ego.unsqueeze(0), CONFIG)
    assert scores[0].item() < -1.0, f"Expected strongly negative, got {scores[0]}"
    print(f"  PASS  zigzag_high_jerk: {scores[0]:.2f}")


def test_constant_acceleration():
    t = torch.arange(T, dtype=torch.float32)
    x = 0.5 * 0.01 * t ** 2
    y = torch.zeros(T)
    ego = torch.stack([x, y, torch.ones(T), torch.zeros(T)], dim=-1)
    scores = compute_smoothness_score_batch(ego.unsqueeze(0), CONFIG)
    assert abs(scores[0].item()) < 0.01, f"Expected near zero jerk, got {scores[0]}"
    print(f"  PASS  constant_acceleration: {scores[0]:.6f}")


# -------------------------------------------------------------------------
# Feasibility score tests
# -------------------------------------------------------------------------

def test_within_lane_no_penalty():
    ego = _straight_line(speed_m_per_step=0.5).unsqueeze(0)
    data = _make_lane_data(center_y=0.0)
    scores, off_road = compute_feasibility_score_batch(ego, _default_ego_shape(), data, CONFIG)
    assert off_road[0].item() < 0.1, f"Expected low off_road, got {off_road[0]}"
    print(f"  PASS  within_lane_no_penalty: off_road={off_road[0]:.3f}, score={scores[0]:.3f}")



def test_high_acceleration_penalty():
    t = torch.arange(T, dtype=torch.float32)
    x = 0.5 * 100.0 * (t * 0.1) ** 2
    y = torch.zeros(T)
    ego = torch.stack([x, y, torch.ones(T), torch.zeros(T)], dim=-1)
    data = _make_lane_data()
    scores, _ = compute_feasibility_score_batch(ego.unsqueeze(0), _default_ego_shape(), data, CONFIG)
    assert scores[0].item() < -0.1, f"Expected penalty, got {scores[0]}"
    print(f"  PASS  high_acceleration_penalty: score={scores[0]:.3f}")


def test_moderate_driving_no_violation():
    ego = _straight_line(speed_m_per_step=0.3).unsqueeze(0)
    data = _make_lane_data(center_y=0.0)
    scores, off_road = compute_feasibility_score_batch(ego, _default_ego_shape(), data, CONFIG)
    assert scores[0].item() > -0.5, f"Expected minimal penalty, got {scores[0]}"
    print(f"  PASS  moderate_driving_no_violation: score={scores[0]:.3f}")


# -------------------------------------------------------------------------
# Road border tests
# -------------------------------------------------------------------------

def test_road_border_on_road():
    """Trajectory centered between road borders should have no crossing."""
    ego = _straight_line(speed_m_per_step=0.5).unsqueeze(0)
    data = _make_road_border_data(border_y_left=5.0, border_y_right=-5.0)
    rb_gate, near_frac, wide_frac, _, _ = compute_road_border_penalty(
        ego, _default_ego_shape(), data,
    )
    assert rb_gate[0].item() > 0.5, f"Expected no crossing (gate=1), got {rb_gate[0]}"
    print(f"  PASS  road_border_on_road: gate={rb_gate[0]:.1f}, near={near_frac[0]:.3f}")


def test_road_border_crossing():
    """Trajectory that drives directly on a road border should trigger crossing gate."""
    ego = _straight_line(speed_m_per_step=0.5)
    ego[:, 1] = 3.0  # drive right on the left border
    data = _make_road_border_data(border_y_left=3.0, border_y_right=-3.0)
    rb_gate, near_frac, wide_frac, _, _ = compute_road_border_penalty(
        ego.unsqueeze(0), _default_ego_shape(), data,
    )
    assert rb_gate[0].item() < 0.5, f"Expected crossing (gate=0), got {rb_gate[0]}"
    print(f"  PASS  road_border_crossing: gate={rb_gate[0]:.1f}, near={near_frac[0]:.3f}")


def test_road_border_near_penalty():
    """Trajectory near (but not crossing) a road border should have wide_frac > 0.

    wide_frac measures the fraction of timesteps where ego edge is within 40cm
    of the border. near_frac uses a tighter 25cm threshold.
    """
    ego = _straight_line(speed_m_per_step=0.5)
    # Drive close to right border at y=-3.0, ego width ~1.7 so edge at y-0.85
    ego[:, 1] = -1.9  # ego edge at ~-2.75, border at -3.0 → ~25cm gap
    data = _make_road_border_data(border_y_left=5.0, border_y_right=-3.0)
    rb_gate, near_frac, wide_frac, _, _ = compute_road_border_penalty(
        ego.unsqueeze(0), _default_ego_shape(), data,
    )
    # Should not cross but should have wide proximity penalty (within 40cm)
    assert wide_frac[0].item() > 0.0, f"Expected wide proximity (40cm) > 0, got {wide_frac[0]}"
    print(f"  PASS  road_border_near_penalty: gate={rb_gate[0]:.1f}, near={near_frac[0]:.3f}, wide={wide_frac[0]:.3f}")


def test_road_border_no_data():
    """Missing line_strings should return safe defaults."""
    ego = _straight_line(speed_m_per_step=0.5).unsqueeze(0)
    data = _make_lane_data()  # no line_strings key
    rb_gate, near_frac, wide_frac, _, _ = compute_road_border_penalty(
        ego, _default_ego_shape(), data,
    )
    assert rb_gate[0].item() == 1.0, "No data should return gate=1 (safe)"
    assert near_frac[0].item() == 0.0, "No data should return near_frac=0"
    print(f"  PASS  road_border_no_data: gate={rb_gate[0]:.1f}")


def test_road_border_batch():
    """Batch of trajectories: one safe, one crossing."""
    safe = _straight_line(speed_m_per_step=0.5)
    crossing_traj = _straight_line(speed_m_per_step=0.5)
    crossing_traj[:, 1] = 3.0  # on the left border
    trajs = torch.stack([safe, crossing_traj])
    data = _make_road_border_data(border_y_left=3.0, border_y_right=-3.0)
    rb_gate, near_frac, wide_frac, _, _ = compute_road_border_penalty(
        trajs, _default_ego_shape(), data,
    )
    assert rb_gate[0].item() > 0.5, "Safe traj should not cross"
    assert rb_gate[1].item() < 0.5, "Crossing traj should trigger gate"
    print(f"  PASS  road_border_batch: gates={rb_gate.tolist()}")


# -------------------------------------------------------------------------
# Batched integration tests (N > 1)
# -------------------------------------------------------------------------

def test_batch_multiple_trajectories():
    trajs = torch.stack([
        _straight_line(speed_m_per_step=0.5),
        _straight_line(speed_m_per_step=0.1),
        _straight_line(speed_m_per_step=0.0),
        _straight_line(speed_m_per_step=0.5),
    ])

    data = _make_lane_data()
    data["goal_pose"] = torch.tensor([[100.0, 0.0, 1.0, 0.0]])

    breakdowns = compute_reward_batch(trajs, data)
    assert len(breakdowns) == 4
    assert breakdowns[0].total > breakdowns[2].total, "Fast should beat stationary"
    print(f"  PASS  batch_multiple_trajectories: totals={[f'{b.total:.1f}' for b in breakdowns]}")


def test_batch_collision_mixed():
    npc_at_zero = _npc_straight(offset_y=0.0).unsqueeze(0)
    npc_valid = torch.ones(1, T, dtype=torch.bool)

    safe_traj = _straight_line(speed_m_per_step=0.5)
    safe_traj[:, 1] = 30.0

    colliding_traj = _straight_line(speed_m_per_step=0.5)

    trajs = torch.stack([safe_traj, colliding_traj])

    scores, steps = compute_safety_score_batch(
        trajs, _default_ego_shape(), npc_at_zero, _default_neighbor_shapes(1), npc_valid, CONFIG
    )
    assert scores[0].item() == 0.0, f"Safe traj should have no collision, got {scores[0]}"
    assert scores[1].item() <= CONFIG.collision_penalty, f"Colliding traj should have penalty"
    assert steps[0] is None
    assert steps[1] is not None
    print(f"  PASS  batch_collision_mixed: scores={scores.tolist()}, steps={steps}")


def test_batch_shape_consistency():
    N = 8
    trajs = torch.stack([_straight_line(speed_m_per_step=0.3 + i * 0.05) for i in range(N)])
    data = _make_lane_data()
    data["goal_pose"] = torch.tensor([[50.0, 0.0, 1.0, 0.0]])

    breakdowns = compute_reward_batch(trajs, data)
    assert len(breakdowns) == N
    for rb in breakdowns:
        assert isinstance(rb.safety, float)
        assert isinstance(rb.total, float)
    print(f"  PASS  batch_shape_consistency: N={N}")


def test_compute_reward_full_pipeline():
    ego = _straight_line(speed_m_per_step=0.5).unsqueeze(0)
    data = _make_lane_data()
    data["goal_pose"] = torch.tensor([[100.0, 0.0, 1.0, 0.0]])
    data["ego_shape"] = torch.tensor([[2.79, 4.34, 1.70]])
    breakdowns = compute_reward_batch(ego, data)
    rb = breakdowns[0]
    assert isinstance(rb, RewardBreakdown)
    # Reward uses multiplicative safety gates (NAVSIM PDMS-style):
    # total = safety_product * quality_score + (1 - safety_product) * floor
    # For a safe on-road trajectory, total should be positive.
    assert rb.total > 0, f"Expected positive total for safe trajectory, got {rb.total}"
    assert not rb.rb_crossing, "On-road trajectory should not cross road border"
    print(f"  PASS  compute_reward_full_pipeline: total={rb.total:.2f}")


# -------------------------------------------------------------------------
# Advantage tests
# -------------------------------------------------------------------------

def test_advantages_zero_mean():
    rewards = [
        RewardBreakdown(0, 5.0, -0.5, -0.1, 0.0, 0.0, 4.4, None, 0.0),
        RewardBreakdown(0, 3.0, -1.0, -0.2, 0.0, 0.0, 1.8, None, 0.1),
        RewardBreakdown(0, 8.0, -0.3, -0.0, 0.0, 0.0, 7.7, None, 0.0),
        RewardBreakdown(-10, 2.0, -2.0, -0.5, 0.0, 0.0, -10.5, 5, 0.3),
    ]
    adv = compute_group_advantages(rewards)
    assert abs(adv.mean()) < 1e-6, f"Expected zero mean, got {adv.mean()}"
    print(f"  PASS  advantages_zero_mean: mean={adv.mean():.8f}")


def test_advantages_unit_variance():
    rewards = [
        RewardBreakdown(0, i * 2.0, -0.5, -0.1, 0.0, 0.0, i * 2.0 - 0.6, None, 0.0)
        for i in range(10)
    ]
    adv = compute_group_advantages(rewards)
    assert abs(adv.std() - 1.0) < 0.1, f"Expected ~1.0 std, got {adv.std()}"
    print(f"  PASS  advantages_unit_variance: std={adv.std():.4f}")


def test_identical_rewards():
    rewards = [
        RewardBreakdown(0, 5.0, -0.5, -0.1, 0.0, 0.0, 4.4, None, 0.0)
        for _ in range(5)
    ]
    adv = compute_group_advantages(rewards)
    assert np.allclose(adv, 0.0), f"Expected all zeros, got {adv}"
    print("  PASS  identical_rewards")


def test_one_outlier():
    rewards = [
        RewardBreakdown(0, 1.0, 0, 0, 0.0, 0.0, 1.0, None, 0.0),
        RewardBreakdown(0, 1.0, 0, 0, 0.0, 0.0, 1.0, None, 0.0),
        RewardBreakdown(0, 1.0, 0, 0, 0.0, 0.0, 1.0, None, 0.0),
        RewardBreakdown(0, 100.0, 0, 0, 0.0, 0.0, 100.0, None, 0.0),
    ]
    adv = compute_group_advantages(rewards)
    assert adv[3] > 0, f"Outlier should have positive advantage, got {adv[3]}"
    assert all(adv[i] < 0 for i in range(3)), f"Others should be negative: {adv[:3]}"
    print(f"  PASS  one_outlier: adv={adv}")


# -------------------------------------------------------------------------
# VD-GRPO advantage tests
# -------------------------------------------------------------------------

def test_vd_grpo_fixed_scale():
    """VD-GRPO should preserve crash signal magnitude."""
    rewards = [
        RewardBreakdown(0, 5.0, 0, 0, 0.0, 0.0, -50.0, 0, 0.0),  # crash
        RewardBreakdown(0, 5.0, 0, 0, 0.0, 0.0, 5.0, None, 0.0),
        RewardBreakdown(0, 5.0, 0, 0, 0.0, 0.0, 6.0, None, 0.0),
        RewardBreakdown(0, 5.0, 0, 0, 0.0, 0.0, 7.0, None, 0.0),
    ]
    adv = compute_group_advantages(rewards, mode="vd_grpo", fixed_scale=10.0)
    # Crash advantage should be large negative (not compressed to ~-1.5)
    assert adv[0] < -3.0, f"Crash should have large negative adv, got {adv[0]}"
    assert adv[3] > 0, f"Best traj should have positive adv, got {adv[3]}"
    print(f"  PASS  vd_grpo_fixed_scale: adv={adv}")


def test_vd_grpo_invalid_scale():
    """VD-GRPO should raise on zero/negative fixed_scale."""
    rewards = [RewardBreakdown(0, 1.0, 0, 0, 0.0, 0.0, 1.0, None, 0.0)]
    try:
        compute_group_advantages(rewards, mode="vd_grpo", fixed_scale=0.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print(f"  PASS  vd_grpo_invalid_scale: raised ValueError")


def test_invalid_advantage_mode():
    """Unknown advantage mode should raise ValueError."""
    rewards = [RewardBreakdown(0, 1.0, 0, 0, 0.0, 0.0, 1.0, None, 0.0)]
    try:
        compute_group_advantages(rewards, mode="bogus")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print(f"  PASS  invalid_advantage_mode: raised ValueError")


# -------------------------------------------------------------------------
# Survival reward tests
# -------------------------------------------------------------------------

def test_survival_reward_late_crash_beats_early():
    """Same trajectory hitting an early vs late NPC: survival mode differentiates, gate doesn't."""
    # Same ego speed, two NPCs at different distances.
    # Ego at 0.3m/step hits NPC at x=5 around t=12, NPC at x=20 around t=63.
    ego = _straight_line(speed_m_per_step=0.3)

    # Traj 0: collides with NPC at x=5 (early crash)
    # Traj 1: collides with NPC at x=20 (late crash)
    # Use the same ego trajectory for both, but different NPC positions per eval.
    data_early = _make_lane_data()
    data_early["goal_pose"] = torch.tensor([[100.0, 0.0, 1.0, 0.0]])
    npc_early = torch.zeros(1, 1, T, 4)
    npc_early[:, :, :, 0] = 5.0
    npc_early[:, :, :, 2] = 1.0
    data_early["neighbor_agents_future"] = npc_early
    data_early["neighbor_agents_past"] = torch.zeros(1, 1, 21, 11)
    data_early["neighbor_agents_past"][:, :, -1, 6] = 2.0
    data_early["neighbor_agents_past"][:, :, -1, 7] = 4.5
    data_early["neighbor_agents_past"][:, :, :, 0] = 5.0
    data_early["neighbor_agents_past"][:, :, :, 2] = 1.0

    data_late = _make_lane_data()
    data_late["goal_pose"] = torch.tensor([[100.0, 0.0, 1.0, 0.0]])
    npc_late = torch.zeros(1, 1, T, 4)
    npc_late[:, :, :, 0] = 20.0
    npc_late[:, :, :, 2] = 1.0
    data_late["neighbor_agents_future"] = npc_late
    data_late["neighbor_agents_past"] = torch.zeros(1, 1, 21, 11)
    data_late["neighbor_agents_past"][:, :, -1, 6] = 2.0
    data_late["neighbor_agents_past"][:, :, -1, 7] = 4.5
    data_late["neighbor_agents_past"][:, :, :, 0] = 20.0
    data_late["neighbor_agents_past"][:, :, :, 2] = 1.0

    cfg_gate = RewardConfig(reward_mode="gate")
    cfg_surv = RewardConfig(reward_mode="survival")

    rw_gate_early = compute_reward_batch(ego.unsqueeze(0), data_early, cfg_gate)[0]
    rw_gate_late = compute_reward_batch(ego.unsqueeze(0), data_late, cfg_gate)[0]
    rw_surv_early = compute_reward_batch(ego.unsqueeze(0), data_early, cfg_surv)[0]
    rw_surv_late = compute_reward_batch(ego.unsqueeze(0), data_late, cfg_surv)[0]

    assert rw_gate_early.collision_step is not None, "Early NPC should cause collision"
    assert rw_gate_late.collision_step is not None, "Late NPC should cause collision"
    assert rw_gate_late.collision_step > rw_gate_early.collision_step, \
        f"Late NPC should crash later: {rw_gate_late.collision_step} vs {rw_gate_early.collision_step}"
    # Gate mode: both get same floor
    assert rw_gate_early.total == rw_gate_late.total, \
        f"Gate mode should give same floor: {rw_gate_early.total} vs {rw_gate_late.total}"
    # Survival mode: late crash should score higher (more survived quality)
    assert rw_surv_late.total > rw_surv_early.total, \
        f"Late crash should beat early: late={rw_surv_late.total:.1f} vs early={rw_surv_early.total:.1f}"
    print(f"  PASS  survival_reward_late_crash_beats_early: "
          f"gate=[{rw_gate_early.total:.1f}, {rw_gate_late.total:.1f}] "
          f"surv=[{rw_surv_early.total:.1f}, {rw_surv_late.total:.1f}] "
          f"collision_steps=[{rw_gate_early.collision_step}, {rw_gate_late.collision_step}]")


def test_survival_reward_safe_matches_gate():
    """For safe trajectories, survival mode should equal gate mode."""
    ego = _straight_line(speed_m_per_step=0.5).unsqueeze(0)
    data = _make_lane_data()
    data["goal_pose"] = torch.tensor([[100.0, 0.0, 1.0, 0.0]])

    rw_gate = compute_reward_batch(ego, data, RewardConfig(reward_mode="gate"))
    rw_surv = compute_reward_batch(ego, data, RewardConfig(reward_mode="survival"))

    # No failure → survival_frac=1.0 → same as gate with safety_product=1.0
    assert abs(rw_gate[0].total - rw_surv[0].total) < 0.01, \
        f"Safe traj should match: gate={rw_gate[0].total:.2f} vs surv={rw_surv[0].total:.2f}"
    print(f"  PASS  survival_reward_safe_matches_gate: gate={rw_gate[0].total:.2f}, surv={rw_surv[0].total:.2f}")


# -------------------------------------------------------------------------
# Red light tests
# -------------------------------------------------------------------------

def test_red_light_no_violation():
    """Trajectory that stays away from red-light lane points gets no penalty."""
    N, T = 1, 80
    # Ego goes straight ahead at y=0
    ego = torch.zeros(N, T, 4)
    ego[:, :, 0] = torch.linspace(0, 40, T)
    ego[:, :, 2] = 1.0  # cos heading = forward

    # Red light on a perpendicular lane at (20, -15), direction (0, 1) — not on ego's path
    route_lanes = torch.zeros(1, 25, 20, 33)
    for pt in range(10):
        route_lanes[0, 2, pt, 0] = 20.0       # x
        route_lanes[0, 2, pt, 1] = -15 + pt   # y
        route_lanes[0, 2, pt, 2] = 0.0         # dx (perpendicular)
        route_lanes[0, 2, pt, 3] = 1.5         # dy
        route_lanes[0, 2, pt, 10] = 1.0        # RED light

    data = {"route_lanes": route_lanes}
    config = RewardConfig()
    scores = compute_red_light_score_batch(ego, data, config)
    assert scores[0].item() == 0.0, f"Expected no penalty, got {scores[0].item()}"
    print(f"  PASS  red_light_no_violation: score={scores[0].item():.1f}")


def test_red_light_violation():
    """Trajectory that enters a red-light lane gets penalized."""
    N, T = 1, 80
    # Ego goes straight, passing through red-light lane points at (10-20, 0)
    ego = torch.zeros(N, T, 4)
    ego[:, :, 0] = torch.linspace(0, 30, T)
    ego[:, :, 2] = 1.0  # cos heading = forward

    # Red light directly ahead on ego's lane, direction (1, 0) — aligned with ego
    route_lanes = torch.zeros(1, 25, 20, 33)
    for pt in range(10):
        route_lanes[0, 1, pt, 0] = 10 + pt * 1.5  # x = 10 to 23.5
        route_lanes[0, 1, pt, 1] = 0.0              # y = 0 (on ego's path)
        route_lanes[0, 1, pt, 2] = 1.5              # dx (forward)
        route_lanes[0, 1, pt, 3] = 0.0              # dy
        route_lanes[0, 1, pt, 10] = 1.0             # RED light

    data = {"route_lanes": route_lanes}
    config = RewardConfig()
    scores = compute_red_light_score_batch(ego, data, config)
    assert scores[0].item() < -5.0, f"Expected strong penalty, got {scores[0].item()}"
    print(f"  PASS  red_light_violation: score={scores[0].item():.1f}")


def test_red_light_stopped_no_penalty():
    """Ego stopped near red light but not moving through it — no penalty."""
    N, T = 1, 80
    # Ego is stationary at (0, 0)
    ego = torch.zeros(N, T, 4)
    ego[:, :, 0] = 0.0  # not moving
    ego[:, :, 2] = 1.0  # heading forward

    # Red light at (5, 0) directly ahead
    route_lanes = torch.zeros(1, 25, 20, 33)
    for pt in range(5):
        route_lanes[0, 1, pt, 0] = 5 + pt
        route_lanes[0, 1, pt, 1] = 0.0
        route_lanes[0, 1, pt, 2] = 1.5
        route_lanes[0, 1, pt, 3] = 0.0
        route_lanes[0, 1, pt, 10] = 1.0

    data = {"route_lanes": route_lanes}
    config = RewardConfig()
    scores = compute_red_light_score_batch(ego, data, config)
    assert scores[0].item() == 0.0, f"Stopped ego should not be penalized, got {scores[0].item()}"
    print(f"  PASS  red_light_stopped_no_penalty: score={scores[0].item():.1f}")


def test_red_light_no_data():
    """No route_lanes in data — no penalty."""
    N, T = 1, 80
    ego = torch.zeros(N, T, 4)
    ego[:, :, 0] = torch.linspace(0, 30, T)
    ego[:, :, 2] = 1.0

    data = {}
    config = RewardConfig()
    scores = compute_red_light_score_batch(ego, data, config)
    assert scores[0].item() == 0.0
    print(f"  PASS  red_light_no_data: score={scores[0].item():.1f}")


# -------------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------------

def test_advantage_raw_all_bad():
    """Raw mode: all-bad group should have all negative or zero advantages."""
    class FakeReward:
        def __init__(self, t): self.total = t
    # All trajectories are bad (e.g., all gated at -50)
    rewards = [FakeReward(-50.0 + i * 0.1) for i in range(8)]
    adv = compute_group_advantages(rewards, mode="raw", fixed_scale=10.0)
    # Centered: mean ~ -49.65, so all are close to zero (centered)
    # But importantly, NOT normalized to have half positive
    assert adv.std() < 0.1, f"Raw advantages should have small spread: std={adv.std():.3f}"
    print("  PASS  test_advantage_raw_all_bad")


def test_advantage_positive_only():
    """Positive-only mode: negative advantages should be clipped to zero."""
    class FakeReward:
        def __init__(self, t): self.total = t
    rewards = [FakeReward(1.0), FakeReward(2.0), FakeReward(3.0), FakeReward(10.0)]
    adv = compute_group_advantages(rewards, mode="positive_only")
    # Only the above-mean trajectories should have positive advantages
    assert all(a >= 0 for a in adv), f"All advantages should be >= 0: {adv}"
    # The best trajectory should have the highest advantage
    assert adv[3] > adv[0], f"Best traj should have highest adv: {adv}"
    # At least one should be zero (below mean)
    assert any(a == 0 for a in adv), f"Some advantages should be exactly 0: {adv}"
    print("  PASS  test_advantage_positive_only")


def test_lane_departure_in_lane():
    """Trajectory staying in-lane should not trigger lane departure."""
    from rlvr.reward import compute_lane_departure_penalty
    device = torch.device("cpu")
    T = 20
    # Ego trajectory going straight at y=0, well within lane
    ego = torch.zeros(1, T, 4, device=device)
    for t in range(T):
        ego[0, t, 0] = 2.0 + t * 0.5  # x moves forward, start at x=2
        ego[0, t, 2] = 1.0             # cos(heading) = 1
    # Lane centerline at y=0, width=3.5m (half_width=1.75m)
    # Lane extends well beyond ego to avoid polygon boundary issues
    lanes = torch.zeros(1, 10, 20, 33, device=device)
    for pt in range(20):
        lanes[0, 0, pt, 0] = pt * 1.0    # center X (0 to 19, covers ego range)
        lanes[0, 0, pt, 1] = 0.01         # center Y slightly off-zero (avoid origin validity filter)
        lanes[0, 0, pt, 2] = 1.0          # direction cos
        lanes[0, 0, pt, 4] = 0.0          # left boundary dX
        lanes[0, 0, pt, 5] = 1.74         # left boundary dY
        lanes[0, 0, pt, 6] = 0.0          # right boundary dX
        lanes[0, 0, pt, 7] = -1.76        # right boundary dY
    ego_shape = torch.tensor([2.75, 4.34, 1.70])
    data = {"lanes": lanes}
    crossing_gate, near_frac, wide_frac, _, cont = compute_lane_departure_penalty(ego, ego_shape, data)
    assert crossing_gate[0] == 1.0, f"In-lane trajectory should not cross: gate={crossing_gate[0]}"
    print("  PASS  test_lane_departure_in_lane")


def test_lane_departure_out_of_lane():
    """Trajectory far outside lane should trigger lane departure."""
    from rlvr.reward import compute_lane_departure_penalty
    device = torch.device("cpu")
    T = 20
    # Ego trajectory at y=5.0 (well outside 1.75m half-width lane)
    ego = torch.zeros(1, T, 4, device=device)
    for t in range(T):
        ego[0, t, 0] = t * 0.5
        ego[0, t, 1] = 5.0      # far outside lane
        ego[0, t, 2] = 1.0
    lanes = torch.zeros(1, 10, 20, 33, device=device)
    for pt in range(20):
        lanes[0, 0, pt, 0] = pt * 0.5
        lanes[0, 0, pt, 1] = 0.0
        lanes[0, 0, pt, 2] = 1.0
        lanes[0, 0, pt, 4] = 0.0
        lanes[0, 0, pt, 5] = 1.75
        lanes[0, 0, pt, 6] = 0.0
        lanes[0, 0, pt, 7] = -1.75
    ego_shape = torch.tensor([2.75, 4.34, 1.70])
    data = {"lanes": lanes}
    crossing_gate, near_frac, wide_frac, _, cont = compute_lane_departure_penalty(ego, ego_shape, data)
    assert crossing_gate[0] == 0.0, f"Out-of-lane trajectory should cross: gate={crossing_gate[0]}"
    print("  PASS  test_lane_departure_out_of_lane")


def test_advantage_absolute():
    """Absolute mode: no centering, positive reward → positive advantage."""
    from rlvr.reward import compute_group_advantages, RewardBreakdown
    rewards = [RewardBreakdown(safety=0, progress=0, smoothness=0, feasibility=0, centerline=0,
                               red_light=0, total=t, collision_step=None, off_road_fraction=0)
               for t in [+10, +5, -5, -20]]
    adv = compute_group_advantages(rewards, mode="absolute", fixed_scale=10.0)
    assert adv[0] > 0, f"Positive reward should give positive advantage: {adv[0]}"
    assert adv[1] > 0, f"Positive reward should give positive advantage: {adv[1]}"
    assert adv[2] < 0, f"Negative reward should give negative advantage: {adv[2]}"
    assert adv[3] < 0, f"Negative reward should give negative advantage: {adv[3]}"
    assert abs(adv[0] - 1.0) < 1e-6, f"10/10 should be 1.0: {adv[0]}"
    print("  PASS  test_advantage_absolute")


def test_advantage_softmax():
    """Softmax mode: rank 1 gets disproportionately high weight."""
    from rlvr.reward import compute_group_advantages, RewardBreakdown
    rewards = [RewardBreakdown(safety=0, progress=0, smoothness=0, feasibility=0, centerline=0,
                               red_light=0, total=t, collision_step=None, off_road_fraction=0)
               for t in [+20, +5, 0, -10, -30]]
    adv = compute_group_advantages(rewards, mode="softmax", fixed_scale=5.0)
    # Rank 1 should have the highest advantage
    assert adv[0] > adv[1], f"Rank 1 should beat rank 2: {adv[0]} vs {adv[1]}"
    # Rank 1 should be much larger than rank 2 (softmax concentrates)
    assert adv[0] > 2 * adv[1], f"Softmax T=5 should concentrate on rank 1: {adv[0]} vs {adv[1]}"
    # Last rank should be negative (centered)
    assert adv[-1] < 0, f"Worst trajectory should have negative advantage: {adv[-1]}"
    print("  PASS  test_advantage_softmax")


if __name__ == "__main__":
    tests = [
        test_no_collision_straight_line,
        test_collision_detected,
        test_no_neighbors,
        test_multiple_neighbors_one_collision,
        test_near_miss_no_penalty,
        test_proximity_penalty,
        test_progress_toward_goal,
        test_progress_away_from_goal,
        test_no_goal_fallback_path_length,
        test_stationary_trajectory,
        test_straight_line_smooth,
        test_zigzag_high_jerk,
        test_constant_acceleration,
        test_within_lane_no_penalty,
        test_high_acceleration_penalty,
        test_moderate_driving_no_violation,
        test_road_border_on_road,
        test_road_border_crossing,
        test_road_border_near_penalty,
        test_road_border_no_data,
        test_road_border_batch,
        test_batch_multiple_trajectories,
        test_batch_collision_mixed,
        test_batch_shape_consistency,
        test_compute_reward_full_pipeline,
        test_advantages_zero_mean,
        test_advantages_unit_variance,
        test_identical_rewards,
        test_one_outlier,
        test_vd_grpo_fixed_scale,
        test_vd_grpo_invalid_scale,
        test_invalid_advantage_mode,
        test_survival_reward_late_crash_beats_early,
        test_survival_reward_safe_matches_gate,
        test_red_light_no_violation,
        test_red_light_violation,
        test_red_light_stopped_no_penalty,
        test_red_light_no_data,
        test_advantage_raw_all_bad,
        test_advantage_positive_only,
        test_lane_departure_in_lane,
        test_lane_departure_out_of_lane,
        test_advantage_absolute,
        test_advantage_softmax,
    ]

    print("=" * 60)
    print("RLVR Reward Test Suite")
    print("=" * 60 + "\n")

    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    if failed == 0:
        print(f"ALL {len(tests)} TESTS PASSED!")
    else:
        print(f"{failed}/{len(tests)} TESTS FAILED")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
