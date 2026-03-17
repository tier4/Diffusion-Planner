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
    compute_reward_batch,
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
    assert scores[0].item() == CONFIG.collision_penalty, f"Expected {CONFIG.collision_penalty}, got {scores[0]}"
    assert steps[0] is not None and 0 <= steps[0] < T
    print(f"  PASS  collision_detected (step={steps[0]})")


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
    assert scores[0].item() == CONFIG.collision_penalty
    print(f"  PASS  multiple_neighbors_one_collision (step={steps[0]})")


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
    assert abs(scores[0].item()) < 1e-3, f"Expected ~0, got {scores[0]}"
    print(f"  PASS  straight_line_smooth: {scores[0]:.6f}")


def test_zigzag_high_jerk():
    ego = _straight_line()
    for t in range(T):
        ego[t, 1] = 2.0 * ((-1) ** t)
    scores = compute_smoothness_score_batch(ego.unsqueeze(0), CONFIG)
    assert scores[0].item() < -1.0, f"Expected strongly negative, got {scores[0]}"
    print(f"  PASS  zigzag_high_jerk: {scores[0]:.2f}")


def test_constant_acceleration():
    t = torch.arange(T, dtype=torch.float32)
    x = 0.5 * 0.01 * t ** 2
    y = torch.zeros(T)
    ego = torch.stack([x, y, torch.ones(T), torch.zeros(T)], dim=-1)
    scores = compute_smoothness_score_batch(ego.unsqueeze(0), CONFIG)
    assert abs(scores[0].item()) < 1.0, f"Expected near zero jerk, got {scores[0]}"
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


def test_off_road_trajectory():
    ego = _straight_line()
    ego[:, 1] = 50.0
    data = _make_lane_data(center_y=0.0)
    scores, off_road = compute_feasibility_score_batch(ego.unsqueeze(0), _default_ego_shape(), data, CONFIG)
    assert off_road[0].item() > 0.5, f"Expected high off_road, got {off_road[0]}"
    print(f"  PASS  off_road_trajectory: off_road={off_road[0]:.3f}, score={scores[0]:.3f}")


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
# Batched integration tests (N > 1)
# -------------------------------------------------------------------------

def test_batch_multiple_trajectories():
    trajs = torch.stack([
        _straight_line(speed_m_per_step=0.5),
        _straight_line(speed_m_per_step=0.1),
        _straight_line(speed_m_per_step=0.0),
        _straight_line(speed_m_per_step=0.5),
    ])
    trajs[3, :, 1] = 50.0  # off-road

    data = _make_lane_data()
    data["goal_pose"] = torch.tensor([[100.0, 0.0, 1.0, 0.0]])

    breakdowns = compute_reward_batch(trajs, data)
    assert len(breakdowns) == 4
    assert breakdowns[0].total > breakdowns[2].total, "Fast should beat stationary"
    assert breakdowns[3].feasibility < breakdowns[0].feasibility, "Off-road should have worse feasibility"
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
    assert scores[1].item() == CONFIG.collision_penalty, f"Colliding traj should have penalty"
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
    expected = (
        CONFIG.w_safety * rb.safety
        + CONFIG.w_progress * rb.progress
        + CONFIG.w_smooth * rb.smoothness
        + CONFIG.w_feasibility * rb.feasibility
        + CONFIG.w_centerline * rb.centerline
    )
    assert abs(rb.total - expected) < 1e-4, f"Total mismatch: {rb.total} vs {expected}"
    print(f"  PASS  compute_reward_full_pipeline: total={rb.total:.2f}")


# -------------------------------------------------------------------------
# Advantage tests
# -------------------------------------------------------------------------

def test_advantages_zero_mean():
    rewards = [
        RewardBreakdown(0, 5.0, -0.5, -0.1, 0.0, 4.4, None, 0.0),
        RewardBreakdown(0, 3.0, -1.0, -0.2, 0.0, 1.8, None, 0.1),
        RewardBreakdown(0, 8.0, -0.3, -0.0, 0.0, 7.7, None, 0.0),
        RewardBreakdown(-10, 2.0, -2.0, -0.5, 0.0, -10.5, 5, 0.3),
    ]
    adv = compute_group_advantages(rewards)
    assert abs(adv.mean()) < 1e-6, f"Expected zero mean, got {adv.mean()}"
    print(f"  PASS  advantages_zero_mean: mean={adv.mean():.8f}")


def test_advantages_unit_variance():
    rewards = [
        RewardBreakdown(0, i * 2.0, -0.5, -0.1, 0.0, i * 2.0 - 0.6, None, 0.0)
        for i in range(10)
    ]
    adv = compute_group_advantages(rewards)
    assert abs(adv.std() - 1.0) < 0.1, f"Expected ~1.0 std, got {adv.std()}"
    print(f"  PASS  advantages_unit_variance: std={adv.std():.4f}")


def test_identical_rewards():
    rewards = [
        RewardBreakdown(0, 5.0, -0.5, -0.1, 0.0, 4.4, None, 0.0)
        for _ in range(5)
    ]
    adv = compute_group_advantages(rewards)
    assert np.allclose(adv, 0.0), f"Expected all zeros, got {adv}"
    print("  PASS  identical_rewards")


def test_one_outlier():
    rewards = [
        RewardBreakdown(0, 1.0, 0, 0, 0.0, 1.0, None, 0.0),
        RewardBreakdown(0, 1.0, 0, 0, 0.0, 1.0, None, 0.0),
        RewardBreakdown(0, 1.0, 0, 0, 0.0, 1.0, None, 0.0),
        RewardBreakdown(0, 100.0, 0, 0, 0.0, 100.0, None, 0.0),
    ]
    adv = compute_group_advantages(rewards)
    assert adv[3] > 0, f"Outlier should have positive advantage, got {adv[3]}"
    assert all(adv[i] < 0 for i in range(3)), f"Others should be negative: {adv[:3]}"
    print(f"  PASS  one_outlier: adv={adv}")


# -------------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_no_collision_straight_line,
        test_collision_detected,
        test_no_neighbors,
        test_multiple_neighbors_one_collision,
        test_near_miss_no_penalty,
        test_progress_toward_goal,
        test_progress_away_from_goal,
        test_no_goal_fallback_path_length,
        test_stationary_trajectory,
        test_straight_line_smooth,
        test_zigzag_high_jerk,
        test_constant_acceleration,
        test_within_lane_no_penalty,
        test_off_road_trajectory,
        test_high_acceleration_penalty,
        test_moderate_driving_no_violation,
        test_batch_multiple_trajectories,
        test_batch_collision_mixed,
        test_batch_shape_consistency,
        test_compute_reward_full_pipeline,
        test_advantages_zero_mean,
        test_advantages_unit_variance,
        test_identical_rewards,
        test_one_outlier,
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
