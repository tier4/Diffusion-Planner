"""Smoke test: underprogress / overprogress / stopped penalties must fire
regardless of w_progress.

Regression for the 2026-04-24 path-collapse bug: these penalties were
accumulated into `clamped_progress`, which got multiplied by `w_progress`
in `quality_score`. When `w_progress=0` (CL-only configs), the penalties
silently evaporated and path collapsed unchecked.
"""
from __future__ import annotations

import math

import pytest
import torch

from rlvr.reward import (
    RewardConfig,
    compute_reward_batch,
)


def _make_scene(
    path_lens: list[float],
    gt_len: float = 60.0,
    T: int = 80,
    dt: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Build N straight-line ego trajectories of given path lengths + GT.

    Each traj is a straight line along +x starting from origin, yaw=0.
    No neighbors, no lanes (no-op for this test).
    """
    N = len(path_lens)
    trajs = torch.zeros(N, T, 4)
    for i, L in enumerate(path_lens):
        v = L / ((T - 1) * dt)  # constant speed to cover L over T-1 steps
        xs = torch.arange(T, dtype=torch.float32) * v * dt
        trajs[i, :, 0] = xs
        trajs[i, :, 2] = 1.0  # cos(0) = 1, yaw = 0

    # GT: straight line of length gt_len along +x, T=80 points
    gt_future = torch.zeros(T, 3)
    v_gt = gt_len / ((T - 1) * dt)
    gt_future[:, 0] = torch.arange(T, dtype=torch.float32) * v_gt * dt

    data = {
        "ego_agent_future": gt_future.unsqueeze(0),  # (1, T, 3)
        "ego_shape": torch.tensor([[2.79, 4.34, 1.70]]),
    }
    return trajs, data


def _cfg_cl_only(**overrides) -> RewardConfig:
    """CL-only config mirroring the RUN J-v2 setup."""
    base = dict(
        w_progress=0.0,
        w_safety=0.0,
        w_smooth=0.0,
        w_centerline=0.0,
        progress_norm_scale=20.0,
        # gates off so survival floor is the only other moving piece
        rb_gate_enabled=False,
        lane_gate_enabled=False,
        enable_lane_departure=False,
        static_collision_enabled=False,
        reward_mode="direct",  # skip survival so totals = quality_score
        # penalty settings under test
        enable_overprogress=True,
        underprogress_penalty=50.0,
        underprogress_threshold=0.85,
        underprogress_reference="det",
        overprogress_penalty=30.0,
        overprogress_margin=1.1,
        stopped_penalty=100.0,
    )
    base.update(overrides)
    return RewardConfig(**base)


def _totals(cfg, trajs, data):
    return [float(r.total) for r in compute_reward_batch(trajs, data, cfg)]


def test_underprogress_fires_with_w_progress_zero():
    """Core regression: w_progress=0, short traj must score worse than long traj."""
    # det = 60m (matches GT), test traj = 30m (half, < 0.85*60=51)
    trajs, data = _make_scene([60.0, 30.0], gt_len=60.0)
    cfg = _cfg_cl_only()
    totals = _totals(cfg, trajs, data)

    # Short traj underprogress = relu(0.85 - 30/60) = 0.35 → penalty = 50*0.35 = 17.5
    # Det traj underprogress = relu(0.85 - 1.0) = 0 → no penalty
    # Expected gap: totals[1] - totals[0] ≈ -17.5
    gap = totals[1] - totals[0]
    assert gap == pytest.approx(-17.5, abs=0.5), (
        f"underprogress did not fire with w_progress=0. "
        f"totals={totals}, gap={gap}, expected ~-17.5"
    )


def test_stopped_fires_with_w_progress_zero():
    """Stopped traj (<1m) when GT moves (>5m) must trigger stopped_penalty=100."""
    trajs, data = _make_scene([60.0, 0.5], gt_len=60.0)  # 2nd is stopped
    cfg = _cfg_cl_only(underprogress_penalty=0.0)  # isolate stopped penalty
    totals = _totals(cfg, trajs, data)

    # stopped_penalty=100 hit exactly once for traj[1]
    gap = totals[1] - totals[0]
    assert gap == pytest.approx(-100.0, abs=1.0), (
        f"stopped penalty did not fire with w_progress=0. "
        f"totals={totals}, gap={gap}, expected ~-100"
    )


def test_overprogress_fires_with_w_progress_zero():
    """Overprogress (path > margin*GT) must trigger penalty regardless of w_progress."""
    # GT≈60m (t=0 filtered by gt_valid → effective ~59.24m), margin=1.1.
    # Traj 1 = 80m → overprogress ≈ relu(80/59.24 - 1.1) ≈ 0.250 → penalty ≈ 7.5.
    # Traj 0 = 60m → overprogress = relu(60/59.24 - 1.1) = 0 → no penalty.
    # Main assertion: gap is meaningfully negative and in the expected ballpark.
    trajs, data = _make_scene([60.0, 80.0], gt_len=60.0)
    cfg = _cfg_cl_only(underprogress_penalty=0.0)  # isolate overprogress
    totals = _totals(cfg, trajs, data)

    gap = totals[1] - totals[0]
    # Accept ~7.0 (using nominal GT=60) through ~7.6 (using effective GT=59.24).
    assert -8.5 < gap < -6.0, (
        f"overprogress did not fire with w_progress=0. "
        f"totals={totals}, gap={gap}, expected -8.5 < gap < -6.0"
    )


def test_penalties_still_apply_with_nonzero_w_progress():
    """Sanity: the refactor must not break the w_progress>0 case."""
    trajs, data = _make_scene([60.0, 30.0], gt_len=60.0)
    cfg_zero = _cfg_cl_only(w_progress=0.0)
    cfg_one = _cfg_cl_only(w_progress=1.0)

    totals_zero = _totals(cfg_zero, trajs, data)
    totals_one = _totals(cfg_one, trajs, data)

    # w_progress=1 adds positive progress (progress_frac*20=20 for det, 0.5*20=10 for short)
    # so absolute totals shift up; but the relative gap from underprogress is identical.
    gap_zero = totals_zero[1] - totals_zero[0]
    gap_one = totals_one[1] - totals_one[0]

    # gap_one = gap_zero + (10 - 20) = gap_zero - 10 (short has less positive progress)
    assert gap_one == pytest.approx(gap_zero - 10.0, abs=0.5), (
        f"relative underprogress magnitude changed with w_progress. "
        f"gap_zero={gap_zero}, gap_one={gap_one}"
    )


def test_underprogress_uses_baseline_ref_when_configured():
    """underprogress_reference='baseline' reads data['baseline_path_len']."""
    # det=30m (short!), test=30m, but baseline_path_len=60m frozen anchor.
    # With det reference: ratio=30/30=1.0 → no penalty. Both trajs score identical.
    # With baseline reference: ratio=30/60=0.5 → penalty fires on BOTH (including det).
    trajs, data = _make_scene([30.0, 30.0], gt_len=60.0)
    data["baseline_path_len"] = torch.tensor(60.0)

    cfg_det = _cfg_cl_only(underprogress_reference="det")
    cfg_base = _cfg_cl_only(underprogress_reference="baseline")

    totals_det = _totals(cfg_det, trajs, data)
    totals_base = _totals(cfg_base, trajs, data)

    # det reference: neither traj penalized (det is traj[0], ratio=1.0)
    # baseline reference: both penalized by 50*(0.85-0.5) = 17.5
    assert totals_det[0] == pytest.approx(totals_det[1], abs=0.1), (
        f"det ref: trajs should be equal, got {totals_det}"
    )
    # baseline shift: totals_base should be ~17.5 lower than totals_det for BOTH trajs
    shift_0 = totals_det[0] - totals_base[0]
    shift_1 = totals_det[1] - totals_base[1]
    assert shift_0 == pytest.approx(17.5, abs=0.5), f"traj[0] shift: {shift_0}"
    assert shift_1 == pytest.approx(17.5, abs=0.5), f"traj[1] shift: {shift_1}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
