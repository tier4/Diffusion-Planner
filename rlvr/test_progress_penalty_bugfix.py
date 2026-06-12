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
        # reward_mode: stick to default "gate" branch (collision_gate + red_light_gate
        # both pass with neutral data → totals = quality_score, same as survival w/o
        # failures). Avoids relying on any unsupported mode value.
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
        f"underprogress did not fire with w_progress=0. totals={totals}, gap={gap}, expected ~-17.5"
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
    # so absolute totals shift up; but the relative gap from underprogress is identical
    # (both at the floor penalty_mult = max(w_progress, 1.0) = 1.0).
    gap_zero = totals_zero[1] - totals_zero[0]
    gap_one = totals_one[1] - totals_one[0]

    # gap_one = gap_zero + (10 - 20) = gap_zero - 10 (short has less positive progress)
    assert gap_one == pytest.approx(gap_zero - 10.0, abs=0.5), (
        f"relative underprogress magnitude changed with w_progress. "
        f"gap_zero={gap_zero}, gap_one={gap_one}"
    )


def test_penalty_magnitude_scales_with_w_progress_above_one():
    """With w_progress=2 the underprogress penalty should fire at 2x magnitude
    (matching legacy behavior where penalties lived inside the w_progress sum).
    With w_progress=7 it should fire at 7x. This guards the penalty_mult floor
    that preserves backward compat for configs with w_progress in {2, 7}."""
    trajs, data = _make_scene([60.0, 30.0], gt_len=60.0)
    cfg_base = _cfg_cl_only(w_progress=1.0)
    cfg_2 = _cfg_cl_only(w_progress=2.0)
    cfg_7 = _cfg_cl_only(w_progress=7.0)

    totals_1 = _totals(cfg_base, trajs, data)
    totals_2 = _totals(cfg_2, trajs, data)
    totals_7 = _totals(cfg_7, trajs, data)

    # Underprogress: ratio = 30/60 = 0.5, thresh 0.85 → underprogress = 0.35.
    # penalty_at_w1 = 50 * 0.35 = 17.5 on traj[1].
    # penalty_at_w2 = 2 * 17.5 = 35.0 on traj[1] → extra 17.5 swing vs w=1.
    # Positive progress contribution also scales: det gets 1*20→2*20 (+20),
    # short gets 1*10→2*10 (+10), so gap-from-progress changes by -10.
    # Net gap change (w=2 vs w=1): -17.5 (penalty) + -10 (progress) = -27.5.
    gap_1 = totals_1[1] - totals_1[0]
    gap_2 = totals_2[1] - totals_2[0]
    gap_7 = totals_7[1] - totals_7[0]

    # At w=2: expect gap shift of −27.5 vs w=1.
    assert (gap_2 - gap_1) == pytest.approx(-27.5, abs=1.0), (
        f"penalty did not scale to 2x at w_progress=2. "
        f"gap_1={gap_1:.2f} gap_2={gap_2:.2f} delta={gap_2 - gap_1:.2f} (expected -27.5)"
    )
    # At w=7: penalty 7*17.5=122.5 (extra 105 vs w=1), progress 7*(10)=70 (extra 60 vs w=1).
    # Net gap shift: -105 + -60 = -165.
    assert (gap_7 - gap_1) == pytest.approx(-165.0, abs=2.0), (
        f"penalty did not scale to 7x at w_progress=7. "
        f"gap_1={gap_1:.2f} gap_7={gap_7:.2f} delta={gap_7 - gap_1:.2f} (expected -165)"
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


def _make_scene_no_gt(path_lens, T=80, dt=0.1):
    """Build N ego trajectories with ZERO GT (simulating sim-dump NPZs
    where ego_agent_future is zero)."""
    N = len(path_lens)
    trajs = torch.zeros(N, T, 4)
    for i, L in enumerate(path_lens):
        v = L / ((T - 1) * dt)
        xs = torch.arange(T, dtype=torch.float32) * v * dt
        trajs[i, :, 0] = xs
        trajs[i, :, 2] = 1.0
    # Zero GT — matches synthetic NPZ dumps from scenario_generation.replay.
    gt_future = torch.zeros(T, 3)
    data = {
        "ego_agent_future": gt_future.unsqueeze(0),
        "ego_shape": torch.tensor([[2.79, 4.34, 1.70]]),
    }
    return trajs, data


def test_underprogress_fires_without_gt_when_baseline_ref_set():
    """Regression for the 2026-04-24 synthetic-data collapse: when the
    scene has no GT (common for scenario_generation.replay NPZs), an
    underprogress_reference='baseline' config must still anchor on
    ``data['baseline_path_len']`` and fire the penalty. Previously the
    whole progress block was gated on ``gt_valid.sum() >= 10``, so no-GT
    scenes silently dropped every path penalty."""
    trajs, data = _make_scene_no_gt([30.0, 30.0])  # short path
    data["baseline_path_len"] = torch.tensor(60.0)  # frozen anchor says we should do 60 m

    cfg = _cfg_cl_only(
        underprogress_reference="baseline",
        underprogress_penalty=50.0,
        underprogress_threshold=0.85,
        stopped_penalty=0.0,  # isolate underprogress
    )
    totals = _totals(cfg, trajs, data)

    # underprogress = relu(0.85 - 30/60) = 0.35 → penalty = 50*0.35 = 17.5 on both.
    # Relative comparison vs a matching-baseline case: same trajs with baseline=30
    # should fire no penalty. We verify the penalty exists by comparing.
    data_match = dict(data)
    data_match["baseline_path_len"] = torch.tensor(30.0)
    totals_match = _totals(cfg, trajs, data_match)

    shift = totals_match[0] - totals[0]
    assert shift == pytest.approx(17.5, abs=0.5), (
        f"underprogress did not fire without GT. "
        f"totals_short_baseline={totals_match}, totals_long_baseline={totals}, "
        f"shift={shift} (expected ~17.5)"
    )


def test_stopped_penalty_fires_without_gt_when_baseline_moves():
    """stopped_penalty must fire on a nearly-frozen ego when there is no
    GT but baseline_path_len says the scene is one where movement was
    possible. Previously the penalty was gated by ``if gt_path_len > 5.0``,
    so sim-dumped no-GT scenes never got a stopped-penalty signal and
    the model could reward-hack by collapsing path."""
    trajs, data = _make_scene_no_gt([60.0, 0.5])  # second is stopped
    data["baseline_path_len"] = torch.tensor(60.0)

    cfg = _cfg_cl_only(
        underprogress_penalty=0.0,  # isolate stopped
        underprogress_reference="baseline",
        stopped_penalty=100.0,
    )
    totals = _totals(cfg, trajs, data)
    gap = totals[1] - totals[0]
    assert gap == pytest.approx(-100.0, abs=1.0), (
        f"stopped_penalty did not fire on stopped ego without GT. "
        f"totals={totals}, gap={gap}, expected ~-100"
    )


def test_stopped_does_not_fire_without_gt_or_baseline_anchor():
    """With no GT AND no baseline_path_len in data, we can't tell a
    collapse from a legitimate red-light stop — stopped_penalty must
    NOT fire. Underprogress still fires via the det-fallback (ratio
    vs traj[0]), which is the long-standing existing behaviour."""
    trajs, data = _make_scene_no_gt([60.0, 0.5])
    # no baseline_path_len in data.
    cfg = _cfg_cl_only(
        stopped_penalty=100.0,
        underprogress_penalty=0.0,  # isolate stopped
        underprogress_reference="baseline",
    )
    totals = _totals(cfg, trajs, data)
    gap = totals[1] - totals[0]
    assert abs(gap) < 1.0, f"stopped leaked without an anchor. totals={totals}, gap={gap}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
