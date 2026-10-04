"""Coverage for the live-vs-recorded speed/accel diff metric (``gt_speed`` block)."""

from __future__ import annotations

import numpy as np

from scenario_generation.closed_loop_eval import _pool_gt_speed
from scenario_generation.reproducer_rollout import (
    DT,
    _gt_accels,
    _gt_deviation_m,
    _gt_speed_accel,
    gt_speed_block,
)


class _FakeTL:
    """Straight road along +x; recorded ego speeds given per frame."""

    def __init__(self, speeds: np.ndarray):
        n = len(speeds)
        x = np.concatenate([[0.0], np.cumsum(speeds[:-1] * DT)])
        self.poses = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
        self.speeds = np.asarray(speeds, dtype=np.float64)
        self.frame_indices = np.arange(n)


def test_gt_speed_accel_reads_nearest_segment():
    tl = _FakeTL(np.full(60, 5.0))
    dev = _gt_deviation_m(tl, tl.poses[30, :2] + np.array([0.01, 0.5]), 0.0, 30)
    v, a = _gt_speed_accel(tl, dev)
    assert abs(v - 5.0) < 1e-6
    assert abs(a) < 1e-6


def test_gt_speed_accel_none_when_yaw_gated():
    tl = _FakeTL(np.full(60, 5.0))
    dev = _gt_deviation_m(tl, tl.poses[30, :2], np.pi, 30)  # facing the wrong way
    assert _gt_speed_accel(tl, dev) is None


def test_gt_accels_recovers_constant_accel():
    tl = _FakeTL(np.arange(60) * 0.2)  # 2 m/s^2
    assert np.allclose(_gt_accels(tl)[10:-10], 2.0, atol=0.15)


def test_block_splits_signs_and_drops_nan():
    dv = np.array([-2.0, 1.0, np.nan, 0.0], dtype=np.float32)
    da = np.array([1.0, -1.5, np.nan, 0.0], dtype=np.float32)
    b = gt_speed_block(dv, da)
    assert b["n"] == 3
    assert b["slow_sum"] == 2.0 and b["fast_sum"] == 1.0
    assert b["brake_sum"] == 1.5 and b["accel_sum"] == 1.0  # da<0 -> braking harder than GT
    assert b["slow_max"] == 2.0


def test_pool_p99_and_missing_block():
    # 99 zero-diff steps + 1 step 3 m/s too slow -> p99 sits at 0, max at 3.
    b = gt_speed_block(np.array([0.0] * 99 + [-3.0]), np.zeros(100))
    pooled = _pool_gt_speed([{"gt_speed": b}])
    assert pooled["n_steps"] == 100
    assert pooled["slow"]["p99"] == 0.0 and pooled["slow"]["max"] == 3.0
    # 2 of 100 slow -> p99 lands in the 3 m/s bin (upper edge).
    b2 = gt_speed_block(np.array([0.0] * 98 + [-3.0] * 2), np.zeros(100))
    assert abs(_pool_gt_speed([{"gt_speed": b2}])["slow"]["p99"] - 3.05) < 1e-6
    assert _pool_gt_speed([{"n_steps_run": 1}]) is None


def test_quad_split_separates_weak_accel_from_hard_brake():
    # step0: GT +1, live +0.5 (weak accel); step1: GT 0, live -3 (hard brake);
    # step2: GT -1, live -2 (brakes harder); step3: GT -2, live +1 (live accelerates).
    ga = np.array([1.0, 0.0, -1.0, -2.0])
    live = np.array([0.5, -3.0, -2.0, 1.0])
    b = gt_speed_block(np.zeros(4), live - ga, ga)
    q = b["quad"]
    assert q["gtacc_liveacc"]["n"] == 1 and abs(q["gtacc_liveacc"]["sum"] - 0.5) < 1e-9
    assert q["gtacc_livebrk"]["n"] == 1 and abs(q["gtacc_livebrk"]["max"] - 3.0) < 1e-9
    assert q["gtbrk_livebrk"]["n"] == 1 and q["gtbrk_liveacc"]["n"] == 1
    pooled = _pool_gt_speed([{"gt_speed": b}])["accel_quad"]
    assert pooled["gtacc_livebrk"]["steps"] == 1
    assert abs(pooled["gtbrk_liveacc"]["mean"] - 3.0) < 1e-9
    assert "quad" not in gt_speed_block(np.zeros(2), np.zeros(2))
