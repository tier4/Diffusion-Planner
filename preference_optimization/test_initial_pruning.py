"""
Unit tests for initial pose pruning utilities.

Tests calculate_initial_position_displacement, calculate_initial_yaw_difference,
and should_prune_by_initial_pose from preference_optimization/utils.py.

Run with:
    python3 preference_optimization/test_initial_pruning.py
or:
    pytest preference_optimization/test_initial_pruning.py
"""

import sys

import numpy as np


def _import_utils():
    """Import utils, adding repo root to path if needed."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from preference_optimization.utils import (
        calculate_initial_position_displacement,
        calculate_initial_yaw_difference,
        should_prune_by_initial_pose,
    )

    return (
        calculate_initial_position_displacement,
        calculate_initial_yaw_difference,
        should_prune_by_initial_pose,
    )


_calc_disp, _calc_yaw, _should_prune = _import_utils()


def _make_traj(x0: float, y0: float, yaw_deg: float, T: int = 80) -> np.ndarray:
    """Create a synthetic trajectory with given initial position and heading.

    Args:
        x0: Initial x position
        y0: Initial y position
        yaw_deg: Initial heading in degrees
        T: Trajectory length

    Returns:
        Trajectory array [T, 4] with (x, y, cos(yaw), sin(yaw))
    """
    yaw_rad = np.radians(yaw_deg)
    traj = np.zeros((T, 4))
    traj[:, 0] = x0 + np.arange(T) * 0.1  # simple forward motion along x
    traj[:, 1] = y0
    traj[:, 2] = np.cos(yaw_rad)
    traj[:, 3] = np.sin(yaw_rad)
    return traj


# --- calculate_initial_position_displacement ---


def test_displacement_identical():
    """Displacement between a trajectory and itself is zero."""
    traj = _make_traj(1.0, 2.0, 0.0)
    result = _calc_disp(traj, traj)
    assert abs(result) < 1e-9, f"Expected 0.0, got {result}"
    print(f"  PASS  displacement_identical: {result:.6f}m")


def test_displacement_known_3_4_5():
    """3-4-5 right triangle: displacement should be exactly 5.0m."""
    t1 = _make_traj(0.0, 0.0, 0.0)
    t2 = _make_traj(3.0, 4.0, 0.0)
    result = _calc_disp(t1, t2)
    assert abs(result - 5.0) < 1e-6, f"Expected 5.0, got {result}"
    print(f"  PASS  displacement_known_3_4_5: {result:.6f}m")


def test_displacement_small():
    """Small displacement within typical pruning range."""
    t1 = _make_traj(0.0, 0.0, 0.0)
    t2 = _make_traj(0.03, 0.04, 0.0)  # 0.05m
    result = _calc_disp(t1, t2)
    assert abs(result - 0.05) < 1e-6, f"Expected 0.05, got {result}"
    print(f"  PASS  displacement_small: {result:.6f}m")


def test_displacement_symmetric():
    """Displacement is symmetric: d(A,B) == d(B,A)."""
    t1 = _make_traj(1.0, 2.0, 30.0)
    t2 = _make_traj(3.0, 5.0, 45.0)
    assert abs(_calc_disp(t1, t2) - _calc_disp(t2, t1)) < 1e-9
    print(f"  PASS  displacement_symmetric")


# --- calculate_initial_yaw_difference ---


def test_yaw_identical():
    """Same heading gives zero yaw difference."""
    traj = _make_traj(0.0, 0.0, 45.0)
    result = _calc_yaw(traj, traj)
    assert abs(result) < 1e-9, f"Expected 0.0°, got {result}°"
    print(f"  PASS  yaw_identical: {result:.6f}°")


def test_yaw_known_positive():
    """Known yaw difference of 0.5 degrees."""
    t1 = _make_traj(0.0, 0.0, 0.0)
    t2 = _make_traj(0.0, 0.0, 0.5)
    result = _calc_yaw(t1, t2)
    assert abs(result - 0.5) < 1e-4, f"Expected 0.5°, got {result}°"
    print(f"  PASS  yaw_known_positive: {result:.6f}°")


def test_yaw_known_negative_symmetry():
    """Yaw difference is absolute: -0.5° and +0.5° give the same result."""
    t1 = _make_traj(0.0, 0.0, 0.0)
    t_pos = _make_traj(0.0, 0.0, 0.5)
    t_neg = _make_traj(0.0, 0.0, -0.5)
    result_pos = _calc_yaw(t1, t_pos)
    result_neg = _calc_yaw(t1, t_neg)
    assert abs(result_pos - result_neg) < 1e-9, (
        f"Expected symmetry, got {result_pos}° vs {result_neg}°"
    )
    print(f"  PASS  yaw_known_negative_symmetry: {result_pos:.6f}°")


def test_yaw_wrap_near_180():
    """Trajectories at ±179° differ by ~2°, not 358° (correct wrap to [-π, π])."""
    t1 = _make_traj(0.0, 0.0, 179.0)
    t2 = _make_traj(0.0, 0.0, -179.0)
    result = _calc_yaw(t1, t2)
    assert abs(result - 2.0) < 1e-3, f"Expected ~2.0°, got {result}°"
    print(f"  PASS  yaw_wrap_near_180: {result:.6f}°")


def test_yaw_90_degrees():
    """90° yaw difference is unambiguous."""
    t1 = _make_traj(0.0, 0.0, 0.0)
    t2 = _make_traj(0.0, 0.0, 90.0)
    result = _calc_yaw(t1, t2)
    assert abs(result - 90.0) < 1e-4, f"Expected 90.0°, got {result}°"
    print(f"  PASS  yaw_90_degrees: {result:.6f}°")


def test_yaw_180_degrees():
    """180° is the maximum possible absolute yaw difference."""
    t1 = _make_traj(0.0, 0.0, 0.0)
    t2 = _make_traj(0.0, 0.0, 180.0)
    result = _calc_yaw(t1, t2)
    assert abs(result - 180.0) < 1e-3, f"Expected 180.0°, got {result}°"
    print(f"  PASS  yaw_180_degrees: {result:.6f}°")


# --- should_prune_by_initial_pose ---


def test_prune_both_under_threshold():
    """Both metrics under threshold: should NOT prune."""
    t1 = _make_traj(0.0, 0.0, 0.0)
    t2 = _make_traj(0.02, 0.0, 0.2)  # 0.02m disp, 0.2° yaw
    should_prune, disp, yaw = _should_prune(t1, t2, pos_threshold_m=0.055, yaw_threshold_deg=0.55)
    assert not should_prune, f"Should not prune, disp={disp:.4f}, yaw={yaw:.4f}"
    assert abs(disp - 0.02) < 1e-6
    print(f"  PASS  prune_both_under: prune={should_prune}, disp={disp:.4f}m, yaw={yaw:.4f}°")


def test_prune_position_over_only():
    """Position exceeds threshold, yaw is fine: should prune (OR logic)."""
    t1 = _make_traj(0.0, 0.0, 0.0)
    t2 = _make_traj(0.08, 0.0, 0.2)  # 0.08m > 0.055 threshold, yaw fine
    should_prune, disp, yaw = _should_prune(t1, t2, pos_threshold_m=0.055, yaw_threshold_deg=0.55)
    assert should_prune, f"Should prune due to position, disp={disp:.4f}"
    print(
        f"  PASS  prune_position_over_only: prune={should_prune}, disp={disp:.4f}m, yaw={yaw:.4f}°"
    )


def test_prune_yaw_over_only():
    """Yaw exceeds threshold, position is fine: should prune (OR logic)."""
    t1 = _make_traj(0.0, 0.0, 0.0)
    t2 = _make_traj(0.01, 0.0, 0.8)  # 0.01m disp fine, 0.8° > 0.55 threshold
    should_prune, disp, yaw = _should_prune(t1, t2, pos_threshold_m=0.055, yaw_threshold_deg=0.55)
    assert should_prune, f"Should prune due to yaw, yaw={yaw:.4f}"
    print(f"  PASS  prune_yaw_over_only: prune={should_prune}, disp={disp:.4f}m, yaw={yaw:.4f}°")


def test_prune_both_over_threshold():
    """Both metrics exceed threshold: should prune."""
    t1 = _make_traj(0.0, 0.0, 0.0)
    t2 = _make_traj(0.09, 0.0, 0.9)  # both over
    should_prune, disp, yaw = _should_prune(t1, t2, pos_threshold_m=0.055, yaw_threshold_deg=0.55)
    assert should_prune, f"Should prune when both over, disp={disp:.4f}, yaw={yaw:.4f}"
    print(f"  PASS  prune_both_over: prune={should_prune}, disp={disp:.4f}m, yaw={yaw:.4f}°")


def test_prune_exactly_at_threshold():
    """Values exactly at threshold should NOT prune (strict >)."""
    t1 = _make_traj(0.0, 0.0, 0.0)
    t2 = _make_traj(0.055, 0.0, 0.55)  # exactly at threshold
    should_prune, disp, yaw = _should_prune(t1, t2, pos_threshold_m=0.055, yaw_threshold_deg=0.55)
    assert not should_prune, f"Should not prune at exact threshold, disp={disp:.4f}, yaw={yaw:.4f}"
    print(
        f"  PASS  prune_exactly_at_threshold: prune={should_prune}, disp={disp:.4f}m, yaw={yaw:.4f}°"
    )


def test_prune_returns_correct_metrics():
    """Return values match independently computed metrics."""
    t1 = _make_traj(0.0, 0.0, 0.0)
    t2 = _make_traj(0.03, 0.04, 0.3)

    from preference_optimization.utils import (
        calculate_initial_position_displacement,
        calculate_initial_yaw_difference,
    )

    expected_disp = calculate_initial_position_displacement(t1, t2)
    expected_yaw = calculate_initial_yaw_difference(t1, t2)

    _, disp, yaw = _should_prune(t1, t2, pos_threshold_m=0.055, yaw_threshold_deg=0.55)
    assert abs(disp - expected_disp) < 1e-9, f"Displacement mismatch: {disp} vs {expected_disp}"
    assert abs(yaw - expected_yaw) < 1e-9, f"Yaw mismatch: {yaw} vs {expected_yaw}"
    print(f"  PASS  prune_returns_correct_metrics: disp={disp:.4f}m, yaw={yaw:.4f}°")


if __name__ == "__main__":
    tests = [
        # displacement
        test_displacement_identical,
        test_displacement_known_3_4_5,
        test_displacement_small,
        test_displacement_symmetric,
        # yaw
        test_yaw_identical,
        test_yaw_known_positive,
        test_yaw_known_negative_symmetry,
        test_yaw_wrap_near_180,
        test_yaw_90_degrees,
        test_yaw_180_degrees,
        # should_prune (OR logic)
        test_prune_both_under_threshold,
        test_prune_position_over_only,
        test_prune_yaw_over_only,
        test_prune_both_over_threshold,
        test_prune_exactly_at_threshold,
        test_prune_returns_correct_metrics,
    ]

    print("=" * 60)
    print("Initial Pose Pruning Test Suite")
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
        print(f"ALL {len(tests)} TESTS PASSED! ✓")
    else:
        print(f"{failed}/{len(tests)} TESTS FAILED ✗")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
