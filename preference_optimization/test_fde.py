"""
Quick test script for FDE calculation functionality.
Run this to verify the implementation works correctly.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import calculate_fde


def test_fde_identical_trajectories():
    """Test FDE with identical trajectories - should return 0."""
    traj = np.array([
        [0.0, 0.0, 0.0, 5.0],
        [1.0, 1.0, 0.1, 5.0],
        [2.0, 2.0, 0.2, 5.0],
        [3.0, 3.0, 0.3, 5.0],
    ])

    fde = calculate_fde(traj, traj)
    print(f"Test 1 - Identical trajectories: FDE = {fde:.4f} (expected: 0.0000)")
    assert abs(fde) < 1e-6, "FDE should be 0 for identical trajectories"
    print("✓ PASSED\n")


def test_fde_known_distance():
    """Test FDE with known final position distance."""
    traj_1 = np.array([
        [0.0, 0.0, 0.0, 5.0],
        [1.0, 0.0, 0.0, 5.0],
        [2.0, 0.0, 0.0, 5.0],
        [3.0, 0.0, 0.0, 5.0],  # Final position: (3, 0)
    ])

    traj_2 = np.array([
        [0.0, 0.0, 0.0, 5.0],
        [1.0, 1.0, 0.1, 5.0],
        [2.0, 2.0, 0.2, 5.0],
        [3.0, 4.0, 0.3, 5.0],  # Final position: (3, 4)
    ])

    # Expected FDE: sqrt((3-3)^2 + (0-4)^2) = 4.0
    fde = calculate_fde(traj_1, traj_2)
    expected = 4.0
    print(f"Test 2 - Known distance: FDE = {fde:.4f} (expected: {expected:.4f})")
    assert abs(fde - expected) < 1e-6, f"FDE should be {expected}"
    print("✓ PASSED\n")


def test_fde_diagonal_distance():
    """Test FDE with diagonal distance (3-4-5 triangle)."""
    traj_1 = np.array([
        [0.0, 0.0, 0.0, 5.0],
        [5.0, 5.0, 0.5, 5.0],  # Final position: (5, 5)
    ])

    traj_2 = np.array([
        [0.0, 0.0, 0.0, 5.0],
        [8.0, 9.0, 0.8, 5.0],  # Final position: (8, 9)
    ])

    # Expected FDE: sqrt((8-5)^2 + (9-5)^2) = sqrt(9 + 16) = 5.0
    fde = calculate_fde(traj_1, traj_2)
    expected = 5.0
    print(f"Test 3 - Diagonal distance: FDE = {fde:.4f} (expected: {expected:.4f})")
    assert abs(fde - expected) < 1e-6, f"FDE should be {expected}"
    print("✓ PASSED\n")


def test_fde_symmetry():
    """Test FDE symmetry - distance(A,B) should equal distance(B,A)."""
    traj_1 = np.random.randn(10, 4)
    traj_2 = np.random.randn(10, 4)

    fde_12 = calculate_fde(traj_1, traj_2)
    fde_21 = calculate_fde(traj_2, traj_1)

    print(f"Test 4 - Symmetry: FDE(1,2) = {fde_12:.4f}, FDE(2,1) = {fde_21:.4f}")
    assert abs(fde_12 - fde_21) < 1e-6, "FDE should be symmetric"
    print("✓ PASSED\n")


def test_fde_only_final_position_matters():
    """Test that only final position matters, not the path taken."""
    # Two different paths to the same endpoint
    traj_1 = np.array([
        [0.0, 0.0, 0.0, 5.0],
        [1.0, 0.0, 0.0, 5.0],  # Go along x-axis
        [2.0, 0.0, 0.0, 5.0],
        [3.0, 0.0, 0.0, 5.0],
        [4.0, 0.0, 0.0, 5.0],
        [5.0, 0.0, 0.0, 5.0],  # Then along y-axis
        [5.0, 1.0, 0.5, 5.0],
        [5.0, 2.0, 0.5, 5.0],
        [5.0, 3.0, 0.5, 5.0],  # Final: (5, 3)
    ])

    traj_2 = np.array([
        [0.0, 0.0, 0.0, 5.0],
        [0.0, 1.0, 0.5, 5.0],  # Go along y-axis first
        [0.0, 2.0, 0.5, 5.0],
        [0.0, 3.0, 0.5, 5.0],
        [1.0, 3.0, 0.0, 5.0],  # Then along x-axis
        [2.0, 3.0, 0.0, 5.0],
        [3.0, 3.0, 0.0, 5.0],
        [4.0, 3.0, 0.0, 5.0],
        [5.0, 3.0, 0.0, 5.0],  # Final: (5, 3)
    ])

    fde = calculate_fde(traj_1, traj_2)
    print(f"Test 5 - Same endpoint, different paths: FDE = {fde:.4f} (expected: 0.0000)")
    assert abs(fde) < 1e-6, "FDE should be 0 when endpoints are the same"
    print("✓ PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("FDE Calculation Test Suite")
    print("=" * 60 + "\n")

    try:
        test_fde_identical_trajectories()
        test_fde_known_distance()
        test_fde_diagonal_distance()
        test_fde_symmetry()
        test_fde_only_final_position_matters()

        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
