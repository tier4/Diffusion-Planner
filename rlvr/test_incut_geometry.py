"""Sign-convention tests for the in-cut geometry helpers."""

import numpy as np

from rlvr.autoresearch.tools.incut_geometry import (
    incut_from_signed_lat,
    signed_curvature_from_poses,
)

R = 20.0
_TH = np.linspace(0.0, np.pi / 2, 50)
LEFT_ARC = np.stack([R * np.sin(_TH), R * (1 - np.cos(_TH)), _TH], axis=1)
RIGHT_ARC = np.stack([R * np.sin(_TH), -R * (1 - np.cos(_TH)), -_TH], axis=1)


def test_left_turn_has_positive_curvature_of_one_over_radius():
    kappa = signed_curvature_from_poses(LEFT_ARC)
    assert kappa > 0.0
    assert abs(kappa - 1.0 / R) < 0.02


def test_right_turn_has_negative_curvature():
    assert signed_curvature_from_poses(RIGHT_ARC) < 0.0


def test_straight_window_has_zero_curvature():
    straight = np.stack([np.linspace(0, 50, 50), np.zeros(50), np.zeros(50)], axis=1)
    assert abs(signed_curvature_from_poses(straight)) < 1e-6


def test_degenerate_windows_return_zero_rather_than_raising():
    assert signed_curvature_from_poses(np.zeros((5, 3))) == 0.0
    assert signed_curvature_from_poses(np.zeros((1, 3))) == 0.0


def test_offset_toward_the_bend_reads_as_incut_on_both_turn_directions():
    k_left = signed_curvature_from_poses(LEFT_ARC)
    k_right = signed_curvature_from_poses(RIGHT_ARC)
    # 0.4 m LEFT of the centreline on a LEFT bend == cutting the corner
    assert incut_from_signed_lat(np.array([0.4]), k_left)[0] > 0.0
    # the same offset on a RIGHT bend == swinging wide
    assert incut_from_signed_lat(np.array([0.4]), k_right)[0] < 0.0
    # 0.4 m RIGHT on a RIGHT bend == cutting
    assert incut_from_signed_lat(np.array([-0.4]), k_right)[0] > 0.0


def test_straight_window_has_no_inside():
    assert np.isnan(incut_from_signed_lat(np.array([0.4]), 0.0)[0])
