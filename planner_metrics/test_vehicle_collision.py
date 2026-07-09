"""Parity tests for the single canonical vehicle-collision wrapper.

Proves ``planner_metrics.vehicle_collision`` is numerically identical to the
canonical torch primitives it wraps (so re-pointing every detection/labeling/
GUI/viz site at it changes no results) and to the legacy hand-rolled SAT it
replaces.
"""

import math

import numpy as np
import torch

from planner_metrics.collision_geometry import batch_signed_distance_rect, center_rect_to_points
from planner_metrics.geometry import _build_ego_bbox_corners
from planner_metrics.vehicle_collision import (
    obb_corners,
    obb_overlap,
    obb_signed_distance,
)


def _legacy_corners(x, y, h, length, width, wheelbase=None):
    cos_h, sin_h = math.cos(h), math.sin(h)
    if wheelbase is not None:
        ro = (length - wheelbase) / 2.0
        dxl, dxh = -ro, length - ro
    else:
        dxl, dxh = -length / 2.0, length / 2.0
    dyl, dyh = -width / 2.0, width / 2.0
    local = np.array([[dxl, dyl], [dxh, dyl], [dxh, dyh], [dxl, dyh]])
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    return (local @ rot.T) + np.array([x, y])


def _legacy_sat_collides(ca, cb):
    for corners in (ca, cb):
        for i in range(4):
            edge = corners[(i + 1) % 4] - corners[i]
            axis = np.array([-edge[1], edge[0]])
            n = np.linalg.norm(axis)
            if n < 1e-9:
                continue
            axis /= n
            mina, maxa = (ca @ axis).min(), (ca @ axis).max()
            minb, maxb = (cb @ axis).min(), (cb @ axis).max()
            if maxa < minb or maxb < mina:
                return False
    return True


def _corner_set(c):
    return sorted((round(float(px), 6), round(float(py), 6)) for px, py in c)


def test_obb_corners_match_legacy_ego_and_centroid():
    cases = [
        (3.0, -1.0, 0.6, 4.76, 2.29, 2.9),
        (0, 0, 0, 5, 3, None),
        (-2, 5, -1.2, 4.0, 1.8, None),
    ]
    for x, y, h, ln, w, wb in cases:
        got = obb_corners(x, y, h, ln, w, wb)
        assert _corner_set(got) == _corner_set(_legacy_corners(x, y, h, ln, w, wb))


def test_obb_corners_match_canonical_primitives():
    # centroid convention == center_rect_to_points
    x, y, h, ln, w = 1.5, -0.3, 0.9, 4.2, 1.9
    rect = torch.tensor([[x, y, math.cos(h), math.sin(h), ln, w]], dtype=torch.float64)
    canonical = center_rect_to_points(rect)[0].numpy()
    assert _corner_set(obb_corners(x, y, h, ln, w, None)) == _corner_set(canonical)

    # rear-axle convention == _build_ego_bbox_corners
    wb = 2.9
    ego = torch.tensor([[[x, y, math.cos(h), math.sin(h)]]], dtype=torch.float64)
    shape = torch.tensor([wb, ln, w], dtype=torch.float64)
    canonical_ego = _build_ego_bbox_corners(ego, shape)[0, 0].numpy()
    assert _corner_set(obb_corners(x, y, h, ln, w, wb)) == _corner_set(canonical_ego)


def test_obb_overlap_matches_legacy_sat_random():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        a = obb_corners(rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-3, 3), 4.76, 2.29, 2.9)
        b = obb_corners(rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-3, 3), 4.5, 2.0, None)
        assert obb_overlap(a, b) == _legacy_sat_collides(a, b)


def test_obb_signed_distance_matches_canonical():
    a = obb_corners(0, 0, 0, 4.0, 2.0, None)
    b = obb_corners(6.0, 0, 0, 4.0, 2.0, None)  # separated by 2m gap
    ta = torch.as_tensor(a, dtype=torch.float64).reshape(1, 4, 2)
    tb = torch.as_tensor(b, dtype=torch.float64).reshape(1, 4, 2)
    canonical = float(batch_signed_distance_rect(ta, tb)[0].item())
    assert obb_signed_distance(a, b) == canonical
    assert obb_signed_distance(a, b) > 0  # separated
    assert not obb_overlap(a, b)


def test_obb_overlap_true_on_penetration():
    a = obb_corners(0, 0, 0, 4.0, 2.0, None)
    b = obb_corners(1.0, 0, 0, 4.0, 2.0, None)  # heavily overlapping
    assert obb_overlap(a, b)
    assert obb_signed_distance(a, b) < 0
