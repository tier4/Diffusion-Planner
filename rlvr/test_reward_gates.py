"""Unit tests for signed-distance lane gate + kinematic gate + baseline underprogress.

Run: python -m pytest rlvr/test_reward_gates.py -x -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from rlvr.reward import (
    RewardConfig,
    _point_to_segments_signed_min_dist,
    _points_inside_intersection_areas,
    _classify_outer_boundaries,
    compute_kinematic_gate,
)


# ---------------------------------------------------------------------------
# _point_to_segments_signed_min_dist
# ---------------------------------------------------------------------------

def test_signed_dist_inside_returns_positive():
    # Single vertical segment at x=1, y∈[0,2], outward pointing +x (to the right).
    p1 = torch.tensor([[1.0, 0.0]])
    p2 = torch.tensor([[1.0, 2.0]])
    outward = torch.tensor([[1.0, 0.0]])
    # Query on the left of segment (x<1) → "inside" → signed negative (since outward is +x)
    # Query on the right (x>1) → "outside" → signed positive
    points = torch.tensor([[0.5, 1.0], [1.5, 1.0]])
    unsigned, signed = _point_to_segments_signed_min_dist(p1, p2, outward=outward, points=points) \
        if False else _point_to_segments_signed_min_dist(points, p1, p2, outward)
    assert torch.allclose(unsigned, torch.tensor([0.5, 0.5]), atol=1e-5)
    assert signed[0].item() < 0  # "inside" lane
    assert signed[1].item() > 0  # "outside" lane
    assert abs(abs(signed[0].item()) - 0.5) < 1e-5
    assert abs(abs(signed[1].item()) - 0.5) < 1e-5


def test_signed_dist_endpoint_projection_returns_floor():
    # Query past the endpoint → nearest projection is clamped → should return -100
    p1 = torch.tensor([[0.0, 0.0]])
    p2 = torch.tensor([[1.0, 0.0]])
    outward = torch.tensor([[0.0, 1.0]])
    # Query at (2.0, 0.5): nearest projection clamps to (1.0, 0.0). Should return -100.
    points = torch.tensor([[2.0, 0.5]])
    _, signed = _point_to_segments_signed_min_dist(points, p1, p2, outward)
    assert signed.item() == -100.0


def test_signed_dist_empty_segments():
    points = torch.tensor([[0.0, 0.0]])
    p1 = torch.zeros(0, 2)
    p2 = torch.zeros(0, 2)
    outward = torch.zeros(0, 2)
    unsigned, signed = _point_to_segments_signed_min_dist(points, p1, p2, outward)
    assert unsigned.item() == 100.0
    assert signed.item() == -100.0


# ---------------------------------------------------------------------------
# _points_inside_intersection_areas
# ---------------------------------------------------------------------------

def test_inside_square_polygon():
    # Unit square polygon (Np=1, P=4 vertices, K=0 extra channels beyond xy).
    verts = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    ])  # (1, 4, 2)
    points = torch.tensor([[0.5, 0.5], [1.5, 0.5], [0.5, -0.5]])
    inside = _points_inside_intersection_areas(points, verts)
    assert inside[0].item() is True or inside[0].item() == True
    assert not inside[1].item()
    assert not inside[2].item()


def test_intersection_ignores_degenerate_polygon():
    # Polygon with fewer than 3 valid points (one valid vertex) is ignored.
    verts = torch.zeros(1, 4, 2)
    verts[0, 0] = torch.tensor([0.5, 0.5])  # only one vertex has norm > 1e-3
    points = torch.tensor([[0.5, 0.5]])
    inside = _points_inside_intersection_areas(points, verts)
    assert not inside.item()


# ---------------------------------------------------------------------------
# _classify_outer_boundaries returns outward vector
# ---------------------------------------------------------------------------

def test_classify_returns_outward():
    # Build a 2-lane strip: lane 0 is y∈[0,1], lane 1 is y∈[1,2], each x∈[0,10].
    # Each lane has 4 boundary segments (left=x=0, right=x=10, bottom=y=low, top=y=high).
    # _classify_outer_boundaries expects even=left, odd=right segments per lane.
    # It needs seg_p1, seg_p2, seg_dir, seg_lane, plus polygon edge info.
    # Construct a single lane for simplicity.
    # Lane 0 polygon corners: (0,0), (10,0), (10,1), (0,1).
    # Segments (left/right of centerline): we fake with two segments.
    seg_p1 = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    seg_p2 = torch.tensor([[10.0, 0.0], [10.0, 1.0]])
    seg_dir = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    seg_lane = torch.tensor([0, 0], dtype=torch.int64)
    # Edge vertices — 4 polygon edges (the lane polygon), each defined as 2 endpoints.
    edge_v1 = torch.tensor([
        [0.0, 0.0], [10.0, 0.0], [10.0, 1.0], [0.0, 1.0]
    ])
    edge_v2 = torch.tensor([
        [10.0, 0.0], [10.0, 1.0], [0.0, 1.0], [0.0, 0.0]
    ])
    edge_poly_id = torch.zeros(4, dtype=torch.int64)
    n_polys = 1

    is_outer, outward = _classify_outer_boundaries(
        seg_p1, seg_p2, seg_dir, seg_lane,
        edge_v1, edge_v2, edge_poly_id, n_polys,
    )
    assert is_outer.shape == (2,)
    assert outward.shape == (2, 2)
    # Outward should be unit-normalized for any outer-classified segment.
    outer_mags = outward[is_outer].norm(dim=-1)
    if outer_mags.numel() > 0:
        assert torch.allclose(outer_mags, torch.ones_like(outer_mags), atol=1e-4)


# ---------------------------------------------------------------------------
# compute_kinematic_gate
# ---------------------------------------------------------------------------

def _straight_traj(speed_mps: float = 5.0, T: int = 80, dt: float = 0.1) -> torch.Tensor:
    """(1, T, 4) trajectory going straight along +x at constant speed, heading=0."""
    xs = torch.arange(T, dtype=torch.float32) * dt * speed_mps
    ys = torch.zeros(T)
    cos_h = torch.ones(T)
    sin_h = torch.zeros(T)
    return torch.stack([xs, ys, cos_h, sin_h], dim=-1).unsqueeze(0)


def _pivot_in_place_traj(T: int = 80, dt: float = 0.1) -> torch.Tensor:
    """(1, T, 4) ego spins in place at 3 rad/s — yaw rate violation."""
    xs = torch.zeros(T)
    ys = torch.zeros(T)
    theta = torch.arange(T, dtype=torch.float32) * dt * 3.0
    cos_h = torch.cos(theta)
    sin_h = torch.sin(theta)
    return torch.stack([xs, ys, cos_h, sin_h], dim=-1).unsqueeze(0)


def test_kinematic_gate_straight_passes():
    cfg = RewardConfig()
    ego_shape = torch.tensor([3.0, 5.0, 2.0])  # wheelbase, length, width
    gate = compute_kinematic_gate(_straight_traj(), cfg, ego_shape)
    assert gate.item() == 1.0


def test_kinematic_gate_pivot_fails():
    cfg = RewardConfig()
    ego_shape = torch.tensor([3.0, 5.0, 2.0])
    gate = compute_kinematic_gate(_pivot_in_place_traj(), cfg, ego_shape)
    assert gate.item() == 0.0


def test_kinematic_gate_no_ego_shape_skips_curvature():
    # With ego_shape=None the curvature check is skipped; only abs yaw rate applies.
    cfg = RewardConfig()
    # Gentle turn: 0.5 rad/s — below max_yaw_rate=1.0 → passes even without curvature check
    T = 80
    dt = 0.1
    theta = torch.arange(T, dtype=torch.float32) * dt * 0.5
    xs = torch.arange(T, dtype=torch.float32) * dt
    ys = torch.zeros(T)
    traj = torch.stack([xs, ys, torch.cos(theta), torch.sin(theta)], dim=-1).unsqueeze(0)
    gate = compute_kinematic_gate(traj, cfg, ego_shape=None)
    assert gate.item() == 1.0


def test_kinematic_gate_max_yaw_absolute_cap():
    # Yaw rate 2.0 rad/s > max_yaw_rate=1.0 → fail even with ego_shape=None
    cfg = RewardConfig(max_yaw_rate=1.0)
    T = 80
    dt = 0.1
    theta = torch.arange(T, dtype=torch.float32) * dt * 2.0
    xs = torch.arange(T, dtype=torch.float32) * dt
    ys = torch.zeros(T)
    traj = torch.stack([xs, ys, torch.cos(theta), torch.sin(theta)], dim=-1).unsqueeze(0)
    gate = compute_kinematic_gate(traj, cfg, ego_shape=None)
    assert gate.item() == 0.0


def test_kinematic_gate_short_traj_passes():
    # T<5 short-circuits to pass.
    cfg = RewardConfig()
    short = torch.zeros(1, 3, 4)
    short[:, :, 2] = 1.0  # cos_h=1
    gate = compute_kinematic_gate(short, cfg)
    assert gate.item() == 1.0


# ---------------------------------------------------------------------------
# Config defaults: verify removed fields don't resurrect
# ---------------------------------------------------------------------------

def test_reward_config_no_dead_fields():
    cfg = RewardConfig()
    # These used to exist but are now removed because nothing consumed them.
    assert not hasattr(cfg, "yaw_rate_scale")
    assert not hasattr(cfg, "kinematic_scale")


def test_reward_config_baseline_reference_default():
    cfg = RewardConfig()
    # Default reference is adaptive "det"; "baseline" is opt-in.
    assert cfg.underprogress_reference == "det"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-x", "-q"]))
