# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for diffusion_planner/utils/data_augmentation.py.

Covers:
- vector_transform: identity, 90-deg rotation, bias, batch norm-preservation
- heading_transform: identity, 90-deg, 180-deg
- StatePerturbation.normalize_angle: in-range, wrapping, numpy input
- StatePerturbation.get_transform_matrix_batch: identity heading, 90-deg heading
- StatePerturbation.augment: prob=0, prob=1 fast/slow vehicles, velocity >= 0,
  output shape, cos/sin unit-norm
- StatePerturbation.interpolation_future_trajectory: output shapes,
  end-point proximity
- StatePerturbation.centric_transform: identity ego (positions unchanged),
  zero-mask preserved, translation, ego xy zeroed
- _cross2d: basic values, parallel vectors, batched
- _rect_corners: heading=0 corner positions, heading=90° rotation
- _sat_signed_distance: overlapping (negative), separated (positive), touching (~0)
- _segments_intersect_rect: crossing, endpoint inside, outside, fully inside,
  valid mask filtering, batch
- StatePerturbation._check_aug_validity: no keys, neighbor overlap/clear,
  left/right boundary cross/clear, zero offset ignored, batch mixed
- StatePerturbation.augment (integration): collision suppresses aug_flag,
  no-collision preserves aug_flag

Usage:
    python tests/test_data_augmentation.py          # standalone
    pytest tests/test_data_augmentation.py -v       # with pytest
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion_planner.utils.data_augmentation import (
    StatePerturbation,
    _cross2d,
    _rect_corners,
    _sat_signed_distance,
    _segments_intersect_rect,
    heading_transform,
    vector_transform,
)

# Standard ego vehicle shape used across tests: (wheelbase, length, width) in metres.
_EGO_SHAPE_DEFAULT = torch.tensor([[2.75, 5.0, 2.0]])

ATOL = 1e-5


# ─────────────────────────────── helpers ────────────────────────────────────


def _rot(B: int, angle: float) -> torch.Tensor:
    """Batch of 2D CCW rotation matrices, shape (B, 2, 2)."""
    c, s = math.cos(angle), math.sin(angle)
    mat = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    return mat.unsqueeze(0).expand(B, -1, -1).clone()


def _ego_state(
    B: int, x: float = 0.0, y: float = 0.0, heading: float = 0.0, vx: float = 5.0
) -> torch.Tensor:
    """Minimal ego_current_state tensor of shape (B, 10)."""
    state = torch.zeros(B, 10, dtype=torch.float32)
    state[:, 0] = x
    state[:, 1] = y
    state[:, 2] = math.cos(heading)
    state[:, 3] = math.sin(heading)
    state[:, 4] = vx
    return state


def _make_inputs(B: int = 1, N_nbr: int = 3, T_past: int = 5, T_fut: int = 80):
    """Minimal inputs dict + ego_future + neighbors_future for centric_transform."""
    ego_current_state = _ego_state(B, vx=5.0)

    # Past trajectory approaching origin from behind
    ego_agent_past = torch.zeros(B, T_past, 3, dtype=torch.float32)
    for t in range(T_past):
        ego_agent_past[:, t, 0] = (t - T_past) * 0.1

    neighbor_agents_past = torch.zeros(B, N_nbr, T_past, 11, dtype=torch.float32)
    lanes = torch.zeros(B, 2, 5, 8, dtype=torch.float32)
    route_lanes = torch.zeros(B, 2, 5, 8, dtype=torch.float32)
    polygons = torch.zeros(B, 2, 4, 2, dtype=torch.float32)
    line_strings = torch.zeros(B, 2, 5, 2, dtype=torch.float32)
    static_objects = torch.zeros(B, 2, 10, dtype=torch.float32)

    inputs = {
        "ego_current_state": ego_current_state,
        "ego_agent_past": ego_agent_past,
        "neighbor_agents_past": neighbor_agents_past,
        "lanes": lanes,
        "route_lanes": route_lanes,
        "polygons": polygons,
        "line_strings": line_strings,
        "static_objects": static_objects,
    }

    # Future trajectory: straight ahead at ~0.5 m per step
    ego_future = torch.zeros(B, T_fut, 3, dtype=torch.float32)
    for t in range(T_fut):
        ego_future[:, t, 0] = (t + 1) * 0.5

    neighbors_future = torch.zeros(B, N_nbr, T_fut, 3, dtype=torch.float32)
    return inputs, ego_future, neighbors_future


# ──────────────────────────── vector_transform ──────────────────────────────


def test_vector_transform_identity():
    B = 2
    v = torch.randn(B, 5, 2)
    I = torch.eye(2).unsqueeze(0).expand(B, -1, -1).clone()
    out = vector_transform(v, I)
    assert torch.allclose(out, v, atol=ATOL), (
        f"Identity rotation changed vectors (max diff {(out - v).abs().max():.2e})"
    )
    print("  [PASS] vector_transform identity")


def test_vector_transform_rotation_90():
    """CCW 90-deg: (1, 0) -> (0, 1)."""
    v = torch.tensor([[[1.0, 0.0]]])  # (1, 1, 2)
    R = _rot(1, math.pi / 2)
    out = vector_transform(v, R)
    assert torch.allclose(out, torch.tensor([[[0.0, 1.0]]]), atol=1e-5), (
        f"90-deg rotation: expected (0,1), got {out}"
    )
    print("  [PASS] vector_transform 90-degree rotation")


def test_vector_transform_with_bias():
    """Bias is subtracted before rotation (identity rotation)."""
    v = torch.tensor([[[3.0, 0.0]]])  # (1, 1, 2)
    bias = torch.tensor([[1.0, 0.0]])  # (1, 2)
    I = torch.eye(2).unsqueeze(0)
    out = vector_transform(v, I, bias)
    assert torch.allclose(out, torch.tensor([[[2.0, 0.0]]]), atol=ATOL), (
        f"Bias subtraction failed: got {out}"
    )
    print("  [PASS] vector_transform with bias")


def test_vector_transform_norm_preserved():
    """Rotation preserves vector norms."""
    B = 3
    v = torch.randn(B, 4, 2)
    R = _rot(B, math.pi / 4)
    out = vector_transform(v, R)
    assert out.shape == v.shape
    assert torch.allclose(v.norm(dim=-1), out.norm(dim=-1), atol=1e-5), (
        "Rotation changed vector norms"
    )
    print("  [PASS] vector_transform norm preserved")


# ──────────────────────────── heading_transform ─────────────────────────────


def test_heading_transform_identity():
    B = 2
    h = torch.randn(B, 5)
    I = torch.eye(2).unsqueeze(0).expand(B, -1, -1).clone()
    out = heading_transform(h, I)
    assert out.shape == h.shape
    assert torch.allclose(out, h, atol=1e-5), (
        f"Identity heading transform changed values (max diff {(out - h).abs().max():.2e})"
    )
    print("  [PASS] heading_transform identity")


def test_heading_transform_rotation_90():
    """Heading 0 + 90-deg CCW rotation -> pi/2."""
    h = torch.tensor([[0.0]])
    R = _rot(1, math.pi / 2)
    out = heading_transform(h, R)
    assert abs(out.item() - math.pi / 2) < 1e-5, (
        f"90-deg heading: expected {math.pi / 2:.4f}, got {out.item():.4f}"
    )
    print("  [PASS] heading_transform 90-degree rotation")


def test_heading_transform_rotation_180():
    """Heading pi/4 + 180-deg rotation -> -3*pi/4 (wrapped)."""
    h = torch.tensor([[math.pi / 4]])
    R = _rot(1, math.pi)
    out = heading_transform(h, R)
    expected = math.pi / 4 - math.pi  # = -3*pi/4
    assert abs(out.item() - expected) < 1e-5, (
        f"180-deg heading: expected {expected:.4f}, got {out.item():.4f}"
    )
    print("  [PASS] heading_transform 180-degree rotation")


# ──────────────────── StatePerturbation init & helpers ──────────────────────


def test_state_perturbation_init():
    aug = StatePerturbation(augment_prob=0.7, wheel_base=3.0, device="cpu")
    assert aug._augment_prob == 0.7
    assert aug._wheel_base == 3.0
    assert aug.num_refine == 20
    assert aug.time_interval == 0.1
    assert aug.coeff_matrix.shape == (6, 6), f"coeff_matrix shape: {aug.coeff_matrix.shape}"
    assert aug.t_matrix.shape == (20, 6), f"t_matrix shape: {aug.t_matrix.shape}"
    print("  [PASS] StatePerturbation init")


def test_normalize_angle_in_range():
    aug = StatePerturbation()
    angles = torch.tensor([0.0, math.pi / 2, -math.pi / 2, math.pi * 0.999])
    out = aug.normalize_angle(angles)
    assert torch.allclose(out, angles, atol=1e-5), (
        f"normalize_angle changed in-range angles (max diff {(out - angles).abs().max():.2e})"
    )
    print("  [PASS] normalize_angle in-range unchanged")


def test_normalize_angle_wrapping():
    """2pi -> 0, -2pi -> 0, 3pi -> -pi."""
    aug = StatePerturbation()
    angles = torch.tensor([2 * math.pi, -2 * math.pi, 3 * math.pi])
    out = aug.normalize_angle(angles)
    expected = torch.tensor([0.0, 0.0, -math.pi])
    assert torch.allclose(out, expected, atol=1e-5), (
        f"normalize_angle wrapping failed: got {out.tolist()}, expected {expected.tolist()}"
    )
    print("  [PASS] normalize_angle wrapping")


def test_normalize_angle_numpy():
    aug = StatePerturbation()
    arr = np.array([0.0, 2 * np.pi, -2 * np.pi])
    out = aug.normalize_angle(arr)
    assert isinstance(out, np.ndarray), "Should return ndarray for ndarray input"
    assert np.allclose(out, np.array([0.0, 0.0, 0.0]), atol=1e-5), (
        f"numpy normalize_angle failed: got {out}"
    )
    print("  [PASS] normalize_angle numpy input")


def test_get_transform_matrix_batch_identity():
    """cos=1, sin=0 (heading=0) -> identity matrix."""
    aug = StatePerturbation()
    cur_state = torch.zeros(2, 10)
    cur_state[:, 2] = 1.0
    cur_state[:, 3] = 0.0
    mat = aug.get_transform_matrix_batch(cur_state)
    I = torch.eye(2).unsqueeze(0).expand(2, -1, -1)
    assert torch.allclose(mat, I, atol=1e-5), (
        f"Identity heading produced non-identity matrix:\n{mat}"
    )
    print("  [PASS] get_transform_matrix_batch identity")


def test_get_transform_matrix_batch_90deg():
    """cos=0, sin=1 (heading=pi/2) -> [[0, 1], [-1, 0]] (inverse rotation)."""
    aug = StatePerturbation()
    cur_state = torch.zeros(1, 10)
    cur_state[:, 2] = 0.0  # cos(pi/2)
    cur_state[:, 3] = 1.0  # sin(pi/2)
    mat = aug.get_transform_matrix_batch(cur_state)
    # [[cos, sin], [-sin, cos]] = [[0, 1], [-1, 0]]
    expected = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]])
    assert torch.allclose(mat, expected, atol=1e-5), f"90-deg heading gave wrong matrix:\n{mat}"
    print("  [PASS] get_transform_matrix_batch 90-degree")


# ────────────────────────────── augment ─────────────────────────────────────


def _augment_inputs(B: int, vx: float = 10.0) -> dict:
    """Minimal inputs for augment() tests: ego state + ego_shape, no neighbours or lanes."""
    return {
        "ego_current_state": _ego_state(B, vx=vx),
        "ego_shape": _EGO_SHAPE_DEFAULT.expand(B, -1),
    }


def test_augment_prob_zero():
    """augment_prob=0: no samples augmented regardless of velocity."""
    torch.manual_seed(42)
    aug = StatePerturbation(augment_prob=0.0)
    aug_flag, _ = aug.augment(_augment_inputs(8))
    assert not aug_flag.any(), "augment_prob=0 should not augment any sample"
    print("  [PASS] augment prob=0 no augmentation")


def test_augment_prob_one_fast_vehicle():
    """augment_prob=1, |vx|>=2: all samples augmented and state changes."""
    torch.manual_seed(0)
    aug = StatePerturbation(augment_prob=1.0)
    B = 4
    inputs = _augment_inputs(B)
    original = inputs["ego_current_state"].clone()
    aug_flag, new_state = aug.augment(inputs)
    assert aug_flag.all(), "augment_prob=1 with fast vehicle should flag all samples"
    assert not torch.allclose(new_state[:, :4], original[:, :4], atol=1e-3), (
        "Augmented state should differ from original"
    )
    print("  [PASS] augment prob=1 fast vehicle")


def test_augment_slow_vehicle_not_augmented():
    """Slow vehicle (|vx| < 2) is never augmented even with prob=1."""
    torch.manual_seed(0)
    aug = StatePerturbation(augment_prob=1.0)
    aug_flag, _ = aug.augment(_augment_inputs(4, vx=0.5))
    assert not aug_flag.any(), "Slow vehicle (vx=0.5) should not be augmented"
    print("  [PASS] augment slow vehicle not augmented")


def test_augment_velocity_nonneg():
    """Augmented vx >= 0 (velocity is clamped at 0)."""
    torch.manual_seed(123)
    aug = StatePerturbation(augment_prob=1.0)
    _, new_state = aug.augment(_augment_inputs(32, vx=2.5))
    vx = new_state[:, 4]
    assert (vx >= -1e-6).all(), f"Augmented vx has negative values: min={vx.min():.4f}"
    print("  [PASS] augment velocity non-negative")


def test_augment_output_shape():
    aug = StatePerturbation()
    B = 3
    inputs = _augment_inputs(B, vx=5.0)
    aug_flag, new_state = aug.augment(inputs)
    assert aug_flag.shape == (B,), f"aug_flag shape mismatch: {aug_flag.shape}"
    assert new_state.shape == inputs["ego_current_state"].shape, (
        f"State shape changed: {new_state.shape}"
    )
    print("  [PASS] augment output shapes correct")


def test_augment_cos_sin_unit_norm():
    """After augmentation, cos and sin values must lie on the unit circle."""
    torch.manual_seed(42)
    aug = StatePerturbation(augment_prob=1.0)
    B = 8
    _, new_state = aug.augment(_augment_inputs(B, vx=5.0))
    norms = torch.hypot(new_state[:, 2], new_state[:, 3])
    assert torch.allclose(norms, torch.ones(B), atol=1e-5), (
        f"cos/sin not on unit circle after augment: norms={norms.tolist()}"
    )
    print("  [PASS] augment cos/sin unit norm")


# ──────────────────── interpolation_future_trajectory ───────────────────────


def test_interpolation_shape_keep_remaining():
    """keep_remaining=True preserves trailing waypoints: output shape == input shape."""
    aug = StatePerturbation()
    B, T = 2, 80
    aug_state = _ego_state(B, vx=5.0)
    ego_future = torch.zeros(B, T, 3)
    for t in range(T):
        ego_future[:, t, 0] = (t + 1) * 0.5
    out = aug.interpolation_future_trajectory(aug_state, ego_future, keep_remaining=True)
    assert out.shape == (B, T, 3), f"keep_remaining=True: expected ({B}, {T}, 3), got {out.shape}"
    print("  [PASS] interpolation shape keep_remaining=True")


def test_interpolation_shape_no_remaining():
    """keep_remaining=False: output length == num_refine (P=20)."""
    aug = StatePerturbation()
    B, T, P = 2, 80, aug.num_refine
    aug_state = _ego_state(B, vx=5.0)
    ego_future = torch.zeros(B, T, 3)
    for t in range(T):
        ego_future[:, t, 0] = (t + 1) * 0.5
    out = aug.interpolation_future_trajectory(aug_state, ego_future, keep_remaining=False)
    assert out.shape == (B, P, 3), f"keep_remaining=False: expected ({B}, {P}, 3), got {out.shape}"
    print("  [PASS] interpolation shape keep_remaining=False")


def test_interpolation_endpoint_proximity():
    """Interpolated trajectory endpoint is within one timestep of the P-th waypoint.

    The quintic polynomial is fitted to reach ego_future[:, P] at t = (P+1)*dt,
    so the last *sampled* point (at t = P*dt) differs by roughly one step of travel.
    With vx=5 m/s and dt=0.1 s the expected gap is ~0.5 m.
    """
    aug = StatePerturbation()
    B, T, P = 1, 80, aug.num_refine
    aug_state = _ego_state(B, vx=5.0)
    ego_future = torch.zeros(B, T, 3)
    for t in range(T):
        ego_future[:, t, 0] = (t + 1) * 0.5
    out = aug.interpolation_future_trajectory(aug_state, ego_future)
    last_interp = out[:, P - 1, :2]
    target = ego_future[:, P, :2]
    err = (last_interp - target).abs().max().item()
    # Allow up to one full timestep of travel at vx=5 m/s (= 0.5 m) plus margin
    assert err <= 0.5 + 1e-4, f"Interpolation end-point error too large: {err:.4f} m"
    print("  [PASS] interpolation end-point proximity")


# ─────────────────────────── centric_transform ──────────────────────────────


def test_centric_transform_identity_ego():
    """Ego at origin with zero heading: neighbor and lane positions unchanged."""
    aug = StatePerturbation()
    B = 1
    inputs, ego_future, nbrs_future = _make_inputs(B)

    # Put a visible neighbor at (1, 2) with cos=1, sin=0
    inputs["neighbor_agents_past"][:, 0, :, :6] = torch.tensor([[1.0, 2.0, 1.0, 0.0, 0.0, 0.0]])
    nbr_xy_before = inputs["neighbor_agents_past"][:, 0, :, :2].clone()

    # Put a lane segment at (3, 4)
    inputs["lanes"][:, 0, :, :8] = torch.tensor([[3.0, 4.0, 1.0, 0.0, 3.0, 4.0, 3.0, 4.0]])
    lane_xy_before = inputs["lanes"][:, 0, :, :2].clone()

    result_inputs, _, _ = aug.centric_transform(inputs, ego_future, nbrs_future)

    nbr_xy_after = result_inputs["neighbor_agents_past"][:, 0, :, :2]
    assert torch.allclose(nbr_xy_after, nbr_xy_before, atol=1e-4), (
        f"Neighbor xy changed under identity transform "
        f"(max diff {(nbr_xy_after - nbr_xy_before).abs().max():.2e})"
    )

    lane_xy_after = result_inputs["lanes"][:, 0, :, :2]
    assert torch.allclose(lane_xy_after, lane_xy_before, atol=1e-4), (
        f"Lane xy changed under identity transform "
        f"(max diff {(lane_xy_after - lane_xy_before).abs().max():.2e})"
    )

    print("  [PASS] centric_transform identity ego (positions preserved)")


def test_centric_transform_zero_mask_preserved():
    """All-zero neighbor entries remain zero after transform (mask respected)."""
    aug = StatePerturbation()
    inputs, ego_future, nbrs_future = _make_inputs(1)
    # neighbor_agents_past is all zeros by default
    result_inputs, _, _ = aug.centric_transform(inputs, ego_future, nbrs_future)
    assert torch.all(result_inputs["neighbor_agents_past"] == 0.0), (
        "Zero-masked neighbor entries were non-zero after centric_transform"
    )
    print("  [PASS] centric_transform zero mask preserved")


def test_centric_transform_translation():
    """Ego at (5, 3), neighbor at (6, 3) -> neighbor becomes (1, 0) after transform."""
    aug = StatePerturbation()
    inputs, ego_future, nbrs_future = _make_inputs(1)

    inputs["ego_current_state"][:, 0] = 5.0
    inputs["ego_current_state"][:, 1] = 3.0
    inputs["ego_current_state"][:, 2] = 1.0  # cos(0)
    inputs["ego_current_state"][:, 3] = 0.0  # sin(0)

    # Visible neighbor at (6, 3)
    inputs["neighbor_agents_past"][:, 0, :, :6] = torch.tensor([[6.0, 3.0, 1.0, 0.0, 0.0, 0.0]])

    result_inputs, _, _ = aug.centric_transform(inputs, ego_future, nbrs_future)

    nbr_xy = result_inputs["neighbor_agents_past"][:, 0, 0, :2]
    expected = torch.tensor([[1.0, 0.0]])
    assert torch.allclose(nbr_xy, expected, atol=1e-4), (
        f"Translation test: expected {expected.tolist()}, got {nbr_xy.tolist()}"
    )
    print("  [PASS] centric_transform translation")


def test_centric_transform_ego_xy_zeroed():
    """After centric_transform, ego xy should always be (0, 0)."""
    aug = StatePerturbation()
    inputs, ego_future, nbrs_future = _make_inputs(1)

    inputs["ego_current_state"][:, 0] = 10.0
    inputs["ego_current_state"][:, 1] = -5.0

    result_inputs, _, _ = aug.centric_transform(inputs, ego_future, nbrs_future)

    ego_xy = result_inputs["ego_current_state"][:, :2]
    assert torch.allclose(ego_xy, torch.zeros(1, 2), atol=1e-4), (
        f"Ego xy not zeroed after centric_transform: got {ego_xy.tolist()}"
    )
    print("  [PASS] centric_transform ego xy zeroed")


# ─────────────────────── collision-detection helpers ────────────────────────


def _nbr(
    B: int,
    x: float,
    y: float = 0.0,
    width: float = 2.0,
    length: float = 4.5,
    N: int = 5,
    T: int = 31,
) -> torch.Tensor:
    """Build a neighbor_agents_past tensor with one visible agent at (x, y)."""
    out = torch.zeros(B, N, T, 11, dtype=torch.float32)
    out[:, 0, -1, 0] = x  # position x
    out[:, 0, -1, 1] = y  # position y
    out[:, 0, -1, 2] = 1.0  # cos_h (heading = 0)
    out[:, 0, -1, 6] = width  # feature index 6 = width
    out[:, 0, -1, 7] = length  # feature index 7 = length
    return out


def _lanes(
    B: int, center_xs: list[float], left_y_off: float = 0.0, right_y_off: float = 0.0, L: int = 5
) -> torch.Tensor:
    """Build a lanes tensor with one populated lane segment along x."""
    P = len(center_xs)
    out = torch.zeros(B, L, P, 33, dtype=torch.float32)
    for i, cx in enumerate(center_xs):
        out[:, 0, i, 0] = cx  # center x
        # center y is 0 (default)
        out[:, 0, i, 5] = left_y_off  # left  boundary y-offset  (feature 5)
        out[:, 0, i, 7] = right_y_off  # right boundary y-offset  (feature 7)
    return out


def _check_inputs(
    B: int,
    nbr_tensor: torch.Tensor,
    lane_tensor: torch.Tensor,
    vx: float = 5.0,
    ego_shape: torch.Tensor | None = None,
) -> dict:
    """Minimal inputs dict for _check_aug_validity."""
    if ego_shape is None:
        ego_shape = _EGO_SHAPE_DEFAULT.expand(B, -1)
    return {
        "ego_current_state": _ego_state(B, vx=vx),
        "ego_shape": ego_shape,
        "neighbor_agents_past": nbr_tensor,
        "lanes": lane_tensor,
    }


# ──────────────────────────────── _cross2d ──────────────────────────────────


def test_cross2d_ccw_positive():
    """(1,0) × (0,1) = +1 (CCW orientation)."""
    u = torch.tensor([[1.0, 0.0]])
    v = torch.tensor([[0.0, 1.0]])
    assert abs(_cross2d(u, v).item() - 1.0) < ATOL, f"Expected 1.0, got {_cross2d(u, v).item()}"
    assert abs(_cross2d(v, u).item() + 1.0) < ATOL, f"Expected -1.0, got {_cross2d(v, u).item()}"
    print("  [PASS] _cross2d CCW positive / CW negative")


def test_cross2d_parallel_zero():
    """Parallel vectors have zero cross product."""
    u = torch.tensor([[2.0, 3.0]])
    v = torch.tensor([[4.0, 6.0]])  # v = 2*u
    assert abs(_cross2d(u, v).item()) < ATOL, (
        f"Parallel vectors: expected 0, got {_cross2d(u, v).item()}"
    )
    print("  [PASS] _cross2d parallel zero")


def test_cross2d_batched():
    """Batched input produces per-element cross products."""
    u = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    out = _cross2d(u, v)
    assert out.shape == (2,)
    assert abs(out[0].item() - 1.0) < ATOL  # (1,0)×(0,1) = +1
    assert abs(out[1].item() + 1.0) < ATOL  # (0,1)×(1,0) = -1
    print("  [PASS] _cross2d batched")


# ────────────────────────────── _rect_corners ───────────────────────────────


def test_rect_corners_shape():
    """Output shape is [B, 4, 2]."""
    rect = torch.zeros(3, 6)
    rect[:, 2] = 1.0  # cos_h = 1
    rect[:, 4] = 4.0  # length
    rect[:, 5] = 2.0  # width
    assert _rect_corners(rect).shape == (3, 4, 2)
    print("  [PASS] _rect_corners output shape")


def test_rect_corners_heading_zero():
    """Heading=0: corners at expected symmetric offsets from center."""
    # center=(1, 2), heading=0, length=4, width=2
    rect = torch.tensor([[1.0, 2.0, 1.0, 0.0, 4.0, 2.0]])
    corners = _rect_corners(rect)  # [1, 4, 2]
    # signs pattern: [+l/2,+w/2], [-l/2,+w/2], [-l/2,-w/2], [+l/2,-w/2]
    expected = torch.tensor([[[3.0, 3.0], [-1.0, 3.0], [-1.0, 1.0], [3.0, 1.0]]])
    assert torch.allclose(corners, expected, atol=ATOL), (
        f"Heading=0 corners wrong:\n{corners}\n≠\n{expected}"
    )
    print("  [PASS] _rect_corners heading=0")


def test_rect_corners_heading_90():
    """Heading=90° (cos=0, sin=1): corners are rotated 90° CCW."""
    # center=(0,0), cos=0, sin=1, length=4, width=2
    rect = torch.tensor([[0.0, 0.0, 0.0, 1.0, 4.0, 2.0]])
    corners = _rect_corners(rect)  # [1, 4, 2]
    # Local [+2,+1] rotated 90° CCW → [-1, +2], etc.
    expected = torch.tensor([[[-1.0, 2.0], [-1.0, -2.0], [1.0, -2.0], [1.0, 2.0]]])
    assert torch.allclose(corners, expected, atol=ATOL), (
        f"Heading=90° corners wrong:\n{corners}\n≠\n{expected}"
    )
    print("  [PASS] _rect_corners heading=90°")


def test_rect_corners_center_preserved():
    """Mean of the 4 corners equals the rectangle center."""
    rect = torch.tensor([[3.0, -2.0, 1.0, 0.0, 6.0, 3.0]])
    corners = _rect_corners(rect)  # [1, 4, 2]
    center = corners.mean(dim=1)  # [1, 2]
    expected = rect[:, :2]
    assert torch.allclose(center, expected, atol=ATOL), (
        f"Corner mean {center.tolist()} ≠ center {expected.tolist()}"
    )
    print("  [PASS] _rect_corners center preserved")


# ─────────────────────────── _sat_signed_distance ───────────────────────────


def test_sat_signed_distance_overlap_negative():
    """Identical (fully overlapping) rectangles → negative signed distance."""
    rect = torch.tensor([[0.0, 0.0, 1.0, 0.0, 4.0, 2.0]])
    c = _rect_corners(rect)
    dist = _sat_signed_distance(c, c)
    assert dist.item() < 0, (
        f"Overlapping rects must have negative SAT distance, got {dist.item():.4f}"
    )
    print("  [PASS] _sat_signed_distance overlap (negative)")


def test_sat_signed_distance_separated_positive():
    """Well-separated rectangles → positive signed distance matching the gap."""
    r1 = torch.tensor([[0.0, 0.0, 1.0, 0.0, 2.0, 2.0]])  # spans x∈[-1,1]
    r2 = torch.tensor([[10.0, 0.0, 1.0, 0.0, 2.0, 2.0]])  # spans x∈[9,11]
    dist = _sat_signed_distance(_rect_corners(r1), _rect_corners(r2))
    # Separation along x: 9 - 1 = 8 m
    assert abs(dist.item() - 8.0) < 0.1, f"Expected gap ~8.0 m, got {dist.item():.4f}"
    print("  [PASS] _sat_signed_distance separated (positive, correct gap)")


def test_sat_signed_distance_touching_zero():
    """Rectangles that just touch → signed distance ≈ 0."""
    r1 = torch.tensor([[0.0, 0.0, 1.0, 0.0, 2.0, 2.0]])
    r2 = torch.tensor([[2.0, 0.0, 1.0, 0.0, 2.0, 2.0]])  # r2 starts where r1 ends
    dist = _sat_signed_distance(_rect_corners(r1), _rect_corners(r2))
    assert abs(dist.item()) < 0.05, f"Touching rects: expected ≈0, got {dist.item():.4f}"
    print("  [PASS] _sat_signed_distance touching (~0)")


def test_sat_signed_distance_batch():
    """Batch B=2: first pair overlaps, second pair is separated."""
    r1 = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 4.0, 2.0],  # b=0
            [0.0, 0.0, 1.0, 0.0, 4.0, 2.0],  # b=1
        ]
    )
    r2 = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 4.0, 2.0],  # b=0: identical → overlap
            [20.0, 0.0, 1.0, 0.0, 4.0, 2.0],  # b=1: far away → separated
        ]
    )
    dist = _sat_signed_distance(_rect_corners(r1), _rect_corners(r2))
    assert dist.shape == (2,)
    assert dist[0].item() < 0, f"b=0 overlap: expected negative, got {dist[0].item()}"
    assert dist[1].item() > 0, f"b=1 separated: expected positive, got {dist[1].item()}"
    print("  [PASS] _sat_signed_distance batch")


# ─────────────────────────── _segments_intersect_rect ───────────────────────
# Reference rect for these tests:
# center=(0,0), heading=0, l=4, w=2 → x∈[-2,2], y∈[-1,1]

_UNIT_RECT = None  # lazy-initialised once below


def _get_unit_rect(B: int = 1) -> torch.Tensor:
    spec = torch.tensor([[0.0, 0.0, 1.0, 0.0, 4.0, 2.0]]).expand(B, -1)
    return _rect_corners(spec)


def test_segments_intersect_rect_crossing():
    """Segment crossing both sides of the rectangle → True."""
    rect = _get_unit_rect()
    s = torch.tensor([[[-5.0, 0.0]]])  # [1, 1, 2]
    e = torch.tensor([[[5.0, 0.0]]])
    assert _segments_intersect_rect(s, e, rect).item(), "Through-crossing segment should intersect"
    print("  [PASS] _segments_intersect_rect crossing")


def test_segments_intersect_rect_endpoint_inside():
    """Segment with one endpoint inside the rectangle → True."""
    rect = _get_unit_rect()
    s = torch.tensor([[[0.0, 0.0]]])  # inside rect (x∈[-2,2], y∈[-1,1])
    e = torch.tensor([[[10.0, 0.0]]])  # outside
    assert _segments_intersect_rect(s, e, rect).item(), (
        "Segment with inside endpoint should intersect"
    )
    print("  [PASS] _segments_intersect_rect endpoint inside")


def test_segments_intersect_rect_fully_inside():
    """Segment fully inside the rectangle → True (both endpoints inside)."""
    rect = _get_unit_rect()
    s = torch.tensor([[[-0.5, 0.0]]])
    e = torch.tensor([[[0.5, 0.0]]])
    assert _segments_intersect_rect(s, e, rect).item(), "Fully-inside segment should be detected"
    print("  [PASS] _segments_intersect_rect fully inside")


def test_segments_intersect_rect_outside():
    """Segment entirely outside the rectangle → False."""
    rect = _get_unit_rect()
    s = torch.tensor([[[-5.0, 5.0]]])  # y=5 is well above rect (y∈[-1,1])
    e = torch.tensor([[[5.0, 5.0]]])
    assert not _segments_intersect_rect(s, e, rect).item(), "Outside segment should not intersect"
    print("  [PASS] _segments_intersect_rect outside")


def test_segments_intersect_rect_parallel_outside():
    """Segment parallel to a rect edge but just outside → False."""
    rect = _get_unit_rect()
    # Rect top edge at y=1; segment at y=1.5 (above)
    s = torch.tensor([[[-5.0, 1.5]]])
    e = torch.tensor([[[5.0, 1.5]]])
    assert not _segments_intersect_rect(s, e, rect).item(), (
        "Parallel segment above rect should not intersect"
    )
    print("  [PASS] _segments_intersect_rect parallel outside")


def test_segments_intersect_rect_valid_mask_excludes_crossing():
    """valid=False for an otherwise-crossing segment → no intersection reported."""
    rect = _get_unit_rect()
    s = torch.tensor([[[-5.0, 0.0], [10.0, 5.0]]])  # seg0 crosses, seg1 doesn't
    e = torch.tensor([[[5.0, 0.0], [20.0, 5.0]]])
    # All invalid
    assert not _segments_intersect_rect(s, e, rect, torch.tensor([[False, False]])).item(), (
        "All-invalid mask: no intersection"
    )
    # Only the non-crossing segment is valid
    assert not _segments_intersect_rect(s, e, rect, torch.tensor([[False, True]])).item(), (
        "Only non-crossing segment valid: no intersection"
    )
    # Only the crossing segment is valid
    assert _segments_intersect_rect(s, e, rect, torch.tensor([[True, False]])).item(), (
        "Only crossing segment valid: intersection"
    )
    print("  [PASS] _segments_intersect_rect valid mask")


def test_segments_intersect_rect_batch():
    """Batch B=2: b=0 has crossing segment, b=1 has outside segment."""
    rect = _get_unit_rect(B=2)  # [2, 4, 2]
    # b=0: (-5,0)→(5,0) crosses; b=1: (-5,5)→(5,5) is above
    s = torch.tensor([[[-5.0, 0.0]], [[-5.0, 5.0]]])  # [2, 1, 2]
    e = torch.tensor([[[5.0, 0.0]], [[5.0, 5.0]]])
    result = _segments_intersect_rect(s, e, rect)
    assert result.shape == (2,)
    assert result[0].item() and not result[1].item(), (
        f"Expected [True, False], got {result.tolist()}"
    )
    print("  [PASS] _segments_intersect_rect batch")


def test_segments_intersect_rect_multiple_segments_any():
    """With N segments: True if ANY valid segment intersects."""
    rect = _get_unit_rect()
    # Two segments: one outside, one crossing; no mask
    s = torch.tensor([[[-5.0, 5.0], [-5.0, 0.0]]])  # [1, 2, 2]
    e = torch.tensor([[[5.0, 5.0], [5.0, 0.0]]])
    assert _segments_intersect_rect(s, e, rect).item(), (
        "Should return True when any segment crosses"
    )
    print("  [PASS] _segments_intersect_rect multiple (any)")


# ────────────────────────── _check_aug_validity ─────────────────────────────


def test_check_aug_validity_no_collision_sources():
    """ego_shape present but no neighbor or lane data: always valid (returns all False)."""
    aug = StatePerturbation()
    B = 3
    ego = _ego_state(B, vx=5.0)
    inputs = {
        "ego_current_state": ego,
        "ego_shape": _EGO_SHAPE_DEFAULT.expand(B, -1),
    }
    collision = aug._check_aug_validity(ego, inputs)
    assert collision.shape == (B,)
    assert not collision.any(), "No neighbor/lane keys → no collision"
    print("  [PASS] _check_aug_validity no collision sources")


def test_check_aug_validity_neighbor_at_ego_position():
    """Neighbor at exactly the ego position: overlap → collision."""
    aug = StatePerturbation()
    B = 1
    ego = _ego_state(B, vx=5.0)  # ego at (0, 0)
    inputs = _check_inputs(B, _nbr(B, x=0.0, y=0.0), _lanes(B, []))
    collision = aug._check_aug_validity(ego, inputs)
    assert collision.item(), "Neighbor at ego center must trigger collision"
    print("  [PASS] _check_aug_validity neighbor at ego position")


def test_check_aug_validity_neighbor_far():
    """Neighbor 50 m away: no overlap → no collision."""
    aug = StatePerturbation()
    B = 1
    ego = _ego_state(B, vx=5.0)
    inputs = _check_inputs(B, _nbr(B, x=50.0), _lanes(B, []))
    collision = aug._check_aug_validity(ego, inputs)
    assert not collision.item(), "Distant neighbor should not trigger collision"
    print("  [PASS] _check_aug_validity neighbor far")


def test_check_aug_validity_all_zero_neighbors_ignored():
    """All-zero neighbor tensor (padding): treated as absent → no collision."""
    aug = StatePerturbation()
    B = 1
    ego = _ego_state(B, vx=5.0)
    empty_nbr = torch.zeros(B, 5, 31, 11)
    inputs = _check_inputs(B, empty_nbr, _lanes(B, []))
    collision = aug._check_aug_validity(ego, inputs)
    assert not collision.item(), "All-zero neighbors must be ignored"
    print("  [PASS] _check_aug_validity all-zero neighbors ignored")


def test_check_aug_validity_lane_left_boundary_cross():
    """Ego shifted left until it straddles the left lane boundary → collision."""
    aug = StatePerturbation()
    B = 1
    # Ego at y=0.8, width=2 → spans y∈[-0.2, 1.8].
    # Left boundary at absolute y=1.0 (center y=0, left_off=+1.0) → inside ego.
    ego = _ego_state(B, y=0.8, vx=5.0)
    inputs = _check_inputs(
        B,
        torch.zeros(B, 5, 31, 11),
        _lanes(B, list(range(-10, 10)), left_y_off=1.0, right_y_off=-3.0),
    )
    collision = aug._check_aug_validity(ego, inputs)
    assert collision.item(), "Left boundary inside ego must trigger collision"
    print("  [PASS] _check_aug_validity left boundary cross")


def test_check_aug_validity_lane_right_boundary_cross():
    """Ego shifted left until it straddles the right lane boundary → collision."""
    aug = StatePerturbation()
    B = 1
    # Ego at y=0.8 → spans y∈[-0.2, 1.8].
    # Right boundary at absolute y=-0.1 (right_off=-0.1) → inside ego.
    ego = _ego_state(B, y=0.8, vx=5.0)
    inputs = _check_inputs(
        B,
        torch.zeros(B, 5, 31, 11),
        _lanes(B, list(range(-10, 10)), left_y_off=3.0, right_y_off=-0.1),
    )
    collision = aug._check_aug_validity(ego, inputs)
    assert collision.item(), "Right boundary inside ego must trigger collision"
    print("  [PASS] _check_aug_validity right boundary cross")


def test_check_aug_validity_lane_both_boundaries_clear():
    """Ego centered in lane with boundaries at ±2 m: no collision."""
    aug = StatePerturbation()
    B = 1
    # Ego at (0, 0), width=2 → spans y∈[-1, 1].
    # Left boundary at y=+2.0, right at y=-2.0: both well outside.
    ego = _ego_state(B, vx=5.0)
    inputs = _check_inputs(
        B,
        torch.zeros(B, 5, 31, 11),
        _lanes(B, list(range(-10, 10)), left_y_off=2.0, right_y_off=-2.0),
    )
    collision = aug._check_aug_validity(ego, inputs)
    assert not collision.item(), "Ego centered in lane should not collide"
    print("  [PASS] _check_aug_validity lane both boundaries clear")


def test_check_aug_validity_zero_boundary_offset_ignored():
    """Boundary offset ≤ 0.01 m is treated as 'no data' and ignored."""
    aug = StatePerturbation()
    B = 1
    # Ego at (0, 0). If zero right_off were honoured, abs right boundary = center = (cx, 0)
    # which would be inside ego.  It must be skipped.
    ego = _ego_state(B, vx=5.0)
    inputs = _check_inputs(
        B, torch.zeros(B, 5, 31, 11), _lanes(B, list(range(-3, 4)), left_y_off=2.0, right_y_off=0.0)
    )
    collision = aug._check_aug_validity(ego, inputs)
    assert not collision.item(), "Zero boundary offset must not trigger collision"
    print("  [PASS] _check_aug_validity zero boundary offset ignored")


def test_check_aug_validity_batch_mixed():
    """B=2: b=0 collides with neighbor, b=1 is clear."""
    aug = StatePerturbation()
    B = 2
    ego = _ego_state(B, vx=5.0)  # both at (0, 0)
    nbr = torch.zeros(B, 5, 31, 11)
    # b=0: neighbor at (0, 0) → collision
    nbr[0, 0, -1, 0] = 0.0
    nbr[0, 0, -1, 2] = 1.0
    nbr[0, 0, -1, 6] = 2.0
    nbr[0, 0, -1, 7] = 4.5
    # b=1: neighbor at (50, 0) → no collision
    nbr[1, 0, -1, 0] = 50.0
    nbr[1, 0, -1, 2] = 1.0
    nbr[1, 0, -1, 6] = 2.0
    nbr[1, 0, -1, 7] = 4.5
    inputs = {
        "ego_current_state": ego,
        "ego_shape": _EGO_SHAPE_DEFAULT.expand(B, -1),
        "neighbor_agents_past": nbr,
        "lanes": torch.zeros(B, 5, 10, 33),
    }
    collision = aug._check_aug_validity(ego, inputs)
    assert collision[0].item() and not collision[1].item(), (
        f"Expected [True, False], got {collision.tolist()}"
    )
    print("  [PASS] _check_aug_validity batch mixed")


def test_check_aug_validity_ego_shape_controls_size():
    """Different ego_shape values change which neighbours are detected as collisions."""
    aug = StatePerturbation()
    B = 1
    ego = _ego_state(B, vx=5.0)  # ego at (0, 0), heading=0

    # Neighbour at y=3.5 m: outside a 5×2 m ego (half-width=1 m) but inside a 10×8 m ego.
    nbr = _nbr(B, x=0.0, y=3.5, width=1.0, length=1.0)
    lanes = torch.zeros(B, 5, 10, 33)

    # Small ego (5×2 m): neighbour at y=3.5 is OUTSIDE → no collision
    small_shape = torch.tensor([[2.75, 5.0, 2.0]])
    inputs_small = {
        "ego_current_state": ego,
        "ego_shape": small_shape,
        "neighbor_agents_past": nbr,
        "lanes": lanes,
    }
    assert not aug._check_aug_validity(ego, inputs_small).item(), (
        "5×2 ego: neighbour at y=3.5 must not collide"
    )

    # Large ego (10×8 m): neighbour at y=3.5 is INSIDE → collision
    large_shape = torch.tensor([[2.75, 10.0, 8.0]])
    inputs_large = {
        "ego_current_state": ego,
        "ego_shape": large_shape,
        "neighbor_agents_past": nbr,
        "lanes": lanes,
    }
    assert aug._check_aug_validity(ego, inputs_large).item(), (
        "10×8 ego: neighbour at y=3.5 must collide"
    )
    print("  [PASS] _check_aug_validity ego_shape controls collision size")


# ────────────── augment + collision filter (integration) ────────────────────


def test_augment_collision_suppresses_aug_flag():
    """Neighbor at ego origin: any perturbation still overlaps → aug_flag forced False."""
    torch.manual_seed(0)
    aug = StatePerturbation(augment_prob=1.0)
    B = 1
    # Ego at (0, 0), |vx|=10 (above the speed threshold).
    # Neighbor is also at (0, 0) with size 4.5×2.0 m.
    # After perturbation (max ±0.75 m lateral) the two boxes always overlap:
    #   ego (5×2 m) and neighbour (4.5×2 m) share x and y extent even at max offset.
    inputs = {
        "ego_current_state": _ego_state(B, vx=10.0),
        "ego_shape": _EGO_SHAPE_DEFAULT.expand(B, -1),
        "neighbor_agents_past": _nbr(B, x=0.0, y=0.0),
        "lanes": torch.zeros(B, 5, 10, 33),
    }
    aug_flag, _ = aug.augment(inputs)
    assert not aug_flag.item(), "Collision with overlapping neighbour must suppress aug_flag"
    print("  [PASS] augment collision suppresses aug_flag")


def test_augment_no_collision_preserves_aug_flag():
    """Fast vehicle with no collision sources: aug_flag all True (prob=1)."""
    torch.manual_seed(0)
    aug = StatePerturbation(augment_prob=1.0)
    B = 4
    inputs = {
        "ego_current_state": _ego_state(B, vx=10.0),
        "ego_shape": _EGO_SHAPE_DEFAULT.expand(B, -1),
        "neighbor_agents_past": torch.zeros(B, 5, 31, 11),  # no neighbours
        "lanes": torch.zeros(B, 5, 10, 33),  # no lane boundaries
    }
    aug_flag, _ = aug.augment(inputs)
    assert aug_flag.all(), "Fast vehicle with no collision sources must keep all aug_flag True"
    print("  [PASS] augment no collision preserves aug_flag")


def test_augment_collision_batch_selectively_suppresses():
    """B=2: b=0 collides (flag→False), b=1 is clear (flag stays True)."""
    torch.manual_seed(0)
    aug = StatePerturbation(augment_prob=1.0)
    B = 2
    nbr = torch.zeros(B, 5, 31, 11)
    # b=0: neighbour at (0, 0) → always collides after any perturbation
    nbr[0, 0, -1, 0] = 0.0
    nbr[0, 0, -1, 2] = 1.0
    nbr[0, 0, -1, 6] = 2.0
    nbr[0, 0, -1, 7] = 4.5
    # b=1: neighbour 50 m ahead → no collision
    nbr[1, 0, -1, 0] = 50.0
    nbr[1, 0, -1, 2] = 1.0
    nbr[1, 0, -1, 6] = 2.0
    nbr[1, 0, -1, 7] = 4.5
    inputs = {
        "ego_current_state": _ego_state(B, vx=10.0),
        "ego_shape": _EGO_SHAPE_DEFAULT.expand(B, -1),
        "neighbor_agents_past": nbr,
        "lanes": torch.zeros(B, 5, 10, 33),
    }
    aug_flag, _ = aug.augment(inputs)
    assert not aug_flag[0].item(), "b=0 (collision) must have aug_flag=False"
    assert aug_flag[1].item(), "b=1 (no collision) must have aug_flag=True"
    print("  [PASS] augment batch selectively suppresses aug_flag")


# ──────────────────────────────── runner ────────────────────────────────────


ALL_TESTS = [
    # ── vector_transform ──
    test_vector_transform_identity,
    test_vector_transform_rotation_90,
    test_vector_transform_with_bias,
    test_vector_transform_norm_preserved,
    # ── heading_transform ──
    test_heading_transform_identity,
    test_heading_transform_rotation_90,
    test_heading_transform_rotation_180,
    # ── StatePerturbation helpers ──
    test_state_perturbation_init,
    test_normalize_angle_in_range,
    test_normalize_angle_wrapping,
    test_normalize_angle_numpy,
    test_get_transform_matrix_batch_identity,
    test_get_transform_matrix_batch_90deg,
    # ── augment ──
    test_augment_prob_zero,
    test_augment_prob_one_fast_vehicle,
    test_augment_slow_vehicle_not_augmented,
    test_augment_velocity_nonneg,
    test_augment_output_shape,
    test_augment_cos_sin_unit_norm,
    # ── interpolation ──
    test_interpolation_shape_keep_remaining,
    test_interpolation_shape_no_remaining,
    test_interpolation_endpoint_proximity,
    # ── centric_transform ──
    test_centric_transform_identity_ego,
    test_centric_transform_zero_mask_preserved,
    test_centric_transform_translation,
    test_centric_transform_ego_xy_zeroed,
    # ── _cross2d ──
    test_cross2d_ccw_positive,
    test_cross2d_parallel_zero,
    test_cross2d_batched,
    # ── _rect_corners ──
    test_rect_corners_shape,
    test_rect_corners_heading_zero,
    test_rect_corners_heading_90,
    test_rect_corners_center_preserved,
    # ── _sat_signed_distance ──
    test_sat_signed_distance_overlap_negative,
    test_sat_signed_distance_separated_positive,
    test_sat_signed_distance_touching_zero,
    test_sat_signed_distance_batch,
    # ── _segments_intersect_rect ──
    test_segments_intersect_rect_crossing,
    test_segments_intersect_rect_endpoint_inside,
    test_segments_intersect_rect_fully_inside,
    test_segments_intersect_rect_outside,
    test_segments_intersect_rect_parallel_outside,
    test_segments_intersect_rect_valid_mask_excludes_crossing,
    test_segments_intersect_rect_batch,
    test_segments_intersect_rect_multiple_segments_any,
    # ── _check_aug_validity ──
    test_check_aug_validity_no_collision_sources,
    test_check_aug_validity_neighbor_at_ego_position,
    test_check_aug_validity_neighbor_far,
    test_check_aug_validity_all_zero_neighbors_ignored,
    test_check_aug_validity_lane_left_boundary_cross,
    test_check_aug_validity_lane_right_boundary_cross,
    test_check_aug_validity_lane_both_boundaries_clear,
    test_check_aug_validity_zero_boundary_offset_ignored,
    test_check_aug_validity_batch_mixed,
    test_check_aug_validity_ego_shape_controls_size,
    # ── augment + collision filter (integration) ──
    test_augment_collision_suppresses_aug_flag,
    test_augment_no_collision_preserves_aug_flag,
    test_augment_collision_batch_selectively_suppresses,
]


if __name__ == "__main__":
    print(f"Running {len(ALL_TESTS)} tests for data_augmentation.py\n")
    passed, failed, errors = 0, 0, []

    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((fn.__name__, e))
            print(f"  [FAIL] {fn.__name__}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(ALL_TESTS)} passed, {failed} failed")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print("All tests passed!")
