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
    heading_transform,
    vector_transform,
)

ATOL = 1e-5


# ─────────────────────────────── helpers ────────────────────────────────────


def _rot(B: int, angle: float) -> torch.Tensor:
    """Batch of 2D CCW rotation matrices, shape (B, 2, 2)."""
    c, s = math.cos(angle), math.sin(angle)
    mat = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    return mat.unsqueeze(0).expand(B, -1, -1).clone()


def _ego_state(B: int, x: float = 0.0, y: float = 0.0,
               heading: float = 0.0, vx: float = 5.0) -> torch.Tensor:
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
    assert torch.allclose(out, v, atol=ATOL), \
        f"Identity rotation changed vectors (max diff {(out-v).abs().max():.2e})"
    print("  [PASS] vector_transform identity")


def test_vector_transform_rotation_90():
    """CCW 90-deg: (1, 0) -> (0, 1)."""
    v = torch.tensor([[[1.0, 0.0]]])           # (1, 1, 2)
    R = _rot(1, math.pi / 2)
    out = vector_transform(v, R)
    assert torch.allclose(out, torch.tensor([[[0.0, 1.0]]]), atol=1e-5), \
        f"90-deg rotation: expected (0,1), got {out}"
    print("  [PASS] vector_transform 90-degree rotation")


def test_vector_transform_with_bias():
    """Bias is subtracted before rotation (identity rotation)."""
    v = torch.tensor([[[3.0, 0.0]]])           # (1, 1, 2)
    bias = torch.tensor([[1.0, 0.0]])           # (1, 2)
    I = torch.eye(2).unsqueeze(0)
    out = vector_transform(v, I, bias)
    assert torch.allclose(out, torch.tensor([[[2.0, 0.0]]]), atol=ATOL), \
        f"Bias subtraction failed: got {out}"
    print("  [PASS] vector_transform with bias")


def test_vector_transform_norm_preserved():
    """Rotation preserves vector norms."""
    B = 3
    v = torch.randn(B, 4, 2)
    R = _rot(B, math.pi / 4)
    out = vector_transform(v, R)
    assert out.shape == v.shape
    assert torch.allclose(v.norm(dim=-1), out.norm(dim=-1), atol=1e-5), \
        "Rotation changed vector norms"
    print("  [PASS] vector_transform norm preserved")


# ──────────────────────────── heading_transform ─────────────────────────────


def test_heading_transform_identity():
    B = 2
    h = torch.randn(B, 5)
    I = torch.eye(2).unsqueeze(0).expand(B, -1, -1).clone()
    out = heading_transform(h, I)
    assert out.shape == h.shape
    assert torch.allclose(out, h, atol=1e-5), \
        f"Identity heading transform changed values (max diff {(out-h).abs().max():.2e})"
    print("  [PASS] heading_transform identity")


def test_heading_transform_rotation_90():
    """Heading 0 + 90-deg CCW rotation -> pi/2."""
    h = torch.tensor([[0.0]])
    R = _rot(1, math.pi / 2)
    out = heading_transform(h, R)
    assert abs(out.item() - math.pi / 2) < 1e-5, \
        f"90-deg heading: expected {math.pi/2:.4f}, got {out.item():.4f}"
    print("  [PASS] heading_transform 90-degree rotation")


def test_heading_transform_rotation_180():
    """Heading pi/4 + 180-deg rotation -> -3*pi/4 (wrapped)."""
    h = torch.tensor([[math.pi / 4]])
    R = _rot(1, math.pi)
    out = heading_transform(h, R)
    expected = math.pi / 4 - math.pi  # = -3*pi/4
    assert abs(out.item() - expected) < 1e-5, \
        f"180-deg heading: expected {expected:.4f}, got {out.item():.4f}"
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
    assert torch.allclose(out, angles, atol=1e-5), \
        f"normalize_angle changed in-range angles (max diff {(out-angles).abs().max():.2e})"
    print("  [PASS] normalize_angle in-range unchanged")


def test_normalize_angle_wrapping():
    """2pi -> 0, -2pi -> 0, 3pi -> -pi."""
    aug = StatePerturbation()
    angles = torch.tensor([2 * math.pi, -2 * math.pi, 3 * math.pi])
    out = aug.normalize_angle(angles)
    expected = torch.tensor([0.0, 0.0, -math.pi])
    assert torch.allclose(out, expected, atol=1e-5), \
        f"normalize_angle wrapping failed: got {out.tolist()}, expected {expected.tolist()}"
    print("  [PASS] normalize_angle wrapping")


def test_normalize_angle_numpy():
    aug = StatePerturbation()
    arr = np.array([0.0, 2 * np.pi, -2 * np.pi])
    out = aug.normalize_angle(arr)
    assert isinstance(out, np.ndarray), "Should return ndarray for ndarray input"
    assert np.allclose(out, np.array([0.0, 0.0, 0.0]), atol=1e-5), \
        f"numpy normalize_angle failed: got {out}"
    print("  [PASS] normalize_angle numpy input")


def test_get_transform_matrix_batch_identity():
    """cos=1, sin=0 (heading=0) -> identity matrix."""
    aug = StatePerturbation()
    cur_state = torch.zeros(2, 10)
    cur_state[:, 2] = 1.0
    cur_state[:, 3] = 0.0
    mat = aug.get_transform_matrix_batch(cur_state)
    I = torch.eye(2).unsqueeze(0).expand(2, -1, -1)
    assert torch.allclose(mat, I, atol=1e-5), \
        f"Identity heading produced non-identity matrix:\n{mat}"
    print("  [PASS] get_transform_matrix_batch identity")


def test_get_transform_matrix_batch_90deg():
    """cos=0, sin=1 (heading=pi/2) -> [[0, 1], [-1, 0]] (inverse rotation)."""
    aug = StatePerturbation()
    cur_state = torch.zeros(1, 10)
    cur_state[:, 2] = 0.0   # cos(pi/2)
    cur_state[:, 3] = 1.0   # sin(pi/2)
    mat = aug.get_transform_matrix_batch(cur_state)
    # [[cos, sin], [-sin, cos]] = [[0, 1], [-1, 0]]
    expected = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]])
    assert torch.allclose(mat, expected, atol=1e-5), \
        f"90-deg heading gave wrong matrix:\n{mat}"
    print("  [PASS] get_transform_matrix_batch 90-degree")


# ────────────────────────────── augment ─────────────────────────────────────


def test_augment_prob_zero():
    """augment_prob=0: no samples augmented regardless of velocity."""
    torch.manual_seed(42)
    aug = StatePerturbation(augment_prob=0.0)
    inputs = {"ego_current_state": _ego_state(8, vx=10.0)}
    aug_flag, _ = aug.augment(inputs)
    assert not aug_flag.any(), "augment_prob=0 should not augment any sample"
    print("  [PASS] augment prob=0 no augmentation")


def test_augment_prob_one_fast_vehicle():
    """augment_prob=1, |vx|>=2: all samples augmented and state changes."""
    torch.manual_seed(0)
    aug = StatePerturbation(augment_prob=1.0)
    B = 4
    inputs = {"ego_current_state": _ego_state(B, vx=10.0)}
    original = inputs["ego_current_state"].clone()
    aug_flag, new_state = aug.augment(inputs)
    assert aug_flag.all(), "augment_prob=1 with fast vehicle should flag all samples"
    assert not torch.allclose(new_state[:, :4], original[:, :4], atol=1e-3), \
        "Augmented state should differ from original"
    print("  [PASS] augment prob=1 fast vehicle")


def test_augment_slow_vehicle_not_augmented():
    """Slow vehicle (|vx| < 2) is never augmented even with prob=1."""
    torch.manual_seed(0)
    aug = StatePerturbation(augment_prob=1.0)
    inputs = {"ego_current_state": _ego_state(4, vx=0.5)}
    aug_flag, _ = aug.augment(inputs)
    assert not aug_flag.any(), "Slow vehicle (vx=0.5) should not be augmented"
    print("  [PASS] augment slow vehicle not augmented")


def test_augment_velocity_nonneg():
    """Augmented vx >= 0 (velocity is clamped at 0)."""
    torch.manual_seed(123)
    aug = StatePerturbation(augment_prob=1.0)
    inputs = {"ego_current_state": _ego_state(32, vx=2.5)}
    _, new_state = aug.augment(inputs)
    vx = new_state[:, 4]
    assert (vx >= -1e-6).all(), f"Augmented vx has negative values: min={vx.min():.4f}"
    print("  [PASS] augment velocity non-negative")


def test_augment_output_shape():
    aug = StatePerturbation()
    B = 3
    inputs = {"ego_current_state": _ego_state(B, vx=5.0)}
    aug_flag, new_state = aug.augment(inputs)
    assert aug_flag.shape == (B,), f"aug_flag shape mismatch: {aug_flag.shape}"
    assert new_state.shape == inputs["ego_current_state"].shape, \
        f"State shape changed: {new_state.shape}"
    print("  [PASS] augment output shapes correct")


def test_augment_cos_sin_unit_norm():
    """After augmentation, cos and sin values must lie on the unit circle."""
    torch.manual_seed(42)
    aug = StatePerturbation(augment_prob=1.0)
    B = 8
    inputs = {"ego_current_state": _ego_state(B, vx=5.0)}
    _, new_state = aug.augment(inputs)
    norms = torch.hypot(new_state[:, 2], new_state[:, 3])
    assert torch.allclose(norms, torch.ones(B), atol=1e-5), \
        f"cos/sin not on unit circle after augment: norms={norms.tolist()}"
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
    assert out.shape == (B, T, 3), \
        f"keep_remaining=True: expected ({B}, {T}, 3), got {out.shape}"
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
    assert out.shape == (B, P, 3), \
        f"keep_remaining=False: expected ({B}, {P}, 3), got {out.shape}"
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
    inputs["neighbor_agents_past"][:, 0, :, :6] = torch.tensor(
        [[1.0, 2.0, 1.0, 0.0, 0.0, 0.0]]
    )
    nbr_xy_before = inputs["neighbor_agents_past"][:, 0, :, :2].clone()

    # Put a lane segment at (3, 4)
    inputs["lanes"][:, 0, :, :8] = torch.tensor([[3.0, 4.0, 1.0, 0.0, 3.0, 4.0, 3.0, 4.0]])
    lane_xy_before = inputs["lanes"][:, 0, :, :2].clone()

    result_inputs, _, _ = aug.centric_transform(inputs, ego_future, nbrs_future)

    nbr_xy_after = result_inputs["neighbor_agents_past"][:, 0, :, :2]
    assert torch.allclose(nbr_xy_after, nbr_xy_before, atol=1e-4), \
        f"Neighbor xy changed under identity transform " \
        f"(max diff {(nbr_xy_after - nbr_xy_before).abs().max():.2e})"

    lane_xy_after = result_inputs["lanes"][:, 0, :, :2]
    assert torch.allclose(lane_xy_after, lane_xy_before, atol=1e-4), \
        f"Lane xy changed under identity transform " \
        f"(max diff {(lane_xy_after - lane_xy_before).abs().max():.2e})"

    print("  [PASS] centric_transform identity ego (positions preserved)")


def test_centric_transform_zero_mask_preserved():
    """All-zero neighbor entries remain zero after transform (mask respected)."""
    aug = StatePerturbation()
    inputs, ego_future, nbrs_future = _make_inputs(1)
    # neighbor_agents_past is all zeros by default
    result_inputs, _, _ = aug.centric_transform(inputs, ego_future, nbrs_future)
    assert torch.all(result_inputs["neighbor_agents_past"] == 0.0), \
        "Zero-masked neighbor entries were non-zero after centric_transform"
    print("  [PASS] centric_transform zero mask preserved")


def test_centric_transform_translation():
    """Ego at (5, 3), neighbor at (6, 3) -> neighbor becomes (1, 0) after transform."""
    aug = StatePerturbation()
    inputs, ego_future, nbrs_future = _make_inputs(1)

    inputs["ego_current_state"][:, 0] = 5.0
    inputs["ego_current_state"][:, 1] = 3.0
    inputs["ego_current_state"][:, 2] = 1.0   # cos(0)
    inputs["ego_current_state"][:, 3] = 0.0   # sin(0)

    # Visible neighbor at (6, 3)
    inputs["neighbor_agents_past"][:, 0, :, :6] = torch.tensor(
        [[6.0, 3.0, 1.0, 0.0, 0.0, 0.0]]
    )

    result_inputs, _, _ = aug.centric_transform(inputs, ego_future, nbrs_future)

    nbr_xy = result_inputs["neighbor_agents_past"][:, 0, 0, :2]
    expected = torch.tensor([[1.0, 0.0]])
    assert torch.allclose(nbr_xy, expected, atol=1e-4), \
        f"Translation test: expected {expected.tolist()}, got {nbr_xy.tolist()}"
    print("  [PASS] centric_transform translation")


def test_centric_transform_ego_xy_zeroed():
    """After centric_transform, ego xy should always be (0, 0)."""
    aug = StatePerturbation()
    inputs, ego_future, nbrs_future = _make_inputs(1)

    inputs["ego_current_state"][:, 0] = 10.0
    inputs["ego_current_state"][:, 1] = -5.0

    result_inputs, _, _ = aug.centric_transform(inputs, ego_future, nbrs_future)

    ego_xy = result_inputs["ego_current_state"][:, :2]
    assert torch.allclose(ego_xy, torch.zeros(1, 2), atol=1e-4), \
        f"Ego xy not zeroed after centric_transform: got {ego_xy.tolist()}"
    print("  [PASS] centric_transform ego xy zeroed")


# ──────────────────────────────── runner ────────────────────────────────────


ALL_TESTS = [
    test_vector_transform_identity,
    test_vector_transform_rotation_90,
    test_vector_transform_with_bias,
    test_vector_transform_norm_preserved,
    test_heading_transform_identity,
    test_heading_transform_rotation_90,
    test_heading_transform_rotation_180,
    test_state_perturbation_init,
    test_normalize_angle_in_range,
    test_normalize_angle_wrapping,
    test_normalize_angle_numpy,
    test_get_transform_matrix_batch_identity,
    test_get_transform_matrix_batch_90deg,
    test_augment_prob_zero,
    test_augment_prob_one_fast_vehicle,
    test_augment_slow_vehicle_not_augmented,
    test_augment_velocity_nonneg,
    test_augment_output_shape,
    test_augment_cos_sin_unit_norm,
    test_interpolation_shape_keep_remaining,
    test_interpolation_shape_no_remaining,
    test_interpolation_endpoint_proximity,
    test_centric_transform_identity_ego,
    test_centric_transform_zero_mask_preserved,
    test_centric_transform_translation,
    test_centric_transform_ego_xy_zeroed,
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
