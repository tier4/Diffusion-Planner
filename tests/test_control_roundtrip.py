#!/usr/bin/env python3
"""Roundtrip tests for the trajectory <-> control pipeline.

Runnable with:
    pytest tests/test_control_roundtrip.py          # synthetic data only
    python3 tests/test_control_roundtrip.py <path_list.json>  # actual data
"""

import json
import math
import random
import sys

import numpy as np
import torch
from diffusion_planner.dimensions import INPUT_T, OUTPUT_T
from diffusion_planner.loss import control_to_waypoints, waypoints_to_control
from diffusion_planner.utils.coordinate_transform import (
    transform_to_ego_frame,
    transform_to_local_frame,
)
from diffusion_planner.utils.normalizer import ControlNormalizer

T_HIST = INPUT_T + 1  # 31
T_FUTURE = OUTPUT_T  # 80
DT = 0.1


# ---------------------------------------------------------------------------
# Synthetic trajectory generators
# ---------------------------------------------------------------------------


def make_straight_line(
    v: float = 5.0,
    heading: float = 0.0,
    T: int = T_FUTURE,
    dt: float = DT,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Generate a straight-line trajectory.

    Returns:
        past:  [1, T_HIST, 4]  (x, y, cos, sin)
        future: [1, T, 4]
        v0: initial speed (m/s)
    """
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    total_steps = T_HIST + T

    xs, ys = [], []
    for i in range(total_steps):
        t = (i - T_HIST + 1) * dt  # t=0 at last history step
        xs.append(v * t * cos_h)
        ys.append(v * t * sin_h)

    traj = torch.zeros(1, total_steps, 4)
    traj[0, :, 0] = torch.tensor(xs)
    traj[0, :, 1] = torch.tensor(ys)
    traj[0, :, 2] = cos_h
    traj[0, :, 3] = sin_h

    past = traj[:, :T_HIST]
    future = traj[:, T_HIST:]
    return past, future, v


def make_circular_arc(
    v: float = 5.0,
    radius: float = 50.0,
    T: int = T_FUTURE,
    dt: float = DT,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Generate a circular-arc trajectory (constant speed, constant curvature).

    Returns:
        past:  [1, T_HIST, 4]
        future: [1, T, 4]
        v0: initial speed
    """
    omega = v / radius  # angular velocity
    total_steps = T_HIST + T

    traj = torch.zeros(1, total_steps, 4)
    for i in range(total_steps):
        t = (i - T_HIST + 1) * dt
        theta = omega * t
        traj[0, i, 0] = radius * math.sin(theta)
        traj[0, i, 1] = radius * (1.0 - math.cos(theta))
        traj[0, i, 2] = math.cos(theta)
        traj[0, i, 3] = math.sin(theta)

    past = traj[:, :T_HIST]
    future = traj[:, T_HIST:]
    return past, future, v


def _place_as_neighbor(
    past: torch.Tensor,
    future: torch.Tensor,
    heading: float,
    offset_x: float,
    offset_y: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate and translate a local-frame trajectory, then reshape as neighbor.

    Args:
        past: [1, T_HIST, 4] trajectory generated at heading=0 around origin
        future: [1, T, 4] same
        heading: global heading to rotate the trajectory to
        offset_x, offset_y: global position offset

    Returns:
        (past, future) reshaped to [1, 1, T, 4] for neighbor tests
    """
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    for traj in (past, future):
        old_x = traj[..., 0].clone()
        old_y = traj[..., 1].clone()
        traj[..., 0] = old_x * cos_h - old_y * sin_h + offset_x
        traj[..., 1] = old_x * sin_h + old_y * cos_h + offset_y
        old_cos = traj[..., 2].clone()
        old_sin = traj[..., 3].clone()
        traj[..., 2] = old_cos * cos_h - old_sin * sin_h
        traj[..., 3] = old_cos * sin_h + old_sin * cos_h

    return past.unsqueeze(1), future.unsqueeze(1)  # [1, 1, T, 4]


# ---------------------------------------------------------------------------
# Pytest tests (synthetic data)
# ---------------------------------------------------------------------------


class TestEgoRoundtrip:
    """trajectory -> control -> trajectory for ego."""

    def _run(self, past, future, v0, atol=0.01):
        ctrl = waypoints_to_control(past, future, t0_states={"v": torch.tensor([v0])})
        assert not ctrl.isnan().any(), "NaN in ego control"
        assert not ctrl.isinf().any(), "Inf in ego control"

        recon = control_to_waypoints(ctrl, past, t0_states={"v": torch.tensor([v0])})
        assert torch.allclose(future[..., :2], recon[..., :2], atol=atol), (
            f"Ego roundtrip pos error too large: max={(future[..., :2] - recon[..., :2]).abs().max().item():.6f}"
        )

    def test_straight_line(self):
        past, future, v0 = make_straight_line(v=5.0, heading=0.0)
        self._run(past, future, v0)

    def test_circular_arc(self):
        # Larger atol for arcs: unicycle discrete integration accumulates error
        past, future, v0 = make_circular_arc(v=5.0, radius=50.0)
        self._run(past, future, v0, atol=0.25)

    def test_circular_arc_tight(self):
        past, future, v0 = make_circular_arc(v=3.0, radius=20.0)
        self._run(past, future, v0, atol=0.20)


class TestNeighborLocalFrameRoundtrip:
    """neighbor ego-frame -> local frame -> control -> waypoints -> ego frame."""

    def _run(self, n_past, n_future, atol=0.05):
        # n_past: [1, 1, T_HIST, 4], n_future: [1, 1, T, 4]
        hist_local, fut_local = transform_to_local_frame(n_past, n_future)

        ctrl = waypoints_to_control(hist_local, fut_local)
        ctrl = torch.nan_to_num(ctrl, nan=0.0)

        recon_local = control_to_waypoints(ctrl, hist_local)

        ref_pos = n_past[..., -1, :2]  # [1, 1, 2]
        ref_cos = n_past[..., -1, 2:3]  # [1, 1, 1]
        ref_sin = n_past[..., -1, 3:4]  # [1, 1, 1]
        recon_ego = transform_to_ego_frame(recon_local, ref_pos, ref_cos, ref_sin)

        assert torch.allclose(n_future[..., :2], recon_ego[..., :2], atol=atol), (
            f"Neighbor roundtrip pos error too large: max="
            f"{(n_future[..., :2] - recon_ego[..., :2]).abs().max().item():.6f}"
        )

    def test_straight_neighbor(self):
        past, future, _ = make_straight_line(v=4.0, heading=0.0)
        n_past, n_future = _place_as_neighbor(past, future, heading=0.0, offset_x=5.0, offset_y=3.0)
        self._run(n_past, n_future)

    def test_angled_neighbor(self):
        past, future, _ = make_straight_line(v=4.0, heading=0.0)
        n_past, n_future = _place_as_neighbor(past, future, heading=0.5, offset_x=5.0, offset_y=3.0)
        self._run(n_past, n_future)

    def test_curved_neighbor(self):
        past, future, _ = make_circular_arc(v=4.0, radius=40.0)
        n_past, n_future = _place_as_neighbor(past, future, heading=0.2, offset_x=3.0, offset_y=2.0)
        self._run(n_past, n_future, atol=0.20)


class TestControlNormalizerRoundtrip:
    """control -> normalize -> inverse -> compare."""

    def test_roundtrip(self):
        mean = [0.1, 0.002]
        std = [1.5, 0.05]
        norm = ControlNormalizer(mean, std)

        ctrl = torch.randn(2, T_FUTURE, 2) * 2.0
        normed = norm(ctrl)
        recovered = norm.inverse(normed)

        assert torch.allclose(ctrl, recovered, atol=1e-6), (
            f"ControlNormalizer roundtrip error: max={(ctrl - recovered).abs().max().item():.2e}"
        )

    def test_zero_preserving(self):
        norm = ControlNormalizer([0.0, 0.0], [1.0, 1.0])
        ctrl = torch.zeros(1, T_FUTURE, 2)
        normed = norm(ctrl)
        recovered = norm.inverse(normed)
        assert torch.allclose(ctrl, recovered, atol=1e-6)


class TestInvalidTimestepPreservation:
    """Zero-padded neighbor future stays zero after local frame transform."""

    def test_zeros_preserved(self):
        past, future, _ = make_straight_line(v=4.0, heading=0.0)
        n_past, n_future = _place_as_neighbor(past, future, heading=0.3, offset_x=5.0, offset_y=3.0)

        # Zero out second half of future (simulate invalid timesteps)
        n_future[:, :, T_FUTURE // 2 :, :] = 0.0

        _, fut_local = transform_to_local_frame(n_past, n_future, preserve_invalid=True)

        invalid_region = fut_local[:, :, T_FUTURE // 2 :, :]
        assert (invalid_region == 0.0).all(), (
            f"Invalid timesteps not preserved: max nonzero = {invalid_region.abs().max().item():.2e}"
        )

    def test_valid_part_unchanged(self):
        past, future, _ = make_straight_line(v=4.0, heading=0.0)
        n_past, n_future_full = _place_as_neighbor(
            past, future, heading=0.3, offset_x=5.0, offset_y=3.0
        )

        n_future_partial = n_future_full.clone()
        n_future_partial[:, :, T_FUTURE // 2 :, :] = 0.0

        _, fut_local_full = transform_to_local_frame(n_past, n_future_full, preserve_invalid=True)
        _, fut_local_partial = transform_to_local_frame(
            n_past, n_future_partial, preserve_invalid=True
        )

        valid_half = T_FUTURE // 2
        assert torch.allclose(
            fut_local_full[:, :, :valid_half],
            fut_local_partial[:, :, :valid_half],
            atol=1e-6,
        ), "Valid portion of future changed when zeros were added"


class TestNeighborLocalFrameIdentity:
    """At reference timestep, local frame position is (0,0) and heading is (1,0)."""

    def test_identity_at_reference(self):
        past, future, _ = make_straight_line(v=5.0, heading=0.0)
        n_past, n_future = _place_as_neighbor(past, future, heading=0.5, offset_x=5.0, offset_y=3.0)

        hist_local, _ = transform_to_local_frame(n_past, n_future)

        ref = hist_local[0, 0, -1]  # last history step
        assert torch.allclose(ref[:2], torch.zeros(2), atol=1e-6), (
            f"Local frame ref position not (0,0): {ref[:2].tolist()}"
        )
        assert torch.allclose(ref[2:], torch.tensor([1.0, 0.0]), atol=1e-6), (
            f"Local frame ref heading not (1,0): {ref[2:].tolist()}"
        )

    def test_identity_multiple_headings(self):
        for heading in [0.0, 0.3, -0.5, math.pi / 2, math.pi, -math.pi]:
            past, future, _ = make_straight_line(v=3.0, heading=0.0)
            n_past, n_future = _place_as_neighbor(
                past, future, heading=heading, offset_x=2.0, offset_y=1.0
            )
            hist_local, _ = transform_to_local_frame(n_past, n_future)
            ref = hist_local[0, 0, -1]
            assert torch.allclose(ref[:2], torch.zeros(2), atol=1e-6)
            assert torch.allclose(ref[2:], torch.tensor([1.0, 0.0]), atol=1e-6)


# ---------------------------------------------------------------------------
# Standalone mode: test on actual npz data files
# ---------------------------------------------------------------------------


def _load_sample(path: str) -> dict:
    """Load a single npz file and prepare tensors for roundtrip testing."""
    from diffusion_planner.train_epoch import heading_to_cos_sin

    d = np.load(path)

    ego_past = heading_to_cos_sin(
        torch.from_numpy(d["ego_agent_past"]).unsqueeze(0).float()
    )  # [1, T_HIST, 4]
    ego_v0 = float(d["ego_current_state"][4])
    ego_future = heading_to_cos_sin(
        torch.from_numpy(d["ego_agent_future"]).unsqueeze(0).float()
    )  # [1, T, 4]

    neighbor_past_raw = torch.from_numpy(d["neighbor_agents_past"]).unsqueeze(0).float()
    neighbor_past_4d = neighbor_past_raw[:, :, :, :4]  # already (x, y, cos, sin)

    neighbors_future_raw = torch.from_numpy(d["neighbor_agents_future"]).unsqueeze(0).float()
    neighbor_future_mask = torch.sum(torch.ne(neighbors_future_raw[..., :3], 0), dim=-1) == 0
    neighbors_future = heading_to_cos_sin(neighbors_future_raw)
    neighbors_future[neighbor_future_mask] = 0.0

    return {
        "path": path,
        "ego_past": ego_past,
        "ego_v0": ego_v0,
        "ego_future": ego_future,
        "neighbor_past_4d": neighbor_past_4d,
        "neighbors_future": neighbors_future,
        "neighbor_future_mask": neighbor_future_mask,
    }


def _run_standalone(path_list_json: str):
    """Run roundtrip tests on real data and print detailed results."""
    with open(path_list_json) as f:
        paths = json.load(f)

    N = min(10, len(paths))
    indices = sorted(random.sample(range(len(paths)), N))

    print(f"Testing {N} samples from {path_list_json}")
    print(f"{'=' * 70}")

    ego_errors = []
    neighbor_errors = []

    for idx in indices:
        sample = _load_sample(paths[idx])
        print(f"\nSample {idx}: {sample['path']}")

        # --- Ego roundtrip ---
        ego_ctrl = waypoints_to_control(
            sample["ego_past"],
            sample["ego_future"],
            t0_states={"v": torch.tensor([sample["ego_v0"]])},
        )
        ego_recon = control_to_waypoints(
            ego_ctrl,
            sample["ego_past"],
            t0_states={"v": torch.tensor([sample["ego_v0"]])},
        )
        ego_err = (sample["ego_future"][0, :, :2] - ego_recon[0, :, :2]).abs()
        ego_max = ego_err.max().item()
        ego_mean = ego_err.mean().item()
        ego_errors.append(ego_max)
        status = "PASS" if ego_max < 0.01 else "FAIL"
        print(f"  EGO roundtrip: [{status}] mean={ego_mean:.6f}m  max={ego_max:.6f}m")

        if ego_ctrl.isnan().any() or ego_ctrl.isinf().any():
            print(f"  EGO ctrl has NaN/Inf!")

        # --- Neighbor roundtrip ---
        n_past = sample["neighbor_past_4d"]
        n_future = sample["neighbors_future"]
        Pn = n_future.shape[1]

        valid_mask = n_past[0, :, -1, :].abs().sum(dim=-1) > 0
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]

        if len(valid_indices) == 0:
            print(f"  NEIGHBORS: None valid")
            continue

        for ni in valid_indices[:5].tolist():
            ni_past = n_past[:, ni : ni + 1]  # [1, 1, T_HIST, 4]
            ni_future = n_future[:, ni : ni + 1]  # [1, 1, T, 4]

            # Check if this neighbor has valid future data
            n_valid_count = ni_future.ne(0).any(dim=-1).sum().item()
            if n_valid_count == 0:
                print(f"  Neighbor {ni}: no valid future -- skip")
                continue

            # Transform to local frame
            hist_local, fut_local = transform_to_local_frame(ni_past, ni_future)

            # Verify local frame identity
            ref = hist_local[0, 0, -1]
            id_ok = (
                ref[:2].abs().max().item() < 1e-6
                and (ref[2:] - torch.tensor([1.0, 0.0])).abs().max().item() < 1e-6
            )

            # Control roundtrip in local frame
            n_ctrl = waypoints_to_control(hist_local, fut_local)
            n_ctrl = torch.nan_to_num(n_ctrl, nan=0.0)

            if n_ctrl.isnan().any() or n_ctrl.isinf().any():
                print(f"  Neighbor {ni}: ctrl has NaN/Inf -- skip")
                continue

            recon_local = control_to_waypoints(n_ctrl, hist_local)

            # Back to ego frame
            ref_pos = ni_past[..., -1, :2]
            ref_cos = ni_past[..., -1, 2:3]
            ref_sin = ni_past[..., -1, 3:4]
            recon_ego = transform_to_ego_frame(recon_local, ref_pos, ref_cos, ref_sin)

            # Only compare valid timesteps
            valid_t = ni_future.ne(0).any(dim=-1).squeeze(0).squeeze(0)  # [T]
            if valid_t.sum() == 0:
                continue

            n_err = (ni_future[0, 0, valid_t, :2] - recon_ego[0, 0, valid_t, :2]).abs()
            n_max = n_err.max().item()
            n_mean = n_err.mean().item()
            neighbor_errors.append(n_max)

            status = "PASS" if n_max < 0.05 else "FAIL"
            id_str = "OK" if id_ok else "FAIL"
            print(
                f"  Neighbor {ni}: [{status}] mean={n_mean:.6f}m  max={n_max:.6f}m  "
                f"local_id={id_str}  valid={n_valid_count}/{ni_future.shape[2]}"
            )

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(
        f"  Ego   : {len(ego_errors)} samples, "
        f"max_err range [{min(ego_errors):.6f}, {max(ego_errors):.6f}]m, "
        f"all<0.01: {all(e < 0.01 for e in ego_errors)}"
    )
    if neighbor_errors:
        print(
            f"  Neigh : {len(neighbor_errors)} agents, "
            f"max_err range [{min(neighbor_errors):.6f}, {max(neighbor_errors):.6f}]m, "
            f"all<0.05: {all(e < 0.05 for e in neighbor_errors)}"
        )
    else:
        print("  Neigh : no valid neighbors tested")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        _run_standalone(sys.argv[1])
    else:
        # Run pytest programmatically
        import pytest

        sys.exit(pytest.main([__file__, "-v"]))
