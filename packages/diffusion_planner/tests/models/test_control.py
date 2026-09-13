"""Round-trip tests for the pose <-> control conversion.

Every trajectory here is built in the frame the conversion assumes: ego-centric,
with the last history pose at the origin and heading zero (see
``diffusion_planner.models.control``).
"""

from __future__ import annotations

import torch

from diffusion_planner.data.dimensions import (
    CONTROL_DIM,
    EGO_HISTORY_LENGTH,
    TRAJECTORY_DIM,
    TRAJECTORY_LENGTH,
)
from diffusion_planner.models.control import (
    ControlNormalizer,
    control_to_waypoints,
    denormalize_positions,
    waypoints_to_control,
)

DT = 0.1


def make_straight_line(speed: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return `(history, future, initial_speed)` for a constant-velocity line."""
    total = EGO_HISTORY_LENGTH + TRAJECTORY_LENGTH
    times = (torch.arange(total, dtype=torch.float32) - EGO_HISTORY_LENGTH + 1) * DT
    trajectory = torch.zeros(1, total, TRAJECTORY_DIM)
    trajectory[0, :, 0] = speed * times
    trajectory[0, :, 2] = 1.0
    return trajectory[:, :EGO_HISTORY_LENGTH], trajectory[:, EGO_HISTORY_LENGTH:], speed


def make_circular_arc(
    speed: float, radius: float
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return `(history, future, initial_speed)` for a constant-curvature arc."""
    total = EGO_HISTORY_LENGTH + TRAJECTORY_LENGTH
    times = (torch.arange(total, dtype=torch.float32) - EGO_HISTORY_LENGTH + 1) * DT
    angles = (speed / radius) * times
    trajectory = torch.zeros(1, total, TRAJECTORY_DIM)
    trajectory[0, :, 0] = radius * torch.sin(angles)
    trajectory[0, :, 1] = radius * (1.0 - torch.cos(angles))
    trajectory[0, :, 2] = torch.cos(angles)
    trajectory[0, :, 3] = torch.sin(angles)
    return trajectory[:, :EGO_HISTORY_LENGTH], trajectory[:, EGO_HISTORY_LENGTH:], speed


class TestRoundTrip:
    """pose -> control -> pose must return the original trajectory."""

    def check(self, history, future, speed, tolerance):
        initial_speed = torch.tensor([speed])
        control = waypoints_to_control(history, future, initial_speed)
        assert control.shape == (1, TRAJECTORY_LENGTH, CONTROL_DIM)
        assert torch.isfinite(control).all()

        reconstructed = control_to_waypoints(control, history, initial_speed)
        error = (future[..., :2] - reconstructed[..., :2]).abs().max().item()
        assert error < tolerance, f"position round-trip error {error:.4f} m"

    def test_straight_line(self):
        self.check(*make_straight_line(speed=5.0), tolerance=0.01)

    def test_fast_straight_line(self):
        self.check(*make_straight_line(speed=15.0), tolerance=0.01)

    def test_circular_arc(self):
        # Discrete unicycle integration accumulates error along an arc.
        self.check(*make_circular_arc(speed=5.0, radius=50.0), tolerance=0.25)

    def test_circular_arc_tight(self):
        self.check(*make_circular_arc(speed=3.0, radius=20.0), tolerance=0.20)

    def test_stationary(self):
        history, future, _ = make_straight_line(speed=0.0)
        self.check(history, future, 0.0, tolerance=0.01)

    def test_batch_is_independent_of_neighbours(self):
        """Two trajectories fitted together must match them fitted separately."""
        slow_history, slow_future, slow_speed = make_straight_line(speed=3.0)
        fast_history, fast_future, fast_speed = make_straight_line(speed=12.0)
        history = torch.cat((slow_history, fast_history), dim=0)
        future = torch.cat((slow_future, fast_future), dim=0)
        speeds = torch.tensor([slow_speed, fast_speed])

        batched = waypoints_to_control(history, future, speeds)
        separate = torch.cat(
            (
                waypoints_to_control(slow_history, slow_future, speeds[:1]),
                waypoints_to_control(fast_history, fast_future, speeds[1:]),
            ),
            dim=0,
        )
        assert torch.allclose(batched, separate, atol=1e-5)


class TestControlValues:
    """The fitted control must match the analytic values of the generator."""

    def test_straight_line_is_zero_control(self):
        history, future, speed = make_straight_line(speed=6.0)
        control = waypoints_to_control(history, future, torch.tensor([speed]))
        assert control.abs().max().item() < 1e-2

    def test_arc_curvature_matches_radius(self):
        radius = 40.0
        history, future, speed = make_circular_arc(speed=5.0, radius=radius)
        control = waypoints_to_control(history, future, torch.tensor([speed]))
        curvature = control[0, :, 1]
        assert abs(curvature.mean().item() - 1.0 / radius) < 5e-3
        assert control[0, :, 0].abs().max().item() < 0.1


class TestDenormalizePositions:
    def test_scales_only_xy(self):
        trajectory = torch.tensor([[[0.1, 0.2, 0.6, 0.8]]])
        result = denormalize_positions(trajectory, position_scale=50.0)
        assert torch.allclose(result[..., :2], torch.tensor([[[5.0, 10.0]]]))
        assert torch.allclose(result[..., 2:], trajectory[..., 2:])

    def test_round_trip_through_control(self):
        """A normalized trajectory must survive scale-up, conversion, and scale-down."""
        history, future, speed = make_straight_line(speed=5.0)
        scale = 50.0
        history_normalized = torch.cat(
            (history[..., :2] / scale, history[..., 2:]), dim=-1
        )
        future_normalized = torch.cat(
            (future[..., :2] / scale, future[..., 2:]), dim=-1
        )

        control = waypoints_to_control(
            denormalize_positions(history_normalized, scale),
            denormalize_positions(future_normalized, scale),
            torch.tensor([speed]),
        )
        reconstructed = control_to_waypoints(
            control,
            denormalize_positions(history_normalized, scale),
            torch.tensor([speed]),
        )
        error = (future[..., :2] - reconstructed[..., :2]).abs().max().item()
        assert error < 0.01


class TestControlNormalizer:
    def test_round_trip(self):
        normalizer = ControlNormalizer([0.1, 0.002], [1.5, 0.05])
        control = torch.randn(2, TRAJECTORY_LENGTH, CONTROL_DIM) * 2.0
        recovered = normalizer.inverse(normalizer(control))
        assert torch.allclose(control, recovered, atol=1e-5)

    def test_identity_statistics_are_a_no_op(self):
        normalizer = ControlNormalizer([0.0, 0.0], [1.0, 1.0])
        control = torch.randn(1, TRAJECTORY_LENGTH, CONTROL_DIM)
        assert torch.allclose(normalizer(control), control)

    def test_to_dict(self):
        normalizer = ControlNormalizer([0.5, 1.5], [2.0, 4.0])
        assert normalizer.to_dict() == {"mean": [0.5, 1.5], "std": [2.0, 4.0]}
