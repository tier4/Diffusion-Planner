"""Tests for the numpy-core bridge augmentation (data_augmentation_bridge).

Covered:
- augment: shapes, aug_flag, continuity between past/current/future
- lateral offset magnitude stays within the configured range
- future endpoint stays close to the GT endpoint (progress alignment)
- constraint feasibility of the augmented sample
- slow vehicles are not augmented
- determinism under torch.manual_seed
- __call__ end-to-end with map tensors (centric_transform recentering)
- per-sample runtime stays fast enough for training
"""

import time

import numpy as np
import pytest
import torch
from diffusion_planner.utils import data_augmentation_bridge_core as bridge_core
from diffusion_planner.utils.data_augmentation_bridge import StatePerturbation

PAST_LEN = 21  # 2.0 s history including the current frame
FUTURE_LEN = 80  # 8.0 s future
DT = 0.1
SPEED = 8.0
CURVATURE = 0.01


def _make_ego_sample() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Constant-speed gentle arc: past (P,4), current (10,), future (F,3)."""
    num_points = PAST_LEN + FUTURE_LEN
    times = (np.arange(num_points) - (PAST_LEN - 1)) * DT
    s = SPEED * times
    heading = CURVATURE * s
    x = np.where(np.abs(heading) < 1e-9, s, np.sin(heading) / CURVATURE)
    y = np.where(np.abs(heading) < 1e-9, 0.0, (1.0 - np.cos(heading)) / CURVATURE)

    past = np.stack(
        (
            x[:PAST_LEN],
            y[:PAST_LEN],
            np.cos(heading[:PAST_LEN]),
            np.sin(heading[:PAST_LEN]),
        ),
        axis=-1,
    )
    current_heading = heading[PAST_LEN - 1]
    current = np.zeros(10)
    current[0] = x[PAST_LEN - 1]
    current[1] = y[PAST_LEN - 1]
    current[2] = np.cos(current_heading)
    current[3] = np.sin(current_heading)
    current[4] = SPEED
    future = np.stack(
        (x[PAST_LEN:], y[PAST_LEN:], heading[PAST_LEN:]),
        axis=-1,
    )
    return (
        torch.tensor(past, dtype=torch.float32),
        torch.tensor(current, dtype=torch.float32),
        torch.tensor(future, dtype=torch.float32),
    )


def _make_inputs(batch_size: int = 1) -> tuple[dict, torch.Tensor, torch.Tensor]:
    past, current, future = _make_ego_sample()
    inputs = {
        "ego_current_state": current.unsqueeze(0).repeat(batch_size, 1),
        "ego_agent_past": past.unsqueeze(0).repeat(batch_size, 1, 1),
        "neighbor_agents_past": torch.zeros(batch_size, 4, PAST_LEN, 11),
        "lanes": torch.zeros(batch_size, 10, 20, 12),
        "route_lanes": torch.zeros(batch_size, 5, 20, 12),
        "polygons": torch.zeros(batch_size, 5, 8, 3),
        "line_strings": torch.zeros(batch_size, 5, 8, 3),
        "static_objects": torch.zeros(batch_size, 5, 10),
    }
    ego_future = future.unsqueeze(0).repeat(batch_size, 1, 1)
    neighbors_future = torch.zeros(batch_size, 4, FUTURE_LEN, 3)
    return inputs, ego_future, neighbors_future


def test_augment_shapes_and_continuity():
    torch.manual_seed(0)
    aug = StatePerturbation(augment_prob=1.0, device="cpu")
    inputs, ego_future, _ = _make_inputs()
    original_current = inputs["ego_current_state"].clone()

    aug_flag, aug_current, aug_past, aug_future = aug.augment(inputs, ego_future)

    assert bool(aug_flag[0])
    assert aug_past.shape == inputs["ego_agent_past"].shape
    assert aug_future.shape == ego_future.shape
    assert aug_current.shape == inputs["ego_current_state"].shape

    # The last past row is the current frame.
    np.testing.assert_allclose(aug_past[0, -1, :2].numpy(), aug_current[0, :2].numpy(), atol=1e-5)
    np.testing.assert_allclose(aug_past[0, -1, 2:4].numpy(), aug_current[0, 2:4].numpy(), atol=1e-5)

    # Lateral displacement of the current pose stays within the configured range.
    displacement = torch.linalg.norm(aug_current[0, :2] - original_current[0, :2]).item()
    assert 1e-3 < displacement <= 0.75 + 1e-3

    # Progress alignment keeps the future endpoint near the GT endpoint.
    end_delta = torch.linalg.norm(aug_future[0, -1, :2] - ego_future[0, -1, :2]).item()
    assert end_delta < 1.0


def test_augmented_sample_passes_constraints():
    torch.manual_seed(1)
    aug = StatePerturbation(augment_prob=1.0, device="cpu")
    inputs, ego_future, _ = _make_inputs()

    aug_flag, aug_current, aug_past, aug_future = aug.augment(inputs, ego_future)
    assert bool(aug_flag[0])

    past_np = aug_past[0].numpy()
    x = np.concatenate((past_np[:, 0], aug_future[0, :, 0].numpy()))
    y = np.concatenate((past_np[:, 1], aug_future[0, :, 1].numpy()))
    sigma = bridge_core.cumulative_distance(x, y)
    curvature = bridge_core.curvature_from_xy(x, y, sigma)
    times = (np.arange(len(x), dtype=float) - (PAST_LEN - 1)) * DT
    speed = bridge_core.speed_from_progress(sigma, times)
    lateral_accel = np.abs(speed * speed * curvature)
    assert float(np.max(lateral_accel)) <= 3.0 + 1e-6


def test_slow_vehicle_is_not_augmented():
    torch.manual_seed(0)
    aug = StatePerturbation(augment_prob=1.0, device="cpu")
    inputs, ego_future, _ = _make_inputs()
    inputs["ego_current_state"][0, 4] = 0.5  # below the 2 m/s gate

    aug_flag, _, _, _ = aug.augment(inputs, ego_future)
    assert not bool(aug_flag[0])


def test_augment_is_deterministic_under_torch_seed():
    results = []
    for _ in range(2):
        torch.manual_seed(123)
        aug = StatePerturbation(augment_prob=1.0, device="cpu")
        inputs, ego_future, _ = _make_inputs()
        _, aug_current, aug_past, aug_future = aug.augment(inputs, ego_future)
        results.append((aug_current.clone(), aug_past.clone(), aug_future.clone()))

    for a, b in zip(results[0], results[1]):
        assert torch.equal(a, b)


def test_call_end_to_end_recenters_ego():
    torch.manual_seed(0)
    aug = StatePerturbation(augment_prob=1.0, device="cpu")
    inputs, ego_future, neighbors_future = _make_inputs()

    out_inputs, out_future, _ = aug(inputs, ego_future, neighbors_future)

    # After centric_transform the current pose is at the origin with zero heading.
    current = out_inputs["ego_current_state"][0]
    np.testing.assert_allclose(current[:2].numpy(), np.zeros(2), atol=1e-5)
    np.testing.assert_allclose(current[2:4].numpy(), np.array([1.0, 0.0]), atol=1e-5)
    assert out_future.shape == (1, FUTURE_LEN, 3)


def test_runtime_is_fast_enough():
    torch.manual_seed(0)
    aug = StatePerturbation(augment_prob=1.0, device="cpu")
    num_trials = 10
    start = time.perf_counter()
    for _ in range(num_trials):
        inputs, ego_future, _ = _make_inputs()
        aug_flag, _, _, _ = aug.augment(inputs, ego_future)
        assert bool(aug_flag[0])
    per_sample_s = (time.perf_counter() - start) / num_trials
    assert per_sample_s < 0.5, f"augmentation too slow: {per_sample_s * 1000:.0f} ms/sample"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
