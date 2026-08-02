"""NeighborNoiseAugmentation: add tracking-like Gaussian noise to the neighbor
history only, keeping futures clean and padding rows all-zero."""

import torch
from diffusion_planner.utils.data_augmentation import NeighborNoiseAugmentation

B, N, T, TF = 2, 5, 4, 6


def _make_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    past = torch.randn(B, N, T, 11, generator=g)
    # give valid frames a unit-norm (cos, sin) pair like real data
    heading = torch.rand(B, N, T, generator=g) * 6.28
    past[..., 2] = torch.cos(heading)
    past[..., 3] = torch.sin(heading)
    # last two neighbors are padding (all-zero)
    past[:, -2:] = 0.0
    inputs = {
        "neighbor_agents_past": past,
        "ego_current_state": torch.randn(B, 10, generator=g),
        "ego_agent_past": torch.randn(B, T, 4, generator=g),
    }
    ego_future = torch.randn(B, TF, 3, generator=g)
    neighbors_future = torch.randn(B, N, TF, 3, generator=g)
    neighbors_future[:, -2:] = 0.0
    return inputs, ego_future, neighbors_future


def _aug(pos=0.2, vel=0.3, heading=0.05):
    return NeighborNoiseAugmentation(
        pos_noise_std=pos, vel_noise_std=vel, heading_noise_std=heading, device="cpu"
    )


def test_noise_perturbs_only_past_pose_and_velocity():
    torch.manual_seed(0)
    inputs, ego_future, neighbors_future = _make_batch()
    past_input = inputs["neighbor_agents_past"]
    orig = {k: v.clone() for k, v in inputs.items()}
    orig_future = neighbors_future.clone()
    orig_ego_future = ego_future.clone()

    inputs, ego_future, neighbors_future = _aug()(inputs, ego_future, neighbors_future)

    # The disposable training input is updated in place to avoid cloning the full
    # (B, N, T, D) neighbor-history tensor.
    assert inputs["neighbor_agents_past"] is past_input
    past, orig_past = inputs["neighbor_agents_past"], orig["neighbor_agents_past"]
    valid = orig_past[:, :-2]
    noisy = past[:, :-2]
    # x, y, cos, sin, vx, vy all changed on valid frames
    for i in range(6):
        assert not torch.equal(noisy[..., i], valid[..., i])
    # width, length and type one-hot untouched
    assert torch.equal(past[..., 6:], orig_past[..., 6:])
    # everything else in the batch is untouched
    assert torch.equal(inputs["ego_current_state"], orig["ego_current_state"])
    assert torch.equal(inputs["ego_agent_past"], orig["ego_agent_past"])
    assert torch.equal(ego_future, orig_ego_future)
    assert torch.equal(neighbors_future, orig_future)


def test_heading_stays_unit_norm():
    torch.manual_seed(0)
    inputs, ego_future, neighbors_future = _make_batch()

    inputs, _, _ = _aug()(inputs, ego_future, neighbors_future)

    cos_sin = inputs["neighbor_agents_past"][:, :-2, :, 2:4]
    norms = cos_sin.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_padding_rows_stay_zero():
    torch.manual_seed(0)
    inputs, ego_future, neighbors_future = _make_batch()

    inputs, _, neighbors_future = _aug()(inputs, ego_future, neighbors_future)

    pad = inputs["neighbor_agents_past"][:, -2:]
    assert torch.equal(pad, torch.zeros_like(pad))
    assert torch.equal(neighbors_future[:, -2:], torch.zeros_like(neighbors_future[:, -2:]))


def test_zero_std_is_identity():
    inputs, ego_future, neighbors_future = _make_batch()
    orig_past = inputs["neighbor_agents_past"].clone()

    inputs, _, _ = _aug(pos=0.0, vel=0.0, heading=0.0)(inputs, ego_future, neighbors_future)

    assert torch.allclose(inputs["neighbor_agents_past"], orig_past, atol=1e-6)


def test_noise_magnitude_matches_std():
    torch.manual_seed(0)
    inputs, ego_future, neighbors_future = _make_batch()
    # larger batch of frames for a stable estimate
    inputs["neighbor_agents_past"] = torch.randn(8, 16, 21, 11)
    inputs["neighbor_agents_past"][..., 2] = 1.0  # cos
    inputs["neighbor_agents_past"][..., 3] = 0.0  # sin
    orig = inputs["neighbor_agents_past"].clone()
    neighbors_future = torch.randn(8, 16, TF, 3)

    inputs, _, _ = _aug(pos=0.2, vel=0.3)(inputs, ego_future, neighbors_future)

    diff = inputs["neighbor_agents_past"] - orig
    assert abs(diff[..., 0:2].std().item() - 0.2) < 0.02
    assert abs(diff[..., 4:6].std().item() - 0.3) < 0.03
