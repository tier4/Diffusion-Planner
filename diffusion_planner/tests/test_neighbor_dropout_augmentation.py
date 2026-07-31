"""NeighborDropoutAugmentation: zero out whole neighbor agents (past and future
together) to simulate perception misses, without touching padding rows."""

import torch
from diffusion_planner.utils.data_augmentation import NeighborDropoutAugmentation

B, N, T, TF = 2, 5, 4, 6


def _make_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    inputs = {
        "neighbor_agents_past": torch.randn(B, N, T, 11, generator=g),
        "ego_current_state": torch.randn(B, 10, generator=g),
    }
    # last two neighbors are padding (all-zero)
    inputs["neighbor_agents_past"][:, -2:] = 0.0
    ego_future = torch.randn(B, TF, 3, generator=g)
    neighbors_future = torch.randn(B, N, TF, 3, generator=g)
    neighbors_future[:, -2:] = 0.0
    return inputs, ego_future, neighbors_future


def test_dropout_prob_one_drops_all_valid_neighbors():
    inputs, ego_future, neighbors_future = _make_batch()
    past_input = inputs["neighbor_agents_past"]
    future_input = neighbors_future
    orig_ego_current = inputs["ego_current_state"].clone()
    orig_ego_future = ego_future.clone()

    aug = NeighborDropoutAugmentation(dropout_prob=1.0, device="cpu")
    inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

    # Disposable training tensors are updated in place to avoid full-size copies.
    assert inputs["neighbor_agents_past"] is past_input
    assert neighbors_future is future_input
    assert torch.equal(
        inputs["neighbor_agents_past"], torch.zeros_like(inputs["neighbor_agents_past"])
    )
    assert torch.equal(neighbors_future, torch.zeros_like(neighbors_future))
    # ego is untouched
    assert torch.equal(inputs["ego_current_state"], orig_ego_current)
    assert torch.equal(ego_future, orig_ego_future)


def test_dropout_prob_zero_is_identity():
    inputs, ego_future, neighbors_future = _make_batch()
    orig_past = inputs["neighbor_agents_past"].clone()
    orig_future = neighbors_future.clone()

    aug = NeighborDropoutAugmentation(dropout_prob=0.0, device="cpu")
    inputs, _, neighbors_future = aug(inputs, ego_future, neighbors_future)

    assert torch.equal(inputs["neighbor_agents_past"], orig_past)
    assert torch.equal(neighbors_future, orig_future)


def test_past_and_future_are_dropped_together():
    torch.manual_seed(0)
    inputs, ego_future, neighbors_future = _make_batch()
    orig_past = inputs["neighbor_agents_past"].clone()
    orig_future = neighbors_future.clone()

    aug = NeighborDropoutAugmentation(dropout_prob=0.5, device="cpu")
    inputs, _, neighbors_future = aug(inputs, ego_future, neighbors_future)

    past = inputs["neighbor_agents_past"]
    for b in range(B):
        for n in range(N - 2):  # valid neighbors only
            past_zeroed = torch.equal(past[b, n], torch.zeros_like(past[b, n]))
            future_zeroed = torch.equal(
                neighbors_future[b, n], torch.zeros_like(neighbors_future[b, n])
            )
            # an agent is either fully kept or fully dropped, consistently
            assert past_zeroed == future_zeroed
            if not past_zeroed:
                assert torch.equal(past[b, n], orig_past[b, n])
                assert torch.equal(neighbors_future[b, n], orig_future[b, n])


def test_padding_rows_stay_zero():
    inputs, ego_future, neighbors_future = _make_batch()

    aug = NeighborDropoutAugmentation(dropout_prob=0.5, device="cpu")
    inputs, _, neighbors_future = aug(inputs, ego_future, neighbors_future)

    assert torch.equal(
        inputs["neighbor_agents_past"][:, -2:],
        torch.zeros_like(inputs["neighbor_agents_past"][:, -2:]),
    )
    assert torch.equal(neighbors_future[:, -2:], torch.zeros_like(neighbors_future[:, -2:]))


def test_some_neighbors_survive_at_moderate_prob():
    torch.manual_seed(0)
    inputs, ego_future, neighbors_future = _make_batch()

    aug = NeighborDropoutAugmentation(dropout_prob=0.3, device="cpu")
    inputs, _, _ = aug(inputs, ego_future, neighbors_future)

    # with 6 valid agents and p=0.3, all being dropped is astronomically
    # unlikely under the fixed seed; guards against inverted keep/drop logic
    assert inputs["neighbor_agents_past"].abs().sum() > 0
