import torch

from diffusion_planner.model.module.decoder import (
    apply_inference_prefix,
    future_prefix_mask,
    generate_prefix_mask,
    replace_current_state,
)


def test_delay_zero_is_exactly_the_existing_current_state_constraint():
    xt = torch.arange(2 * 3 * 5 * 4, dtype=torch.float32).reshape(2, 3, 5, 4)
    action_prefix = torch.full_like(xt, -123.0)
    current = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0]] * 3,
            [[5.0, 6.0, 7.0, 8.0]] * 3,
        ]
    )
    mask = future_prefix_mask(generate_prefix_mask(torch.zeros(2), 3, 5))

    actual = apply_inference_prefix(xt, action_prefix, current, mask)
    expected = replace_current_state(xt, current)

    assert not mask.any()
    assert torch.equal(actual, expected)


def test_positive_delay_fixes_only_ego_future_indices_one_through_delay():
    xt = torch.zeros((1, 3, 6, 4), dtype=torch.float32)
    action_prefix = torch.arange(1 * 3 * 6 * 4, dtype=torch.float32).reshape(1, 3, 6, 4)
    current = torch.full((1, 3, 4), 777.0)
    mask = future_prefix_mask(generate_prefix_mask(torch.tensor([2]), 3, 6))

    actual = apply_inference_prefix(xt, action_prefix, current, mask)

    assert torch.equal(actual[:, :, 0], current)
    assert torch.equal(actual[:, 0, 1:3], action_prefix[:, 0, 1:3])
    assert torch.equal(actual[:, 0, 3:], torch.zeros_like(actual[:, 0, 3:]))
    assert torch.equal(actual[:, 1:, 1:], torch.zeros_like(actual[:, 1:, 1:]))


def test_each_batch_row_uses_its_own_delay():
    mask = future_prefix_mask(generate_prefix_mask(torch.tensor([0, 1, 3]), 2, 5))
    assert mask[:, 0, :, 0].sum(dim=1).tolist() == [0, 1, 3]
    assert not mask[:, 1].any()
