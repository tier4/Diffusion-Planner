"""TurnIndicatorDropoutAugmentation: zero the turn indicator input per sample
while preserving the original sequence for the loss GT."""

import torch
from diffusion_planner.loss import make_turn_indicator_gt
from diffusion_planner.utils.data_augmentation import TurnIndicatorDropoutAugmentation

B, T = 4, 6


def _make_batch():
    inputs = {
        "turn_indicators": torch.tensor(
            [
                [1, 1, 2, 2, 2, 2],  # keeps LEFT -> gt KEEP
                [1, 1, 1, 1, 1, 2],  # changes to LEFT -> gt LEFT
                [1, 1, 1, 3, 3, 3],  # keeps RIGHT -> gt KEEP
                [3, 3, 3, 3, 3, 1],  # changes to DISABLE -> gt DISABLE
            ]
        ),
    }
    ego_future = torch.randn(B, 5, 3)
    neighbors_future = torch.randn(B, 3, 5, 3)
    return inputs, ego_future, neighbors_future


def test_dropout_zeroes_input_but_preserves_gt_source():
    inputs, ego_future, neighbors_future = _make_batch()
    orig = inputs["turn_indicators"].clone()

    aug = TurnIndicatorDropoutAugmentation(dropout_prob=1.0, device="cpu")
    inputs, _, _ = aug(inputs, ego_future, neighbors_future)

    assert torch.equal(inputs["turn_indicators"], torch.zeros_like(orig))
    assert torch.equal(inputs["turn_indicators_gt_source"], orig)


def test_gt_from_source_matches_original_gt():
    inputs, ego_future, neighbors_future = _make_batch()
    gt_before = make_turn_indicator_gt(inputs["turn_indicators"])

    aug = TurnIndicatorDropoutAugmentation(dropout_prob=1.0, device="cpu")
    inputs, _, _ = aug(inputs, ego_future, neighbors_future)

    # the loss reads turn_indicators_gt_source when present
    source = inputs.get("turn_indicators_gt_source", inputs["turn_indicators"])
    assert torch.equal(make_turn_indicator_gt(source), gt_before)
    # zeroed input alone would corrupt the GT — that is exactly what the
    # gt_source key guards against
    assert not torch.equal(make_turn_indicator_gt(inputs["turn_indicators"]), gt_before)


def test_dropout_prob_zero_keeps_input():
    inputs, ego_future, neighbors_future = _make_batch()
    orig = inputs["turn_indicators"].clone()

    aug = TurnIndicatorDropoutAugmentation(dropout_prob=0.0, device="cpu")
    inputs, _, _ = aug(inputs, ego_future, neighbors_future)

    assert torch.equal(inputs["turn_indicators"], orig)
    assert torch.equal(inputs["turn_indicators_gt_source"], orig)


def test_dropout_is_per_sample():
    torch.manual_seed(0)
    inputs, ego_future, neighbors_future = _make_batch()
    orig = inputs["turn_indicators"].clone()

    aug = TurnIndicatorDropoutAugmentation(dropout_prob=0.5, device="cpu")
    inputs, _, _ = aug(inputs, ego_future, neighbors_future)

    for b in range(B):
        row = inputs["turn_indicators"][b]
        zeroed = torch.equal(row, torch.zeros_like(row))
        kept = torch.equal(row, orig[b])
        # each sample is either fully zeroed or fully kept
        assert zeroed or kept
    # gt source is always the original, regardless of the drop decision
    assert torch.equal(inputs["turn_indicators_gt_source"], orig)
