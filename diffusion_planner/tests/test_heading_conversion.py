import pytest
import torch
from diffusion_planner.train_epoch import heading_to_cos_sin


def test_heading_conversion_preserves_a_valid_origin_pose():
    raw = torch.zeros(1, 2, 3)

    converted = heading_to_cos_sin(raw)

    torch.testing.assert_close(converted[..., :2], torch.zeros(1, 2, 2))
    torch.testing.assert_close(converted[..., 2], torch.ones(1, 2))
    torch.testing.assert_close(converted[..., 3], torch.zeros(1, 2))


def test_heading_conversion_clones_four_column_inputs():
    raw = torch.zeros(1, 2, 4)

    converted = heading_to_cos_sin(raw)
    converted[..., 0] = 1.0

    assert converted.data_ptr() != raw.data_ptr()
    assert torch.count_nonzero(raw) == 0


def test_heading_conversion_rejects_invalid_feature_width():
    with pytest.raises(ValueError, match="last dimension 3 or 4"):
        heading_to_cos_sin(torch.zeros(1, 2, 5))
