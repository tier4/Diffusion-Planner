import pytest
import torch
from diffusion_planner.model.module.encoder import _keep_recent_history


def test_keep_recent_history_uses_current_and_previous_frames():
    history = torch.arange(21 * 4, dtype=torch.float32).reshape(1, 21, 4)

    selected = _keep_recent_history(history, 6)

    assert selected.shape == history.shape
    torch.testing.assert_close(selected[:, :-6], torch.zeros(1, 15, 4))
    torch.testing.assert_close(selected[:, -6:], history[:, -6:])


def test_keep_recent_history_rejects_invalid_window():
    history = torch.zeros(1, 6, 4)

    with pytest.raises(ValueError, match="history_frames must be"):
        _keep_recent_history(history, 7)
