"""Tests for KL (output regularization) loss in curated RSFT.

Verifies that:
1. kl_coef=0 produces zero KL loss (default, backward compatible)
2. kl_coef>0 with base_model produces non-zero KL loss
3. kl_coef>0 without base_model or LoRA raises ValueError
"""

import pytest
import torch
import torch.nn as nn


class _FakeModelArgs:
    """Minimal model_args stub for _compute_sft_diffusion_loss."""

    def __init__(self, predicted_neighbor_num: int = 2, future_len: int = 10):
        self.predicted_neighbor_num = predicted_neighbor_num
        self.future_len = future_len

        class _Norm:
            def __init__(self):
                P = 1 + predicted_neighbor_num
                self.mean = torch.zeros(P, 4)
                self.std = torch.ones(P, 4)

        self.state_normalizer = _Norm()

        class _ObsNorm:
            def __call__(self, d):
                return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in d.items()}

        self.observation_normalizer = _ObsNorm()


class _DummyModel(nn.Module):
    """Returns random output with correct shape."""

    def __init__(self, P: int, T: int):
        super().__init__()
        self._P = P
        self._T = T
        self.linear = nn.Linear(4, 4)

    def forward(self, inputs):
        B = inputs["ego_current_state"].shape[0]
        out = torch.randn(B, self._P, self._T + 1, 4, device=inputs["ego_current_state"].device)
        return None, {"model_output": out}


def _make_batch(B: int = 2, Pn: int = 2, T: int = 10, device: str = "cpu"):
    """Create a minimal batch for _compute_sft_diffusion_loss."""
    data = {
        "ego_current_state": torch.randn(B, 10, device=device),
        "neighbor_agents_past": torch.randn(B, Pn, 31, 11, device=device),
    }
    ego_gt = torch.randn(B, T, 4, device=device)
    neighbor_gt = torch.randn(B, Pn, T, 4, device=device)
    neighbor_mask = torch.zeros(B, Pn, T, dtype=torch.bool, device=device)
    return data, ego_gt, neighbor_gt, neighbor_mask


def test_kl_zero_by_default():
    """kl_coef=0 should produce sft_kl_loss=0."""
    from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss

    B, Pn, T = 2, 2, 10
    model_args = _FakeModelArgs(predicted_neighbor_num=Pn, future_len=T)
    model = _DummyModel(1 + Pn, T)
    data, ego_gt, neighbor_gt, neighbor_mask = _make_batch(B, Pn, T)

    loss, metrics = _compute_sft_diffusion_loss(
        model,
        model_args,
        data,
        ego_gt,
        neighbor_gt,
        neighbor_mask,
        device=torch.device("cpu"),
        K=1,
        kl_coef=0.0,
    )
    assert metrics["sft_kl_loss"] == 0.0


def test_kl_nonzero_with_base_model():
    """kl_coef>0 with base_model should produce non-zero KL loss."""
    from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss

    B, Pn, T = 2, 2, 10
    model_args = _FakeModelArgs(predicted_neighbor_num=Pn, future_len=T)
    model = _DummyModel(1 + Pn, T)
    base_model = _DummyModel(1 + Pn, T)
    data, ego_gt, neighbor_gt, neighbor_mask = _make_batch(B, Pn, T)

    loss, metrics = _compute_sft_diffusion_loss(
        model,
        model_args,
        data,
        ego_gt,
        neighbor_gt,
        neighbor_mask,
        device=torch.device("cpu"),
        K=1,
        kl_coef=0.1,
        base_model=base_model,
    )
    assert metrics["sft_kl_loss"] > 0.0


def test_kl_no_base_model_raises():
    """kl_coef>0 without base_model or LoRA should raise ValueError."""
    from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss

    B, Pn, T = 2, 2, 10
    model_args = _FakeModelArgs(predicted_neighbor_num=Pn, future_len=T)
    model = _DummyModel(1 + Pn, T)
    data, ego_gt, neighbor_gt, neighbor_mask = _make_batch(B, Pn, T)

    with pytest.raises(ValueError, match="kl_coef"):
        _compute_sft_diffusion_loss(
            model,
            model_args,
            data,
            ego_gt,
            neighbor_gt,
            neighbor_mask,
            device=torch.device("cpu"),
            K=1,
            kl_coef=0.1,
            base_model=None,
        )
