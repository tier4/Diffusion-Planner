"""Unit tests for neighbor regularization in ranked SFT and GRPO loss paths.

Tests that:
1. neighbor_reg_weight > 0 produces a non-zero regularization loss
2. Gradients flow only through the LoRA model (base model is no_grad)
3. neighbor_reg_only=True skips the neighbor SFT loss when reg is active
4. neighbor_reg_only=True falls back to neighbor SFT loss when reg can't run
5. B=1 assertion in GRPO path
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Stub model that mimics PeftModel with disable_adapter()
# ---------------------------------------------------------------------------
class _StubDiT(nn.Module):
    """Minimal model that returns different outputs with/without LoRA."""

    def __init__(self, P=5, T=80):
        super().__init__()
        self.P = P
        self.T = T
        # "LoRA" parameter — when disabled, output shifts
        self.lora_delta = nn.Parameter(torch.ones(1) * 0.1)
        self._adapter_disabled = False

    @contextlib.contextmanager
    def disable_adapter(self):
        self._adapter_disabled = True
        try:
            yield
        finally:
            self._adapter_disabled = False

    def forward(self, inputs):
        B = inputs["ego_current_state"].shape[0]
        P, T = self.P, self.T
        # Base output: zeros
        output = torch.zeros(B, P, T + 1, 4, device=inputs["ego_current_state"].device)
        if not self._adapter_disabled:
            # LoRA adds a small delta to all outputs
            output = output + self.lora_delta
        return None, {"model_output": output}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_model_args(P=5, T=80):
    """Create a minimal model_args mock."""
    args = MagicMock()
    args.predicted_neighbor_num = P - 1
    args.future_len = T

    # State normalizer (identity)
    norm = MagicMock()
    norm.mean = [torch.zeros(4)]
    norm.std = [torch.ones(4)]
    args.state_normalizer = norm

    # Observation normalizer (pass-through)
    args.observation_normalizer = lambda x: x

    return args


def _make_scene_data(B=1, P=5, T=80, device="cpu"):
    """Create minimal scene data dict."""
    Pn = P - 1
    data = {
        "ego_current_state": torch.randn(B, 10, device=device),
        "neighbor_agents_past": torch.randn(B, Pn, 31, 11, device=device),
        "neighbor_agents_future": torch.randn(B, Pn, T, 3, device=device),
        "ego_agent_future": torch.randn(B, T, 4, device=device),
    }
    # Make neighbor data non-zero so validity checks pass
    data["neighbor_agents_past"][:, :, -1, :4] = torch.ones(B, Pn, 4, device=device) * 0.5
    data["neighbor_agents_future"][:, :, :, :2] = torch.ones(B, Pn, T, 2, device=device) * 0.5
    return data


# ---------------------------------------------------------------------------
# Tests for ranked SFT path (grpo_sft_trainer.py)
# ---------------------------------------------------------------------------
class TestRankedSFTNeighborReg:
    """Tests for _compute_sft_diffusion_loss neighbor regularization."""

    def test_reg_produces_nonzero_loss(self):
        """neighbor_reg_weight > 0 should produce a non-zero reg loss."""
        from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss

        model = _StubDiT(P=5, T=80)
        model_args = _make_model_args(P=5, T=80)
        data = _make_scene_data(B=1, P=5, T=80)

        ego_gt = torch.randn(1, 80, 4)
        neighbor_gt = torch.randn(1, 4, 80, 4)
        neighbor_mask = torch.zeros(1, 4, 80, dtype=torch.bool)

        loss, metrics = _compute_sft_diffusion_loss(
            model=model, model_args=model_args, data=data,
            ego_gt=ego_gt, neighbor_gt=neighbor_gt, neighbor_mask=neighbor_mask,
            device=torch.device("cpu"), K=1,
            neighbor_reg_weight=1.0, neighbor_reg_only=False,
        )
        assert metrics["sft_neighbor_reg_loss"] > 0, "Reg loss should be non-zero"
        assert loss.requires_grad, "Loss should have gradients"

    def test_reg_only_skips_neighbor_sft(self):
        """neighbor_reg_only=True should zero out the neighbor SFT loss."""
        from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss

        model = _StubDiT(P=5, T=80)
        model_args = _make_model_args(P=5, T=80)
        data = _make_scene_data(B=1, P=5, T=80)

        ego_gt = torch.randn(1, 80, 4)
        neighbor_gt = torch.randn(1, 4, 80, 4)
        neighbor_mask = torch.zeros(1, 4, 80, dtype=torch.bool)

        _, metrics = _compute_sft_diffusion_loss(
            model=model, model_args=model_args, data=data,
            ego_gt=ego_gt, neighbor_gt=neighbor_gt, neighbor_mask=neighbor_mask,
            device=torch.device("cpu"), K=1,
            neighbor_reg_weight=1.0, neighbor_reg_only=True,
        )
        assert metrics["sft_neighbor_loss"] == 0.0, \
            "Neighbor SFT loss should be 0 when reg_only=True and reg is active"

    def test_reg_only_fallback_without_adapter(self):
        """When model lacks disable_adapter, reg_only should fall back to neighbor SFT."""
        from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss

        # Plain model without disable_adapter
        model = nn.Linear(10, 10)

        # Override forward to return expected format
        P, T = 5, 80

        def fake_forward(inputs):
            B = 1
            output = torch.zeros(B, P, T + 1, 4)
            return None, {"model_output": output}

        model.forward = fake_forward

        model_args = _make_model_args(P=P, T=T)
        data = _make_scene_data(B=1, P=P, T=T)

        ego_gt = torch.randn(1, T, 4)
        neighbor_gt = torch.randn(1, P - 1, T, 4)
        neighbor_mask = torch.zeros(1, P - 1, T, dtype=torch.bool)

        _, metrics = _compute_sft_diffusion_loss(
            model=model, model_args=model_args, data=data,
            ego_gt=ego_gt, neighbor_gt=neighbor_gt, neighbor_mask=neighbor_mask,
            device=torch.device("cpu"), K=1,
            neighbor_reg_weight=1.0, neighbor_reg_only=True,
        )
        # Should fall back: reg is 0 (can't run), but neighbor SFT should be non-zero
        assert metrics["sft_neighbor_reg_loss"] == 0.0, "Reg should be 0 without adapter"
        assert metrics["sft_neighbor_loss"] > 0, \
            "Should fall back to neighbor SFT when reg can't run"

    def test_gradients_only_through_lora(self):
        """Base model forward (disable_adapter) should be no_grad."""
        from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss

        model = _StubDiT(P=5, T=80)
        model_args = _make_model_args(P=5, T=80)
        data = _make_scene_data(B=1, P=5, T=80)

        ego_gt = torch.randn(1, 80, 4)
        neighbor_gt = torch.randn(1, 4, 80, 4)
        neighbor_mask = torch.zeros(1, 4, 80, dtype=torch.bool)

        model.zero_grad()
        loss, _ = _compute_sft_diffusion_loss(
            model=model, model_args=model_args, data=data,
            ego_gt=ego_gt, neighbor_gt=neighbor_gt, neighbor_mask=neighbor_mask,
            device=torch.device("cpu"), K=1,
            neighbor_reg_weight=1.0, neighbor_reg_only=True,
        )
        loss.backward()
        # lora_delta should have a gradient (it's in the LoRA forward path)
        assert model.lora_delta.grad is not None, "LoRA param should have gradient"
        assert model.lora_delta.grad.abs().sum() > 0, "Gradient should be non-zero"


# ---------------------------------------------------------------------------
# Tests for GRPO path (grpo_loss.py)
# ---------------------------------------------------------------------------
class TestGRPONeighborReg:
    """Tests for _compute_neighbor_reg_loss in GRPO loss path."""

    def test_b1_assertion(self):
        """Should raise AssertionError when B > 1."""
        from rlvr.grpo_loss import _compute_neighbor_reg_loss
        from rlvr.grpo_config import GRPOConfig

        model = _StubDiT(P=5, T=80)
        data = _make_scene_data(B=2, P=5, T=80)  # B=2 should fail
        model_args = _make_model_args(P=5, T=80)
        config = GRPOConfig(neighbor_reg_weight=1.0)

        with pytest.raises(AssertionError, match="B=1"):
            _compute_neighbor_reg_loss(
                model, data, model_args, torch.device("cpu"),
                K=1, P=5, future_len=80, config=config,
            )

    def test_reg_loss_nonzero(self):
        """Should produce non-zero loss for B=1."""
        from rlvr.grpo_loss import _compute_neighbor_reg_loss
        from rlvr.grpo_config import GRPOConfig

        model = _StubDiT(P=5, T=80)
        data = _make_scene_data(B=1, P=5, T=80)
        model_args = _make_model_args(P=5, T=80)
        config = GRPOConfig(neighbor_reg_weight=1.0)

        loss = _compute_neighbor_reg_loss(
            model, data, model_args, torch.device("cpu"),
            K=1, P=5, future_len=80, config=config,
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0, "Reg loss should be non-zero when LoRA changes output"

    def test_no_adapter_returns_zero(self):
        """Model without disable_adapter should return zero loss."""
        from rlvr.grpo_loss import _compute_neighbor_reg_loss
        from rlvr.grpo_config import GRPOConfig

        model = nn.Linear(10, 10)
        data = _make_scene_data(B=1, P=5, T=80)
        model_args = _make_model_args(P=5, T=80)
        config = GRPOConfig(neighbor_reg_weight=1.0)

        loss = _compute_neighbor_reg_loss(
            model, data, model_args, torch.device("cpu"),
            K=1, P=5, future_len=80, config=config,
        )
        assert loss.item() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
