"""Unit tests for neighbor regularization in ranked SFT and GRPO loss paths.

Tests that:
1. neighbor_reg_weight > 0 produces a non-zero regularization loss
2. Gradients flow only through the LoRA model (base model is no_grad)
3. neighbor_reg_only=True skips the neighbor SFT loss when reg is active
4. neighbor_reg_only=True falls back to neighbor SFT loss when reg can't run
5. B>1 handling in GRPO path (same-scene slice, mixed-scene rejection)
6. Batch dimension validation in compute_batched_trajectory_losses
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


class _StubExternalBase(nn.Module):
    """External frozen baseline anchor: returns a constant distinct from _StubDiT's
    outputs, and records the train/eval mode it was forwarded in (so tests can assert
    the decoder-train-mode toggle + its restoration)."""

    def __init__(self, P=5, T=80, const=0.5):
        super().__init__()
        self.P, self.T, self.const = P, T, const
        self.forwarded_in_training = None
        self.n_forward = 0

    def forward(self, inputs):
        B = inputs["ego_current_state"].shape[0]
        self.forwarded_in_training = self.training  # capture mode at forward time
        self.n_forward += 1
        out = torch.full(
            (B, self.P, self.T + 1, 4),
            self.const,
            device=inputs["ego_current_state"].device,
        )
        return None, {"model_output": out}


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
            model=model,
            model_args=model_args,
            data=data,
            ego_gt=ego_gt,
            neighbor_gt=neighbor_gt,
            neighbor_mask=neighbor_mask,
            device=torch.device("cpu"),
            K=1,
            neighbor_reg_weight=1.0,
            neighbor_reg_only=False,
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
            model=model,
            model_args=model_args,
            data=data,
            ego_gt=ego_gt,
            neighbor_gt=neighbor_gt,
            neighbor_mask=neighbor_mask,
            device=torch.device("cpu"),
            K=1,
            neighbor_reg_weight=1.0,
            neighbor_reg_only=True,
        )
        assert metrics["sft_neighbor_loss"] == 0.0, (
            "Neighbor SFT loss should be 0 when reg_only=True and reg is active"
        )

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
            model=model,
            model_args=model_args,
            data=data,
            ego_gt=ego_gt,
            neighbor_gt=neighbor_gt,
            neighbor_mask=neighbor_mask,
            device=torch.device("cpu"),
            K=1,
            neighbor_reg_weight=1.0,
            neighbor_reg_only=True,
        )
        # Should fall back: reg is 0 (can't run), but neighbor SFT should be non-zero
        assert metrics["sft_neighbor_reg_loss"] == 0.0, "Reg should be 0 without adapter"
        assert metrics["sft_neighbor_loss"] > 0, (
            "Should fall back to neighbor SFT when reg can't run"
        )

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
            model=model,
            model_args=model_args,
            data=data,
            ego_gt=ego_gt,
            neighbor_gt=neighbor_gt,
            neighbor_mask=neighbor_mask,
            device=torch.device("cpu"),
            K=1,
            neighbor_reg_weight=1.0,
            neighbor_reg_only=True,
        )
        loss.backward()
        # lora_delta should have a gradient (it's in the LoRA forward path)
        assert model.lora_delta.grad is not None, "LoRA param should have gradient"
        assert model.lora_delta.grad.abs().sum() > 0, "Gradient should be non-zero"

    def test_prefer_external_base_routes_and_restores_mode(self):
        """prefer_external_base=True routes the base pass through the EXTERNAL base
        model (not disable_adapter), forwards it in train mode (so the decoder
        returns 'model_output'), and restores its original eval mode afterward."""
        from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss

        model = _StubDiT(P=5, T=80)  # LoRA policy (has disable_adapter)
        base = _StubExternalBase(P=5, T=80)
        base.eval()  # frozen baseline starts in eval mode
        model_args = _make_model_args(P=5, T=80)
        data = _make_scene_data(B=1, P=5, T=80)

        ego_gt = torch.randn(1, 80, 4)
        neighbor_gt = torch.randn(1, 4, 80, 4)
        neighbor_mask = torch.zeros(1, 4, 80, dtype=torch.bool)

        _, metrics = _compute_sft_diffusion_loss(
            model=model,
            model_args=model_args,
            data=data,
            ego_gt=ego_gt,
            neighbor_gt=neighbor_gt,
            neighbor_mask=neighbor_mask,
            device=torch.device("cpu"),
            K=1,
            neighbor_reg_weight=1.0,
            neighbor_reg_only=True,
            base_model=base,
            prefer_external_base=True,
        )
        assert base.n_forward > 0, "External base must be used for the base pass"
        assert base.forwarded_in_training is True, (
            "Decoder dispatches on self.training; base must forward in train mode for 'model_output'"
        )
        assert base.training is False, (
            "External base train/eval mode must be restored (it started in eval)"
        )
        assert metrics["sft_neighbor_reg_loss"] > 0, "Reg loss should be non-zero"

    def test_default_does_not_use_external_base(self):
        """Default (prefer_external_base=False) keeps the disable_adapter path even
        when a base_model is supplied — byte-identical to prior behavior."""
        from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss

        model = _StubDiT(P=5, T=80)
        base = _StubExternalBase(P=5, T=80)
        base.eval()
        model_args = _make_model_args(P=5, T=80)
        data = _make_scene_data(B=1, P=5, T=80)

        ego_gt = torch.randn(1, 80, 4)
        neighbor_gt = torch.randn(1, 4, 80, 4)
        neighbor_mask = torch.zeros(1, 4, 80, dtype=torch.bool)

        _compute_sft_diffusion_loss(
            model=model,
            model_args=model_args,
            data=data,
            ego_gt=ego_gt,
            neighbor_gt=neighbor_gt,
            neighbor_mask=neighbor_mask,
            device=torch.device("cpu"),
            K=1,
            neighbor_reg_weight=1.0,
            neighbor_reg_only=True,
            base_model=base,
            prefer_external_base=False,
        )
        assert base.n_forward == 0, "Default must use disable_adapter, not the external base_model"

    def test_prefer_external_base_requires_base_model(self):
        """prefer_external_base=True with base_model=None must fail loudly."""
        from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss

        model = _StubDiT(P=5, T=80)
        model_args = _make_model_args(P=5, T=80)
        data = _make_scene_data(B=1, P=5, T=80)

        ego_gt = torch.randn(1, 80, 4)
        neighbor_gt = torch.randn(1, 4, 80, 4)
        neighbor_mask = torch.zeros(1, 4, 80, dtype=torch.bool)

        with pytest.raises(ValueError, match="requires an external base_model"):
            _compute_sft_diffusion_loss(
                model=model,
                model_args=model_args,
                data=data,
                ego_gt=ego_gt,
                neighbor_gt=neighbor_gt,
                neighbor_mask=neighbor_mask,
                device=torch.device("cpu"),
                K=1,
                neighbor_reg_weight=1.0,
                neighbor_reg_only=True,
                base_model=None,
                prefer_external_base=True,
            )


# ---------------------------------------------------------------------------
# Tests for GRPO path (grpo_loss.py)
# ---------------------------------------------------------------------------
class TestGRPONeighborReg:
    """Tests for _compute_neighbor_reg_loss in GRPO loss path."""

    def test_b_gt1_same_scene_slices_to_first(self):
        """Should handle B > 1 by slicing to first element when all entries are same scene."""
        from rlvr.grpo_loss import _compute_neighbor_reg_loss

        model = _StubDiT(P=5, T=80)
        # Build B=1 data then expand to B=4 (same scene repeated)
        data_b1 = _make_scene_data(B=1, P=5, T=80)
        data = {
            k: v.expand(4, *v.shape[1:]).contiguous() if isinstance(v, torch.Tensor) else v
            for k, v in data_b1.items()
        }
        model_args = _make_model_args(P=5, T=80)

        loss = _compute_neighbor_reg_loss(
            model,
            data,
            model_args,
            torch.device("cpu"),
            K=1,
            P=5,
            future_len=80,
        )
        assert isinstance(loss, torch.Tensor)

    def test_b_gt1_mixed_scenes_raises(self):
        """Should raise ValueError when B > 1 with different scenes."""
        from rlvr.grpo_loss import _compute_neighbor_reg_loss

        model = _StubDiT(P=5, T=80)
        data = _make_scene_data(B=4, P=5, T=80)  # random data = different scenes
        model_args = _make_model_args(P=5, T=80)

        with pytest.raises(ValueError, match="mixed scenes"):
            _compute_neighbor_reg_loss(
                model,
                data,
                model_args,
                torch.device("cpu"),
                K=1,
                P=5,
                future_len=80,
            )

    def test_reg_loss_nonzero(self):
        """Should produce non-zero loss for B=1."""
        from rlvr.grpo_loss import _compute_neighbor_reg_loss

        model = _StubDiT(P=5, T=80)
        data = _make_scene_data(B=1, P=5, T=80)
        model_args = _make_model_args(P=5, T=80)

        loss = _compute_neighbor_reg_loss(
            model,
            data,
            model_args,
            torch.device("cpu"),
            K=1,
            P=5,
            future_len=80,
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0, "Reg loss should be non-zero when LoRA changes output"

    def test_no_adapter_returns_zero(self):
        """Model without disable_adapter should return zero loss."""
        from rlvr.grpo_loss import _compute_neighbor_reg_loss

        model = nn.Linear(10, 10)
        data = _make_scene_data(B=1, P=5, T=80)
        model_args = _make_model_args(P=5, T=80)

        loss = _compute_neighbor_reg_loss(
            model,
            data,
            model_args,
            torch.device("cpu"),
            K=1,
            P=5,
            future_len=80,
        )
        assert loss.item() == 0.0


class TestBatchedTrajectoryLossesValidation:
    """Tests for data batch dimension validation in compute_batched_trajectory_losses."""

    def test_invalid_batch_dim_raises(self):
        """Should raise ValueError when data tensor has B != 1 and B != N."""
        from rlvr.grpo_loss import compute_batched_trajectory_losses

        model = _StubDiT(P=5, T=80)
        model_args = _make_model_args(P=5, T=80)
        N = 4
        trajectories = torch.randn(N, 80, 4)
        noise = torch.randn(1, 5, 80, 4)
        t = torch.tensor([0.5])
        # B=3 is neither 1 nor N=4
        data = _make_scene_data(B=3, P=5, T=80)

        with pytest.raises(ValueError, match="expected 1 or N=4"):
            compute_batched_trajectory_losses(
                model,
                data,
                trajectories,
                model_args,
                noise,
                t,
                torch.device("cpu"),
            )

    def test_non_batched_metadata_tensors_are_allowed(self):
        """Non-batched metadata tensors should not be rejected or reshaped."""
        from rlvr.grpo_loss import compute_batched_trajectory_losses

        model = _StubDiT(P=5, T=80)
        model_args = _make_model_args(P=5, T=80)
        N = 4
        trajectories = torch.randn(N, 80, 4)
        noise = torch.randn(1, 5, 80, 4)
        t = torch.tensor([0.5])

        data = _make_scene_data(B=1, P=5, T=80)
        data["ego_shape"] = torch.tensor([4.8, 2.0, 1.7])
        data["lane_geometry"] = torch.randn(8, 20, 2)

        losses = compute_batched_trajectory_losses(
            model,
            data,
            trajectories,
            model_args,
            noise,
            t,
            torch.device("cpu"),
        )
        assert losses is not None


# ---------------------------------------------------------------------------
# Tests for ego IL regularization
# ---------------------------------------------------------------------------
class TestEgoIL:
    """Tests for ego IL modes in _compute_sft_diffusion_loss."""

    def test_gt_mode_produces_nonzero_il_loss(self):
        """ego_il_weight > 0 with mode='gt' should produce a non-zero IL loss."""
        from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss

        model = _StubDiT(P=5, T=80)
        model_args = _make_model_args(P=5, T=80)
        data = _make_scene_data(B=1, P=5, T=80)

        ego_gt = torch.randn(1, 80, 4)
        neighbor_gt = torch.randn(1, 4, 80, 4)
        neighbor_mask = torch.zeros(1, 4, 80, dtype=torch.bool)
        ego_gt_real = torch.randn(1, 80, 4)  # different from ego_gt

        loss, metrics = _compute_sft_diffusion_loss(
            model=model,
            model_args=model_args,
            data=data,
            ego_gt=ego_gt,
            neighbor_gt=neighbor_gt,
            neighbor_mask=neighbor_mask,
            device=torch.device("cpu"),
            K=1,
            neighbor_reg_weight=0.0,
            neighbor_reg_only=False,
            ego_il_weight=1.0,
            ego_il_mode="gt",
            ego_gt_real=ego_gt_real,
        )
        assert metrics["sft_ego_il_loss"] > 0, "GT ego IL loss should be non-zero"
        assert loss.requires_grad

    def test_baseline_mode_produces_nonzero_il_loss(self):
        """ego_il_weight > 0 with mode='baseline' should produce a non-zero IL loss."""
        from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss

        model = _StubDiT(P=5, T=80)
        model_args = _make_model_args(P=5, T=80)
        data = _make_scene_data(B=1, P=5, T=80)

        ego_gt = torch.randn(1, 80, 4)
        neighbor_gt = torch.randn(1, 4, 80, 4)
        neighbor_mask = torch.zeros(1, 4, 80, dtype=torch.bool)

        loss, metrics = _compute_sft_diffusion_loss(
            model=model,
            model_args=model_args,
            data=data,
            ego_gt=ego_gt,
            neighbor_gt=neighbor_gt,
            neighbor_mask=neighbor_mask,
            device=torch.device("cpu"),
            K=1,
            neighbor_reg_weight=0.0,
            neighbor_reg_only=False,
            ego_il_weight=1.0,
            ego_il_mode="baseline",
        )
        assert metrics["sft_ego_il_loss"] > 0, "Baseline ego IL loss should be non-zero"

    def test_gt_mode_raises_without_ego_gt_real(self):
        """GT mode should raise ValueError when ego_gt_real is not provided."""
        from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss

        model = _StubDiT(P=5, T=80)
        model_args = _make_model_args(P=5, T=80)
        data = _make_scene_data(B=1, P=5, T=80)

        with pytest.raises(ValueError, match="ego_gt_real is required"):
            _compute_sft_diffusion_loss(
                model=model,
                model_args=model_args,
                data=data,
                ego_gt=torch.randn(1, 80, 4),
                neighbor_gt=torch.randn(1, 4, 80, 4),
                neighbor_mask=torch.zeros(1, 4, 80, dtype=torch.bool),
                device=torch.device("cpu"),
                K=1,
                ego_il_weight=1.0,
                ego_il_mode="gt",
                ego_gt_real=None,
            )

    def test_invalid_mode_raises(self):
        """Invalid ego_il_mode should raise ValueError."""
        from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss

        model = _StubDiT(P=5, T=80)
        model_args = _make_model_args(P=5, T=80)
        data = _make_scene_data(B=1, P=5, T=80)

        with pytest.raises(ValueError, match="ego_il_mode must be"):
            _compute_sft_diffusion_loss(
                model=model,
                model_args=model_args,
                data=data,
                ego_gt=torch.randn(1, 80, 4),
                neighbor_gt=torch.randn(1, 4, 80, 4),
                neighbor_mask=torch.zeros(1, 4, 80, dtype=torch.bool),
                device=torch.device("cpu"),
                K=1,
                ego_il_weight=1.0,
                ego_il_mode="invalid",
            )

    def test_il_disabled_when_weight_zero(self):
        """ego_il_weight=0 should produce zero IL loss."""
        from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss

        model = _StubDiT(P=5, T=80)
        model_args = _make_model_args(P=5, T=80)
        data = _make_scene_data(B=1, P=5, T=80)

        _, metrics = _compute_sft_diffusion_loss(
            model=model,
            model_args=model_args,
            data=data,
            ego_gt=torch.randn(1, 80, 4),
            neighbor_gt=torch.randn(1, 4, 80, 4),
            neighbor_mask=torch.zeros(1, 4, 80, dtype=torch.bool),
            device=torch.device("cpu"),
            K=1,
            ego_il_weight=0.0,
            ego_il_mode="gt",
        )
        assert metrics["sft_ego_il_loss"] == 0.0

    def test_baseline_il_with_neighbor_reg_shares_forward_pass(self):
        """Baseline IL + neighbor reg should both work (shared base forward pass)."""
        from rlvr.grpo_sft_trainer import _compute_sft_diffusion_loss

        model = _StubDiT(P=5, T=80)
        model_args = _make_model_args(P=5, T=80)
        data = _make_scene_data(B=1, P=5, T=80)

        loss, metrics = _compute_sft_diffusion_loss(
            model=model,
            model_args=model_args,
            data=data,
            ego_gt=torch.randn(1, 80, 4),
            neighbor_gt=torch.randn(1, 4, 80, 4),
            neighbor_mask=torch.zeros(1, 4, 80, dtype=torch.bool),
            device=torch.device("cpu"),
            K=1,
            neighbor_reg_weight=1.0,
            neighbor_reg_only=True,
            ego_il_weight=0.5,
            ego_il_mode="baseline",
        )
        assert metrics["sft_ego_il_loss"] > 0, "Baseline IL should work alongside neighbor reg"
        assert metrics["sft_neighbor_reg_loss"] > 0, (
            "Neighbor reg should work alongside baseline IL"
        )


# ---------------------------------------------------------------------------
# Tests for config validation
# ---------------------------------------------------------------------------
class TestConfigValidation:
    """Tests for GRPOConfig field validation."""

    def test_invalid_ego_il_mode(self):
        from rlvr.grpo_config import GRPOConfig

        with pytest.raises(ValueError, match="ego_il_mode"):
            GRPOConfig(ego_il_mode="bad")

    def test_invalid_selective_mode(self):
        from rlvr.grpo_config import GRPOConfig

        with pytest.raises(ValueError, match="selective_mode"):
            GRPOConfig(selective_mode="bad")

    def test_invalid_neighbor_reg_anchor(self):
        from rlvr.grpo_config import GRPOConfig

        with pytest.raises(ValueError, match="neighbor_reg_anchor"):
            GRPOConfig(
                neighbor_reg_anchor="basline"
            )  # typo must fail loudly, not silently warmstart

    def test_valid_modes_pass(self):
        from rlvr.grpo_config import GRPOConfig

        c = GRPOConfig(
            ego_il_mode="baseline", selective_mode="advantage", neighbor_reg_anchor="baseline"
        )
        assert c.ego_il_mode == "baseline"
        assert c.selective_mode == "advantage"
        assert c.neighbor_reg_anchor == "baseline"

    def test_schedule_constant_no_end(self):
        from rlvr.grpo_config import GRPOConfig

        c = GRPOConfig(schedules={"speed_stretch": {"type": "constant", "start": 1.1}})
        val = c.get_scheduled_value("speed_stretch", 5, 30)
        assert val == 1.1

    def test_schedule_linear_requires_end(self):
        from rlvr.grpo_config import GRPOConfig

        c = GRPOConfig(schedules={"w_progress": {"type": "linear", "start": 0.0}})
        with pytest.raises(ValueError, match="requires 'end'"):
            c.get_scheduled_value("w_progress", 5, 30)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
