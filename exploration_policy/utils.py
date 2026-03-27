"""Utility functions for the Exploration Policy.

Provides helpers to:
1. Extract the frozen encoder from a (possibly LoRA-wrapped) planner model.
2. Generate a reference trajectory using the base SFT model (LoRA disabled).
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

from guidance_gui.generate_samples import generate_samples


def get_inner_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP/DataParallel to get the inner model."""
    return model.module if hasattr(model, "module") else model


def get_frozen_encoder(model: torch.nn.Module) -> torch.nn.Module:
    """Extract the encoder from a planner model (PeftModel or base).

    The encoder has no LoRA layers, so it produces the same output regardless
    of whether adapters are enabled or disabled. We extract it for direct use
    without going through the full forward pass.

    Args:
        model: Diffusion_Planner instance, possibly wrapped by PeftModel and/or DDP.

    Returns:
        The Encoder module (not a copy — shares weights with the model).
    """
    inner = get_inner_model(model)

    # PeftModel wraps the base model: inner.base_model.model is Diffusion_Planner
    if hasattr(inner, "base_model") and hasattr(inner.base_model, "model"):
        planner = inner.base_model.model
    else:
        planner = inner

    return planner.encoder


def run_frozen_encoder(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run the frozen encoder to get scene encoding.

    Args:
        model: Diffusion_Planner instance (possibly LoRA-wrapped).
        data: Normalized observation dict (B=1).

    Returns:
        scene_encoding: [B, N, D_enc] tensor (detached, no grad).
    """
    encoder = get_frozen_encoder(model)
    with torch.no_grad():
        scene_encoding = encoder(data)
    return scene_encoding.detach()


def generate_reference_trajectory(
    model: torch.nn.Module,
    model_args,
    data: dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    """Generate a deterministic reference trajectory using the base SFT model.

    Disables LoRA adapters (if present) to get the frozen base model's output,
    which serves as x_ref for the exploration policy.

    Args:
        model: Diffusion_Planner instance (possibly LoRA-wrapped).
        model_args: Config object from load_model.
        data: Normalized observation dict (B=1).
        device: Torch device.

    Returns:
        x_ref: (T, 4) numpy array — reference trajectory [x, y, cos, sin].
    """
    inner = get_inner_model(model)
    use_lora_disable = hasattr(inner, "disable_adapter")

    disable_ctx = inner.disable_adapter() if use_lora_disable else contextlib.nullcontext()

    with disable_ctx, torch.no_grad():
        samples = generate_samples(
            model=model,
            model_args=model_args,
            data=data,
            noise_scale=0.0,
            n_samples=1,
            composer=None,
            device=device,
        )

    return samples[0]  # (T, 4)
