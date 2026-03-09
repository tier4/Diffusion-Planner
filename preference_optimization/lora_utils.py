"""LoRA utilities for DPO (and future GRPO) training on Diffusion_Planner.

Uses HuggingFace PEFT. Requires: pip install peft

The LoRA targets are restricted to DiT decoder attention projections via a regex
pattern. This prevents accidentally targeting the encoder's nn.MultiheadAttention
out_proj layers (which are also named out_proj but live under model.encoder.*).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model


# Regex pattern matching only the four projection linears inside
# decoder.dit_blocks (both self-attention and cross-attention).
# Format: decoder.blocks.{i}.{attn|cross_attn}.{q|k|v|out}_proj
# Using a regex string (not a list) so PEFT matches against the full module path.
LORA_TARGET_MODULES_REGEX = r"decoder\..*\.(q_proj|k_proj|v_proj|out_proj)"


def apply_lora(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: str = LORA_TARGET_MODULES_REGEX,
) -> nn.Module:
    """Wrap model with PEFT LoRA adapters targeting DiT decoder attention projections.

    After this call:
    - All original parameters: requires_grad=False (frozen base)
    - LoRA A matrices: requires_grad=True
    - LoRA B matrices: requires_grad=True (initialized to zero → identity delta at start)

    Args:
        model:          Diffusion_Planner instance (with UnfusedMultiheadAttention in DiT)
        r:              LoRA rank. Lower rank → less capacity and less catastrophic forgetting.
        lora_alpha:     LoRA alpha. Effective weight scaling = alpha / r.
        lora_dropout:   Dropout probability on LoRA activations.
        target_modules: Regex or list of module name patterns to adapt. Defaults to
                        decoder-only attention projections to avoid touching the encoder.

    Returns:
        PEFT-wrapped model with only LoRA parameters trainable.
    """
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def save_lora_checkpoint(model: nn.Module, save_dir: str) -> None:
    """Save only LoRA adapter weights (adapter_model.bin + adapter_config.json).

    The base model weights are not saved — load the original checkpoint and then
    call load_lora_checkpoint() to reconstruct the fine-tuned model.

    Handles DDP-wrapped models by unwrapping via .module before saving.
    """
    os.makedirs(save_dir, exist_ok=True)
    inner = model.module if hasattr(model, "module") else model
    inner.save_pretrained(save_dir)


def load_lora_checkpoint(base_model: nn.Module, lora_dir: str, is_trainable: bool = False) -> nn.Module:
    """Load LoRA adapter weights on top of a freshly loaded base model.

    Args:
        base_model:    Diffusion_Planner instance with base weights already loaded.
        lora_dir:      Directory containing adapter_config.json and adapter weights.
        is_trainable:  If True, keeps LoRA parameters trainable (use for continued
                       training). If False (default), freezes all weights (inference).

    Usage:
        # Inference
        base_model, model_args = load_model(base_path, device)
        model = load_lora_checkpoint(base_model, lora_checkpoint_dir)

        # Resume training
        model = load_lora_checkpoint(base_model, lora_checkpoint_dir, is_trainable=True)
    """
    return PeftModel.from_pretrained(base_model, lora_dir, is_trainable=is_trainable)


def merge_lora_and_unload(model: nn.Module) -> nn.Module:
    """Merge LoRA delta weights into base model weights and return a plain model.

    Applies W_merged = W_base + (alpha/r) * B @ A per adapted layer, then removes
    all PEFT scaffolding. Use before ONNX export or deployment.

    Handles DDP-wrapped models by unwrapping via .module before merging.
    """
    inner = model.module if hasattr(model, "module") else model
    return inner.merge_and_unload()
