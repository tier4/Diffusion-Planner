"""LoRA utilities for DPO (and future GRPO) training on Diffusion_Planner.

Uses HuggingFace PEFT. Requires: pip install peft

PEFT's MultiheadAttentionLoRA (used when targeting nn.MultiheadAttention directly)
implements a fragile merge→forward→unmerge cycle on every forward call: it deletes
in_proj_weight and re-registers it as a plain tensor, which corrupts internal state
after extended training. To avoid this, apply_lora() first replaces each DiT block's
nn.MultiheadAttention with UnfusedMHA — a numerically identical module that exposes
separate q_proj/k_proj/v_proj/out_proj nn.Linear layers. PEFT then applies its
stable LinearLoRA to those layers instead of the fragile MHA path.

The regex targets those four Linear sub-layers within decoder DiT blocks, leaving the
encoder's attention modules (which also have out_proj) completely untouched.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model


# Regex targeting q/k/v/out_proj Linear layers inside DiT decoder blocks.
# PEFT matches this via re.fullmatch against the full dotted module path.
# UnfusedMHA (defined below) exposes these four sub-layers; apply_lora()
# replaces nn.MultiheadAttention with UnfusedMHA before calling get_peft_model().
LORA_TARGET_MODULES_REGEX = (
    r"decoder\.dit\.blocks\.[0-9]+\.(attn|cross_attn)\.(q_proj|k_proj|v_proj|out_proj)"
)


class UnfusedMHA(nn.Module):
    """nn.MultiheadAttention equivalent with separate q/k/v/out projection Linear layers.

    Numerically identical to nn.MultiheadAttention but exposes four named nn.Linear
    sub-modules (q_proj, k_proj, v_proj, out_proj) instead of the fused in_proj_weight
    tensor. This allows PEFT to apply its stable LinearLoRA to each projection
    independently, avoiding the fragile merge/unmerge cycle in MultiheadAttentionLoRA.

    Args:
        embed_dim:   Total embedding dimension (must equal num_heads * head_dim).
        num_heads:   Number of attention heads.
        dropout:     Dropout probability on attention weights (active only in training).
        batch_first: If True, input/output tensors are [B, T, D]; otherwise [T, B, D].
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    @classmethod
    def from_mha(cls, mha: nn.MultiheadAttention) -> "UnfusedMHA":
        """Construct UnfusedMHA from an nn.MultiheadAttention, copying all weights.

        Splits the fused in_proj_weight [3D, D] into three D×D slices and copies
        out_proj weights directly.
        """
        D = mha.embed_dim
        module = cls(
            embed_dim=D,
            num_heads=mha.num_heads,
            dropout=mha.dropout,
            batch_first=mha.batch_first,
        ).to(device=mha.in_proj_weight.device, dtype=mha.in_proj_weight.dtype)
        with torch.no_grad():
            module.q_proj.weight.copy_(mha.in_proj_weight[:D])
            module.k_proj.weight.copy_(mha.in_proj_weight[D : 2 * D])
            module.v_proj.weight.copy_(mha.in_proj_weight[2 * D :])
            if mha.in_proj_bias is not None:
                module.q_proj.bias.copy_(mha.in_proj_bias[:D])
                module.k_proj.bias.copy_(mha.in_proj_bias[D : 2 * D])
                module.v_proj.bias.copy_(mha.in_proj_bias[2 * D :])
            module.out_proj.weight.copy_(mha.out_proj.weight)
            module.out_proj.bias.copy_(mha.out_proj.bias)
        return module

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Multi-head scaled dot-product attention.

        Calls q_proj/k_proj/v_proj as nn.Linear modules so that any LoRA delta
        attached by PEFT is transparently applied to the projections.

        Args:
            query:            [B, T_q, D] if batch_first else [T_q, B, D]
            key:              [B, T_k, D] if batch_first else [T_k, B, D]
            value:            [B, T_k, D] if batch_first else [T_k, B, D]
            key_padding_mask: [B, T_k] bool; True positions are ignored.
            attn_mask:        Not used (kept for API compatibility).
            need_weights:     Not used (weights are never returned).

        Returns:
            Tuple of (attn_output, None) matching nn.MultiheadAttention return type.
        """
        if self.batch_first:
            query = query.transpose(0, 1)  # [T_q, B, D]
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        T_q, B, _ = query.shape
        T_k = key.shape[0]
        H, Hd = self.num_heads, self.head_dim

        # Project; LoRA delta is applied transparently inside each nn.Linear.forward
        q = self.q_proj(query)  # [T_q, B, D]
        k = self.k_proj(key)    # [T_k, B, D]
        v = self.v_proj(value)  # [T_k, B, D]

        # Split heads: [T, B, D] -> [B, H, T, Hd]
        q = q.view(T_q, B, H, Hd).permute(1, 2, 0, 3)
        k = k.view(T_k, B, H, Hd).permute(1, 2, 0, 3)
        v = v.view(T_k, B, H, Hd).permute(1, 2, 0, 3)

        # Scaled dot-product attention: [B, H, T_q, T_k]
        attn = torch.matmul(q, k.transpose(-2, -1)) * (Hd ** -0.5)

        if key_padding_mask is not None:
            # [B, T_k] -> [B, 1, 1, T_k] for broadcasting
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        if self.training and self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout)

        # [B, H, T_q, Hd] -> [T_q, B, D]
        out = torch.matmul(attn, v).permute(2, 0, 1, 3).reshape(T_q, B, H * Hd)
        out = self.out_proj(out)

        if self.batch_first:
            out = out.transpose(0, 1)

        return out, None


def _replace_dit_mha_with_unfused(model: nn.Module) -> None:
    """Replace nn.MultiheadAttention in DiT decoder blocks with UnfusedMHA in-place.

    Iterates model.decoder.dit.blocks and replaces each block's .attn and .cross_attn
    with an UnfusedMHA that carries the same weights. After this call, PEFT targets
    the constituent nn.Linear layers with stable LinearLoRA instead of the fragile
    MultiheadAttentionLoRA.

    Handles DDP-wrapped models via .module unwrapping.
    """
    inner = model.module if hasattr(model, "module") else model
    for block in inner.decoder.dit.blocks:
        if isinstance(block.attn, nn.MultiheadAttention):
            block.attn = UnfusedMHA.from_mha(block.attn)
        if isinstance(block.cross_attn, nn.MultiheadAttention):
            block.cross_attn = UnfusedMHA.from_mha(block.cross_attn)


def apply_lora(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: str = LORA_TARGET_MODULES_REGEX,
) -> nn.Module:
    """Wrap model with PEFT LoRA adapters targeting DiT decoder attention projections.

    Replaces nn.MultiheadAttention in DiT blocks with UnfusedMHA first, then applies
    LinearLoRA to q_proj/k_proj/v_proj/out_proj. This avoids PEFT's fragile
    MultiheadAttentionLoRA which corrupts in_proj_weight after extended training.

    After this call:
    - All original parameters: requires_grad=False (frozen base)
    - LoRA A matrices: requires_grad=True
    - LoRA B matrices: requires_grad=True (initialized to zero → identity delta at start)

    Args:
        model:          Diffusion_Planner instance
        r:              LoRA rank. Lower rank → less capacity and less catastrophic forgetting.
        lora_alpha:     LoRA alpha. Effective weight scaling = alpha / r.
        lora_dropout:   Dropout probability on LoRA activations.
        target_modules: Regex matched via re.fullmatch against full module path. Defaults to
                        q/k/v/out_proj inside decoder DiT blocks only.

    Returns:
        PEFT-wrapped model with only LoRA parameters trainable.
    """
    _replace_dit_mha_with_unfused(model)
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


def load_lora_checkpoint(
    base_model: nn.Module, lora_dir: str, is_trainable: bool = False
) -> nn.Module:
    """Load LoRA adapter weights on top of a freshly loaded base model.

    Replaces nn.MultiheadAttention in DiT blocks with UnfusedMHA before loading the
    adapter, matching the architecture used when the adapter was saved.

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
    _replace_dit_mha_with_unfused(base_model)
    return PeftModel.from_pretrained(base_model, lora_dir, is_trainable=is_trainable)


def merge_lora_and_unload(model: nn.Module) -> nn.Module:
    """Merge LoRA delta weights into base model weights and return a plain model.

    Applies W_merged = W_base + (alpha/r) * B @ A per adapted layer, then removes
    all PEFT scaffolding. Use before ONNX export or deployment.

    After merging, DiT blocks contain UnfusedMHA modules (not nn.MultiheadAttention).
    UnfusedMHA is numerically equivalent and ONNX-exportable.

    Handles DDP-wrapped models by unwrapping via .module before merging.
    """
    inner = model.module if hasattr(model, "module") else model
    return inner.merge_and_unload()
