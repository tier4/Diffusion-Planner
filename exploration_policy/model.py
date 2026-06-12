"""Exploration Policy network for adaptive guidance during GRPO training.

Learns to output per-scene (eta_lat, eta_lon) guidance scales from Beta
distributions, replacing uniform random sampling in the GRPO trajectory sampler.

Architecture (Figure 2):
    [Frozen Encoder] -> scene_encoding [B, N, D_enc]
    [LoRA-disabled DiT] -> x_ref [B, T, 4]
                              |
                         [RefTrajectoryMixer]  (MLP-Mixer, L layers, hidden=H)
                              |
                         ref_token [B, H]
                              |
                         [RefFusionAttention]   (cross-attn: ref queries scene)
                              |
                         fused [B, H]
                          /        \\
                   [GuidanceHead]  [ValueHead]
                   (4 Beta params)  (V(s) scalar, for value baseline)
                        |
                 eta ~ Beta(alpha, beta) mapped to [-1, 1]
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Beta

from exploration_policy.module.heads import GuidanceHead, ValueHead
from exploration_policy.module.ref_fusion import RefFusionAttention
from exploration_policy.module.ref_mixer import RefTrajectoryMixer


@dataclass
class ExplorationPolicyConfig:
    """Configuration for the exploration policy network."""

    hidden_dim: int = 128
    n_mixer_layers: int = 2
    n_attn_heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-4
    # Encoder hidden dim (must match the planner's encoder output)
    encoder_hidden_dim: int = 256
    # GuidanceHead output layer init: "zeros" or "normal"
    # "zeros": alpha=beta≈1.693, unbiased eta with std≈0.48 (recommended)
    # "normal": nn.init.normal_(std=head_init_std) for faster policy learning
    head_init: str = "zeros"
    head_init_std: float = 0.01
    # Scale factor applied to raw output before softplus. Amplifies gradient
    # flow through the softplus compression. Default 1.0 (no scaling).
    # Set to 10.0 to make the policy learn ~12x faster.
    head_raw_scale: float = 1.0
    # Guidance heads, one Beta distribution (-> eta in [-1, 1]) each. The
    # policy itself is head-name agnostic — names define count, ordering, and
    # the dict keys of ExplorationPolicyOutput; mapping eta -> guidance params
    # is the caller's job. Default matches the original 2-head layout, so old
    # checkpoints (fc2 [4, H]) load unchanged.
    heads: list[str] = field(default_factory=lambda: ["lateral", "longitudinal"])

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> ExplorationPolicyConfig:
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ExplorationPolicyOutput:
    """Output container for a single forward pass.

    Per-head tensors are keyed by the head names from
    ExplorationPolicyConfig.heads. The eta_lat / eta_lon / lat_dist / lon_dist
    accessors preserve the original 2-head API (KeyError if the corresponding
    head is not configured).
    """

    etas: dict[str, torch.Tensor]  # head -> [B] sampled eta in [-1, 1]
    log_probs: dict[str, torch.Tensor]  # head -> [B] log prob of sampled eta
    value: torch.Tensor  # [B] state value estimate (Phase 2)
    dists: dict[str, Beta]  # head -> Beta distribution object

    @property
    def eta_lat(self) -> torch.Tensor:
        return self.etas["lateral"]

    @property
    def eta_lon(self) -> torch.Tensor:
        return self.etas["longitudinal"]

    @property
    def log_prob_lat(self) -> torch.Tensor:
        return self.log_probs["lateral"]

    @property
    def log_prob_lon(self) -> torch.Tensor:
        return self.log_probs["longitudinal"]

    @property
    def lat_dist(self) -> Beta:
        return self.dists["lateral"]

    @property
    def lon_dist(self) -> Beta:
        return self.dists["longitudinal"]


class ExplorationPolicy(nn.Module):
    """Learned exploration policy for adaptive guidance.

    Assembles RefTrajectoryMixer + RefFusionAttention + GuidanceHead + ValueHead.
    Takes frozen scene encoding and a reference trajectory, outputs
    (eta_lat, eta_lon) from learned Beta distributions for use as
    lateral/longitudinal guidance parameters.

    Mirrors the Diffusion_Planner pattern of a top-level model class
    that composes its sub-modules from exploration_policy.module/.
    """

    def __init__(self, config: ExplorationPolicyConfig, ref_seq_len: int = 80):
        super().__init__()
        self.config = config

        self.ref_mixer = RefTrajectoryMixer(
            seq_len=ref_seq_len,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_mixer_layers,
            dropout=config.dropout,
        )

        self.ref_fusion = RefFusionAttention(
            hidden_dim=config.hidden_dim,
            encoder_dim=config.encoder_hidden_dim,
            n_heads=config.n_attn_heads,
            dropout=config.dropout,
        )

        self.guidance_head = GuidanceHead(
            config.hidden_dim,
            n_heads=len(config.heads),
            init_mode=config.head_init,
            init_std=config.head_init_std,
            raw_scale=config.head_raw_scale,
        )
        self.value_head = ValueHead(config.hidden_dim)

    def forward(
        self,
        scene_encoding: torch.Tensor,
        x_ref: torch.Tensor,
        deterministic: bool = False,
    ) -> ExplorationPolicyOutput:
        """
        Args:
            scene_encoding: [B, N, D_enc] frozen encoder output.
            x_ref: [B, T, 4] reference trajectory from LoRA-disabled model.
            deterministic: If True, use distribution mean instead of sampling.

        Returns:
            ExplorationPolicyOutput with sampled eta values and log probs
            keyed by head name.
        """
        # Compress reference trajectory to a single token
        ref_token = self.ref_mixer(x_ref)  # [B, H]

        # Fuse with scene context via cross-attention
        fused = self.ref_fusion(ref_token, scene_encoding)  # [B, H]

        # One Beta distribution per configured head
        head_dists = self.guidance_head(fused)

        # Get value estimate
        value = self.value_head(fused)  # [B]

        etas: dict[str, torch.Tensor] = {}
        log_probs: dict[str, torch.Tensor] = {}
        dists: dict[str, Beta] = {}
        for name, dist in zip(self.config.heads, head_dists):
            eta_01 = dist.mean if deterministic else dist.rsample()  # [B] in (0, 1)
            etas[name] = 2.0 * eta_01 - 1.0  # map to (-1, 1)
            log_probs[name] = dist.log_prob(eta_01)
            dists[name] = dist

        return ExplorationPolicyOutput(
            etas=etas,
            log_probs=log_probs,
            value=value,
            dists=dists,
        )
