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

from dataclasses import asdict, dataclass
import json
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
    """Output container for a single forward pass."""

    eta_lat: torch.Tensor       # [B] sampled lateral eta in [-1, 1]
    eta_lon: torch.Tensor       # [B] sampled longitudinal eta in [-1, 1]
    log_prob_lat: torch.Tensor  # [B] log probability of sampled eta_lat
    log_prob_lon: torch.Tensor  # [B] log probability of sampled eta_lon
    value: torch.Tensor         # [B] state value estimate (Phase 2)
    lat_dist: Beta              # Beta distribution object for eta_lat
    lon_dist: Beta              # Beta distribution object for eta_lon


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
            ExplorationPolicyOutput with sampled eta values and log probs.
        """
        # Compress reference trajectory to a single token
        ref_token = self.ref_mixer(x_ref)  # [B, H]

        # Fuse with scene context via cross-attention
        fused = self.ref_fusion(ref_token, scene_encoding)  # [B, H]

        # Get Beta distributions for guidance parameters
        lat_dist, lon_dist = self.guidance_head(fused)

        # Get value estimate
        value = self.value_head(fused)  # [B]

        if deterministic:
            eta_lat_01 = lat_dist.mean
            eta_lon_01 = lon_dist.mean
        else:
            # rsample() for reparameterized gradients
            eta_lat_01 = lat_dist.rsample()  # [B] in (0, 1)
            eta_lon_01 = lon_dist.rsample()  # [B] in (0, 1)

        # Map from (0, 1) to (-1, 1)
        eta_lat = 2.0 * eta_lat_01 - 1.0
        eta_lon = 2.0 * eta_lon_01 - 1.0

        # Log probabilities in the original (0, 1) space
        log_prob_lat = lat_dist.log_prob(eta_lat_01)
        log_prob_lon = lon_dist.log_prob(eta_lon_01)

        return ExplorationPolicyOutput(
            eta_lat=eta_lat,
            eta_lon=eta_lon,
            log_prob_lat=log_prob_lat,
            log_prob_lon=log_prob_lon,
            value=value,
            lat_dist=lat_dist,
            lon_dist=lon_dist,
        )
