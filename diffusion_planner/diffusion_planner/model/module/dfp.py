import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


def vp_alpha_sigma(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """VP-SDE marginal coefficients used by DFP chunk noising."""
    beta_min = 0.1
    beta_max = 20.0
    log_alpha = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * beta_min * t
    alpha = torch.exp(log_alpha)
    sigma = torch.sqrt(torch.clamp(1.0 - torch.exp(2.0 * log_alpha), min=1.0e-12))
    return alpha, sigma


def normalize_ego_trajectory(state_normalizer, x: torch.Tensor) -> torch.Tensor:
    mean = state_normalizer.mean.to(x.device, dtype=x.dtype)[0]
    std = state_normalizer.std.to(x.device, dtype=x.dtype)[0]
    return (x - mean) / std


def inverse_normalize_ego_trajectory(state_normalizer, x: torch.Tensor) -> torch.Tensor:
    mean = state_normalizer.mean.to(x.device, dtype=x.dtype)[0]
    std = state_normalizer.std.to(x.device, dtype=x.dtype)[0]
    return x * std + mean


def _modulate_tokenwise(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class DFPFinalLayer(nn.Module):
    """Per-chunk final layer with adaptive timestep conditioning."""

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=-1)
        x = _modulate_tokenwise(self.norm_final(x), shift, scale)
        return self.proj(x)
