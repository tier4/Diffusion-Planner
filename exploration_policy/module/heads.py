"""Output heads for the Exploration Policy.

GuidanceHead: Beta distribution parameters for (eta_lat, eta_lon).
ValueHead: Scalar state value V(s) for value baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class GuidanceHead(nn.Module):
    """Outputs Beta distribution parameters for lateral and longitudinal eta.

    Produces 4 values: (alpha_lat, beta_lat, alpha_lon, beta_lon).
    Beta params constrained >= 1.0 via softplus + 1.0, ensuring unimodal
    distributions.

    Zero-initialization: The last linear layer is zero-initialized so that
    raw output = [0,0,0,0] at init. Through softplus(0)+1 = ln(2)+1 ≈ 1.693,
    this gives alpha=beta for both distributions, meaning:
      - Beta mean = 0.5 in (0,1) → eta mean = 0.0 in (-1,1)
      - Beta std ≈ 0.24 in (0,1) → eta std ≈ 0.48 in (-1,1)
    This ensures unbiased exploration around the reference trajectory and
    prevents performance drops at the start of training.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 4)  # alpha_lat, beta_lat, alpha_lon, beta_lon

        # Small random init for the output layer weights so that output is
        # input-dependent from the start (enables gradient flow).
        # Zero-init was too aggressive — the policy never learned because
        # fc2.weight=0 means the output is constant regardless of input,
        # and gradients through zero weights are too small to overcome.
        # Bias is zero so initial mean eta ≈ 0 (unbiased).
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, fused: torch.Tensor) -> tuple[Beta, Beta]:
        """
        Args:
            fused: [B, H] fused scene + reference representation.

        Returns:
            (lat_dist, lon_dist): Beta distributions with batch_shape=[B].
        """
        raw = self.fc2(self.act(self.fc1(fused)))  # [B, 4]

        # softplus + 1.0 ensures params >= 1.0 (unimodal Beta)
        alpha_lat = F.softplus(raw[:, 0]) + 1.0
        beta_lat = F.softplus(raw[:, 1]) + 1.0
        alpha_lon = F.softplus(raw[:, 2]) + 1.0
        beta_lon = F.softplus(raw[:, 3]) + 1.0

        return Beta(alpha_lat, beta_lat), Beta(alpha_lon, beta_lon)


class ValueHead(nn.Module):
    """Scalar value function V(s) for variance reduction.

    Not actively used in current joint training (GRPO advantages serve as
    baseline). Included for future use. Output layer zero-initialized
    so V(s)=0 at init.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Zero-init output layer: V(s) = 0 at init
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: [B, H] fused scene + reference representation.

        Returns:
            [B] estimated state value.
        """
        return self.fc2(self.act(self.fc1(fused))).squeeze(-1)
