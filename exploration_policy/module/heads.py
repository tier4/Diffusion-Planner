"""Output heads for the Exploration Policy.

GuidanceHead: Beta distribution parameters for (eta_lat, eta_lon).
ValueHead: Scalar state value V(s) for value baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class GuidanceHead(nn.Module):
    """Outputs Beta distribution parameters for n_heads guidance etas.

    Produces 2*n_heads values: (alpha_0, beta_0, alpha_1, beta_1, ...), one
    (alpha, beta) pair per head. The default n_heads=2 reproduces the original
    (lateral, longitudinal) layout exactly — fc2 stays [4, H] and old
    checkpoints load unchanged.

    Beta params constrained >= 1.0 via softplus + 1.0, ensuring unimodal
    distributions.

    Zero-initialization: The last linear layer is zero-initialized so that
    raw output = 0 at init. Through softplus(0)+1 = ln(2)+1 ≈ 1.693,
    this gives alpha=beta for every distribution, meaning:
      - Beta mean = 0.5 in (0,1) → eta mean = 0.0 in (-1,1)
      - Beta std ≈ 0.24 in (0,1) → eta std ≈ 0.48 in (-1,1)
    This ensures unbiased exploration around the reference trajectory and
    prevents performance drops at the start of training.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 2,
        init_mode: str = "zeros",
        init_std: float = 0.01,
        raw_scale: float = 1.0,
        max_conc: float = 10.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        # Cap on the Beta concentration (alpha/beta). With the default 10.0 the
        # deterministic Beta mean saturates at ~10/11, so the mapped eta cannot
        # exceed ~0.82 — scenes whose swept-best guidance sits at the grid edge
        # (|eta|=1.0) are then unreachable. Raise it to let the policy emit more
        # extreme etas on tight scenes (the gate still controls engagement).
        self.max_conc = float(max_conc)
        # The clamp caps softplus(.)+1.0 (always >= 1.0) at max_conc; a cap below
        # 1.0 (or NaN) would push params under 1.0 and break the unimodal-Beta
        # invariant. Fail loudly rather than emit invalid Beta shapes.
        if not (self.max_conc >= 1.0):
            raise ValueError(f"max_conc must be >= 1.0 (got {max_conc!r})")
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        # bias=False removes a direct global offset at the output layer:
        # output = W @ fused_input. This encourages the head to use scene-dependent
        # features rather than relying on an output bias. Note that upstream layers
        # (fc1) still include biases, so this is not a strict guarantee of
        # scene-dependence — but empirically it significantly improves per-scene
        # variation vs having bias=True on fc2.
        # Zero-init still works: zero weights → output=0 → softplus(0)+1 ≈ 1.693.
        self.fc2 = nn.Linear(hidden_dim, 2 * n_heads, bias=False)
        self.raw_scale = raw_scale

        # Output layer initialization controls the initial exploration behavior:
        # "zeros": alpha=beta≈1.693, symmetric Beta with std≈0.48 (unbiased, wide)
        # "normal": non-zero init for faster policy learning (may need tuning)
        if init_mode == "zeros":
            nn.init.zeros_(self.fc2.weight)
        elif init_mode == "normal":
            nn.init.normal_(self.fc2.weight, mean=0.0, std=init_std)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode!r} (expected 'zeros' or 'normal')")

    def forward(self, fused: torch.Tensor) -> list[Beta]:
        """
        Args:
            fused: [B, H] fused scene + reference representation.

        Returns:
            list of n_heads Beta distributions with batch_shape=[B].
        """
        raw = self.fc2(self.act(self.fc1(fused)))  # [B, 2*n_heads]

        # raw_scale amplifies gradients through softplus to overcome compression.
        # Without scaling (raw_scale=1): raw=0.03 → softplus(0.03)+1=1.72, eta≈0.004
        # With raw_scale=10: raw=0.03 → softplus(0.3)+1=1.84, eta≈0.05 (12x more)
        scaled = raw * self.raw_scale

        # softplus + 1.0 ensures params >= 1.0 (unimodal Beta)
        # Clamp to max_conc to prevent distribution collapse (higher = more
        # extreme reachable eta; see __init__).
        params = torch.clamp(F.softplus(scaled) + 1.0, max=self.max_conc)
        return [Beta(params[:, 2 * i], params[:, 2 * i + 1]) for i in range(self.n_heads)]


class StrengthHead(nn.Module):
    """Scalar guidance-strength gate g in (0, 1) (sigmoid).

    The policy emits ONE number per scene that controls how strongly the whole
    guidance field is applied: at inference the composer multiplies the total
    guidance energy (hence the guidance gradient) by g, so g=0 -> exactly the
    unguided plan, g=1 -> the full envelope push. This decouples "engage? / how
    hard" (this scalar) from "which way to swerve" (the per-head etas), and
    gives an intrinsic false-positive gate: on a scene with nothing to avoid the
    network can drive g->0 directly instead of relying on every eta head landing
    on exactly 0.

    Supervised target: 1.0 on real-avoidance scenes, 0.0 on zero-target scenes
    (no obstacle in the ego's path — including far-distractor-only scenes).

    Zero-initialized output layer -> raw 0 -> sigmoid(0)=0.5 at init (an
    unbiased, mild gate before any learning).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """fused [B, H] -> [B] strength in (0, 1)."""
        return torch.sigmoid(self.fc2(self.act(self.fc1(fused))).squeeze(-1))


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
