"""JSON-based GRPO configuration.

Supports two modes:
- On-policy (M=1): Single gradient step per rollout, no importance sampling.
- Multi-epoch (M>1): Reuse rollouts for M inner epochs with PPO-clipped
  importance sampling for higher sample efficiency.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class GRPOConfig:
    """Complete GRPO training configuration.

    Attributes:
        num_generations: N trajectories to sample per scene.
        train_epochs: Total number of outer training epochs.
        inner_epochs: M gradient steps on the same batch before regenerating.
            M=1 is pure on-policy. M>1 reuses rollouts with importance sampling.
        ppo_clip_epsilon: PPO clipping range for importance sampling ratio.
            Only active when inner_epochs > 1.
        kl_coef: Coefficient for KL divergence against fixed SFT reference.
        learning_rate: AdamW learning rate.
        grad_accum_groups: Number of groups (scenes) to accumulate before
            stepping the optimizer.
        sampling_randomization: Whether to randomize noise_scale and guidance
            per trajectory (True) or use fixed configs (False).
        noise_scale_range: (min, max) noise scale for stochastic trajectories.
        guidance_scale_range: (min, max) global guidance scale.
        enable_guidance: Master guidance toggle.
        enable_centerline: Include centerline guidance in random pool.
        enable_anchor: Include anchor guidance in random pool.
        enable_collision: Include collision guidance in random pool.
        enable_route_following: Include route-following guidance in random pool.
        enable_lane_keeping: Include lane-keeping guidance in random pool.
        guidance_prob: Per-type coin-flip probability.
        prototypes_path: Path to anchor prototypes .npy (null to disable anchor).
        w_safety: Reward weight for collision/proximity.
        w_progress: Reward weight for goal progress.
        w_smooth: Reward weight for jerk penalty.
        w_feasibility: Reward weight for lane/acceleration feasibility.
        w_centerline: Reward weight for centerline following.
        use_lora: Whether to use LoRA adapters.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_dropout: LoRA dropout probability.
    """

    # Core GRPO
    num_generations: int = 32
    train_epochs: int = 10
    inner_epochs: int = 1
    ppo_clip_epsilon: float = 0.2
    kl_coef: float = 0.1

    # Optimizer
    learning_rate: float = 1e-5
    grad_accum_groups: int = 4

    # Sampling
    sampling_randomization: bool = True
    noise_scale_range: list[float] = field(default_factory=lambda: [0.5, 4.0])
    guidance_scale_range: list[float] = field(default_factory=lambda: [0.1, 2.0])
    enable_guidance: bool = True
    enable_centerline: bool = True
    enable_anchor: bool = True
    enable_collision: bool = False
    enable_route_following: bool = False
    enable_lane_keeping: bool = False
    guidance_prob: float = 0.5
    prototypes_path: str | None = None

    # Reward weights
    w_safety: float = 5.0
    w_progress: float = 2.0
    w_smooth: float = 0.5
    w_feasibility: float = 5.0
    w_centerline: float = 5.0

    # LoRA
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    @classmethod
    def from_json(cls, path: str | Path) -> GRPOConfig:
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self, path: str | Path) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @property
    def uses_importance_sampling(self) -> bool:
        """Whether this config uses PPO-clipped importance sampling (M > 1)."""
        return self.inner_epochs > 1
