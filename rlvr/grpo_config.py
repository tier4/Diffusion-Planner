"""JSON-based GRPO configuration.

Supports two modes:
- On-policy (M=1): Single gradient step per rollout, no importance sampling.
- Multi-epoch (M>1): Reuse rollouts for M inner epochs with PPO-clipped
  importance sampling for higher sample efficiency.
"""

from __future__ import annotations

import json
import math
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
    enable_road_border: bool = True
    enable_speed: bool = True
    guidance_prob: float = 0.5
    prototypes_path: str | None = None

    # Reward weights
    w_safety: float = 5.0
    w_progress: float = 2.0
    w_smooth: float = 0.5
    w_feasibility: float = 5.0
    w_centerline: float = 5.0

    # Reward tuning (passed to RewardConfig for training)
    near_edge_scale: float = 3.0
    wide_edge_scale: float = 0.2
    max_lat_accel: float = 2.0
    lat_accel_scale: float = 3.0
    enable_overprogress: bool = True
    overprogress_margin: float = 1.1
    overprogress_penalty: float = 0.3
    stopped_penalty: float = 5.0

    # Reward aggregation mode (passed to RewardConfig):
    # "gate": binary safety gates (default). Any terminal event → floor.
    # "survival": PlannerRFT-style proportional credit. Crash at t=60/80 gets
    #   75% quality score. Prevents gradient death on hard scenes.
    reward_mode: str = "gate"

    # Loss mode: controls how gradients flow to affect the deterministic output.
    # "diffusion" (default): standard advantage-weighted diffusion loss at random t.
    # "direct_best": regress the model's deterministic output toward the best-in-group
    #     trajectory via MSE, bypassing diffusion timestep sampling entirely.
    # "diffusion_low_t": sample t from a narrow range near 0 where denoising is
    #     closest to the final clean output.
    # "diffusion_multistep": average loss over K timesteps spread across the schedule
    #     for better coverage of the diffusion probability.
    loss_mode: str = "diffusion"
    direct_loss_weight: float = 1.0
    diffusion_t_range: list[float] = field(default_factory=lambda: [0.001, 0.1])
    diffusion_k_steps: int = 4

    # Advantage computation mode:
    # "normalized" (default): standard GRPO per-group normalization (mean=0, std=1).
    # "vd_grpo": Variance-Decoupled GRPO (Plan-R1). Center only (subtract mean),
    #   divide by a fixed scale instead of per-group std. Preserves absolute
    #   magnitude of negative rewards (e.g. crashes) across groups.
    advantage_mode: str = "normalized"
    advantage_fixed_scale: float = 10.0

    # KL coefficient scheduling over training epochs.
    # "constant" (default): kl_coef stays fixed.
    # "linear": linearly interpolate from kl_coef to kl_coef_final.
    # "cosine": cosine annealing from kl_coef to kl_coef_final.
    # "step": hold kl_coef for kl_warmup_fraction of training, then drop to kl_coef_final.
    kl_schedule: str = "constant"
    kl_coef_initial: float | None = None  # set automatically from kl_coef at first call
    kl_coef_final: float = 0.01
    kl_warmup_fraction: float = 0.5  # fraction of epochs to hold initial kl_coef (for "step" schedule)

    # Rejection sampling: generate num_generations trajectories but keep only
    # the top rejection_keep by reward. Set to 0 or None to disable (keep all).
    rejection_keep: int = 0

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

    def get_kl_coef(self, epoch: int, total_epochs: int) -> float:
        """Compute KL coefficient for the given epoch based on schedule.

        Uses kl_coef_initial (captured from kl_coef on first call) as the
        stable starting point, so the schedule is stateless and independent
        of any mutations to kl_coef during training.

        Args:
            epoch: Current epoch (1-indexed).
            total_epochs: Total number of training epochs.

        Returns:
            KL coefficient for this epoch.
        """
        # Capture the initial kl_coef on first call
        if self.kl_coef_initial is None:
            self.kl_coef_initial = self.kl_coef

        if self.kl_schedule == "constant" or total_epochs <= 1:
            return self.kl_coef_initial

        # progress: 0.0 at epoch 1, 1.0 at final epoch
        progress = (epoch - 1) / (total_epochs - 1)
        start, end = self.kl_coef_initial, self.kl_coef_final

        if self.kl_schedule == "linear":
            return start + (end - start) * progress

        if self.kl_schedule == "cosine":
            return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))

        if self.kl_schedule == "step":
            if progress < self.kl_warmup_fraction:
                return start
            return end

        raise ValueError(
            f"Unknown kl_schedule: {self.kl_schedule!r}. "
            f"Expected 'constant', 'linear', 'cosine', or 'step'."
        )

    @property
    def uses_importance_sampling(self) -> bool:
        """Whether this config uses PPO-clipped importance sampling (M > 1)."""
        return self.inner_epochs > 1
