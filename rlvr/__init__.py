"""RLVR -- Reinforcement Learning with Verifiable Rewards for Diffusion Planner."""

from rlvr.grpo_config import GRPOConfig
from rlvr.grpo_loss import compute_grpo_loss, compute_log_probs
from rlvr.grpo_sampler import SampledTrajectory, SamplerConfig, generate_diverse_group
from rlvr.grpo_trainer import GRPOTrainer
from rlvr.reward import (
    RewardBreakdown,
    RewardConfig,
    compute_group_advantages,
    compute_reward,
    compute_reward_batch,
)

__all__ = [
    "RewardConfig",
    "RewardBreakdown",
    "compute_reward",
    "compute_reward_batch",
    "compute_group_advantages",
    "SamplerConfig",
    "SampledTrajectory",
    "generate_diverse_group",
    "GRPOConfig",
    "compute_grpo_loss",
    "compute_log_probs",
    "GRPOTrainer",
]
