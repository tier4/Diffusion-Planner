"""RLVR -- Reinforcement Learning with Verifiable Rewards for Diffusion Planner."""

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


def __getattr__(name):
    if name == "GRPOConfig":
        from rlvr.grpo_config import GRPOConfig

        return GRPOConfig
    if name in {"compute_grpo_loss", "compute_log_probs"}:
        from rlvr import grpo_loss

        return getattr(grpo_loss, name)
    if name in {"SampledTrajectory", "SamplerConfig", "generate_diverse_group"}:
        from rlvr import grpo_sampler

        return getattr(grpo_sampler, name)
    if name == "GRPOTrainer":
        from rlvr.grpo_trainer import GRPOTrainer

        return GRPOTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
