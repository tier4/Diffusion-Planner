"""Learned Exploration Policy for adaptive GRPO guidance.

Outputs per-scene (eta_lat, eta_lon) guidance scales from Beta distributions
for the GRPO trajectory sampler.
"""

from exploration_policy.model import (
    ExplorationPolicy,
    ExplorationPolicyConfig,
    ExplorationPolicyOutput,
)
from exploration_policy.utils import (
    generate_reference_trajectory,
    get_frozen_encoder,
    run_frozen_encoder,
)

__all__ = [
    "ExplorationPolicy",
    "ExplorationPolicyConfig",
    "ExplorationPolicyOutput",
    "get_frozen_encoder",
    "run_frozen_encoder",
    "generate_reference_trajectory",
]
