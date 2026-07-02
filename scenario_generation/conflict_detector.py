"""Optional expert-disagreement conflict detection for scene classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rlvr.deviation import rollout_gt_deviation


@dataclass(frozen=True)
class ConflictResult:
    expert_disagreement: bool
    expert_disagreement_step: int | None
    max_deviation: float


def detect_expert_disagreement(
    rollout: torch.Tensor,
    gt: torch.Tensor,
    *,
    enabled: bool,
    threshold_m: float,
    sustain_steps: int,
) -> ConflictResult:
    """Detect sustained xy mismatch between a rollout and expert/GT trajectory."""
    if not enabled:
        return ConflictResult(False, None, 0.0)
    if threshold_m <= 0:
        raise ValueError(f"threshold_m must be > 0, got {threshold_m}")
    if sustain_steps < 1:
        raise ValueError(f"sustain_steps must be >= 1, got {sustain_steps}")
    if rollout.shape[-1] < 2 or gt.shape[-1] < 2:
        raise ValueError("rollout and gt must have xy in the first two channels")
    max_dev, step = rollout_gt_deviation(
        rollout,
        gt,
        threshold_m=threshold_m,
        sustain_steps=sustain_steps,
    )
    return ConflictResult(step is not None, step, max_dev)
