"""Trajectory deviation helpers for replay/conflict mining."""

from __future__ import annotations

import torch


def rollout_gt_deviation(
    rollout: torch.Tensor,
    gt: torch.Tensor,
    *,
    threshold_m: float | None = None,
    sustain_steps: int = 1,
) -> tuple[float, int | None]:
    """Return max xy deviation and first sustained threshold-crossing step.

    ``rollout`` and ``gt`` are shaped ``(..., T, C)`` with xy in the first two
    channels. If ``threshold_m`` is provided, the returned step is the first
    index where deviation stays above the threshold for ``sustain_steps``
    consecutive frames. If no threshold is provided, the step is ``None``.
    """
    if sustain_steps < 1:
        raise ValueError(f"sustain_steps must be >= 1, got {sustain_steps}")
    if threshold_m is not None and threshold_m <= 0:
        raise ValueError(f"threshold_m must be > 0, got {threshold_m}")
    if rollout.shape[-1] < 2 or gt.shape[-1] < 2:
        raise ValueError("rollout and gt must have xy in the first two channels")
    if rollout.shape[-2] == 0 or gt.shape[-2] == 0:
        return 0.0, None
    T = min(int(rollout.shape[-2]), int(gt.shape[-2]))
    dist = torch.linalg.norm(rollout[..., :T, :2] - gt[..., :T, :2], dim=-1).reshape(-1, T)
    per_step = dist.max(dim=0).values
    max_dev = float(per_step.max().item()) if per_step.numel() else 0.0
    if threshold_m is None or T < sustain_steps:
        return max_dev, None
    above = per_step > threshold_m
    if int(above.sum().item()) < sustain_steps:
        return max_dev, None
    kernel = torch.ones(sustain_steps, dtype=torch.float32, device=above.device)
    conv = torch.nn.functional.conv1d(
        above.to(torch.float32).view(1, 1, -1),
        kernel.view(1, 1, -1),
    )[0, 0]
    hits = torch.nonzero(conv >= sustain_steps, as_tuple=False)
    return max_dev, int(hits[0].item()) if hits.numel() else None
