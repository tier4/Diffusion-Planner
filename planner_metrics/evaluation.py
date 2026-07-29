"""Shared result type for scenario-specific metric evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class MetricEvaluation:
    """Per-sample scores and optional metric-specific detail fields."""

    scores: dict[str, torch.Tensor]
    details: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
