"""Common result types for scenario-specific metric evaluation.

The Open-loop evaluation runner consumes :class:`MetricEvaluation` so metric
implementations can return both aggregateable per-sample scores and optional
metric-specific details without coupling the runner to individual metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class MetricEvaluation:
    """Per-sample scores and optional metric-specific detail fields.

    Attributes:
        scores: Mapping from score names to one-dimensional tensors with one
            value per evaluated sample. These values are aggregated into the
            validation summary.
        details: Mapping from detail sections to fields, where each field is a
            one-dimensional tensor aligned with ``scores``. These values are
            written to per-sample detail records and are not included in the
            aggregate summary.
    """

    scores: dict[str, torch.Tensor]
    details: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)


__all__ = ["MetricEvaluation"]
