"""Metric registry for Override Open-loop validation.

Each scorer receives the ego prediction for one batch, the raw NPZ batch, and the
metric-specific configuration dictionary.  Detailed metric implementations can be
added independently without changing the training integration.
"""

from collections.abc import Callable

import torch

from diffusion_planner.override_validation.metrics.centerline import evaluate_centerline
from diffusion_planner.override_validation.metrics.departure import evaluate_departure

OverrideMetric = Callable[[torch.Tensor, dict[str, torch.Tensor], dict], dict[str, torch.Tensor]]

METRICS: dict[str, OverrideMetric] = {
    "centerline": evaluate_centerline,
    "departure": evaluate_departure,
}

__all__ = ["METRICS", "OverrideMetric", "evaluate_centerline", "evaluate_departure"]
