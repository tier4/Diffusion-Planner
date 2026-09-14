"""Lane-change metric placeholder for scenario-based open-loop evaluation."""

from __future__ import annotations

import torch

from planner_metrics.evaluation import MetricEvaluation


@torch.no_grad()
def evaluate_lane_change_with_details(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    parameters: dict,
) -> MetricEvaluation:
    """Evaluate lane-change behavior.

    The lane-change criterion and its scene-data/parameter contract are not
    defined yet. Keep this registered scorer explicit so a configured
    lane-change evaluation cannot silently produce a misleading placeholder
    score.
    """
    del ego_trajs, data, parameters
    raise NotImplementedError("lane_change metric is not implemented yet")


__all__ = ["evaluate_lane_change_with_details"]
