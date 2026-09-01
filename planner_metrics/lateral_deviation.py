"""Shared plumbing for "lateral/longitudinal error against a per-sample
reference path" metrics — used by both ``centerline.py`` (route-lane
centerline reference) and ``gt_lateral_deviation.py`` (GT-trajectory
reference). The two differ only in how each sample's reference-path segments
are built; this module holds the per-sample loop that is otherwise identical
between them. Horizon resolution lives in ``planner_metrics/horizon.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from planner_metrics.geometry import _point_to_segments_error_components

SegmentsForSample = Callable[[int], tuple[torch.Tensor, torch.Tensor]]


def compute_lateral_longitudinal_error_batch(
    ego_trajs: torch.Tensor,
    horizon_steps: int,
    segments_for_sample: SegmentsForSample,
) -> dict[str, torch.Tensor]:
    """Return per-sample lateral/longitudinal error against each sample's reference path.

    ``segments_for_sample(index)`` returns that sample's ``(seg_p1, seg_p2)``
    reference-path segments.
    """
    lateral_errors = []
    longitudinal_errors = []
    for index in range(ego_trajs.shape[0]):
        seg_p1, seg_p2 = segments_for_sample(index)
        points = ego_trajs[index, :horizon_steps, :2]
        lateral, longitudinal = _point_to_segments_error_components(
            points, seg_p1.to(points), seg_p2.to(points)
        )
        lateral_errors.append(lateral)
        longitudinal_errors.append(longitudinal)
    return {
        "lateral_error_m": torch.stack(lateral_errors, dim=0),
        "longitudinal_error_m": torch.stack(longitudinal_errors, dim=0),
    }


__all__ = ["compute_lateral_longitudinal_error_batch"]
