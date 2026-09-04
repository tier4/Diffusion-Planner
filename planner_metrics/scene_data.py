"""Adapter boundary between metric evaluation and any specific data source.

The evaluators in this package only ever read a fixed, small set of scene
fields (see ``_METRIC_SCENE_FIELDS``). ``extract_metric_scene_data`` is the
single place that pulls exactly those fields out of a raw sample, so callers
never need to hand evaluators a source-specific object (an NPZ dict, a model's
internal batch, or anything else) — only a plain mapping that happens to carry
these keys. This keeps metric evaluation decoupled from where the data came
from.
"""

from __future__ import annotations

from typing import Mapping

import torch

_METRIC_SCENE_FIELDS = (
    "ego_current_state",
    "ego_agent_future",
    "route_lanes",
    "lanes",
    "neighbor_agents_future",
    "neighbor_agents_past",
    "ego_shape",
)


def extract_metric_scene_data(source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return the subset of ``source`` that open-loop metrics require.

    ``source`` may be a raw NPZ-derived sample/batch or any other mapping
    that provides the same field names under the same keys — evaluators never
    look at anything else, so this is the full contract a data source must
    satisfy to be scored.
    """
    return {key: source[key] for key in _METRIC_SCENE_FIELDS if key in source}


__all__ = ["extract_metric_scene_data"]
