"""Thresholds the closed-loop scenario metrics share with the open-loop ones.

The source of truth is ``ScenarioOpenLoopConfig`` (``scenario_<label>_<parameter>``
fields, grouped per label exactly as the open-loop runner groups them), so changing an
open-loop threshold changes the closed-loop verdict too. Scorers take an optional
``config`` -- a ``ScenarioOpenLoopConfig``, or any args object carrying its fields
(e.g. ``TrainConfig``); None means the config's defaults. Each scorer maps the
open-loop parameter names onto its own explicitly; thresholds with no open-loop
counterpart stay constants in the family modules.
"""

from __future__ import annotations

from diffusion_planner.config.scenario_open_loop_config import (
    ScenarioOpenLoopConfig,
    scenario_metric_parameters,
)

_DEFAULT_CONFIG = ScenarioOpenLoopConfig()


def open_loop_parameters(label: str, config=None) -> dict[str, object]:
    """``{<parameter>: value}`` of ``label``'s ``scenario_<label>_<parameter>`` fields."""
    return scenario_metric_parameters(_DEFAULT_CONFIG if config is None else config, (label,))[
        label
    ]
