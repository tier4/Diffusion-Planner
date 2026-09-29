"""Label -> scenario metric registry.

Each family module registers its scorers with ``@register("<label>", ...)``. A label
maps to exactly one scorer; registering it twice is an error so two families can
never silently disagree about the same label.
"""

from __future__ import annotations

from typing import Callable

from scenario_generation.scenario_metrics.base import ClosedLoopScenarioInput, ScenarioResult

ScenarioMetric = Callable[[ClosedLoopScenarioInput], ScenarioResult]

METRICS: dict[str, ScenarioMetric] = {}


def register(*labels: str) -> Callable[[ScenarioMetric], ScenarioMetric]:
    def decorator(fn: ScenarioMetric) -> ScenarioMetric:
        for label in labels:
            if label in METRICS:
                raise ValueError(f"scenario metric for {label!r} registered twice")
            METRICS[label] = fn
        return fn

    return decorator


def score(inp: ClosedLoopScenarioInput) -> ScenarioResult:
    """Score ``inp`` with its label's metric; unknown labels are reported, not raised."""
    fn = METRICS.get(inp.label)
    if fn is None:
        return ScenarioResult(
            metric="none", passed=None, reason=f"no scenario metric for label {inp.label!r}"
        )
    return fn(inp)
