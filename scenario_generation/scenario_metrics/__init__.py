"""Post-hoc closed-loop scenario metrics (see ``base`` for the input contract).

Importing the package registers every family's scorers in ``registry.METRICS``.
"""

from scenario_generation.scenario_metrics import geometry, progress, stop_arrival  # noqa: F401
from scenario_generation.scenario_metrics.base import ClosedLoopScenarioInput, ScenarioResult
from scenario_generation.scenario_metrics.registry import METRICS, register, score

__all__ = ["METRICS", "ClosedLoopScenarioInput", "ScenarioResult", "register", "score"]
