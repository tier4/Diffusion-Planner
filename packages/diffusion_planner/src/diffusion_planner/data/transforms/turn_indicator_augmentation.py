"""NumPy turn-indicator augmentation for planner dataset frames."""

from __future__ import annotations

import numpy as np

from .base import Frame, FrameLike

TURN_INDICATOR_DISABLED = 1
TURN_INDICATOR_LEFT = 2
TURN_INDICATOR_RIGHT = 3


class PlannerTurnIndicatorAugmentation:
    """Perturb the current indicator using its state over the past three seconds.

    An active current indicator is changed to disabled. A disabled current
    indicator is changed to the most recent active indicator in its history.
    Either change is applied with ``probability``.
    """

    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def __call__(self, input_data: FrameLike) -> Frame:
        output = dict(input_data)
        if np.random.random() >= self.probability:
            return output

        turn_indicators = input_data["turn_indicators"]
        current = turn_indicators[-1]

        if current in (TURN_INDICATOR_LEFT, TURN_INDICATOR_RIGHT):
            augmented_current = TURN_INDICATOR_DISABLED
        elif current == TURN_INDICATOR_DISABLED:
            active_indices = np.flatnonzero(
                np.isin(
                    turn_indicators[:-1],
                    (TURN_INDICATOR_LEFT, TURN_INDICATOR_RIGHT),
                )
            )
            if active_indices.size == 0:
                return output
            augmented_current = turn_indicators[active_indices[-1]]
        else:
            return output

        output["turn_indicators"] = turn_indicators.copy()
        output["turn_indicators"][-1] = augmented_current
        return output
