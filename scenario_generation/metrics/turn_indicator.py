"""Turn-indicator scoring for left/right turn scenario groups."""

from __future__ import annotations

import numpy as np

from scenario_generation.simulate import decode_turn_indicator

# Class ids aligned with C++ TurnIndicatorManager / decode_turn_indicator.
TI_NONE = 0
TI_LEFT = 2
TI_RIGHT = 3
TI_KEEP = 4


def expected_turn_class(metric_group: str) -> int | None:
    if metric_group == "left_turn":
        return TI_LEFT
    if metric_group == "right_turn":
        return TI_RIGHT
    return None


def score_turn_indicator_step(
    outputs: dict,
    metric_group: str,
    *,
    keep_bias: float = 0.25,
) -> dict:
    """Compare model-predicted turn indicator to expected side for turn groups."""
    expected = expected_turn_class(metric_group)
    if expected is None:
        return {"turn_pred": None, "turn_expected": None, "turn_match": None}

    if "turn_indicator_logit" not in outputs:
        return {"turn_pred": None, "turn_expected": expected, "turn_match": False}

    pred = int(
        np.asarray(decode_turn_indicator(outputs["turn_indicator_logit"], keep_bias)).reshape(-1)[0]
    )
    # LEFT/RIGHT match; KEEP also acceptable when actively turning.
    match = pred == expected or (pred == TI_KEEP and expected in (TI_LEFT, TI_RIGHT))
    return {
        "turn_pred": pred,
        "turn_expected": expected,
        "turn_match": bool(match),
    }
