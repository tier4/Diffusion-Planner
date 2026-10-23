"""Ego tensor layouts of H5 formats 5 and 6, and of the two model interfaces.

H5 format 6 inserts ``steering_tire_angle`` before ``yaw_rate`` in the ego states and appends
``wheel_base`` to ``ego_shape``::

    format 4/5: ego states [x, y, cos, sin, velocity, yaw_rate]
                ego_shape  [base_link_to_front, length, width]
    format 6:   ego states [x, y, cos, sin, velocity, steering_tire_angle, yaw_rate]
                ego_shape  [base_link_to_front, length, width, wheel_base]

The evaluator works on the format-5 layout throughout (the scorers, the renderer, both
closed-loop backends and the devkit simulator read it), so every frame is read through
``to_format5_layout``. A format-6 frame keeps its two extra values under their own keys,
``STEERING_PAST_KEY`` and ``WHEEL_BASE_KEY``, the same drop ML-Planner applies before its
model (``PlannerDropSteeringAndWheelBase``).

The model interface is told apart by the ONNX ``ego_agent_past`` width: 6 is the format-5
layout, which every model trained after ML-Planner #98 still takes; 7 is the format-6 layout
of the steering-input models, which ``to_model_feed`` rebuilds from the two extra keys. A
format-5 frame has no recorded steering angle, so a steering-input model cannot be fed from
it, and the evaluator refuses rather than estimating one.
"""

from __future__ import annotations

import numpy as np

STEERING_PAST_KEY = "ego_steering_past"
WHEEL_BASE_KEY = "ego_wheel_base"
STEERING_KEYS = (STEERING_PAST_KEY, WHEEL_BASE_KEY)

FORMAT5_EGO_STATE_DIM = 6
FORMAT6_EGO_STATE_DIM = 7
FORMAT5_EGO_SHAPE_DIM = 3
FORMAT6_EGO_SHAPE_DIM = 4
_STEERING_INDEX = 5
_WHEEL_BASE_INDEX = 3
# Model inputs are normalized before the ONNX call; ego_shape is divided by this.
_EGO_SHAPE_SCALE = 10.0


def to_format5_layout(frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return ``frame`` in the format-5 ego layout, moving format-6 extras to their own keys.

    Works on one frame or on a stack of frames (any leading axes). A format-5 frame is
    returned unchanged.
    """
    past = frame["ego_agent_past"]
    if past.shape[-1] == FORMAT5_EGO_STATE_DIM:
        return frame
    if past.shape[-1] != FORMAT6_EGO_STATE_DIM:
        raise ValueError(f"unexpected ego_agent_past width {past.shape[-1]}")
    if frame["ego_shape"].shape[-1] != FORMAT6_EGO_SHAPE_DIM:
        raise ValueError(
            f"format-6 ego_agent_past with ego_shape width {frame['ego_shape'].shape[-1]}"
        )
    out = dict(frame)
    for key in ("ego_agent_past", "ego_agent_future"):
        if key in frame:
            out[key] = np.delete(frame[key], _STEERING_INDEX, axis=-1)
    out[STEERING_PAST_KEY] = np.asarray(past[..., _STEERING_INDEX])
    out["ego_shape"] = np.asarray(frame["ego_shape"][..., :_WHEEL_BASE_INDEX])
    out[WHEEL_BASE_KEY] = np.asarray(frame["ego_shape"][..., _WHEEL_BASE_INDEX])
    return out


def model_takes_steering(session) -> bool:
    """Whether an ONNX session takes the format-6 ego layout (steering and wheel base)."""
    inputs = {item.name: item.shape for item in session.get_inputs()}
    width = inputs["ego_agent_past"][-1]
    shape_width = inputs["ego_shape"][-1]
    layouts = {
        (FORMAT5_EGO_STATE_DIM, FORMAT5_EGO_SHAPE_DIM): False,
        (FORMAT6_EGO_STATE_DIM, FORMAT6_EGO_SHAPE_DIM): True,
    }
    if (width, shape_width) not in layouts:
        raise ValueError(
            f"ONNX ego_agent_past width {width} and ego_shape width {shape_width} match "
            "neither the format-5 nor the format-6 ego layout"
        )
    return layouts[(width, shape_width)]


def to_model_feed(
    feed: dict[str, np.ndarray], extras: dict[str, np.ndarray] | None, takes_steering: bool
) -> dict[str, np.ndarray]:
    """Rebuild the format-6 ego inputs of a normalized batched ``feed`` for a steering model.

    ``extras`` holds the batch's ``STEERING_PAST_KEY`` ``(B, T)`` and ``WHEEL_BASE_KEY``
    ``(B,)``, which are not normalized. A format-5 model gets ``feed`` unchanged.
    """
    if not takes_steering:
        return feed
    if extras is None or any(key not in extras for key in STEERING_KEYS):
        raise ValueError(
            "this model takes the format-6 ego layout (steering_tire_angle and wheel_base), "
            "which format-5 data does not record; evaluate it on format-6 H5 data"
        )
    out = dict(feed)
    steering = np.asarray(extras[STEERING_PAST_KEY], dtype=np.float32)
    out["ego_agent_past"] = np.insert(feed["ego_agent_past"], _STEERING_INDEX, steering, axis=-1)
    wheel_base = np.asarray(extras[WHEEL_BASE_KEY], dtype=np.float32) / _EGO_SHAPE_SCALE
    out["ego_shape"] = np.concatenate((feed["ego_shape"], wheel_base[..., None]), axis=-1)
    return out


__all__ = [
    "FORMAT5_EGO_SHAPE_DIM",
    "FORMAT5_EGO_STATE_DIM",
    "FORMAT6_EGO_SHAPE_DIM",
    "FORMAT6_EGO_STATE_DIM",
    "STEERING_KEYS",
    "STEERING_PAST_KEY",
    "WHEEL_BASE_KEY",
    "model_takes_steering",
    "to_format5_layout",
    "to_model_feed",
]
