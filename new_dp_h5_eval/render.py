"""Open-loop prediction images of native-H5 frames, drawn with the closed-loop step renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .metric_compat import legacy_lanes, legacy_route_lanes
from .schema import EgoPastLayout

VISUALIZATION_HALF_M = 60.0
GROUND_TRUTH_COLOR = "#111827"


def render_view(frame: dict[str, np.ndarray], ego_shape: np.ndarray) -> dict[str, np.ndarray]:
    """The recorded frame in the batched layout the step renderer reads: legacy lanes, road
    borders as line strings and the recorded ego state, ego at the origin."""
    view = {key: np.asarray(value)[None] for key, value in frame.items()}
    view["ego_shape"] = np.asarray(ego_shape, dtype=np.float32)[None]
    borders = frame["road_borders"]
    lines = np.zeros((*borders.shape[:-1], 4), dtype=np.float32)
    lines[..., :2] = borders
    lines[..., 3] = (np.linalg.norm(borders, axis=-1) > 0).astype(np.float32)
    view["line_strings"] = lines[None]
    view["metric_lanes"] = legacy_lanes(frame)[None]
    view["metric_route_lanes"] = legacy_route_lanes(frame)[None]
    ego = frame["ego_agent_past"][-1]
    layout = EgoPastLayout.of(ego.shape[-1])
    current = np.array([[0.0, 0.0, 1.0, 0.0, ego[4], 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    if layout.steering is not None:
        current[0, 8] = ego[layout.steering]
    current[0, 9] = ego[layout.yaw_rate]
    view["ego_current_state"] = current
    return view


def save_prediction_png(
    view: dict[str, np.ndarray], prediction: np.ndarray, path: Path, title: str
) -> None:
    """Draw the ``(T, 4)`` ego prediction and the recorded future over a :func:`render_view` scene."""
    from scenario_generation.reproducer_rollout import _draw_step

    _draw_step(
        view,
        prediction,
        view["ego_shape"][0],
        path,
        title_prefix=title,
        view_half_m=VISUALIZATION_HALF_M,
        extra_ego_trajectories=[
            (view["ego_agent_future"][0][:, :4], GROUND_TRUTH_COLOR, "Ground truth")
        ],
    )
