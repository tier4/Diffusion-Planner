"""Constraint: filter scenes by ego speed at t=0."""

import numpy as np

from scene_search.constraints.base import BaseConstraint
from scene_search.constraints.registry import register


@register("speed_range")
class SpeedRangeConstraint(BaseConstraint):
    name = "Ego Speed"
    description = "Filter by ego speed at t=0 (km/h)"

    def get_params_spec(self) -> dict:
        return {
            "min_speed_kmh": {"type": "float", "default": 0.0, "label": "Min speed (km/h)", "min": 0.0, "max": 100.0, "step": 1.0},
            "max_speed_kmh": {"type": "float", "default": 100.0, "label": "Max speed (km/h)", "min": 0.0, "max": 100.0, "step": 1.0},
        }

    def filter(self, npz_path: str, npz_data: np.lib.npyio.NpzFile, params: dict) -> bool:
        # ego_current_state: [x, y, cos, sin, vx, vy, ax, ay, steer, yaw_rate]
        state = npz_data["ego_current_state"]  # (10,)
        vx, vy = state[4], state[5]
        speed_ms = np.sqrt(vx ** 2 + vy ** 2)
        speed_kmh = float(speed_ms * 3.6)
        return params["min_speed_kmh"] <= speed_kmh <= params["max_speed_kmh"]
