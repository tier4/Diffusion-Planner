"""Constraint: filter scenes by ground-truth trajectory travel distance."""

import numpy as np

from scene_search.constraints.base import BaseConstraint
from scene_search.constraints.registry import register


@register("travel_distance")
class TravelDistanceConstraint(BaseConstraint):
    name = "Travel Distance"
    description = "Filter by total GT trajectory travel distance (meters)"

    def get_params_spec(self) -> dict:
        return {
            "min_distance": {"type": "float", "default": 5.0, "label": "Min distance (m)", "min": 0.0, "max": 200.0, "step": 1.0},
            "max_distance": {"type": "float", "default": 200.0, "label": "Max distance (m)", "min": 0.0, "max": 200.0, "step": 1.0},
        }

    def filter(self, npz_path: str, npz_data: np.lib.npyio.NpzFile, params: dict) -> bool:
        fut = npz_data["ego_agent_future"]  # (80, 3) — [x, y, yaw_rad]
        dx = np.diff(fut[:, 0])
        dy = np.diff(fut[:, 1])
        travel_dist = float(np.sqrt(dx ** 2 + dy ** 2).sum())
        return params["min_distance"] <= travel_dist <= params["max_distance"]
