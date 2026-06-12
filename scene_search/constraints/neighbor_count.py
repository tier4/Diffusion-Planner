"""Constraint: filter scenes by number of active neighbors within a radius."""

import numpy as np

from scene_search.constraints.base import BaseConstraint
from scene_search.constraints.registry import register


@register("neighbor_count")
class NeighborCountConstraint(BaseConstraint):
    name = "Neighbor Count"
    description = "Filter by number of active neighbors within a radius"

    def get_params_spec(self) -> dict:
        return {
            "min_count": {"type": "int", "default": 1, "label": "Min neighbors", "min": 0, "max": 32, "step": 1},
            "max_count": {"type": "int", "default": 32, "label": "Max neighbors", "min": 0, "max": 32, "step": 1},
            "within_radius": {"type": "float", "default": 30.0, "label": "Within radius (m)", "min": 1.0, "max": 100.0, "step": 1.0},
        }

    def filter(self, npz_path: str, npz_data: np.lib.npyio.NpzFile,
               params: dict, entry: dict | None = None) -> bool:
        neighbors = npz_data["neighbor_agents_past"]  # (32, 21, 11)
        current = neighbors[:, -1, :]  # (32, 11) — last timestep (t=0)
        # A neighbor is active if any of its features are nonzero
        active = np.any(current != 0, axis=-1)  # (32,)
        if not np.any(active):
            return params["min_count"] <= 0

        positions = current[active, :2]  # (N, 2) — x, y in ego frame
        dists = np.linalg.norm(positions, axis=-1)
        n_within = int(np.sum(dists <= params["within_radius"]))
        return params["min_count"] <= n_within <= params["max_count"]
