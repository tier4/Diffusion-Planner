"""Constraint plugins for scene search filtering.

Importing this package triggers @register decorators in all constraint modules.
"""

from scene_search.constraints import (  # noqa: F401
    neighbor_count,
    reward_threshold,
    speed_range,
    travel_distance,
)
from scene_search.constraints.registry import build, list_available
