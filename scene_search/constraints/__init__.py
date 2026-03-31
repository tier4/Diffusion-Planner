"""Constraint plugins for scene search filtering.

Importing this package triggers @register decorators in all constraint modules.
"""

from scene_search.constraints import neighbor_count, speed_range, travel_distance  # noqa: F401
from scene_search.constraints.registry import build, list_available
