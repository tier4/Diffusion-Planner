"""Guidance framework for Diffusion Planner.

Public API
----------
GuidanceComposer    -- composes multiple guidance functions; drop-in for GuidanceWrapper
GuidanceSetConfig   -- serialisable config for a set of guidance functions
GuidanceConfig      -- per-function config (name, enabled, scale, params)
register            -- class decorator to register a new BaseGuidance subclass
build               -- instantiate a guidance function from GuidanceConfig
list_available      -- list all registered guidance function names
"""

# Import all guidance modules to trigger @register decorators at import time.
from . import (
    anchor_following,  # noqa: F401
    centerline_following,  # noqa: F401
    collision,  # noqa: F401
    lane_keeping,  # noqa: F401
    lateral_guidance,  # noqa: F401
    longitudinal_guidance,  # noqa: F401
    road_border,  # noqa: F401
    route_centerline_following,  # noqa: F401
    route_following,  # noqa: F401
    speed_guidance,  # noqa: F401
)
from .composer import GuidanceComposer
from .config import GuidanceConfig, GuidanceSetConfig
from .registry import build, list_available, register

__all__ = [
    "GuidanceComposer",
    "GuidanceConfig",
    "GuidanceSetConfig",
    "register",
    "build",
    "list_available",
]
