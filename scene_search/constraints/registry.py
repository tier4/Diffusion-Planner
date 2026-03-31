"""Constraint plugin registry for scene search filtering.

Follows the same @register decorator pattern as diffusion_planner/model/guidance/registry.py.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scene_search.constraints.base import BaseConstraint

_REGISTRY: dict[str, type["BaseConstraint"]] = {}


def register(name: str):
    """Decorator to register a constraint class by name."""
    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(f"Constraint '{name}' already registered")
        _REGISTRY[name] = cls
        return cls
    return decorator


def build(name: str) -> "BaseConstraint":
    """Instantiate a registered constraint by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown constraint: '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]()


def list_available() -> list[str]:
    """Return names of all registered constraints (sorted for stable UI ordering)."""
    return sorted(_REGISTRY.keys())
