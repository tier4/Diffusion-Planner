"""Registry for BaseGuidance subclasses.

Usage
-----
Decorate a BaseGuidance subclass with @register to make it discoverable by name:

    from diffusion_planner.model.guidance.registry import register

    @register
    class MyGuidance(BaseGuidance):
        name = "my_guidance"
        ...

Then build an instance from a GuidanceConfig:

    from diffusion_planner.model.guidance.registry import build
    fn = build(GuidanceConfig(name="my_guidance"))
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseGuidance
    from .config import GuidanceConfig

_REGISTRY: dict[str, type["BaseGuidance"]] = {}


def register(cls):
    """Class decorator that registers a BaseGuidance subclass by its ``name`` attribute."""
    assert hasattr(cls, "name"), f"{cls} must define a `name` class attribute"
    _REGISTRY[cls.name] = cls
    return cls


def build(config: "GuidanceConfig", **kwargs) -> "BaseGuidance":
    """Instantiate a guidance function from its GuidanceConfig."""
    if config.name not in _REGISTRY:
        raise KeyError(
            f"Guidance '{config.name}' not registered. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[config.name](config, **kwargs)


def list_available() -> list[str]:
    """Return sorted list of all registered guidance function names."""
    return sorted(_REGISTRY.keys())
