"""Scenario generation module for Diffusion-Planner.

Provides a structured scene representation (SceneContext) that:
- Stores all data in a world/scene coordinate frame
- Can be loaded from NPZ files used in SFT/GRPO training
- Supports extensible per-agent attributes (routes, goals, turn indicators)
- Converts to normalized model input tensors for any agent as ego
"""

from scenario_generation.npz_loader import from_npz
from scenario_generation.scene_context import Agent, AgentType, MapData, SceneContext
from scenario_generation.tensor_converter import to_model_tensors
from scenario_generation.transforms import world_to_ego_frame
from scenario_generation.visualize import draw_scene, visualize_scene

__all__ = [
    "Agent",
    "AgentType",
    "MapData",
    "SceneContext",
    "draw_scene",
    "from_npz",
    "to_model_tensors",
    "visualize_scene",
    "world_to_ego_frame",
]
