"""Shared pytest fixtures for scenario_generation tests."""
# ruff: noqa: I001, E402

from __future__ import annotations

# Select Agg backend before any pyplot import in the test session (headless CI).
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pytest

from scenario_generation.scene_context import Agent, AgentType, MapData, SceneContext


@pytest.fixture
def synthetic_scene() -> SceneContext:
    """Build a minimal SceneContext with 4 agents for unit testing."""
    T_past = 31

    def _make_agent(
        agent_id: str, x: float, y: float, heading: float, speed: float,
    ) -> Agent:
        traj = np.zeros((T_past, 3), dtype=np.float32)
        vels = np.zeros((T_past, 2), dtype=np.float32)
        for t in range(T_past):
            frac = t / (T_past - 1)
            traj[t] = [x - (1 - frac) * speed * 3.0 * np.cos(heading),
                        y - (1 - frac) * speed * 3.0 * np.sin(heading),
                        heading]
            vels[t] = [speed * np.cos(heading), speed * np.sin(heading)]

        route = np.zeros((25, 20, 33), dtype=np.float32)
        for pt in range(20):
            route[0, pt, 0] = x + pt * 0.5 * np.cos(heading)
            route[0, pt, 1] = y + pt * 0.5 * np.sin(heading)

        return Agent(
            id=agent_id,
            agent_type=AgentType.VEHICLE,
            length=4.5, width=1.8, wheelbase=2.9,
            past_trajectory=traj,
            past_velocities=vels,
            acceleration=np.zeros(2, dtype=np.float32),
            goal_pose=np.array([x + 50 * np.cos(heading),
                                y + 50 * np.sin(heading),
                                heading], dtype=np.float32),
            route_lanes=route,
            route_speed_limit=np.zeros((25, 1), dtype=np.float32),
            route_has_speed_limit=np.zeros((25, 1), dtype=bool),
            turn_indicators=np.zeros(T_past, dtype=np.int32),
        )

    agents = [
        _make_agent("ego", 0.0, 0.0, 0.0, 3.0),
        _make_agent("nb_1", 10.0, 3.0, 0.1, 2.5),
        _make_agent("nb_2", -5.0, -4.0, -0.2, 4.0),
        _make_agent("nb_3", 20.0, 1.0, 0.0, 5.0),
    ]

    lanes = np.zeros((140, 20, 33), dtype=np.float32)
    for i in range(5):
        for pt in range(20):
            lanes[i, pt, 0] = i * 10.0 + pt * 0.5
            lanes[i, pt, 1] = (i - 2) * 3.5
            lanes[i, pt, 2] = 0.5
            lanes[i, pt, 3] = 0.0

    map_data = MapData(
        lanes=lanes,
        lanes_speed_limit=np.full((140, 1), 8.33, dtype=np.float32),
        lanes_has_speed_limit=np.ones((140, 1), dtype=bool),
        polygons=np.zeros((10, 40, 3), dtype=np.float32),
        line_strings=np.zeros((60, 20, 4), dtype=np.float32),
        static_objects=np.zeros((5, 10), dtype=np.float32),
    )

    return SceneContext(
        agents=agents,
        map_data=map_data,
        ego_agent_id="ego",
    )


@pytest.fixture
def map_snippets_dir() -> Path:
    """Return the repo-root .map_snippets/ directory, skip if missing."""
    d = Path(__file__).resolve().parents[2] / ".map_snippets"
    if not d.exists() or not list(d.glob("*.pkl")):
        pytest.skip("No .map_snippets/*.pkl available")
    return d


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Return a temporary output directory for test artifacts."""
    out = tmp_path / "output"
    out.mkdir()
    return out
