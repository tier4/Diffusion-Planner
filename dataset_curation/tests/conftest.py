from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def make_npz(tmp_path: Path):
    """Factory fixture that creates a synthetic NPZ with controllable properties."""
    counter = 0

    def _make(
        *,
        ego_speed: float = 5.0,
        heading_change_deg: float = 0.0,
        n_neighbors: int = 3,
        has_traffic_light: bool = False,
        seed: int | None = None,
    ) -> Path:
        nonlocal counter
        rng = np.random.default_rng(seed if seed is not None else counter)
        counter += 1

        t_past = 31
        t_future = 80

        heading_rad = np.deg2rad(heading_change_deg)
        headings = np.linspace(0, heading_rad, t_future).astype(np.float32)
        dt = 0.1
        xs = np.cumsum(np.cos(headings) * ego_speed * dt).astype(np.float32)
        ys = np.cumsum(np.sin(headings) * ego_speed * dt).astype(np.float32)
        ego_future = np.stack([xs, ys, headings], axis=-1)

        ego_past = np.zeros((t_past, 4), dtype=np.float32)
        for i in range(t_past):
            t = -(t_past - 1 - i) * dt
            ego_past[i] = [t * ego_speed, 0.0, 1.0, 0.0]

        ego_current = np.array(
            [0.0, 0.0, 1.0, 0.0, ego_speed, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

        nbr = np.zeros((320, t_past, 11), dtype=np.float32)
        for i in range(min(n_neighbors, 320)):
            offset_x = rng.uniform(5, 20)
            offset_y = rng.uniform(-5, 5)
            nbr[i, :, 0] = offset_x
            nbr[i, :, 1] = offset_y
            nbr[i, :, 2] = 1.0  # cos(0)
            nbr[i, :, 6] = 2.0  # width
            nbr[i, :, 7] = 4.5  # length
            nbr[i, :, 8] = 1.0  # is_vehicle

        lanes = np.zeros((140, 20, 33), dtype=np.float32)
        for seg in range(10):
            for pt in range(20):
                x = seg * 5.0 + pt * 0.25
                lanes[seg, pt, 0] = x
                lanes[seg, pt, 2] = 1.0  # dX
                lanes[seg, pt, 4] = -1.75  # LB_X offset
                lanes[seg, pt, 6] = 1.75  # RB_X offset
                if has_traffic_light:
                    lanes[seg, pt, 10] = 1.0  # red light

        lanes_speed_limit = np.full((140, 1), 13.9, dtype=np.float32)
        lanes_has_speed_limit = np.ones((140, 1), dtype=np.float32)

        route_lanes = np.zeros((25, 20, 33), dtype=np.float32)
        for seg in range(5):
            for pt in range(20):
                route_lanes[seg, pt, 0] = seg * 10.0 + pt * 0.5

        route_speed_limit = np.full((25, 1), 13.9, dtype=np.float32)
        route_has_speed_limit = np.ones((25, 1), dtype=np.float32)

        goal = np.array(
            [xs[-1], ys[-1], np.cos(headings[-1]), np.sin(headings[-1])], dtype=np.float32
        )

        ego_shape = np.array([2.75, 5.0, 2.0], dtype=np.float32)

        static_objects = np.zeros((5, 10), dtype=np.float32)
        polygons = np.zeros((10, 40, 3), dtype=np.float32)
        line_strings = np.zeros((60, 20, 4), dtype=np.float32)
        turn_indicators = np.zeros((30,), dtype=np.float32)

        path = tmp_path / f"scene_{counter:04d}.npz"
        np.savez(
            path,
            ego_agent_past=ego_past,
            ego_current_state=ego_current,
            ego_agent_future=ego_future,
            neighbor_agents_past=nbr,
            neighbor_agents_future=np.zeros((320, t_future, 3), dtype=np.float32),
            lanes=lanes,
            lanes_speed_limit=lanes_speed_limit,
            lanes_has_speed_limit=lanes_has_speed_limit,
            route_lanes=route_lanes,
            route_lanes_speed_limit=route_speed_limit,
            route_lanes_has_speed_limit=route_has_speed_limit,
            goal_pose=goal,
            ego_shape=ego_shape,
            static_objects=static_objects,
            polygons=polygons,
            line_strings=line_strings,
            turn_indicators=turn_indicators,
        )
        return path

    return _make


@pytest.fixture
def scene_list_json(tmp_path: Path, make_npz):
    """Create a JSON file listing N synthetic NPZ paths."""

    def _make(n: int = 20, **kwargs) -> Path:
        paths = [str(make_npz(**kwargs)) for _ in range(n)]
        json_path = tmp_path / "scenes.json"
        json_path.write_text(json.dumps(paths))
        return json_path

    return _make
