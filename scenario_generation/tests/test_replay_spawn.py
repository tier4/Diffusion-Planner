"""Tests for :class:`scenario_generation.replay.SceneNPCManager` and
:class:`SpawnConfig`.

These exercise the spawn-manager logic without the diffusion model — there is
no ``_predict_batch`` call here. ``run_route_replay`` is covered by a
manual end-to-end run (task #13); unit tests stay lightweight.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from scenario_generation.replay import (
    SceneNPCManager,
    SpawnConfig,
)
from scenario_generation.scene_context import Agent, AgentType, SceneContext

# Set LANELET2_MAP_PATH to override; otherwise fall back to a conventional
# per-user location. Tests that need the map skip when it's absent.
MAP_PATH = Path(
    os.environ.get("LANELET2_MAP_PATH") or (Path.home() / "autoware_map" / "lanelet2_map.osm")
)


# ── SpawnConfig ──────────────────────────────────────────────────────────────


def test_spawn_config_json_round_trip(tmp_path: Path) -> None:
    c = SpawnConfig(max_active_npcs=12, goal_tolerance_m=1.5, max_steps=1000)
    p = tmp_path / "cfg.json"
    c.to_json(p)
    c2 = SpawnConfig.from_json(p)
    assert c == c2


def test_spawn_config_ignores_unknown_keys(tmp_path: Path) -> None:
    """Comment-style underscore-prefixed keys in a config JSON must not
    crash the constructor."""
    raw = {
        "_comment": "example comment",
        "max_active_npcs": 4,
        "_note": "another",
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(raw))
    c = SpawnConfig.from_json(p)
    assert c.max_active_npcs == 4


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"ego_length": -1.0}, "ego dimensions must be positive"),
        ({"ego_width": 0.0}, "ego dimensions must be positive"),
        ({"ego_wheelbase": -0.1}, "ego dimensions must be positive"),
        ({"ego_length": 3.0, "ego_wheelbase": 4.0}, "ego_wheelbase must be <= ego_length"),
        ({"ego_max_steer": 0.0}, "ego_max_steer must be in"),
        ({"ego_max_steer": 1.6}, "ego_max_steer must be in"),
        ({"inference_delay": -1}, "inference_delay must be non-negative"),
        ({"max_steps": 0}, "max_steps must be >= 1"),
        ({"ego_init_speed": -0.5}, "ego_init_speed must be >= 0"),
    ],
)
def test_spawn_config_validate_rejects_invalid_values(kwargs: dict, needle: str) -> None:
    """__post_init__ should fail fast on out-of-range fields."""
    with pytest.raises(ValueError, match=needle):
        SpawnConfig(**kwargs)


def test_spawn_config_validate_catches_post_construction_mutation() -> None:
    """validate() must be re-callable after direct field mutation (the
    CLI-override path in replay.main bypasses __post_init__)."""
    c = SpawnConfig()
    c.max_steps = 0
    with pytest.raises(ValueError, match="max_steps must be >= 1"):
        c.validate()


def test_spawn_config_validate_accepts_valid_ego_init_speed() -> None:
    # None, 0.0, and positive speeds all pass.
    SpawnConfig(ego_init_speed=None)
    SpawnConfig(ego_init_speed=0.0)
    SpawnConfig(ego_init_speed=1.75)


def test_spawn_config_defaults_match_user_spec() -> None:
    """User-required values from the April 2026 planning session."""
    c = SpawnConfig()
    assert c.max_active_npcs == 8  # user confirmed default
    assert c.despawn_distance == 120.0  # user-required
    assert c.goal_tolerance_m == 2.0  # user: "within 2 meters"
    assert c.max_steps == 6000  # user: "more than 6000 timesteps"
    assert c.ego_overlap_ratio == 0.3  # user: "mixed 70/30"
    assert c.goal_pass_window_m == 25.0  # added when v3 showed ego passing goal at d=16.5m


# ── SceneNPCManager with a real map ──────────────────────────────────────────


@pytest.fixture(scope="module")
def builder():
    if not MAP_PATH.exists():
        pytest.skip(f"Lanelet2 map not found at {MAP_PATH}")
    from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder

    return LaneletSceneBuilder(str(MAP_PATH))


def _make_ego_scene(builder, start_ll_id: int) -> tuple[SceneContext, list[int]]:
    """Build a minimal scene containing just an ego on the given lanelet."""
    cl = builder._cache[start_ll_id].raw_centerline
    pos = cl[len(cl) // 2].astype(np.float32)
    heading = 0.0
    history, history_ids = builder.generate_history(pos, heading, 5.0, start_ll_id)
    route_ll_ids = builder.find_route(start_ll_id, 120.0)
    route_lanes, rsl, rhsl = builder._route_to_33dim(route_ll_ids)
    ego = Agent(
        id="ego",
        agent_type=AgentType.VEHICLE,
        length=4.5,
        width=1.9,
        wheelbase=2.9,
        past_trajectory=history,
        past_velocities=np.zeros((history.shape[0], 2), dtype=np.float32),
        acceleration=np.zeros(2, dtype=np.float32),
        goal_pose=np.array([pos[0] + 20, pos[1], heading], dtype=np.float32),
        route_lanes=route_lanes,
        route_speed_limit=rsl,
        route_has_speed_limit=rhsl,
    )
    all_ids = sorted(set(route_ll_ids) | set(history_ids))
    map_data = builder._build_map_data(all_ids)
    scene = SceneContext(agents=[ego], map_data=map_data, ego_agent_id="ego")
    return scene, route_ll_ids


def test_spawn_one_produces_valid_neighbor(builder) -> None:
    # Pick a lanelet that's likely to have neighbors nearby.
    all_ids = list(builder._cache.keys())
    start_id = all_ids[len(all_ids) // 2]
    scene, ego_route = _make_ego_scene(builder, start_id)

    cfg = SpawnConfig(
        max_active_npcs=4,
        spawn_probability=1.0,
        seed=42,
        min_spawn_distance=5.0,
        max_spawn_distance=80.0,
        forward_bias=0.5,
    )
    mgr = SceneNPCManager(builder, ego_route, cfg)
    mgr.register_known_lanelets(ego_route)

    mgr.tick(scene)
    # At least one spawn attempt at prob=1.0 should succeed on a dense map.
    npcs = [a for a in scene.agents if a.id != "ego"]
    if not npcs:
        # Dense-map assumption can fail at some lanelets; retry a few times.
        for _ in range(10):
            mgr.tick(scene)
            npcs = [a for a in scene.agents if a.id != "ego"]
            if npcs:
                break
    assert npcs, "spawn manager never produced a neighbor even after 10 ticks"

    nb = npcs[0]
    # Valid 31-step history with the current pose at index -1.
    assert nb.past_trajectory.shape == (31, 3)
    # Non-degenerate speed.
    v = nb.past_velocities[-1]
    assert np.linalg.norm(v) > 0.0
    # Distance to ego within spawn bounds.
    d = float(np.linalg.norm(nb.current_position - scene.ego_agent.current_position))
    assert cfg.min_spawn_distance - 1.0 <= d <= cfg.max_spawn_distance + 1.0


def test_despawn_drops_far_neighbors(builder) -> None:
    all_ids = list(builder._cache.keys())
    start_id = all_ids[len(all_ids) // 2]
    scene, ego_route = _make_ego_scene(builder, start_id)

    cfg = SpawnConfig(despawn_distance=50.0, spawn_probability=0.0, seed=1)
    mgr = SceneNPCManager(builder, ego_route, cfg)
    mgr.register_known_lanelets(ego_route)

    # Inject a synthetic far-away neighbor.
    ego_pos = scene.ego_agent.current_position
    far_pos = np.array([ego_pos[0] + 200.0, ego_pos[1] + 0.0, 0.0], dtype=np.float32)
    far_history = np.tile(far_pos, (31, 1))
    near_pos = np.array([ego_pos[0] + 10.0, ego_pos[1] + 0.0, 0.0], dtype=np.float32)
    near_history = np.tile(near_pos, (31, 1))
    for label, hist in (("far", far_history), ("near", near_history)):
        scene.agents.append(
            Agent(
                id=label,
                agent_type=AgentType.VEHICLE,
                length=4.5,
                width=1.9,
                wheelbase=2.9,
                past_trajectory=hist,
                past_velocities=np.zeros((31, 2), dtype=np.float32),
                acceleration=np.zeros(2, dtype=np.float32),
            )
        )

    mgr.tick(scene)
    ids = [a.id for a in scene.agents]
    assert "ego" in ids
    assert "near" in ids
    assert "far" not in ids, "neighbor > despawn_distance must be removed"


def test_overlap_ratio_biases_neighbor_routes(builder) -> None:
    """With ``ego_overlap_ratio`` = 1.0 every spawned NPC route should share
    at least one lanelet with the ego route (when reachable). This does not
    require any statistical power beyond a handful of successful spawns."""
    all_ids = list(builder._cache.keys())
    start_id = all_ids[len(all_ids) // 2]
    scene, ego_route = _make_ego_scene(builder, start_id)
    ego_set = set(ego_route)

    cfg = SpawnConfig(
        max_active_npcs=6,
        spawn_probability=1.0,
        seed=7,
        ego_overlap_ratio=1.0,
    )
    mgr = SceneNPCManager(builder, ego_route, cfg)
    mgr.register_known_lanelets(ego_route)

    # Try many ticks to get several spawned neighbors.
    for _ in range(30):
        mgr.tick(scene)

    npcs = [a for a in scene.agents if a.id != "ego"]
    # We cannot assert every neighbor overlaps (the map may have candidates
    # where no forward route touches the ego route after 5 retries — the
    # code falls back to a random route). But at least one should overlap.
    route_sets = []
    for nb in npcs:
        # Walk the non-zero centerline points of nb.route_lanes back to ids by
        # matching against the cache.
        lanes_xy = nb.route_lanes[:, 0, :2]  # first-point of each route slot
        found_ids = set()
        for xy in lanes_xy:
            if abs(xy[0]) < 1e-6 and abs(xy[1]) < 1e-6:
                continue
            snapped = builder.snap_to_nearest_ll(xy)
            if snapped is not None:
                found_ids.add(snapped)
        route_sets.append(found_ids)

    any_overlap = any((rs & ego_set) for rs in route_sets)
    assert npcs and any_overlap, (
        f"ego_overlap_ratio=1.0 produced no neighbor with overlapping route "
        f"across {len(npcs)} spawns"
    )
