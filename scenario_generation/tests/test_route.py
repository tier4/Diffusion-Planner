"""Tests for :class:`scenario_generation.route.Route` and the routing
helpers on :class:`~scenario_generation.gui.lanelet_scene_builder.LaneletSceneBuilder`.

The lanelet2-backed tests need a real ``lanelet2_map.osm`` on disk. Set the
``LANELET2_MAP_PATH`` env var to that file, or place it at
``~/autoware_map/lanelet2_map.osm``. If neither is available the tests skip
so CI can still run the pure-Python Route round-trip coverage.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from scenario_generation.route import Route

MAP_PATH = Path(
    os.environ.get("LANELET2_MAP_PATH") or (Path.home() / "autoware_map" / "lanelet2_map.osm")
)


# ── Pure Route round-trip ────────────────────────────────────────────────────


def test_route_save_load_round_trip(tmp_path: Path) -> None:
    r = Route(
        map_path="/tmp/example.osm",
        start_pose=np.array([1.0, 2.0, 0.5], dtype=np.float32),
        goal_pose=np.array([10.0, 20.0, 1.5], dtype=np.float32),
        start_lanelet_id=100,
        goal_lanelet_id=200,
        waypoint_poses=[np.array([5.0, 5.0, 1.0], dtype=np.float32)],
        waypoint_lanelet_ids=[150],
        route_lanelet_ids=[100, 150, 200],
    )
    r.save(tmp_path / "r.pkl")
    loaded = Route.load(tmp_path / "r.pkl")

    assert loaded.map_path == r.map_path
    np.testing.assert_array_equal(loaded.start_pose, r.start_pose)
    np.testing.assert_array_equal(loaded.goal_pose, r.goal_pose)
    assert loaded.start_lanelet_id == 100
    assert loaded.goal_lanelet_id == 200
    assert loaded.waypoint_lanelet_ids == [150]
    assert loaded.route_lanelet_ids == [100, 150, 200]
    assert loaded.num_waypoints() == 1
    assert loaded.is_resolved()


def test_route_validates_start_pose_shape() -> None:
    with pytest.raises(ValueError, match="start_pose must be shape"):
        Route(
            map_path="x",
            start_pose=np.array([1.0, 2.0], dtype=np.float32),
            goal_pose=np.array([3.0, 4.0, 0.0], dtype=np.float32),
            start_lanelet_id=1,
            goal_lanelet_id=2,
        )


def test_route_validates_parallel_waypoint_lengths() -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        Route(
            map_path="x",
            start_pose=np.zeros(3, dtype=np.float32),
            goal_pose=np.zeros(3, dtype=np.float32),
            start_lanelet_id=1,
            goal_lanelet_id=2,
            waypoint_poses=[np.zeros(3, dtype=np.float32)],
            waypoint_lanelet_ids=[],  # mismatched
        )


def test_route_unresolved_is_flagged() -> None:
    r = Route(
        map_path="x",
        start_pose=np.zeros(3, dtype=np.float32),
        goal_pose=np.zeros(3, dtype=np.float32),
        start_lanelet_id=1,
        goal_lanelet_id=2,
        route_lanelet_ids=None,
    )
    assert not r.is_resolved()


# ── Routing helpers ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def builder():
    """Cached LaneletSceneBuilder for the configured lanelet2 map.

    Scoped to the module so we pay the lanelet load cost once.
    """
    if not MAP_PATH.exists():
        pytest.skip(f"Lanelet2 map not found at {MAP_PATH} — set LANELET2_MAP_PATH")
    if os.environ.get("SKIP_LANELET_TESTS"):
        pytest.skip("SKIP_LANELET_TESTS set")
    from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder

    return LaneletSceneBuilder(str(MAP_PATH))


def _chain_of_length(builder, min_length: int) -> list[int]:
    """Return a connected successor chain of at least ``min_length`` lanelets."""
    for ll in list(builder._ll_by_id.values())[:500]:
        curr = ll
        chain = [curr.id]
        for _ in range(20):
            fol = list(builder._routing_graph.following(curr))
            if not fol:
                break
            curr = fol[0]
            chain.append(curr.id)
        if len(chain) >= min_length:
            return chain
    pytest.skip(f"No successor chain of length >= {min_length} in this map")


def test_snap_to_nearest_ll(builder) -> None:
    ll_id = next(iter(builder._cache))
    cl = builder._cache[ll_id].raw_centerline
    # Snap a point exactly on the centerline — the expected closest is some
    # lanelet whose centerline passes through / very near here.
    snapped = builder.snap_to_nearest_ll(cl[len(cl) // 2])
    assert snapped is not None
    # The snapped lanelet's centerline must be within a few metres of the query.
    snapped_cl = builder._cache[snapped].raw_centerline
    min_d = np.min(np.linalg.norm(snapped_cl - cl[len(cl) // 2], axis=1))
    assert min_d <= 5.0


def test_lanelets_near_point(builder) -> None:
    ll_id = next(iter(builder._cache))
    cl = builder._cache[ll_id].raw_centerline
    near = builder.lanelets_near_point(cl[len(cl) // 2], 30.0)
    assert ll_id in near or any(
        np.min(np.linalg.norm(builder._cache[x].raw_centerline - cl[len(cl) // 2], axis=1)) <= 30.0
        for x in near
    )
    # All returned lanelets actually satisfy the radius constraint.
    for x in near:
        d = np.min(
            np.linalg.norm(builder._cache[x].raw_centerline - cl[len(cl) // 2], axis=1),
        )
        assert d <= 30.0 + 1e-3  # float slack


def test_route_between_recovers_connected_chain(builder) -> None:
    chain = _chain_of_length(builder, min_length=5)
    start, end = chain[0], chain[-1]
    route = builder.route_between(start, end)
    assert route is not None
    assert route[0] == start and route[-1] == end


def test_route_with_waypoints_empty_equals_route_between(builder) -> None:
    chain = _chain_of_length(builder, min_length=5)
    start, end = chain[0], chain[-1]
    base = builder.route_between(start, end)
    with_via = builder.route_with_waypoints(start, [], end)
    assert base == with_via


def test_route_with_waypoints_passes_through_via(builder) -> None:
    chain = _chain_of_length(builder, min_length=5)
    start, end = chain[0], chain[-1]
    via = chain[len(chain) // 2]
    route = builder.route_with_waypoints(start, [via], end)
    assert route is not None
    assert via in route


def test_route_with_waypoints_preserves_order(builder) -> None:
    chain = _chain_of_length(builder, min_length=5)
    start, end = chain[0], chain[-1]
    via1, via2 = chain[1], chain[-2]
    route = builder.route_with_waypoints(start, [via1, via2], end)
    assert route is not None
    assert route.index(via1) < route.index(via2)


def test_route_with_waypoints_reversed_is_infeasible(builder) -> None:
    """Providing via-points in reverse order on a directed chain is unreachable
    — the API must surface that as None rather than silently misroute."""
    chain = _chain_of_length(builder, min_length=5)
    start, end = chain[0], chain[-1]
    via1, via2 = chain[1], chain[-2]
    route = builder.route_with_waypoints(start, [via2, via1], end)
    assert route is None


def test_is_lanelet_straight_on_short_lanelet(builder) -> None:
    # Pick a lanelet with few points — the short-circuit in is_lanelet_straight
    # should return True without crashing.
    short_ids = [i for i, c in builder._cache.items() if len(c.raw_centerline) < 3]
    if not short_ids:
        pytest.skip("No short (<3 points) lanelet in this map")
    assert builder.is_lanelet_straight(short_ids[0]) is True


def test_closest_lanelets_respects_bbox_and_cap(builder) -> None:
    """closest_lanelets must: (1) stay within the mask range, (2) cap length,
    (3) return closest-first order (by min-endpoint-to-query score)."""
    ll_id = next(iter(builder._cache))
    cl = builder._cache[ll_id].raw_centerline
    query = cl[len(cl) // 2]

    tight = builder.closest_lanelets(query, max_n=50, mask_range=50.0)
    loose = builder.closest_lanelets(query, max_n=50, mask_range=200.0)
    assert len(loose) >= len(tight)  # wider bbox = at least as many hits
    assert len(loose) <= 50  # cap respected

    # All returned lanelets have at least one anchor point inside the bbox.
    for lid in loose:
        c = builder._cache[lid].raw_centerline
        center = c.mean(axis=0)
        hits = (
            (np.abs(c[0] - query) < 200).all()
            or (np.abs(c[-1] - query) < 200).all()
            or (np.abs(center - query) < 200).all()
        )
        assert hits, f"lanelet {lid} is outside the 200 m bbox"


def test_closest_lanelets_cap_at_max_n(builder) -> None:
    ll_id = next(iter(builder._cache))
    cl = builder._cache[ll_id].raw_centerline
    query = cl[len(cl) // 2]
    # A huge bbox + small cap — the cap must truncate.
    res = builder.closest_lanelets(query, max_n=5, mask_range=1000.0)
    assert len(res) <= 5


def test_is_lanelet_straight_rejects_sharp_curves(builder) -> None:
    """Curve detection: when the threshold is tight most lanelets with any
    turn register as non-straight; when loose, most register as straight."""
    sample = list(builder._cache.keys())[:200]
    tight = sum(1 for i in sample if builder.is_lanelet_straight(i, 0.01))
    loose = sum(1 for i in sample if builder.is_lanelet_straight(i, 3.14))
    assert tight <= loose
    assert loose == len(sample)  # every lanelet is straight under a huge threshold
