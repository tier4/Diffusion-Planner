"""The track-anchor cache is a pure memoisation of ``_build_neighbor_interp``: it must hit when
the key repeats and miss whenever any part of the key differs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scenario_generation.reproducer_rollout import (
    _INTERP_ANCHOR_CACHE,
    _build_neighbor_interp,
    _neighbor_interp_cached,
)
from scenario_generation.route_timeline import RouteTimeline


@pytest.fixture(autouse=True)
def _clear_cache():
    _INTERP_ANCHOR_CACHE.clear()
    yield
    _INTERP_ANCHOR_CACHE.clear()


def _write_route(npz_dir, n: int, n_slots: int = 3):
    # no sibling JSON, so the sidecar directory is what supplies the track ids
    npz_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        nb = np.zeros((n_slots, 31, 11), dtype=np.float32)
        for s in range(n_slots):
            nb[s, -1, 0] = 10.0 * s + i  # x advances every frame -> anchors are kept, not dropped
            nb[s, -1, 1] = 2.0 * s
            nb[s, -1, 2] = 1.0  # cos(heading)
        p = npz_dir / f"f_{i:010d}.npz"
        np.savez_compressed(p, neighbor_agents_past=nb, ego_shape=np.zeros(3, dtype=np.float32))
        paths.append(p)
    return paths


def _write_sidecars(sidecar_dir, n: int, uuid_prefix: str, n_slots: int = 3):
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (sidecar_dir / f"f_{i:010d}.json").write_text(
            json.dumps(
                {
                    "x": float(i),
                    "y": 0.0,
                    "z": 0.0,
                    "qx": 0,
                    "qy": 0,
                    "qz": 0,
                    "qw": 1,
                    "neighbor_ids": [f"{uuid_prefix}{s}" for s in range(n_slots)],
                }
            )
        )


def _timeline(tmp_path, n=12, uuid_prefix="a", sidecar_name="sc_a"):
    npz_dir = tmp_path / "npz"
    paths = _write_route(npz_dir, n) if not npz_dir.exists() else sorted(npz_dir.glob("*.npz"))
    sidecar_dir = tmp_path / sidecar_name
    _write_sidecars(sidecar_dir, n, uuid_prefix)
    return RouteTimeline(paths, sidecar_dir=sidecar_dir)


def test_second_call_returns_the_cached_object_without_rebuilding(tmp_path):
    tl = _timeline(tmp_path)
    first = _neighbor_interp_cached(tl, 0, len(tl))
    assert len(_INTERP_ANCHOR_CACHE) == 1

    second = _neighbor_interp_cached(tl, 0, len(tl))
    assert second is first  # the same object, not merely an equal one
    assert len(_INTERP_ANCHOR_CACHE) == 1


def test_cached_anchors_equal_a_direct_build(tmp_path):
    tl = _timeline(tmp_path)
    direct = _build_neighbor_interp(tl, 0, len(tl))
    cached = _neighbor_interp_cached(tl, 0, len(tl))

    assert set(cached) == set(direct)
    for uuid, (idxs, xy, hd) in direct.items():
        assert np.array_equal(cached[uuid][0], idxs)
        assert np.array_equal(cached[uuid][1], xy)
        assert np.array_equal(cached[uuid][2], hd)


def test_a_fresh_timeline_over_the_same_files_still_hits(tmp_path):
    tl_first = _timeline(tmp_path)
    first = _neighbor_interp_cached(tl_first, 0, len(tl_first))

    tl_again = _timeline(tmp_path)
    assert tl_again is not tl_first
    assert _neighbor_interp_cached(tl_again, 0, len(tl_again)) is first
    assert len(_INTERP_ANCHOR_CACHE) == 1


def test_a_different_span_of_one_route_does_not_share(tmp_path):
    tl = _timeline(tmp_path)
    whole = _neighbor_interp_cached(tl, 0, len(tl))
    part = _neighbor_interp_cached(tl, 0, len(tl) - 2)

    assert part is not whole
    assert len(_INTERP_ANCHOR_CACHE) == 2


def test_a_different_sidecar_dir_does_not_share(tmp_path):
    """Same NPZ files, different sidecars -> different track ids -> different anchors."""
    tl_a = _timeline(tmp_path, uuid_prefix="a", sidecar_name="sc_a")
    anchors_a = _neighbor_interp_cached(tl_a, 0, len(tl_a))

    tl_b = _timeline(tmp_path, uuid_prefix="b", sidecar_name="sc_b")
    anchors_b = _neighbor_interp_cached(tl_b, 0, len(tl_b))

    assert anchors_b is not anchors_a
    assert len(_INTERP_ANCHOR_CACHE) == 2
    assert set(anchors_a) == {"a0", "a1", "a2"}
    assert set(anchors_b) == {"b0", "b1", "b2"}


def test_eps_is_part_of_the_key(tmp_path):
    tl = _timeline(tmp_path)
    a = _neighbor_interp_cached(tl, 0, len(tl), eps=0.1)
    b = _neighbor_interp_cached(tl, 0, len(tl), eps=5.0)
    assert b is not a
    assert len(_INTERP_ANCHOR_CACHE) == 2


def test_a_hit_is_visible_in_the_timers(tmp_path):
    tl = _timeline(tmp_path)
    _neighbor_interp_cached(tl, 0, len(tl))
    assert tl.timers.calls.get("interp_build") == 1
    assert "interp_build_cached" not in tl.timers.calls

    _neighbor_interp_cached(tl, 0, len(tl))
    assert tl.timers.calls.get("interp_build") == 1  # not rebuilt
    assert tl.timers.calls.get("interp_build_cached") == 1
