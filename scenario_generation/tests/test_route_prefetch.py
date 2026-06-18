"""RouteTimeline.prefetch warms the npz cache (the I/O-overlap speedup) without
changing what npz() returns, and is safe on out-of-range / already-cached frames.
"""

import json

import numpy as np

from scenario_generation.route_timeline import RouteTimeline


def _make_route(tmp_path, n=10):
    paths = []
    for i in range(n):
        p = tmp_path / f"r_{i:010d}.npz"
        np.savez_compressed(
            p,
            ego_agent_past=np.full((31, 3), float(i), dtype=np.float32),
            ego_shape=np.array([4.76, 7.24, 2.29], dtype=np.float32),
        )
        (tmp_path / f"r_{i:010d}.json").write_text(
            json.dumps({"x": float(i), "y": 0.0, "z": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1})
        )
        paths.append(p)
    return RouteTimeline(paths)


def test_prefetch_warms_cache_without_changing_data(tmp_path):
    tl = _make_route(tmp_path, n=10)
    assert tl._npz_cache == {}  # nothing loaded yet

    tl.prefetch(range(2, 5))
    assert set(tl._npz_cache) == {2, 3, 4}

    # The cached object is exactly what a direct npz() load returns (same identity,
    # since npz() returns the cached dict) — prefetch changes nothing about content.
    for i in (2, 3, 4):
        assert tl.npz(i) is tl._npz_cache[i]
        assert np.array_equal(tl.npz(i)["ego_agent_past"], np.full((31, 3), float(i), np.float32))


def test_prefetch_is_idempotent_and_range_safe(tmp_path):
    tl = _make_route(tmp_path, n=5)
    tl.prefetch([1])
    first = tl.npz(1)
    # Re-prefetching must not reload/replace the cached entry.
    tl.prefetch([1, 1, 1])
    assert tl.npz(1) is first
    # Out-of-range indices are ignored, not errors.
    tl.prefetch([-3, 99, 4])
    assert set(tl._npz_cache) >= {1, 4}
    assert -3 not in tl._npz_cache and 99 not in tl._npz_cache
