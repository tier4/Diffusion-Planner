"""``RouteTimeline.neighbor_last``: same values as the full read, without evicting the cache."""

import json

import numpy as np
import pytest

from scenario_generation.route_timeline import RouteTimeline

N_FRAMES = 6
N_NEIGHBORS = 4


def _write_frame(tmp_path, i: int):
    p = tmp_path / f"route_{i:010d}.npz"
    nb = np.arange(N_NEIGHBORS * 31 * 11, dtype=np.float32).reshape(N_NEIGHBORS, 31, 11) + i
    np.savez_compressed(
        p,
        neighbor_agents_past=nb,
        ego_agent_past=np.zeros((31, 3), dtype=np.float32),
        ego_shape=np.array([4.0, 2.0, 1.5], dtype=np.float32),
        turn_indicators=np.zeros(31, dtype=np.int64),
        lanes=np.zeros((140, 20, 33), dtype=np.float32),  # the bulk npz() would decompress
    )
    (tmp_path / f"route_{i:010d}.json").write_text(
        json.dumps(
            {
                "timestamp": float(i) * 0.1,
                "x": float(i),
                "y": 0.0,
                "z": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            }
        )
    )
    return p


@pytest.fixture
def timeline(tmp_path):
    return RouteTimeline([_write_frame(tmp_path, i) for i in range(N_FRAMES)])


def test_values_match_the_full_read(timeline):
    for i in range(N_FRAMES):
        expected = timeline.npz(i)["neighbor_agents_past"][:, -1]
        assert np.array_equal(timeline.neighbor_last(i), expected)


def test_uncached_frame_is_not_cached(timeline):
    assert not timeline._npz_cache, "fixture should start with a cold cache"

    for i in range(N_FRAMES):
        timeline.neighbor_last(i)

    assert not timeline._npz_cache, "the narrow read must leave the frame cache untouched"


def test_cached_frame_is_served_from_the_cache(timeline):
    full = timeline.npz(3)  # populates the cache
    assert 3 in timeline._npz_cache

    got = timeline.neighbor_last(3)

    assert np.array_equal(got, full["neighbor_agents_past"][:, -1])
    assert np.shares_memory(got, full["neighbor_agents_past"])  # a view of the cached array
