"""``merged_percentile`` must not depend on the order digests are pooled in."""

from __future__ import annotations

import random

import numpy as np
import pytest

from scenario_generation.metrics.tdigest import merged_percentile, tdigest_dict_from_values


def digests(n_digests=7, seed=0):
    """Per-segment digests with deliberately different shapes and sample counts."""
    rng = np.random.default_rng(seed)
    return [
        tdigest_dict_from_values(rng.gamma(shape=1.0 + i, scale=2.0, size=50 + 37 * i))
        for i in range(n_digests)
    ]


@pytest.mark.parametrize("percentile", [5, 50, 95])
def test_result_is_independent_of_pooling_order(percentile):
    """Under DDP the shards arrive grouped by rank, so which rank ran a route would
    otherwise shift the metric."""
    ds = digests()
    expected = merged_percentile(ds, percentile)
    rng = random.Random(1234)
    for _ in range(20):
        shuffled = ds[:]
        rng.shuffle(shuffled)
        assert merged_percentile(shuffled, percentile) == expected


def test_result_is_independent_of_how_shards_are_grouped():
    """Round-robin over 1/2/3/4/8 ranks pools the same digests in different orders."""
    ds = digests(n_digests=12)
    expected = merged_percentile(ds, 5)
    for world_size in (1, 2, 3, 4, 8):
        by_rank = [d for rank in range(world_size) for d in ds[rank::world_size]]
        assert merged_percentile(by_rank, 5) == expected


def test_empty_input_reports_infinity():
    assert merged_percentile([], 5) == float("inf")
    assert merged_percentile([{"centroids": []}], 5) == float("inf")


def test_percentile_still_tracks_the_pooled_distribution():
    """Guard against the merge degenerating: p50 must land near the true median."""
    rng = np.random.default_rng(7)
    parts = [rng.normal(10.0, 2.0, size=400) for _ in range(5)]
    ds = [tdigest_dict_from_values(p) for p in parts]
    assert merged_percentile(ds, 50) == pytest.approx(np.median(np.concatenate(parts)), abs=0.2)
