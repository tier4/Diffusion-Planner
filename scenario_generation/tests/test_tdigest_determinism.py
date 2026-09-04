"""The t-digest wrappers must be reproducible: ``clearance_p5_m`` is derived from them, and the
underlying ``tdigest`` package draws from the global ``random`` module.
"""

from __future__ import annotations

import json
import random

import numpy as np

from scenario_generation.metrics.tdigest import (
    merged_percentile,
    tdigest_dict_from_values,
)


def _samples(n: int = 5108, seed: int = 0) -> np.ndarray:
    """A clearance-like series: same size and shape as one closed-loop segment."""
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(5.0, 3.0, n))


def test_digest_from_identical_values_is_identical():
    values = _samples()
    first = tdigest_dict_from_values(values)
    for _ in range(7):
        assert tdigest_dict_from_values(values) == first


def test_percentile_from_identical_values_is_identical():
    values = _samples()
    got = {merged_percentile([tdigest_dict_from_values(values)], 5.0) for _ in range(8)}
    assert len(got) == 1, f"clearance-p5 is not reproducible: {sorted(got)}"


def test_merge_of_several_digests_is_identical():
    """The DDP path merges one digest per shard; that merge must be reproducible too."""
    values = _samples()
    shards = [tdigest_dict_from_values(values[i::4]) for i in range(4)]
    got = {merged_percentile(shards, 5.0) for _ in range(8)}
    assert len(got) == 1, f"merged clearance-p5 is not reproducible: {sorted(got)}"


def test_percentile_still_approximates_the_true_value():
    """Pinning the RNG must not have pinned it to a *wrong* answer."""
    values = _samples()
    got = merged_percentile([tdigest_dict_from_values(values)], 5.0)
    assert abs(got - float(np.percentile(values, 5.0))) < 0.01


def test_global_random_state_is_not_disturbed():
    values = _samples(n=512)

    random.seed(123)
    expected = [random.random() for _ in range(3)]

    random.seed(123)
    tdigest_dict_from_values(values)
    merged_percentile([tdigest_dict_from_values(values)], 5.0)
    got = [random.random() for _ in range(3)]

    assert got == expected


def test_empty_input_still_returns_none_and_inf():
    assert tdigest_dict_from_values(np.array([])) is None
    assert tdigest_dict_from_values(np.array([np.nan, np.inf])) is None
    assert merged_percentile([], 5.0) == float("inf")


def _shard_files(out_dir, n_shards: int = 4, n_routes_per_shard: int = 3):
    """Write per-rank segments_*/tdigests_* pairs the way a DDP run does."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for rank in range(n_shards):
        seg_lines, dig_lines = [], []
        for j in range(n_routes_per_shard):
            route = f"route_{rank}_{j}"
            vals = np.abs(rng.normal(5.0, 3.0, 400))
            digest = tdigest_dict_from_values(vals)
            seg_lines.append(
                json.dumps(
                    {
                        "route": route,
                        "object": {
                            "clearance_min_m": float(vals.min()),
                            "clearance_mean_m": float(vals.mean()),
                            "clearance_p5_m": float(np.percentile(vals, 5)),
                            "clearance_finite_steps": int(vals.size),
                            "miss_thresh_m": 0.5,
                            "collision_steps": 0,
                            "collision_count": 0,
                            "miss_steps": 0,
                            "miss_count": 0,
                        },
                    }
                )
            )
            dig_lines.append(json.dumps({"route": route, "object": digest}))
        (out_dir / f"segments_{rank}.jsonl").write_text("\n".join(seg_lines) + "\n")
        (out_dir / f"tdigests_{rank}.jsonl").write_text("\n".join(dig_lines) + "\n")


def test_shard_merge_reload_gives_the_same_p5(tmp_path):
    from scenario_generation.closed_loop_eval import load_segment_rows_with_tdigests
    from scenario_generation.metrics.tdigest import TDIGEST_KEY

    out_dir = tmp_path / "run"
    _shard_files(out_dir)

    def p5_from_disk() -> float:
        rows = load_segment_rows_with_tdigests(out_dir)
        digests = [r["object"][TDIGEST_KEY] for r in rows]
        return merged_percentile(digests, 5.0)

    got = {p5_from_disk() for _ in range(5)}
    assert len(got) == 1, f"merged shard p5 is not reproducible: {sorted(got)}"


def test_shard_rows_load_in_a_deterministic_order(tmp_path):
    from scenario_generation.closed_loop_eval import load_segment_rows_with_tdigests

    out_dir = tmp_path / "run"
    _shard_files(out_dir)
    order = [r["route"] for r in load_segment_rows_with_tdigests(out_dir)]
    for _ in range(4):
        assert [r["route"] for r in load_segment_rows_with_tdigests(out_dir)] == order
    assert order == sorted(order), "shard files must be walked in sorted() order"
