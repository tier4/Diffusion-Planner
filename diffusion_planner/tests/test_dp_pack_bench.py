import pytest
from diffusion_planner.data_pipeline.partition import PartitionRule
from diffusion_planner.data_pipeline.validation import pack_bench as PB
from tests.dp_fixtures import make_tree

LAYOUT = [
    ("pA/mX/manual/2026-01-01/t1/route_0", 6, "full"),
    ("pB/mY/manual/2026-01-03/t1/route_0", 6, "full"),
]


def test_bench_reports_rate_and_efficiency(tmp_path):
    src = tmp_path / "src"
    make_tree(src, LAYOUT)
    rows = PB.bench(
        source=src,
        dest_root=tmp_path / "bench",
        worker_counts=[1, 2],
        rule=PartitionRule(depth=4),
        path_list=None,
        shard_size_bytes=64 * 1024,
    )
    assert [r["workers"] for r in rows] == [1, 2]
    for r in rows:
        assert r["samples"] == 12
        assert r["samples_per_s"] > 0
        assert r["wall_s"] > 0
        assert r["bytes_written"] > 0
        assert r["partitions"] == 2
        assert r["rss_self_kb"] > 0
    assert rows[0]["efficiency"] == 1.0
    assert rows[1]["efficiency"] > 0
    rendered = PB.render(rows)
    assert "samples/s" in rendered
    assert "cache" in rendered.lower()  # T7-6: cold-vs-warm caveat is stated, not implied
    assert "cumulative" in rendered.lower()  # T7-1: rss fields are labelled, not per-row


def test_efficiency_baseline_is_the_first_worker_count_measured(tmp_path):
    """T7-2: a sweep that never measures workers=1 (e.g. --workers 2,8) must not silently
    assume that baseline. The row for the first worker count actually measured is always
    the reference point, so its own efficiency is exactly 1.0 regardless of its value."""
    src = tmp_path / "src"
    make_tree(src, LAYOUT)
    rows = PB.bench(
        source=src,
        dest_root=tmp_path / "bench_no_one",
        worker_counts=[2, 4],
        rule=PartitionRule(depth=4),
        path_list=None,
        shard_size_bytes=64 * 1024,
    )
    assert rows[0]["workers"] == 2
    assert rows[0]["efficiency"] == 1.0
    assert rows[1]["efficiency"] > 0


def test_rerun_against_same_dest_root_does_not_collide(tmp_path):
    """T7-5: the runbook runs this bench at least three times. A hard-coded tag would make
    the second run raise VersionExistsError against the first run's version of the same
    name; a fresh tag per bench() call must avoid that."""
    src = tmp_path / "src"
    make_tree(src, LAYOUT)
    dest_root = tmp_path / "bench_rerun"
    kwargs = dict(
        source=src,
        dest_root=dest_root,
        worker_counts=[1],
        rule=PartitionRule(depth=4),
        path_list=None,
        shard_size_bytes=64 * 1024,
    )
    PB.bench(**kwargs)
    PB.bench(**kwargs)  # must not raise VersionExistsError


def test_cli_requires_one_partition_rule_flag(tmp_path):
    """T7-4: omitting both --partition-depth and --partition-regex must fail with a clear
    argparse error, not an unhandled ValueError from PartitionRule.__post_init__."""
    with pytest.raises(SystemExit):
        PB.main(["--source", str(tmp_path), "--dest-root", str(tmp_path / "out")])
