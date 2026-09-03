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
    assert rows[0]["efficiency"] == 1.0
    assert rows[1]["efficiency"] > 0
    assert "samples/s" in PB.render(rows)
