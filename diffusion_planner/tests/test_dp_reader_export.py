import json
from unittest.mock import patch

import numpy as np
import pytest
from diffusion_planner.data_pipeline import packer as PK
from diffusion_planner.data_pipeline.encoding import arrays_bitexact, load_npz_bytes
from diffusion_planner.data_pipeline.errors import PlanError
from diffusion_planner.data_pipeline.export import export
from diffusion_planner.data_pipeline.partition import PartitionRule
from diffusion_planner.data_pipeline.reader import ShardReader
from tests.dp_fixtures import make_tree

LAYOUT = [("pA/mX/manual/2026-01-01/t1/r", 30, "full"), ("pB/mY/manual/2026-01-02/t1/r", 8, "psim")]


@pytest.fixture
def packed(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    keys = make_tree(src, LAYOUT)
    PK.pack(
        PK.PackOptions(
            source=src,
            dest=dst,
            base="none",
            tag="v1",
            rule=PartitionRule(depth=4),
            shard_size_bytes=48 * 1024,
        )
    )
    return src, dst, keys


def test_query_get_iter_agree(packed):
    src, dst, keys = packed
    rd = ShardReader(dst, "latest")
    assert rd.version.tag == "v1"
    t = rd.query("project_id = 'projA'", columns=["key", "timestamp"])
    assert t.num_rows == 30 and t.column("key").to_pylist() == sorted(
        k for k in keys if k.startswith("pA")
    )
    k0 = keys[0]
    assert arrays_bitexact(rd.get(k0), load_npz_bytes((src / f"{k0}.npz").read_bytes()))
    with pytest.raises(KeyError):
        rd.get("nope/x")
    seen = dict(rd.iter("project_id IS NULL"))
    assert set(seen) == {k for k in keys if k.startswith("pB")}
    for k, arrays in seen.items():
        assert arrays_bitexact(arrays, load_npz_bytes((src / f"{k}.npz").read_bytes()))
    k, arrays = next(rd.iter("project_id = 'projA'", training_view=True))
    assert len(arrays) == 17 and "version" not in arrays


def test_export_roundtrip_for_legacy_tools(packed, tmp_path):
    src, dst, keys = packed
    out = tmp_path / "exported"
    n = export(
        dst, "v1", "project_id = 'projA' AND timestamp < 1700000000000000000 + 5 * 300000000", out
    )
    assert n == 5
    man = json.loads((out / "export_manifest.json").read_text())
    assert man["n"] == 5 and man["version"] == "v1" and len(man["keys"]) == 5
    for k in man["keys"]:
        exported = dict(np.load(out / f"{k}.npz", allow_pickle=True))  # legacy np.load works
        assert arrays_bitexact(exported, load_npz_bytes((src / f"{k}.npz").read_bytes()))
        js = json.loads((out / f"{k}.json").read_text())
        orig = json.loads((src / f"{k}.json").read_text())
        assert js["is_skipped"] == orig["is_skipped"] and js["timestamp"] == orig["timestamp"]
        assert (
            js["project_id"] == orig["project_id"]
            and js["skipping_info"]["label"] == orig["skipping_info"]["label"]
        )
        assert "neighbor_count" not in js


def test_query_reserved_column_protection(packed):
    """Unquoted reserved column name in WHERE raises PlanError; quoted works."""
    src, dst, keys = packed
    rd = ShardReader(dst, "latest")
    # Unquoted "offset" should raise PlanError
    with pytest.raises(PlanError):
        rd.query("offset > 0")
    # Quoted "offset" should work and return all rows (all offsets > 0)
    t = rd.query('"offset" > 0', ["key"])
    assert t.num_rows == len(keys)


def test_query_rejects_semicolon(packed):
    """WHERE clause with ; is rejected as PlanError."""
    src, dst, keys = packed
    rd = ShardReader(dst, "latest")
    with pytest.raises(PlanError):
        rd.query("1=1; DROP TABLE x")


def test_iter_seek_and_skim_both_exercised(packed):
    """Verify iter() uses seek path for low selectivity and skim path for high selectivity."""
    from diffusion_planner.data_pipeline import defaults as DP_DEFAULTS
    from diffusion_planner.data_pipeline import tar_shards as T

    src, dst, keys = packed
    rd = ShardReader(dst, "latest")

    # Find all keys and group by shard
    all_rows = rd.query("1=1", ["key", "partition_id", "shard_id"]).to_pylist()
    by_shard: dict[tuple[str, int], list[str]] = {}
    by_shard_full: dict[tuple[str, int], list[dict]] = {}
    for r in all_rows:
        by_shard.setdefault((r["partition_id"], r["shard_id"]), []).append(r["key"])
        by_shard_full.setdefault((r["partition_id"], r["shard_id"]), []).append(r)

    # Find a shard with multiple keys
    shard_info = next(
        ((ks, rs) for ks, rs in zip(by_shard.values(), by_shard_full.values()) if len(ks) >= 2),
        None,
    )
    if shard_info is None:
        pytest.skip("No shard with >= 2 keys for seek/skim test")

    shard_with_multi, shard_rows_full = shard_info

    # Get the partition and shard IDs from the first row
    pid = shard_rows_full[0]["partition_id"]
    sid = shard_rows_full[0]["shard_id"]

    # Test seek path: select 1 key with high threshold (so 1/N < threshold), monkeypatch iter_members
    k_single = shard_with_multi[0]
    with patch("diffusion_planner.data_pipeline.reader.SEEK_THRESHOLD", 0.5):
        # With threshold=0.5, selecting 1 key gives selectivity 1/N < 0.5 (for N >= 2)
        with patch(
            "diffusion_planner.data_pipeline.reader.T.iter_members",
            side_effect=RuntimeError("skim should not be used"),
        ):
            seen = dict(rd.iter(f"key = '{k_single}'"))
            assert set(seen) == {k_single}
            assert arrays_bitexact(
                seen[k_single], load_npz_bytes((src / f"{k_single}.npz").read_bytes())
            )

    # Test skim path: select all keys in shard with low threshold, monkeypatch read_member
    key_list = ", ".join(f"'{k}'" for k in shard_with_multi)
    with patch("diffusion_planner.data_pipeline.reader.SEEK_THRESHOLD", 0.1):
        # With threshold=0.1, selecting all keys gives selectivity 1.0 >= 0.1 → skim
        with patch(
            "diffusion_planner.data_pipeline.reader.T.read_member",
            side_effect=RuntimeError("seek should not be used"),
        ):
            seen = dict(rd.iter(f"key IN ({key_list})"))
            assert set(seen) == set(shard_with_multi)
            for k in seen:
                assert arrays_bitexact(seen[k], load_npz_bytes((src / f"{k}.npz").read_bytes()))
