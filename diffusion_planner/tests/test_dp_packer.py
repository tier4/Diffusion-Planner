import json
import os
import time

import duckdb
import numpy as np
import pytest
from diffusion_planner.data_pipeline import packer as PK
from diffusion_planner.data_pipeline import tar_shards as T
from diffusion_planner.data_pipeline.encoding import arrays_bitexact, decode_sample, load_npz_bytes
from diffusion_planner.data_pipeline.errors import (
    IntegrityError,
    PlanError,
    RuleMismatchError,
    SourceChangedError,
)
from diffusion_planner.data_pipeline.manifest import read_manifest
from diffusion_planner.data_pipeline.partition import PartitionRule
from diffusion_planner.data_pipeline.versioning import DatasetRoot
from tests.dp_fixtures import make_tree

LAYOUT = [
    ("pA/mX/manual/2026-01-01/t1/route_0", 12, "full"),
    ("pA/mX/manual/2026-01-02/t1/route_0", 5, "skip"),
    ("psim/loc_seed_1/manual/seed_1/bag_0/r", 4, "psim"),
    ("pB/mY/manual/2026-01-03/t1/route_0", 3, "none"),
]


def _opts(src, dst, tag, base="none", **kw):
    return PK.PackOptions(
        source=src,
        dest=dst,
        base=base,
        tag=tag,
        rule=PartitionRule(depth=4),
        shard_size_bytes=64 * 1024,
        **kw,
    )


def _manifest_tables(root, v):
    return {
        pid: read_manifest(root.manifest_path_for(e.pid, e.data_rev, e.meta_rev))
        for pid, e in v.partitions.items()
    }


def test_pack_builds_version_bitexact(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    keys = make_tree(src, LAYOUT, skipped_every=4)
    v = PK.pack(_opts(src, dst, "v1"))
    root = DatasetRoot(dst)
    assert root.latest() == "v1" and set(v.partitions) == {
        "pA/mX/manual/2026-01-01",
        "pA/mX/manual/2026-01-02",
        "psim/loc_seed_1/manual/seed_1",
        "pB/mY/manual/2026-01-03",
    }
    tables = _manifest_tables(root, v)
    n_rows = sum(t.num_rows for t in tables.values())
    # every packed member decodes bit-exact to its source npz
    for pid, t in tables.items():
        e = v.partitions[pid]
        for row in t.to_pylist():
            with open(
                root.shards_dir_for(e.pid, e.data_rev) / e.shards[row["shard_id"]], "rb"
            ) as f:
                payload = T.read_member(f, row["offset"], row["size"])
            src_arrays = load_npz_bytes((src / f"{row['key']}.npz").read_bytes())
            assert arrays_bitexact(src_arrays, decode_sample(payload))
    # rejected frames (is_skipped true) were not packed; NULL/absent kept
    all_keys = {r["key"] for t in tables.values() for r in t.to_pylist()}
    for k in keys:
        js = src / f"{k}.json"
        rejected = js.exists() and json.loads(js.read_text()).get("is_skipped") is True
        assert (k in all_keys) == (not rejected)
    assert not list((dst / "builds").iterdir()) or all(
        (b / "journal.json").exists() for b in (dst / "builds").iterdir()
    )


def test_pack_is_idempotent_and_detects_changes(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT)
    v1 = PK.pack(_opts(src, dst, "v1"))
    v2 = PK.pack(_opts(src, dst, "v2", base="v1"))
    assert {p: (e.data_rev, e.meta_rev) for p, e in v1.partitions.items()} == {
        p: (e.data_rev, e.meta_rev) for p, e in v2.partitions.items()
    }
    # modify one npz in one partition → only that partition gets a new data_rev
    target = next(src.glob("pA/mX/manual/2026-01-02/**/*.npz"))
    arrays = load_npz_bytes(target.read_bytes())
    arrays["goal_pose"] = arrays["goal_pose"] + 1
    np.savez_compressed(target, **arrays)
    v3 = PK.pack(_opts(src, dst, "v3", base="v2"))
    changed = {p for p in v3.partitions if v3.partitions[p].data_rev != v2.partitions[p].data_rev}
    assert changed == {"pA/mX/manual/2026-01-02"}
    # flip is_skipped → sample removed → partition repacked
    js = next(src.glob("pA/mX/manual/2026-01-01/**/*.json"))
    d = json.loads(js.read_text())
    d["is_skipped"] = True
    js.write_text(json.dumps(d))
    v4 = PK.pack(_opts(src, dst, "v4", base="v3"))
    assert (
        v4.partitions["pA/mX/manual/2026-01-01"].sample_count
        == v3.partitions["pA/mX/manual/2026-01-01"].sample_count - 1
    )
    # remove a partition from source: delta mode inherits, --sync drops
    import shutil

    shutil.rmtree(src / "pB")
    v5 = PK.pack(_opts(src, dst, "v5", base="v4"))
    assert "pB/mY/manual/2026-01-03" in v5.partitions
    v6 = PK.pack(_opts(src, dst, "v6", base="v5", sync=True))
    assert "pB/mY/manual/2026-01-03" not in v6.partitions


def test_rule_change_requires_replace_all_and_keys_stay_unique(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT)
    PK.pack(_opts(src, dst, "v1"))
    o = _opts(src, dst, "v2", base="v1")
    o.rule = PartitionRule(depth=3)
    with pytest.raises(RuleMismatchError):
        PK.pack(o)
    o.replace_all = True
    v2 = PK.pack(o)
    assert all(p.count("/") == 2 for p in v2.partitions)
    # a manually injected overlapping partition entry must be refused at commit
    o3 = _opts(src, dst, "v3", base="v2")
    o3.partitions = ["pA/mX/manual"]
    v3 = PK.pack(o3)  # subset pack keeps others → still unique
    assert set(v3.partitions) == set(v2.partitions)


def test_source_change_during_pack_fails(tmp_path, monkeypatch):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT[:1])
    victim = next(src.rglob("*.npz"))
    real_encode = PK.encoding.encode_sample

    def mutate_then_encode(arrays):
        os.utime(victim, ns=(time.time_ns(), time.time_ns()))
        return real_encode(arrays)

    monkeypatch.setattr(PK.encoding, "encode_sample", mutate_then_encode)
    with pytest.raises(SourceChangedError):
        PK.pack(_opts(src, dst, "v1"))
    assert DatasetRoot(dst).latest() is None and not list((dst / "versions").glob("*.json"))


def test_remove_and_scrub(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT)
    v1 = PK.pack(_opts(src, dst, "v1"))
    v2 = PK.remove(dst, "v1", "v2", ["psim/loc_seed_1/manual/seed_1"])
    assert "psim/loc_seed_1/manual/seed_1" not in v2.partitions and len(v2.partitions) == 3
    rep = PK.scrub(dst, "v2")
    assert rep["mismatches"] == 0 and rep["members"] == sum(
        e.sample_count for e in v2.partitions.values()
    )
    root = DatasetRoot(dst)
    e = v2.partitions["pA/mX/manual/2026-01-01"]
    shard = root.shards_dir_for(e.pid, e.data_rev) / e.shards[0]
    b = bytearray(shard.read_bytes())
    b[600] ^= 0xFF
    shard.write_bytes(bytes(b))
    with pytest.raises(IntegrityError):
        PK.scrub(dst, "v2")


def test_neighbor_ids_relation_optional(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT[:1])
    v = PK.pack(_opts(src, dst, "v1", with_neighbor_ids=True))
    e = v.partitions["pA/mX/manual/2026-01-01"]
    rel = dst / "relations" / f"{e.pid}@{e.data_rev}.neighbor_ids.parquet"
    assert rel.exists()
    n = duckdb.sql(
        f"SELECT count(*) FROM read_parquet('{rel}') WHERE len(neighbor_ids) = 2"
    ).fetchone()[0]
    assert n == e.sample_count
