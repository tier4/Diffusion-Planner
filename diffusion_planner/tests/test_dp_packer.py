import json
import os
import time
from pathlib import Path

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
    # Completed build dirs are cleaned up after publish; the builds directory should be empty.
    assert not list((dst / "builds").iterdir())


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
    # mismatching rule + empty selection must still be refused (unconditional guard)
    o_empty = _opts(src, dst, "v9", base="v2")
    o_empty.rule = PartitionRule(depth=4)
    o_empty.partitions = ["nonexistent/partition"]
    with pytest.raises(RuleMismatchError):
        PK.pack(o_empty)
    assert DatasetRoot(dst).latest() == "v2"
    # a manually injected overlapping partition entry must be refused at commit
    o3 = _opts(src, dst, "v3", base="v2")
    o3.rule = PartitionRule(depth=3)
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


def test_two_pass_build_bounds_npz_reads(tmp_path, monkeypatch):
    """Pins the two-pass build contract: `_build_partition` never holds every kept sample's
    npz bytes in memory at once (see design note in packer.py). Pass 1 reads each npz once
    (hash-only, then dropped); pass 2 re-reads each *kept* npz once more to encode it. A kept
    npz is read a third time only if it happens to land at `sample_index_in_shard == 0` for its
    shard, since spec §4.4's verify-before-commit spot-checks (1 in every 100 members, which
    always includes index 0) re-reads the source npz fresh from disk rather than from a cached
    copy. So: every rejected npz is read exactly once, every kept npz is read either twice, or
    three times if it is one of the (exactly one-per-shard) spot-checked members.
    """
    src, dst = tmp_path / "src", tmp_path / "dst"
    keys = make_tree(src, LAYOUT[:2], skipped_every=4)
    counts: dict[str, int] = {}
    orig_read_bytes = Path.read_bytes

    def counting_read_bytes(self, *a, **kw):
        if self.suffix == ".npz":
            counts[str(self)] = counts.get(str(self), 0) + 1
        return orig_read_bytes(self, *a, **kw)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    v = PK.pack(_opts(src, dst, "v1"))

    n_shards = sum(len(e.shards) for e in v.partitions.values())
    rejected_keys = {
        k
        for k in keys
        if (src / f"{k}.json").exists()
        and json.loads((src / f"{k}.json").read_text()).get("is_skipped") is True
    }
    kept_keys = set(keys) - rejected_keys
    assert kept_keys and rejected_keys  # sanity: fixture actually exercises both paths

    for k in rejected_keys:
        assert counts[str(src / f"{k}.npz")] == 1
    kept_counts = [counts[str(src / f"{k}.npz")] for k in kept_keys]
    assert all(c in (2, 3) for c in kept_counts)
    assert sum(1 for c in kept_counts if c == 3) == n_shards
    assert sum(1 for c in kept_counts if c == 2) == len(kept_keys) - n_shards


def test_relation_move_does_not_overwrite_existing(tmp_path):
    """`_publish` must guard the relation-file move the same way it guards shard dirs and
    manifests: never clobber an already-published relation file with a fresh build's copy."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT[:1])
    v1 = PK.pack(_opts(src, dst, "v1", with_neighbor_ids=True))
    e1 = v1.partitions["pA/mX/manual/2026-01-01"]
    rel = dst / "relations" / f"{e1.pid}@{e1.data_rev}.neighbor_ids.parquet"
    before = rel.stat()
    v2 = PK.pack(_opts(src, dst, "v2", base="v1", with_neighbor_ids=True, force=True))
    e2 = v2.partitions["pA/mX/manual/2026-01-01"]
    assert e2.data_rev == e1.data_rev  # forced rebuild reproduced the identical revision
    after = rel.stat()
    # the published relation file must be untouched by the second (redundant) build
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)


def test_sync_with_partition_filter_keeps_unselected(tmp_path):
    """Finding #1: --sync --partition pA must keep pB/pC; --sync --path-list -> PlanError."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT)
    v1 = PK.pack(_opts(src, dst, "v1"))
    assert set(v1.partitions) == {
        "pA/mX/manual/2026-01-01",
        "pA/mX/manual/2026-01-02",
        "psim/loc_seed_1/manual/seed_1",
        "pB/mY/manual/2026-01-03",
    }
    # --sync --partition pA/.../2026-01-01 keeps all other partitions
    v2 = PK.pack(
        _opts(src, dst, "v2", base="v1", sync=True, partitions=["pA/mX/manual/2026-01-01"])
    )
    assert "pB/mY/manual/2026-01-03" in v2.partitions
    assert "psim/loc_seed_1/manual/seed_1" in v2.partitions
    assert "pA/mX/manual/2026-01-02" in v2.partitions
    # --sync --path-list -> PlanError
    with pytest.raises(PlanError, match="--sync requires a full source scan"):
        PK.pack(
            _opts(
                src,
                dst,
                "v3",
                base="v2",
                sync=True,
                path_list=["pA/mX/manual/2026-01-01/t1/route_0/route_0_00000000.npz"],
            )
        )


def test_sidecar_only_change_meta_only_update(tmp_path):
    """Finding #2: editing one sidecar value -> v2 has same data_rev, new meta_rev,
    no new shard dir; the new manifest has the new value; scrub passes.
    A tensor change still produces a new data_rev."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT[:1])
    v1 = PK.pack(_opts(src, dst, "v1"))
    e1 = v1.partitions["pA/mX/manual/2026-01-01"]
    # list existing shard dirs
    shard_dirs_v1 = set(p.name for p in (dst / "shards").iterdir())
    # modify a sidecar field ("x") in one frame
    js = next(src.glob("pA/mX/manual/2026-01-01/**/*.json"))
    d = json.loads(js.read_text())
    d["x"] = 999.999
    js.write_text(json.dumps(d))
    v2 = PK.pack(_opts(src, dst, "v2", base="v1"))
    e2 = v2.partitions["pA/mX/manual/2026-01-01"]
    assert e2.data_rev == e1.data_rev  # same data
    assert e2.meta_rev != e1.meta_rev  # different metadata
    # No new shard directory created
    shard_dirs_v2 = set(p.name for p in (dst / "shards").iterdir())
    assert shard_dirs_v2 == shard_dirs_v1
    # scrub passes
    rep = PK.scrub(dst, "v2")
    assert rep["mismatches"] == 0
    # tensor change produces new data_rev
    target = next(src.glob("pA/mX/manual/2026-01-01/**/*.npz"))
    arrays = load_npz_bytes(target.read_bytes())
    arrays["goal_pose"] = arrays["goal_pose"] + 1
    np.savez_compressed(target, **arrays)
    v3 = PK.pack(_opts(src, dst, "v3", base="v2"))
    e3 = v3.partitions["pA/mX/manual/2026-01-01"]
    assert e3.data_rev != e2.data_rev


def test_existing_shard_dir_wrong_bytes_raises(tmp_path):
    """Finding #5: pre-create a same-named shard dir with wrong bytes -> pack raises."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT[:1])
    v1 = PK.pack(_opts(src, dst, "v1"))
    e1 = v1.partitions["pA/mX/manual/2026-01-01"]
    shard_dir = dst / "shards" / f"{e1.pid}@{e1.data_rev}"
    # corrupt a shard file
    shard_file = shard_dir / e1.shards[0]
    b = bytearray(shard_file.read_bytes())
    b[100] ^= 0xFF
    shard_file.write_bytes(bytes(b))
    # forced rebuild tries to publish and finds mismatching existing dir
    with pytest.raises(IntegrityError, match="different sha256"):
        PK.pack(_opts(src, dst, "v2", base="v1", force=True))


def test_unknown_partition_raises(tmp_path):
    """Finding #6: unknown --partition -> PlanError listing unknowns."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT[:1])
    with pytest.raises(PlanError, match="--partition names not found"):
        PK.pack(_opts(src, dst, "v1", partitions=["does/not/exist/here"]))


def test_empty_pack_raises(tmp_path):
    """Finding #6: pack with empty groups (not --sync) -> PlanError."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT[:1])
    with pytest.raises(PlanError, match="nothing to pack"):
        PK.pack(_opts(src, dst, "v1", include=["*/zzz_no_match/*"]))
