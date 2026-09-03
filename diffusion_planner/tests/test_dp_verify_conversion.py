import pyarrow.parquet as pq
import pytest
from diffusion_planner.data_pipeline import packer as PK
from diffusion_planner.data_pipeline.errors import IntegrityError
from diffusion_planner.data_pipeline.manifest import read_manifest
from diffusion_planner.data_pipeline.partition import PartitionRule
from diffusion_planner.data_pipeline.validation import verify_conversion as VC
from diffusion_planner.data_pipeline.versioning import DatasetRoot
from tests.dp_fixtures import make_tree

LAYOUT = [("pA/mX/manual/2026-01-01/t1/route_0", 8, "full")]


def _pack(tmp_path):
    src = tmp_path / "src"
    make_tree(src, LAYOUT)
    dst = tmp_path / "dst"
    PK.pack(
        PK.PackOptions(
            source=src,
            dest=dst,
            base="none",
            tag="v1",
            rule=PartitionRule(depth=4),
            shard_size_bytes=64 * 1024,
            progress=False,
        )
    )
    return src, DatasetRoot(dst)


def _append_ghost_row(path, **overrides):
    """Append a copy of manifest row 0 with the given field overrides, preserving schema."""
    table = read_manifest(path)
    cols = table.to_pydict()
    ghost = {k: v[0] for k, v in cols.items()}
    ghost.update(overrides)
    for k in cols:
        cols[k].append(ghost[k])
    pq.write_table(table.__class__.from_pydict(cols, schema=table.schema), path)


def test_offsets_verify_on_a_good_pack(tmp_path):
    _src, root = _pack(tmp_path)
    rep = VC.verify_offsets(root, "v1")
    assert rep["members"] == 8 and rep["mismatches"] == 0


def test_corrupt_offset_is_caught_although_scrub_passes(tmp_path):
    _src, root = _pack(tmp_path)
    v = root.read_version("v1")
    e = next(iter(v.partitions.values()))
    path = root.manifest_path_for(e.pid, e.data_rev, e.meta_rev)
    table = read_manifest(path)
    cols = table.to_pydict()
    cols["offset"] = [o + 512 for o in cols["offset"]]
    pq.write_table(table.__class__.from_pydict(cols, schema=table.schema), path)

    PK.scrub(root.root, "v1")  # scrub does not look at offsets
    with pytest.raises(IntegrityError, match="offset"):
        VC.verify_offsets(root, "v1")


def test_out_of_range_shard_id_is_caught(tmp_path):
    """A manifest row whose shard_id is >= len(e.shards) must not be silently unverified.

    The old per-shard `unconsumed` check only ever compared rows whose shard_id equalled
    the shard currently being iterated, so a row with an out-of-range shard_id was never
    a member of any `unconsumed` set and passed unexamined. This row is added, not
    substituted for a real one, so the 8 real members still match their real rows
    cleanly -- only the extra out-of-range row should trip the check.
    """
    _src, root = _pack(tmp_path)
    v = root.read_version("v1")
    e = next(iter(v.partitions.values()))
    path = root.manifest_path_for(e.pid, e.data_rev, e.meta_rev)
    table = read_manifest(path)
    ghost_key = table.column("key").to_pylist()[0] + "_ghost"
    _append_ghost_row(
        path,
        key=ghost_key,
        shard_id=len(e.shards) + 5,
        sample_index_in_shard=0,
    )

    with pytest.raises(IntegrityError, match="no tar member"):
        VC.verify_offsets(root, "v1")


def test_empty_version_is_a_hard_failure(tmp_path):
    _src, root = _pack(tmp_path)
    v = root.read_version("v1")
    PK.remove(root.root, "v1", "empty", list(v.partitions))
    with pytest.raises(IntegrityError, match="zero partitions"):
        VC.verify_offsets(root, "empty")


def test_membership_diff_is_two_way(tmp_path):
    _src, root = _pack(tmp_path)
    keys = VC.manifest_keys(root, "v1")
    rep = VC.verify_membership(root, "v1", keys)
    assert rep["missing_from_manifest"] == 0 and rep["unexpected_in_manifest"] == 0
    with pytest.raises(IntegrityError, match="membership"):
        VC.verify_membership(root, "v1", keys | {"ghost/key"})
    with pytest.raises(IntegrityError, match="membership"):
        VC.verify_membership(root, "v1", set(sorted(keys)[:-1]))


def test_membership_catches_duplicate_manifest_keys(tmp_path):
    """`manifest_keys` collapses to a set, so comparing distinct-key counts alone would
    let a manifest with a duplicated key pass. Row count must be checked too."""
    _src, root = _pack(tmp_path)
    v = root.read_version("v1")
    e = next(iter(v.partitions.values()))
    path = root.manifest_path_for(e.pid, e.data_rev, e.meta_rev)
    table = read_manifest(path)
    dup_key = table.column("key").to_pylist()[0]
    # Out-of-range shard_id keeps this test isolated from verify_offsets' own checks;
    # it is verify_membership (key-column only) under test here.
    _append_ghost_row(
        path,
        key=dup_key,
        shard_id=len(e.shards) + 9,
        sample_index_in_shard=0,
    )

    expected = VC.manifest_keys(root, "v1")
    assert len(expected) == 8  # the duplicate key adds a row, not a new distinct key
    with pytest.raises(IntegrityError, match="membership"):
        VC.verify_membership(root, "v1", expected)


def test_cli_requires_expected_keys_json(tmp_path):
    _src, root = _pack(tmp_path)
    with pytest.raises(SystemExit):
        VC.main(["--dest", str(root.root), "--tag", "v1"])
