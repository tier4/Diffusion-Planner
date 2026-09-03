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


def test_membership_diff_is_two_way(tmp_path):
    _src, root = _pack(tmp_path)
    keys = VC.manifest_keys(root, "v1")
    rep = VC.verify_membership(root, "v1", keys)
    assert rep["missing_from_manifest"] == 0 and rep["unexpected_in_manifest"] == 0
    with pytest.raises(IntegrityError, match="membership"):
        VC.verify_membership(root, "v1", keys | {"ghost/key"})
