import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from diffusion_planner.data_pipeline import keyset as K
from diffusion_planner.data_pipeline import packer as PK
from diffusion_planner.data_pipeline.errors import KeysetMismatchError, PlanError
from diffusion_planner.data_pipeline.partition import PartitionRule
from diffusion_planner.data_pipeline.versioning import DatasetRoot
from tests.dp_fixtures import make_tree

LAYOUT = [("pA/mX/manual/2026-01-01/t1/r", 10, "full"), ("pB/mY/manual/2026-01-02/t1/r", 6, "skip")]


@pytest.fixture
def packed(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    keys = make_tree(src, LAYOUT, skipped_every=3)
    v = PK.pack(
        PK.PackOptions(
            source=src,
            dest=dst,
            base="none",
            tag="v1",
            rule=PartitionRule(depth=4),
            shard_size_bytes=32 * 1024,
        )
    )
    return DatasetRoot(dst), v, keys


def test_materialize_and_load(packed, tmp_path):
    root, v, _ = packed
    out = K.materialize_keyset(
        root, "v1", "is_skipped IS NOT TRUE AND project_id = 'projA'", tmp_path / "ks.parquet"
    )
    t = K.load_keyset(out, root, "v1")
    assert t.schema.equals(K.KEYSET_SCHEMA, check_metadata=False)
    assert set(t.column("partition_id").to_pylist()) == {"pA/mX/manual/2026-01-01"}
    assert t.num_rows == v.partitions["pA/mX/manual/2026-01-01"].sample_count
    assert len(K.keyset_digest(t)) == 16
    with pytest.raises(PlanError):
        K.materialize_keyset(root, "v1", "project_id = 'nope'", tmp_path / "empty.parquet")
    with pytest.raises(PlanError):
        K.materialize_keyset(root, "v1", "1=1; DROP TABLE x", tmp_path / "inj.parquet")


def test_keys_resolution_rejects_unknown_and_duplicates(packed, tmp_path):
    root, v, keys = packed
    # Get keys from pA manifest (accounts for skipped samples dropped by packer)
    manifest_keys = set()
    for e in v.partitions.values():
        if e.partition_id.startswith("pA"):
            mf = root.manifest_path_for(e.pid, e.data_rev, e.meta_rev)
            t = pq.read_table(mf)
            manifest_keys.update(t.column("key").to_pylist())

    good = sorted(manifest_keys)[:4]
    t = K.load_keyset(K.keyset_from_keys(root, "v1", good, tmp_path / "a.parquet"), root, "v1")
    assert t.num_rows == 4
    with pytest.raises(PlanError):
        K.keyset_from_keys(root, "v1", good + [good[0]], tmp_path / "b.parquet")
    with pytest.raises(PlanError):
        K.keyset_from_keys(root, "v1", good + ["pA/does/not/exist"], tmp_path / "c.parquet")


def test_version_binding(packed, tmp_path):
    root, v, keys = packed
    ks = K.materialize_keyset(root, "v1", "is_skipped IS NOT TRUE", tmp_path / "ks.parquet")
    PK.remove(root.root, "v1", "v2", ["pB/mY/manual/2026-01-02"])
    K.load_keyset(ks, root, "v1")  # still valid against v1
    with pytest.raises(KeysetMismatchError):
        K.load_keyset(ks, root, "v2")
