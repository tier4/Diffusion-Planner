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


def test_load_rejects_duplicate_coordinates(packed, tmp_path):
    root, v, _ = packed
    # Write a keyset with duplicate rows by hand
    dup_table = pa.table(
        {
            "partition_id": pa.array(["pA/mX/manual/2026-01-01", "pA/mX/manual/2026-01-01"]),
            "shard_id": pa.array([0, 0], type=pa.int32()),
            "sample_index_in_shard": pa.array([5, 5], type=pa.int32()),
        }
    )
    meta = {
        b"dp.version_tag": "v1".encode(),
        b"dp.version_hash": root.version_hash("v1").encode(),
        b"dp.created_at": "2026-09-02T00:00:00+00:00".encode(),
    }
    dup_table = dup_table.replace_schema_metadata(meta)
    out = tmp_path / "dup.parquet"
    pq.write_table(dup_table, out, compression="zstd")
    with pytest.raises(PlanError):
        K.load_keyset(out, root, "v1")


def test_load_rejects_missing_version_metadata(packed, tmp_path):
    root, v, _ = packed
    # Write a keyset with valid schema but no metadata
    table = pa.table(
        {
            "partition_id": pa.array(["pA/mX/manual/2026-01-01"]),
            "shard_id": pa.array([0], type=pa.int32()),
            "sample_index_in_shard": pa.array([5], type=pa.int32()),
        }
    )
    out = tmp_path / "nometa.parquet"
    pq.write_table(table, out, compression="zstd")

    with pytest.raises(KeysetMismatchError):
        K.load_keyset(out, root, "v1")


def test_keyset_digest_is_stable_and_content_sensitive(packed, tmp_path):
    root, v, _ = packed
    # Create a keyset from materialize
    ks1 = K.materialize_keyset(root, "v1", "is_skipped IS NOT TRUE", tmp_path / "ks1.parquet")
    t1 = K.load_keyset(ks1, root, "v1")

    # Same table twice should give same digest
    digest1 = K.keyset_digest(t1)
    digest1_again = K.keyset_digest(t1)
    assert digest1 == digest1_again

    # Change one coordinate and digest should differ
    if t1.num_rows > 0:
        rows_list = [t1.slice(i, 1) for i in range(t1.num_rows)]
        # Modify the first row's shard_id
        modified_first = rows_list[0]
        modified_first = pa.table(
            {
                "partition_id": modified_first.column("partition_id"),
                "shard_id": pa.array([999], type=pa.int32()),
                "sample_index_in_shard": modified_first.column("sample_index_in_shard"),
            }
        )
        t2 = pa.concat_tables([modified_first] + rows_list[1:])
        digest2 = K.keyset_digest(t2)
        assert digest1 != digest2

        # Permuted rows should give different digest
        if t1.num_rows > 1:
            permuted_rows = [rows_list[1], rows_list[0]] + rows_list[2:]
            t3 = pa.concat_tables(permuted_rows)
            digest3 = K.keyset_digest(t3)
            assert digest1 != digest3
