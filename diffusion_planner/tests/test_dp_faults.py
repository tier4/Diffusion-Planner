import pytest
from diffusion_planner.data_pipeline import keyset as K
from diffusion_planner.data_pipeline import packer as PK
from diffusion_planner.data_pipeline.errors import IntegrityError
from diffusion_planner.data_pipeline.partition import PartitionRule
from diffusion_planner.data_pipeline.reader import ShardReader
from diffusion_planner.data_pipeline.validation import mixing_test
from diffusion_planner.data_pipeline.versioning import DatasetRoot
from diffusion_planner.utils import shard_dataset as SD
from tests.dp_fixtures import make_tree

LAYOUT = [
    ("pA/mX/manual/2026-01-01/t1/r", 200, "full"),
    ("pB/mY/manual/2026-01-02/t1/r", 200, "psim"),
    ("pC/mZ/manual/2026-01-03/t1/r", 100, "skip"),
]


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
            shard_size_bytes=96 * 1024,
        )
    )
    ks = K.materialize_keyset(
        DatasetRoot(dst), "v1", "is_skipped IS NOT TRUE", tmp_path / "ks.parquet"
    )
    return src, dst, keys, ks


def _corrupt(path, offset):
    b = bytearray(path.read_bytes())
    b[offset] ^= 0xFF
    path.write_bytes(bytes(b))


def test_payload_corruption_never_yields_wrong_sample(packed):
    src, dst, keys, ks = packed
    root = DatasetRoot(dst)
    v = root.read_version("v1")
    e = v.partitions["pA/mX/manual/2026-01-01"]
    shard = root.shards_dir_for(e.pid, e.data_rev) / e.shards[0]
    _corrupt(shard, 512 + 40)  # inside the first member's payload
    rd = ShardReader(dst, "v1")
    with pytest.raises(IntegrityError):
        for _ in rd.iter("partition_id = 'pA/mX/manual/2026-01-01'"):
            pass
    with pytest.raises(IntegrityError):
        PK.scrub(dst, "v1")


def test_header_and_index_corruption_detected(packed):
    src, dst, keys, ks = packed
    root = DatasetRoot(dst)
    v = root.read_version("v1")
    e = v.partitions["pB/mY/manual/2026-01-02"]
    shard = root.shards_dir_for(e.pid, e.data_rev) / e.shards[0]
    _corrupt(shard, 3)  # tar header of member 0 (name field)
    with pytest.raises(IntegrityError):
        PK.scrub(dst, "v1")
    # offset tampering in the manifest → seek lands on garbage → zstd checksum/header fails → IntegrityError
    ds = SD.ShardDataset(
        SD.ShardDatasetConfig(
            root=dst,
            version="v1",
            keyset_path=ks,
            batch_size=4,
            world_size=1,
            rank=0,
            num_workers=0,
            seed=0,
            seek_threshold=1.01,
            chunk_size=16,
            max_pad_fraction=0.5,
            read_retries=0,
        )
    )
    path, table, n = ds._shard_table(next(iter(ds.selection)))
    k0 = next(iter(table))
    table[k0] = (table[k0][0] + 7, table[k0][1])
    with pytest.raises(IntegrityError):
        list(ds)


def test_mixing_statistic_runs_on_fixture(packed):
    src, dst, keys, ks = packed
    rep = mixing_test.run(
        dst,
        "v1",
        ks,
        world_size=1,
        workers=1,
        batch_size=16,
        C_values=(1, 4),
        seeds=(0, 1),
        epochs=1,
        baseline_seeds=3,
        max_pad_fraction=1.0,  # tiny fixture: 500 samples in a single slot pads past the 1% default
    )
    assert (
        set(rep["per_C"]) == {1, 4}
        and "js_mean" in rep["per_C"][1]
        and "pad_fraction" in rep
        and isinstance(rep["pass"], bool)
    )
