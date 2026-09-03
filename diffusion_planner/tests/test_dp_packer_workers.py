import hashlib
import pickle
from pathlib import Path

import pytest
from diffusion_planner.data_pipeline import packer as PK
from diffusion_planner.data_pipeline.errors import PackWorkerError
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


def _boom_worker(job):
    """Stand-in for `_build_partition` that always fails.

    Defined at module scope (not nested in the test) so it survives pickling under the
    `spawn` context: pickle serializes a plain function by (module, qualname) reference,
    and a closure defined inside a test function has no such stable path — the child
    process fails to reconstruct it (PicklingError), which would test pickling, not the
    worker-failure path this test is meant to exercise.
    """
    raise ValueError(f"synthetic failure in {job.partition_id}")


def test_worker_job_excludes_path_list_and_is_picklable(tmp_path):
    src = tmp_path / "src"
    make_tree(src, LAYOUT)
    huge = [f"pA/mX/manual/2026-01-01/t1/route_0/{i:08d}.npz" for i in range(3)]
    opts = _opts(src, tmp_path / "dst", "v1", path_list=huge)
    root = DatasetRoot(opts.dest)
    root.ensure_layout()
    job = PK._job_for(
        opts,
        tmp_path / "build",
        None,
        root,
        "pA/mX/manual/2026-01-01",
        [],
    )
    fields = set(vars(job))
    assert "path_list" not in fields
    assert "include" not in fields and "exclude" not in fields
    assert "partitions" not in fields and "sync" not in fields
    blob = pickle.dumps(job)
    assert pickle.loads(blob).partition_id == "pA/mX/manual/2026-01-01"
    # the job must not drag the path list along by any route
    assert b"00000000.npz" not in blob


def _shard_digests(root, version):
    out = {}
    for e in version.partitions.values():
        d = root.shards_dir_for(e.pid, e.data_rev)
        for name in e.shards:
            out[f"{e.partition_id}/{name}"] = hashlib.sha256((d / name).read_bytes()).hexdigest()
    return out


def test_parallel_pack_is_byte_identical_to_serial(tmp_path):
    src = tmp_path / "src"
    make_tree(src, LAYOUT)
    PK.pack(_opts(src, tmp_path / "a", "v1"))
    PK.pack(_opts(src, tmp_path / "b", "v1", workers=4))
    ra, rb = DatasetRoot(tmp_path / "a"), DatasetRoot(tmp_path / "b")
    va, vb = ra.read_version("v1"), rb.read_version("v1")
    assert set(va.partitions) == set(vb.partitions)
    for pid, ea in va.partitions.items():
        eb = vb.partitions[pid]
        assert (ea.data_rev, ea.meta_rev, ea.sample_count) == (
            eb.data_rev,
            eb.meta_rev,
            eb.sample_count,
        )
    assert _shard_digests(ra, va) == _shard_digests(rb, vb)


def test_worker_failure_publishes_nothing_and_names_the_partition(tmp_path, monkeypatch):
    src = tmp_path / "src"
    make_tree(src, LAYOUT)
    dst = tmp_path / "dst"

    monkeypatch.setattr(PK, "_build_partition", _boom_worker)
    with pytest.raises(PackWorkerError) as ei:
        PK.pack(_opts(src, dst, "v1", workers=2))
    assert "synthetic failure" in str(ei.value)
    assert not (DatasetRoot(dst).versions_dir / "v1.json").exists()


def test_pool_uses_spawn(tmp_path, monkeypatch):
    seen = {}
    real = PK.futures.ProcessPoolExecutor

    class Spy(real):
        def __init__(self, *a, **kw):
            seen["ctx"] = type(kw.get("mp_context")).__name__
            super().__init__(*a, **kw)

    monkeypatch.setattr(PK.futures, "ProcessPoolExecutor", Spy)
    src = tmp_path / "src"
    make_tree(src, LAYOUT)
    PK.pack(_opts(src, tmp_path / "dst", "v1", workers=2))
    assert seen["ctx"] == "SpawnContext"
