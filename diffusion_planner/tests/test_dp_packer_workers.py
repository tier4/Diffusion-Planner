import pickle
from pathlib import Path

from diffusion_planner.data_pipeline import packer as PK
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
