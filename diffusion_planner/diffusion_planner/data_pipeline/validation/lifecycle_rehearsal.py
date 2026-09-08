"""§7.7 version lifecycle rehearsal on a scratch copy: pin v1, publish v2, v1 bytes unchanged, gc semantics."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from diffusion_planner.data_pipeline import keyset as K
from diffusion_planner.data_pipeline import packer as PK
from diffusion_planner.data_pipeline import versioning as V
from diffusion_planner.data_pipeline.encoding import arrays_bitexact, load_npz_bytes
from diffusion_planner.data_pipeline.partition import PartitionRule
from diffusion_planner.data_pipeline.reader import ShardReader


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        required=True,
        type=Path,
        help="read-only source slice (copied to scratch first)",
    )
    ap.add_argument("--dest", required=True, type=Path, help="scratch dataset root (local disk)")
    ap.add_argument("--partition-depth", required=True, type=int)
    ap.add_argument("--keyset-where", default="is_skipped IS NOT TRUE")
    a = ap.parse_args(argv)
    scratch_src = a.dest.parent / (a.dest.name + "_src_copy")
    shutil.copytree(a.source, scratch_src, dirs_exist_ok=True)  # never touch the real source
    rule = PartitionRule(depth=a.partition_depth)
    results = {}
    v1 = PK.pack(PK.PackOptions(source=scratch_src, dest=a.dest, base="none", tag="v1", rule=rule))
    root = V.DatasetRoot(a.dest)
    ks = K.materialize_keyset(root, "v1", a.keyset_where, a.dest / "ks_v1.parquet")
    rd_v1 = ShardReader(a.dest, "v1")
    probe_key = rd_v1.query("1=1", ["key"]).column("key").to_pylist()[0]
    before = rd_v1.get(probe_key)
    target = scratch_src / f"{probe_key}.npz"
    arrays = load_npz_bytes(target.read_bytes())
    arrays["goal_pose"] = arrays["goal_pose"] + 1.0
    np.savez_compressed(target, **arrays)
    v2 = PK.pack(PK.PackOptions(source=scratch_src, dest=a.dest, base="v1", tag="v2", rule=rule))
    results["v1 bytes unchanged after v2"] = arrays_bitexact(before, rd_v1.get(probe_key))
    results["v2 sees the change"] = not arrays_bitexact(
        before, ShardReader(a.dest, "v2").get(probe_key)
    )
    results["gc deletes nothing while v1 exists"] = V.gc(root, dry_run=True) == []
    changed = {p for p in v2.partitions if v2.partitions[p].data_rev != v1.partitions[p].data_rev}
    # `gc` returns shard dirs, manifest files, AND stale build dirs — the only artifacts that
    # become unreferenced by pruning v1 are the pre-image (v1) shard dir and manifest for each
    # partition that v2 actually changed (an unchanged partition still points at the same
    # data_rev/meta_rev from v2, so it stays referenced).
    expected = set()
    for p in changed:
        e1 = v1.partitions[p]
        expected.add(root.shards_dir_for(e1.pid, e1.data_rev))
        expected.add(root.manifest_path_for(e1.pid, e1.data_rev, e1.meta_rev))
    V.prune_version(root, "v1")
    deleted = set(V.gc(root, dry_run=False))
    unexpected = deleted - expected
    missing = expected - deleted
    if unexpected:
        print(f"  unexpected deletions: {sorted(str(p) for p in unexpected)}")
    if missing:
        print(f"  expected deletions that did not happen: {sorted(str(p) for p in missing)}")
    results["gc after prune deletes exactly v1-only revisions"] = deleted == expected
    v2_paths = set()
    for p, e in v2.partitions.items():
        v2_paths.add(root.shards_dir_for(e.pid, e.data_rev))
        v2_paths.add(root.manifest_path_for(e.pid, e.data_rev, e.meta_rev))
    results["v2 shard dirs and manifests still exist after gc"] = all(p.exists() for p in v2_paths)
    ok = all(results.values())
    for k, v in results.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print("OVERALL", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
