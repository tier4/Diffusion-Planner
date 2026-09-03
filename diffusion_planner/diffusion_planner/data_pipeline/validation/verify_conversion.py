"""Post-hoc acceptance checks a published version must pass (spec §8.2, §8.3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from diffusion_planner.data_pipeline import tar_shards as T
from diffusion_planner.data_pipeline import versioning as V
from diffusion_planner.data_pipeline.errors import IntegrityError
from diffusion_planner.data_pipeline.manifest import read_manifest


def verify_offsets(root: V.DatasetRoot, tag: str) -> dict:
    """`scrub()` discards member offsets; a wrong offset serves the wrong sample on the
    loader's seek path. Assert manifest (offset, size) equals the tar's, and that
    (shard_id, sample_index_in_shard) is a bijection onto members."""
    version = root.read_version(tag)
    members = mismatches = 0
    for e in version.partitions.values():
        table = read_manifest(
            root.manifest_path_for(e.pid, e.data_rev, e.meta_rev),
            columns=["shard_id", "sample_index_in_shard", "offset", "size"],
        )
        rows = table.to_pylist()
        by_key: dict[tuple[int, int], dict] = {}
        for r in rows:
            k = (r["shard_id"], r["sample_index_in_shard"])
            if k in by_key:
                raise IntegrityError(f"{e.partition_id}: duplicate manifest row for {k}")
            by_key[k] = r
        for sid, name in enumerate(e.shards):
            listed = T.list_members(root.shards_dir_for(e.pid, e.data_rev) / name)
            seen = set()
            for idx, off, size in listed:
                members += 1
                r = by_key.get((sid, idx))
                if r is None:
                    raise IntegrityError(f"{e.partition_id}/{name}[{idx}]: member has no row")
                if (off, size) != (r["offset"], r["size"]):
                    mismatches += 1
                    raise IntegrityError(
                        f"{e.partition_id}/{name}[{idx}]: offset/size "
                        f"{(off, size)} != manifest {(r['offset'], r['size'])}"
                    )
                seen.add((sid, idx))
            unconsumed = {k for k in by_key if k[0] == sid} - seen
            if unconsumed:
                raise IntegrityError(
                    f"{e.partition_id}/{name}: {len(unconsumed)} manifest rows have no member"
                )
    return {"members": members, "mismatches": mismatches}


def manifest_keys(root: V.DatasetRoot, tag: str) -> set[str]:
    version = root.read_version(tag)
    keys: set[str] = set()
    for e in version.partitions.values():
        table = read_manifest(
            root.manifest_path_for(e.pid, e.data_rev, e.meta_rev), columns=["key"]
        )
        keys.update(table.column("key").to_pylist())
    return keys


def verify_membership(root: V.DatasetRoot, tag: str, expected_keys: set[str]) -> dict:
    """Cardinality is not membership. Diff both directions."""
    got = manifest_keys(root, tag)
    missing = expected_keys - got
    unexpected = got - expected_keys
    report = {
        "expected": len(expected_keys),
        "in_manifest": len(got),
        "missing_from_manifest": len(missing),
        "unexpected_in_manifest": len(unexpected),
    }
    if missing or unexpected:
        raise IntegrityError(
            f"membership differs: {report}; "
            f"e.g. missing {sorted(missing)[:3]} unexpected {sorted(unexpected)[:3]}"
        )
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="verify_conversion")
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument(
        "--expected-keys-json",
        type=Path,
        help="JSON list of source-relative keys (path list entries minus the .npz suffix)",
    )
    a = ap.parse_args(argv)
    root = V.DatasetRoot(a.dest)
    try:
        offsets = verify_offsets(root, a.tag)
        print(f"offsets OK: {offsets}")
        if a.expected_keys_json:
            expected = set(json.loads(a.expected_keys_json.read_text()))
            print(f"membership OK: {verify_membership(root, a.tag, expected)}")
    except IntegrityError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
