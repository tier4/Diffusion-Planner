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
    (shard_id, sample_index_in_shard) is a bijection onto members -- checked across a
    partition's whole `e.shards` list, not shard-by-shard, so a manifest row whose
    shard_id is out of range (or a `shards` list truncated relative to the manifest)
    cannot hide in a per-shard "unconsumed" set that never looks at it.

    Note: only the shard files named in `e.shards` are examined. A tar file that sits in
    the shards directory but is absent from `e.shards` is not read here, so "bijection"
    means onto the partition's recorded shard list, not onto whatever else is on disk.

    A version with zero partitions is a hard failure, not a vacuous pass. Offset/size
    mismatches are accumulated (with a capped set of examples) and raised once at the end
    with the true total count, mirroring how `scrub()` (packer.py) accumulates before
    raising -- "3 bad offsets" and "5.4M bad offsets" are different incidents and the
    raised message says which.
    """
    version = root.read_version(tag)
    if not version.partitions:
        raise IntegrityError(f"version {tag!r} has zero partitions; refusing to pass an empty gate")
    members = 0
    mismatches: list[str] = []
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
        consumed: set[tuple[int, int]] = set()
        partition_members = 0
        for sid, name in enumerate(e.shards):
            listed = T.list_members(root.shards_dir_for(e.pid, e.data_rev) / name)
            for idx, off, size in listed:
                partition_members += 1
                k = (sid, idx)
                r = by_key.get(k)
                if r is None:
                    raise IntegrityError(f"{e.partition_id}/{name}[{idx}]: member has no row")
                if (off, size) != (r["offset"], r["size"]):
                    mismatches.append(
                        f"{e.partition_id}/{name}[{idx}]: offset/size "
                        f"{(off, size)} != manifest {(r['offset'], r['size'])}"
                    )
                consumed.add(k)
        # Row -> member direction of the bijection, over the WHOLE partition's shard list
        # (accumulated across every shard above) rather than per-shard: a row whose
        # shard_id never equals any `sid` in `enumerate(e.shards)` -- e.g. negative, or
        # >= len(e.shards) -- is in `by_key` but would never be excluded by any single
        # shard's `consumed`, so it must be checked here, once, against the union.
        unconsumed = set(by_key) - consumed
        if unconsumed:
            raise IntegrityError(
                f"{e.partition_id}: {len(unconsumed)} manifest row(s) reference no tar member "
                f"(e.g. {sorted(unconsumed)[:5]}); check for an out-of-range shard_id or a "
                "shards list truncated relative to the manifest"
            )
        if not (len(rows) == partition_members == e.sample_count):
            raise IntegrityError(
                f"{e.partition_id}: row/member/sample_count mismatch (manifest rows="
                f"{len(rows)}, tar members={partition_members}, entry.sample_count="
                f"{e.sample_count})"
            )
        members += partition_members
    if mismatches:
        raise IntegrityError(f"{len(mismatches)} offset/size mismatch(es), e.g. {mismatches[:20]}")
    return {"members": members, "mismatches": len(mismatches)}


def _manifest_key_rows(root: V.DatasetRoot, tag: str) -> list[str]:
    version = root.read_version(tag)
    keys: list[str] = []
    for e in version.partitions.values():
        table = read_manifest(
            root.manifest_path_for(e.pid, e.data_rev, e.meta_rev), columns=["key"]
        )
        keys.extend(table.column("key").to_pylist())
    return keys


def manifest_keys(root: V.DatasetRoot, tag: str) -> set[str]:
    return set(_manifest_key_rows(root, tag))


def verify_membership(root: V.DatasetRoot, tag: str, expected_keys: set[str]) -> dict:
    """Cardinality is not membership. Diff both directions.

    `manifest_keys` collapses to a `set`, so a distinct-key comparison alone would let a
    manifest with duplicated keys (`_check_unique_keys` only runs at pack time, not
    against a manifest edited after publication) pass a membership check it should fail.
    Compare the raw row count against the distinct-key count too.
    """
    rows = _manifest_key_rows(root, tag)
    got = set(rows)
    missing = expected_keys - got
    unexpected = got - expected_keys
    duplicate_rows = len(rows) - len(got)
    report = {
        "expected": len(expected_keys),
        "in_manifest": len(got),
        "in_manifest_rows": len(rows),
        "missing_from_manifest": len(missing),
        "unexpected_in_manifest": len(unexpected),
    }
    problems = []
    if missing:
        problems.append(f"missing {sorted(missing)[:3]}")
    if unexpected:
        problems.append(f"unexpected {sorted(unexpected)[:3]}")
    if duplicate_rows:
        problems.append(f"{duplicate_rows} duplicate key row(s) in manifest")
    if problems:
        raise IntegrityError(f"membership differs: {report}; " + "; ".join(problems))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="verify_conversion")
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument(
        "--expected-keys-json",
        required=True,
        type=Path,
        help="JSON list of source-relative keys (path list entries minus the .npz suffix)",
    )
    a = ap.parse_args(argv)
    root = V.DatasetRoot(a.dest)
    try:
        offsets = verify_offsets(root, a.tag)
        print(f"offsets OK: {offsets}")
        expected = set(json.loads(a.expected_keys_json.read_text()))
        print(f"membership OK: {verify_membership(root, a.tag, expected)}")
    except IntegrityError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
