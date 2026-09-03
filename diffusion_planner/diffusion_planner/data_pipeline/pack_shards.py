"""pack_shards CLI (spec §4). Usage: python -m diffusion_planner.data_pipeline.pack_shards <cmd> ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

from diffusion_planner.data_pipeline import export as E
from diffusion_planner.data_pipeline import keyset as K
from diffusion_planner.data_pipeline import packer as PK
from diffusion_planner.data_pipeline import partition as P
from diffusion_planner.data_pipeline import versioning as V
from diffusion_planner.data_pipeline.defaults import SHARD_SIZE_BYTES
from diffusion_planner.data_pipeline.errors import PipelineError


def _add_rule(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--partition-depth", type=int, help="partition = first N path components (no default)"
    )
    g.add_argument(
        "--partition-regex",
        help="partition = named group 'partition' (or group 1) matched on the key",
    )


def _add_selection(p: argparse.ArgumentParser) -> None:
    p.add_argument("--include", action="append", default=[], metavar="GLOB")
    p.add_argument("--exclude", action="append", default=[], metavar="GLOB")


def _rule(a) -> P.PartitionRule:
    return P.PartitionRule(depth=a.partition_depth, regex=a.partition_regex)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pack_shards")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect")
    p.add_argument("--source", required=True, type=Path)
    _add_rule(p)
    _add_selection(p)

    p = sub.add_parser("pack")
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--dest", required=True, type=Path)
    p.add_argument("--base", required=True, help="base version tag, 'latest', or 'none'")
    p.add_argument("--tag", required=True)
    _add_rule(p)
    _add_selection(p)
    p.add_argument("--path-list", type=Path)
    p.add_argument("--partition", action="append", dest="partitions")
    p.add_argument("--sync", action="store_true")
    p.add_argument("--replace-all", action="store_true")
    p.add_argument("--shard-size-gb", type=float, default=SHARD_SIZE_BYTES / 2**30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--keep-skipped",
        action="store_true",
        help="pack is_skipped==true frames too (default: drop)",
    )
    p.add_argument("--with-neighbor-ids", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--require-marker")

    p = sub.add_parser("remove")
    p.add_argument("--dest", required=True, type=Path)
    p.add_argument("--base", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--partition", action="append", required=True, dest="partitions")

    p = sub.add_parser("prune-version")
    p.add_argument("--dest", required=True, type=Path)
    p.add_argument("--tag", required=True)
    p = sub.add_parser("gc")
    p.add_argument("--dest", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("scrub")
    p.add_argument("--dest", required=True, type=Path)
    p.add_argument("--tag", required=True)

    p = sub.add_parser("export")
    p.add_argument("--dest", required=True, type=Path)
    p.add_argument("--tag", required=True)
    p.add_argument("--where", required=True)
    p.add_argument("--out", required=True, type=Path)

    p = sub.add_parser("keyset")
    p.add_argument("--dest", required=True, type=Path)
    p.add_argument("--tag", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--where")
    g.add_argument("--keys-json", type=Path)
    p.add_argument("--out", required=True, type=Path)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "inspect":
            print(P.inspect_tree(a.source, _rule(a), a.include, a.exclude).render())
        elif a.cmd == "pack":
            path_list = json.loads(a.path_list.read_text()) if a.path_list else None
            PK.pack(
                PK.PackOptions(
                    source=a.source,
                    dest=a.dest,
                    base=a.base,
                    tag=a.tag,
                    rule=_rule(a),
                    include=a.include,
                    exclude=a.exclude,
                    path_list=path_list,
                    partitions=a.partitions,
                    sync=a.sync,
                    replace_all=a.replace_all,
                    shard_size_bytes=max(int(a.shard_size_gb * 2**30), 1),
                    seed=a.seed,
                    drop_skipped=not a.keep_skipped,
                    with_neighbor_ids=a.with_neighbor_ids,
                    force=a.force,
                    require_marker=a.require_marker,
                )
            )
        elif a.cmd == "remove":
            PK.remove(a.dest, a.base, a.tag, a.partitions)
        elif a.cmd == "prune-version":
            root = V.DatasetRoot(a.dest)
            with V.writer_lock(root):
                V.prune_version(root, a.tag)
        elif a.cmd == "gc":
            root = V.DatasetRoot(a.dest)
            with V.writer_lock(root):
                for p in V.gc(root, a.dry_run):
                    print(("would delete " if a.dry_run else "deleted ") + str(p))
        elif a.cmd == "scrub":
            result = PK.scrub(a.dest, a.tag)
            print(
                f"scrub OK: {result['members']} members in {result['shards']} shards, "
                f"{result['mismatches']} mismatches"
            )
        elif a.cmd == "export":
            print(f"exported {E.export(a.dest, a.tag, a.where, a.out)} samples to {a.out}")
        elif a.cmd == "keyset":
            root = V.DatasetRoot(a.dest)
            if a.where is not None:
                if not a.where.strip():
                    raise ValueError("--where must not be empty")
                K.materialize_keyset(root, a.tag, a.where, a.out)
            else:
                K.keyset_from_keys(root, a.tag, json.loads(a.keys_json.read_text()), a.out)
            print(f"wrote {a.out}")
        return 0
    except (
        PipelineError,
        ValueError,
        FileNotFoundError,
        KeyError,
        duckdb.Error,
        TimeoutError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
