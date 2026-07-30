"""Command line: ``python -m tag_toolkit <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .store import TagStore, format_buckets
from .taxonomy import list_known_tags


def _parse_clause(text: str):
    stripped = text.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    return stripped


def _store(args: argparse.Namespace) -> TagStore:
    return TagStore(source=args.source)


def cmd_query(args: argparse.Namespace) -> int:
    store = _store(args)
    clause = _parse_clause(args.clause)
    paths = store.query(clause, granularity=args.granularity)
    if args.out:
        Path(args.out).write_text(json.dumps([str(p) for p in paths], indent=2) + "\n")
        print(f"wrote {len(paths)} paths to {args.out}")
    else:
        for path in paths:
            print(path)
    return 0


def cmd_group_by(args: argparse.Namespace) -> int:
    store = _store(args)
    clause = _parse_clause(args.clause) if args.clause else None
    # Keep library default (route → fill members) unless --members forces print.
    buckets = store.group_by(
        args.dimensions,
        members=None,
        clause=clause,
        granularity=args.granularity,
    )
    print(format_buckets(buckets, args.dimensions))
    if args.members:
        for bucket in buckets:
            print(f"\n# {bucket.label()}")
            for path in bucket.members or []:
                print(path)
    return 0


def cmd_tags(args: argparse.Namespace) -> int:
    store = TagStore()
    for path in args.paths:
        print(f"{path}: {store.tags_for(path)}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    n = _store(args).add_tags(None, args.tags)
    print(f"updated {n} sidecars")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    n = _store(args).remove_tags(None, args.tags)
    print(f"updated {n} sidecars")
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    n = _store(args).set_tags(None, args.tags, mode=args.mode)
    print(f"updated {n} sidecars")
    return 0


def cmd_known_tags(args: argparse.Namespace) -> int:
    for tag in list_known_tags(args.taxonomy):
        print(tag)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tag_toolkit", description=__doc__)
    parser.add_argument(
        "--source",
        help="npz / directory / path-list JSON (required for most commands)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("query", help="list routes (default) or npz matching a clause")
    p.add_argument("clause", help="'dim:value' or JSON all/any/not")
    p.add_argument(
        "--granularity",
        choices=("route", "frame"),
        default="route",
        help="route = union of frame tags (default); frame = per-npz",
    )
    p.add_argument("--out", help="write path list JSON")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("group-by", help="bucket routes (default) or frames by dimensions")
    p.add_argument("dimensions", nargs="+")
    p.add_argument("--clause", help="optional filter clause")
    p.add_argument(
        "--granularity",
        choices=("route", "frame"),
        default="route",
        help="route = union of frame tags (default); frame = per-npz",
    )
    p.add_argument("--members", action="store_true")
    p.set_defaults(func=cmd_group_by)

    p = sub.add_parser("tags", help="print tags for npz paths")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_tags)

    p = sub.add_parser("add", help="add tags to every frame under --source")
    p.add_argument("tags", nargs="+")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("remove", help="remove tags under --source")
    p.add_argument("tags", nargs="+")
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser("set", help="merge or replace tags under --source")
    p.add_argument("tags", nargs="+")
    p.add_argument("--mode", choices=("merge", "replace"), default="merge")
    p.set_defaults(func=cmd_set)

    p = sub.add_parser("known-tags", help="list tags from taxonomy.yaml (helper)")
    p.add_argument("--taxonomy", help="path to taxonomy yaml")
    p.set_defaults(func=cmd_known_tags)

    args = parser.parse_args(argv)
    needs_source = args.command in {
        "query",
        "group-by",
        "add",
        "remove",
        "set",
    }
    if needs_source and not args.source:
        parser.error(f"--source is required for {args.command}")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
