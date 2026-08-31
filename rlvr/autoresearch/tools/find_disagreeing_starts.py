"""Find the rollouts where two arms disagree, so a clip is worth rendering.

A clip is only evidence when the two arms did something different from the SAME
start. This reads the per-rollout rows written by
:mod:`rlvr.autoresearch.tools.eval_recovery_route` for two or more labels and lists
the starts where one arm recovered and another was lost — the cases where a
side-by-side clip shows the difference the table reports.

Rows are keyed by ``(route, start, offset)``. The route matters: an eval spans more
than one recorded drive and start indices collide between them, so a
``(start, offset)`` key alone silently mixes two different places.

Usage:
    # every shard JSON of both arms, in any order
    python -m rlvr.autoresearch.tools.find_disagreeing_starts \\
        --rows $CAMP/recovery/*.json \\
        --win treatedEP200 --lose incumbentEP200 --limit 10

    # machine-readable, to drive the clip renderer
    python -m rlvr.autoresearch.tools.find_disagreeing_starts \\
        --rows $CAMP/recovery/*.json --win A --lose B --out_json picks.json
"""

import argparse
import json
from pathlib import Path


def load_rows(paths: list[str]) -> dict[str, dict]:
    """``{label: {(route, start, offset): row}}`` from any number of shard JSONs."""
    per_label: dict[str, dict] = {}
    for p in paths:
        for row in json.loads(Path(p).read_text()):
            key = (row.get("route", ""), row["start"], round(row["offset"], 2))
            per_label.setdefault(row["label"], {})[key] = row
    return per_label


def describe(row: dict) -> str:
    """``LOST`` or the settle value — the same verdict the tables report."""
    if row["lost"]:
        return "LOST"
    settle = row.get("usage_settle")
    return f"{settle:.3f}" if settle is not None else "n/a"


def disagreements(win_rows: dict, lose_rows: dict, *, require_lost: bool) -> list[tuple]:
    """Keys where ``win`` recovered and ``lose`` did not, over their common starts.

    ``require_lost`` demands the losing arm actually left the route, which makes the
    clearest clips; without it, "did not recover" also includes rollouts that stayed
    near the route but never settled.
    """
    common = sorted(set(win_rows) & set(lose_rows))
    out = []
    for key in common:
        w, lose = win_rows[key], lose_rows[key]
        if not w["recovered"]:
            continue
        if lose["lost"] if require_lost else not lose["recovered"]:
            out.append(key)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", nargs="+", required=True, help="recovery row JSONs (shards ok)")
    ap.add_argument("--win", required=True, help="label expected to recover")
    ap.add_argument("--lose", required=True, help="label expected to fail")
    ap.add_argument(
        "--any_non_recovery",
        action="store_true",
        help="count 'did not recover' rather than only 'left the route' for --lose",
    )
    ap.add_argument("--limit", type=int, default=10, help="how many to print")
    ap.add_argument("--out_json", default=None, help="also write the picks here")
    args = ap.parse_args()

    per_label = load_rows(args.rows)
    for label in (args.win, args.lose):
        if label not in per_label:
            raise SystemExit(
                f"no rows for label {label!r}; found {sorted(per_label)} in the given files"
            )

    win_rows, lose_rows = per_label[args.win], per_label[args.lose]
    keys = disagreements(win_rows, lose_rows, require_lost=not args.any_non_recovery)
    common = len(set(win_rows) & set(lose_rows))
    print(
        f"{len(keys)} of {common} shared rollouts have {args.win} recovering "
        f"and {args.lose} {'lost' if not args.any_non_recovery else 'not recovering'}"
    )
    for route, start, offset in keys[: args.limit]:
        print(
            f"  --start {start:<5} --offset {offset:+.1f}   "
            f"{args.win} {describe(win_rows[(route, start, offset)]):>5} | "
            f"{args.lose} {describe(lose_rows[(route, start, offset)]):>5}   route={route}"
        )

    if args.out_json:
        picks = [
            dict(
                route=route,
                start=start,
                offset=offset,
                win=describe(win_rows[(route, start, offset)]),
                lose=describe(lose_rows[(route, start, offset)]),
            )
            for route, start, offset in keys
        ]
        Path(args.out_json).write_text(json.dumps(picks, indent=1))
        print(f"wrote {len(picks)} picks to {args.out_json}")


if __name__ == "__main__":
    main()
