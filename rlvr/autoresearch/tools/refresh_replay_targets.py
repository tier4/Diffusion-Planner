#!/usr/bin/env python3
"""Keep replay memory a FLOOR instead of a leash.

A replayed scene is trained against the trajectory that won selection when the scene was
first repaired — a sample from a policy that no longer exists. As the chain advances that
target drifts off the current policy's manifold, so replay stops protecting old competence
and starts dragging the model back toward an older policy's taste, with target quality
capped at whatever that policy could propose.

This module refreshes those targets with a MONOTONE rule:

    memory_target = argmax_reward( frozen_target , current policy's best candidate )

Taking the max (rather than overwriting with the fresh selection) is what preserves
retention: if the policy has drifted and its fresh candidates for an old scene are all
worse, the frozen target wins and keeps teaching the old fix. The target can only improve
across rounds, never decay.

Two pure-CPU steps; the GPU work in between is the ORDINARY repair phase, unchanged:

  1. ``build-rows``  replay list + previous rounds' repaired rows -> rows jsonl of the
                     ORIGINAL source scenes (re-repairing the source, not the frozen
                     target, so the fresh pass sees exactly the inputs the first pass saw).
  2. ``join``        replay list + previous rows + fresh rows -> final replay list, each
                     entry pointing at whichever NPZ carries the better-scoring target.

Scores are compared on the SAME ruler both times (reward of a fixed trajectory in a fixed
scene is deterministic), so the caller must run the fresh repair with the same reward
config that produced the frozen scores.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _read_json_list(path: Path) -> list[str]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of scene paths")
    return [str(x) for x in data]


def _read_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Map repaired-target NPZ path -> its row.

    Accepts either a repaired-targets ``.jsonl`` or a replay-memory ``.json`` (whose
    ``entries`` carry the same rows). The memory file matters for a chain link: it is the
    only previous-round artifact the runner is handed, so accepting it keeps the refresh
    self-contained instead of reaching into the previous link's output directory.
    """
    rows: dict[str, dict[str, Any]] = {}
    for p in paths:
        p = Path(p)
        if p.suffix == ".json":
            payload = json.loads(p.read_text())
            entries = payload.get("entries") if isinstance(payload, dict) else payload
            if not isinstance(entries, list):
                raise ValueError(f"{p}: expected a memory JSON with 'entries' or a list of rows")
            for row in entries:
                key = row.get("scene_path")
                if key:
                    rows[str(key)] = row
            continue
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = row.get("scene_path")
                if key:
                    rows[str(key)] = row
    return rows


def build_rows(
    replay_scenes: Path, prev_rows: list[Path], out_rows: Path, *, allow_missing: bool = False
) -> dict[str, int]:
    """Emit the rows a repair pass needs to re-generate candidates for replayed scenes."""
    replay = _read_json_list(replay_scenes)
    rows = _read_rows(prev_rows)
    missing = [p for p in replay if p not in rows]
    if missing and not allow_missing:
        raise ValueError(
            f"{len(missing)} of {len(replay)} replay scenes have no repaired row in "
            f"{[str(p) for p in prev_rows]} (first: {missing[0]}). Without the row there is no "
            "source scene to re-repair and no frozen score to compare against; pass "
            "--allow_missing to skip them instead."
        )
    written = 0
    with open(out_rows, "w") as fo:
        for scene in replay:
            row = rows.get(scene)
            if row is None:
                continue
            src = row.get("source_scene_path")
            if not src:
                raise ValueError(f"{scene}: repaired row lacks source_scene_path")
            fresh = dict(row)
            # Re-repair the ORIGINAL mined window: its ego_agent_future is still the
            # recorded future, whereas the frozen target NPZ has it overwritten with the old
            # selection, which would silently re-reference every deviation/progress term.
            fresh["scene_path"] = str(src)
            # Carry the frozen identity so the join can pair fresh rows back to replay entries.
            fresh["refresh_frozen_target"] = scene
            fresh["refresh_frozen_total"] = row.get("selected_total")
            fo.write(json.dumps(fresh) + "\n")
            written += 1
    return {"replay": len(replay), "rows_written": written, "missing": len(missing)}


def join(
    replay_scenes: Path,
    prev_rows: list[Path],
    fresh_rows: list[Path],
    out_list: Path,
    out_stats: Path,
    *,
    min_gain: float = 0.0,
    out_map: Path | None = None,
) -> dict[str, Any]:
    """Pick max(frozen, fresh) per replayed scene and write the final replay list.

    With ``out_map`` also records {frozen_path: {new_path, new_total}} for the scenes the
    fresh policy won, which ``persist_into_memory`` uses to carry the gain into later rounds.
    """
    if float(min_gain) < 0.0:
        raise ValueError(
            f"min_gain must be >= 0 (got {min_gain}); negative values break monotonicity"
        )
    improved_map: dict[str, dict[str, Any]] = {}
    replay = _read_json_list(replay_scenes)
    frozen_rows = _read_rows(prev_rows)
    # Fresh rows are keyed by the SOURCE scene they were re-repaired from.
    fresh_by_source: dict[str, dict[str, Any]] = {}
    for p in fresh_rows:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                src = row.get("source_scene_path")
                if src:
                    fresh_by_source[str(src)] = row

    final: list[str] = []
    stats = {
        "replay_in": len(replay),
        "improved_by_fresh": 0,
        "kept_frozen": 0,
        "no_fresh_candidate": 0,
        "gain_sum": 0.0,
    }
    for scene in replay:
        frow = frozen_rows.get(scene)
        if frow is None:
            raise ValueError(f"{scene}: no frozen repaired row; run build-rows first")
        src = str(frow.get("source_scene_path"))
        frozen_total = frow.get("selected_total")
        fresh = fresh_by_source.get(src)
        if fresh is None:
            # The current policy produced NO gate-passing candidate for this scene. That is
            # information, not an error: the frozen fix is still the best thing we have.
            stats["no_fresh_candidate"] += 1
            final.append(scene)
            continue
        fresh_total = fresh.get("selected_total")
        fresh_path = fresh.get("scene_path")
        # Every repaired row writes selected_total and scene_path unconditionally, so a
        # missing score here is corrupted bookkeeping, not a scene state — keeping the
        # scene silently would let an un-comparable row ride the replay list forever.
        if frozen_total is None or fresh_total is None or not fresh_path:
            missing = (
                "frozen selected_total"
                if frozen_total is None
                else ("fresh selected_total" if fresh_total is None else "fresh scene_path")
            )
            raise ValueError(
                f"{scene}: {missing} is missing — repaired rows always carry both; "
                "the row source is corrupted or hand-edited"
            )
        if float(fresh_total) > float(frozen_total) + float(min_gain):
            stats["improved_by_fresh"] += 1
            stats["gain_sum"] += float(fresh_total) - float(frozen_total)
            final.append(str(fresh_path))
            improved_map[scene] = {
                "new_path": str(fresh_path),
                "new_total": float(fresh_total),
                "frozen_path": scene,
                "frozen_total": float(frozen_total),
            }
        else:
            stats["kept_frozen"] += 1
            final.append(scene)

    stats["replay_out"] = len(final)
    stats["mean_gain_on_improved"] = (
        stats["gain_sum"] / stats["improved_by_fresh"] if stats["improved_by_fresh"] else 0.0
    )
    Path(out_list).write_text(json.dumps(final, indent=2))
    Path(out_stats).write_text(json.dumps(stats, indent=2))
    if out_map is not None:
        Path(out_map).write_text(json.dumps(improved_map, indent=2))
    return stats


def persist_into_memory(memory_json: Path, refresh_map: Path) -> dict[str, int]:
    """Point the replay MEMORY at refreshed targets so the gain carries to later rounds.

    Without this the refresh is per-round scratch: the memory step runs before the refresh
    (it produces the replay list the refresh consumes), so the memory a link hands to its
    successor still names the ORIGINAL targets. Every link then re-generates candidates for
    the same already-improved scenes, and memory never actually rises -- it is re-derived.

    Rewrites BOTH ``scene_path`` and ``selected_total`` together: the next round's join
    compares a fresh candidate against the stored score, so moving the path without the score
    would make it compare a new target against the old target's score and keep/replace wrongly.

    Safe by construction: a refreshed target only enters the map when it beat the frozen one
    on the same frozen ruler, so the stored target is monotone non-decreasing across rounds.
    """
    mem_path = Path(memory_json)
    payload = json.loads(mem_path.read_text())
    entries = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError(f"{mem_path}: expected a memory JSON with 'entries'")
    mapping = json.loads(Path(refresh_map).read_text())
    updated = 0
    for row in entries:
        hit = mapping.get(str(row.get("scene_path")))
        if not hit:
            continue
        # Record provenance BEFORE overwriting scene_path — the fallback must name
        # the frozen target, not the fresh one it was just repointed to.
        row["refreshed_from"] = hit.get("frozen_path", row.get("scene_path"))
        row["scene_path"] = hit["new_path"]
        row["selected_total"] = hit["new_total"]
        updated += 1
    # Atomic replace: this file is what a chain link hands to its successor; a
    # crash mid-write must not leave a truncated memory JSON behind.
    tmp = mem_path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(mem_path)
    return {"memory_entries": len(entries), "repointed_to_refreshed": updated}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-rows", help="replay list -> rows jsonl for a fresh repair pass")
    b.add_argument("--replay_scenes", type=Path, required=True)
    b.add_argument("--prev_rows_jsonl", type=Path, nargs="+", required=True)
    b.add_argument("--out_rows_jsonl", type=Path, required=True)
    b.add_argument("--allow_missing", action="store_true")

    j = sub.add_parser("join", help="pick max(frozen, fresh) per scene")
    j.add_argument("--replay_scenes", type=Path, required=True)
    j.add_argument("--prev_rows_jsonl", type=Path, nargs="+", required=True)
    j.add_argument("--fresh_rows_jsonl", type=Path, nargs="+", required=True)
    j.add_argument("--out_list", type=Path, required=True)
    j.add_argument("--out_stats", type=Path, required=True)
    j.add_argument(
        "--out_map",
        type=Path,
        default=None,
        help="record {frozen: {new_path,new_total}} for scenes the fresh policy won, "
        "for persist-memory to carry the gain into later rounds.",
    )
    j.add_argument(
        "--min_gain",
        type=float,
        default=0.0,
        help="reward margin the fresh target must beat the frozen one by; >0 adds hysteresis "
        "so near-ties keep the frozen target and memory stays stable across rounds.",
    )

    pm = sub.add_parser("persist-memory", help="repoint replay memory at refreshed targets")
    pm.add_argument("--memory_json", type=Path, required=True)
    pm.add_argument("--refresh_map", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "persist-memory":
        print(json.dumps(persist_into_memory(args.memory_json, args.refresh_map), indent=2))
        return
    if args.cmd == "build-rows":
        out = build_rows(
            args.replay_scenes,
            list(args.prev_rows_jsonl),
            args.out_rows_jsonl,
            allow_missing=bool(args.allow_missing),
        )
    else:
        out = join(
            args.replay_scenes,
            list(args.prev_rows_jsonl),
            list(args.fresh_rows_jsonl),
            args.out_list,
            args.out_stats,
            min_gain=float(args.min_gain),
            out_map=args.out_map,
        )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
