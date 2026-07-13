#!/usr/bin/env python3
"""Build bounded lifelong replay memory for R2LPL rounds."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path, default: Any) -> Any:
    if path is None or not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if math.isclose(lo, hi):
        return {k: 1.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def _bucket(row: dict[str, Any], arc_bin_m: float) -> str:
    arc = row.get("route_arc_m", row.get("frame_index", 0))
    try:
        arc_bin = int(float(arc) // arc_bin_m)
    except (TypeError, ValueError):
        arc_bin = 0
    label = str(row.get("label") or row.get("failure_label") or "unknown")
    variant = str(row.get("variant_kind") or row.get("trajectory_source") or "credit")
    return f"arc{arc_bin:05d}|{label}|{variant}"


def _parse_label_quotas(spec: str | None) -> dict[str, float]:
    """Parse ``label=frac,label=frac`` into a dict. Fails loudly on bad input."""
    if not spec:
        return {}
    quotas: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--label_quotas entry {part!r} must be label=fraction")
        label, frac_text = part.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"--label_quotas entry {part!r} has an empty label")
        frac = float(frac_text.strip())
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"--label_quotas fraction for {label!r} must be in (0, 1]: {frac}")
        if label in quotas:
            raise ValueError(f"--label_quotas has duplicate label {label!r}")
        quotas[label] = frac
    if sum(quotas.values()) > 1.0 + 1e-9:
        raise ValueError(f"--label_quotas fractions sum to > 1: {quotas}")
    return quotas


def _row_label(row: dict[str, Any]) -> str:
    return str(row.get("label") or row.get("failure_label") or "unknown")


def build_memory(
    current_rows: list[dict[str, Any]],
    previous_memory: dict[str, Any],
    per_scene_loss: dict[str, float],
    *,
    capacity: int,
    alpha: float,
    beta: float,
    arc_bin_m: float,
    label_quotas: dict[str, float] | None = None,
) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    for row in previous_memory.get("entries", []):
        path = str(row["scene_path"])
        entries[path] = dict(row)
    for row in current_rows:
        path = str(row.get("scene_path") or row.get("npz_path"))
        if not path or path == "None":
            continue
        merged = dict(entries.get(path, {}))
        merged.update(row)
        merged["scene_path"] = path
        entries[path] = merged

    difficulty_raw = {}
    utility_raw = {}
    for path, row in entries.items():
        difficulty_raw[path] = float(
            row.get("difficulty", row.get("credit_width", row.get("window_width", 1.0))) or 1.0
        )
        utility_raw[path] = float(per_scene_loss.get(path, row.get("utility", 0.0)) or 0.0)
    difficulty = _normalize(difficulty_raw)
    utility = _normalize(utility_raw)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path, row in entries.items():
        scored = dict(row)
        scored["difficulty_norm"] = difficulty[path]
        scored["utility_norm"] = utility[path]
        scored["priority"] = alpha * difficulty[path] + beta * utility[path]
        scored["bucket"] = _bucket(scored, arc_bin_m)
        buckets[scored["bucket"]].append(scored)

    for rows in buckets.values():
        rows.sort(key=lambda r: (-float(r["priority"]), str(r["scene_path"])))

    selected: list[dict[str, Any]] = []
    selected_paths: set[str] = set()
    selected_buckets: set[str] = set()

    def _take(row: dict[str, Any]) -> None:
        path = str(row["scene_path"])
        if path in selected_paths:
            return
        selected_paths.add(path)
        selected_buckets.add(str(row["bucket"]))
        selected.append(row)

    # Per-label quotas first: reserve floor(frac * capacity) top-priority slots
    # per label so a dominant label (rb/collision) cannot evict a rarer behavior
    # class (expert_disagreement) across rounds.
    scored_rows = [row for rows in buckets.values() for row in rows]
    if label_quotas:
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scored_rows:
            by_label[_row_label(row)].append(row)
        for label, frac in sorted(label_quotas.items()):
            reserve = int(frac * capacity)
            pool = sorted(
                by_label.get(label, []),
                key=lambda r: (-float(r["priority"]), str(r["scene_path"])),
            )
            for row in pool[:reserve]:
                _take(row)

    # Preserve rare buckets next — one representative per bucket. Buckets that
    # already have a member via the quota pass are REPRESENTED and must be
    # skipped, otherwise a quota label double-dips (its reserve + a second row
    # per bucket) and squeezes the majority labels beyond the quota's promise.
    for key in sorted(buckets):
        if len(selected) >= capacity:
            break
        if key in selected_buckets:
            continue
        for row in buckets[key]:
            if str(row["scene_path"]) not in selected_paths:
                _take(row)
                break
    remaining = [row for row in scored_rows if str(row["scene_path"]) not in selected_paths]
    remaining.sort(key=lambda r: (-float(r["priority"]), str(r["scene_path"])))
    for row in remaining[: max(0, capacity - len(selected))]:
        _take(row)
    selected.sort(key=lambda r: (-float(r["priority"]), str(r["scene_path"])))
    return {
        "capacity": capacity,
        "alpha": alpha,
        "beta": beta,
        "arc_bin_m": arc_bin_m,
        "label_quotas": dict(label_quotas or {}),
        "entries": selected[:capacity],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current_credit_jsonl", type=Path, required=True)
    parser.add_argument("--out_memory", type=Path, required=True)
    parser.add_argument("--out_replay_scenes", type=Path, required=True)
    parser.add_argument("--previous_memory", type=Path, default=None)
    parser.add_argument("--per_scene_loss_json", type=Path, default=None)
    parser.add_argument("--capacity", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--arc_bin_m", type=float, default=25.0)
    parser.add_argument(
        "--label_quotas",
        type=str,
        default=None,
        help=(
            "Reserved capacity fractions per event label, e.g. "
            "'expert_disagreement=0.25,road_border_crossing=0.2'. Each listed label "
            "keeps at least floor(frac*capacity) of its top-priority entries so a "
            "dominant label cannot evict a rarer behavior class across rounds."
        ),
    )
    args = parser.parse_args()
    if args.capacity < 1:
        raise ValueError("--capacity must be >= 1")
    label_quotas = _parse_label_quotas(args.label_quotas)
    rows = _load_jsonl(args.current_credit_jsonl)
    prev = (
        _load_json(args.previous_memory, {"entries": []})
        if args.previous_memory
        else {"entries": []}
    )
    losses = _load_json(args.per_scene_loss_json, {}) if args.per_scene_loss_json else {}
    memory = build_memory(
        rows,
        prev,
        losses,
        capacity=args.capacity,
        alpha=args.alpha,
        beta=args.beta,
        arc_bin_m=args.arc_bin_m,
        label_quotas=label_quotas,
    )
    args.out_memory.parent.mkdir(parents=True, exist_ok=True)
    args.out_replay_scenes.parent.mkdir(parents=True, exist_ok=True)
    args.out_memory.write_text(json.dumps(memory, indent=2))
    args.out_replay_scenes.write_text(
        json.dumps([row["scene_path"] for row in memory["entries"]], indent=2)
    )
    print(f"Wrote {len(memory['entries'])} replay scenes to {args.out_replay_scenes}")


if __name__ == "__main__":
    main()
