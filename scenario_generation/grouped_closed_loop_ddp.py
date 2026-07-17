"""DDP bag sharding for grouped closed-loop eval.

When ``train_predictor.py`` is launched with ``torchrun`` (``RANK`` / ``WORLD_SIZE`` /
``LOCAL_RANK``), each rank evaluates a disjoint subset of bags on its GPU. Rank 0
merges shard JSON files after a barrier — no extra process pool or checkpoint reload.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from scenario_generation.grouped_closed_loop_eval import (
    _build_timing_summary,
    _finalize_grouped_summary,
)
from scenario_generation.metrics.cl_route_by_area import RouteRolloutResult
from scenario_generation.perf_timer import Timers


def shard_bag_items(
    bag_items: list[tuple[str, dict]], rank: int, world_size: int
) -> list[tuple[str, dict]]:
    """Round-robin bag assignment: rank ``r`` gets indices r, r+world_size, ..."""
    if world_size <= 1:
        return bag_items
    return [bag_items[i] for i in range(rank, len(bag_items), world_size)]


def write_ddp_shard(
    out_dir: Path,
    rank: int,
    *,
    segments: list[dict],
    video_mp4s: list[Path],
    rollout_bags: list[str],
    elapsed_sec: float,
    total_steps: int,
    per_bag_timing: list[dict],
    profile: bool,
) -> Path:
    """Persist one rank's partial eval for post-barrier merge on rank 0."""
    shard_dir = out_dir / "ddp_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"rank_{rank:03d}.json"
    payload: dict[str, Any] = {
        "rank": rank,
        "segments": segments,
        "video_mp4s": [str(p) for p in video_mp4s],
        "rollout_bags": rollout_bags,
        "elapsed_sec": elapsed_sec,
        "total_steps": total_steps,
        "per_bag_timing": per_bag_timing if profile else [],
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def merge_ddp_grouped_shards(
    out_dir: Path | str,
    world_size: int,
    *,
    classification_json: Path | str,
    npz_root: Path | str,
    near_miss_thresh: float,
    profile: bool,
) -> dict:
    """Merge per-rank shard files written under ``out_dir/ddp_shards/``."""
    out_dir = Path(out_dir)
    all_rows: list[dict] = []
    video_mp4s: list[Path] = []
    rollouts: dict[str, RouteRolloutResult] = {}
    per_bag_timing: list[dict] = []
    total_steps = 0
    elapsed_sec = 0.0

    for rank in range(world_size):
        path = out_dir / "ddp_shards" / f"rank_{rank:03d}.json"
        if not path.is_file():
            continue
        shard = json.loads(path.read_text(encoding="utf-8"))
        all_rows.extend(shard.get("segments") or [])
        video_mp4s.extend(Path(p) for p in shard.get("video_mp4s") or [])
        total_steps += int(shard.get("total_steps", 0))
        elapsed_sec = max(elapsed_sec, float(shard.get("elapsed_sec", 0.0)))
        per_bag_timing.extend(shard.get("per_bag_timing") or [])
        for bag_name in shard.get("rollout_bags") or []:
            rollouts[str(bag_name)] = RouteRolloutResult(
                route_key=str(bag_name),
                sequence=str(bag_name),
            )

    timing = None
    if profile and per_bag_timing:
        global_timers = Timers()
        for bag_info in per_bag_timing:
            for stage, info in bag_info.get("stages", {}).items():
                global_timers.add(stage, float(info["total_s"]), int(info["calls"]))
        timing = _build_timing_summary(global_timers, per_bag_timing, total_steps, elapsed_sec)
        timing["ddp_world_size"] = world_size
        print("\n" + timing["report"])

    summary = _finalize_grouped_summary(
        classification_json=classification_json,
        npz_root=npz_root,
        near_miss_thresh=near_miss_thresh,
        all_rows=all_rows,
        video_mp4s=video_mp4s,
        rollouts=rollouts,
        elapsed_sec=elapsed_sec,
        timing=timing,
        out_dir=out_dir,
    )
    summary["ddp_shard"] = True
    summary["ddp_world_size"] = world_size
    return summary


def ddp_device_for_rank(fallback: str = "cuda") -> str:
    """Resolve the CUDA device for the current DDP rank from ``LOCAL_RANK``."""
    import os

    if "LOCAL_RANK" in os.environ:
        return f"cuda:{int(os.environ['LOCAL_RANK'])}"
    return fallback
