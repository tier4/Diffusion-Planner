"""Grouped closed-loop validation (scenario_classification_json episodes).

Shared by the standalone CLI and per-epoch training validation in ``train.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from scenario_generation.metrics.cl_route_by_area import (
    RouteRolloutResult,
    aggregate_area_metrics,
    build_area_video,
    run_full_route_rollout,
)
from scenario_generation.metrics.group_report import (
    aggregate_segment_rows,
    write_metrics_summary,
    write_results_table,
)
from scenario_generation.perf_timer import Timers
from scenario_generation.route_timeline import RouteTimeline, _frame_index
from scenario_generation.scenario_classification import (
    episodes_from_entry,
    load_area_metric_groups,
    load_classification_json,
    time_series_from_doc,
    validate_classification_npz_root,
)


def _rollout_kwargs(
    *,
    device: str,
    near_miss_thresh: float,
    search_radius: float,
    warmup_steps: int,
    unstick_after: int,
    unstick_advance_m: float,
    draw_every: int,
    replan_interval: int,
    tracker_mode: str,
    profile: bool,
    profile_sync_gpu: bool,
) -> dict[str, Any]:
    return {
        "device": device,
        "near_miss_thresh": near_miss_thresh,
        "search_radius": search_radius,
        "warmup_steps": warmup_steps,
        "unstick_after": unstick_after,
        "unstick_advance_m": unstick_advance_m,
        "draw_every": draw_every,
        "replan_interval": replan_interval,
        "tracker_mode": tracker_mode,
        "profile": profile,
        "profile_sync_gpu": profile_sync_gpu,
    }


def _aggregate_bag_episodes(
    bag_name: str,
    episodes: list[dict],
    rollout: RouteRolloutResult,
    tl: RouteTimeline,
    *,
    out_dir: Path,
    area_to_metric: dict[str, str],
    allowed_areas: set[str] | None,
    near_miss_thresh: float,
    fps: float,
    verbose: bool,
    timers: Timers | None = None,
) -> tuple[list[dict], list[Path]]:
    """Per-episode metrics + videos for one bag after its rollout."""
    rows: list[dict] = []
    video_mp4s: list[Path] = []
    span_counts: dict[str, int] = {}
    route_key = rollout.route_key

    for ep in episodes:
        area_name = str(ep["area"])
        if allowed_areas is not None and area_name not in allowed_areas:
            continue
        metric_group = area_to_metric.get(area_name)
        if metric_group is None:
            continue
        span_key = f"{bag_name}:{area_name}"
        span_index = span_counts.get(span_key, 0)
        span_counts[span_key] = span_index + 1

        if timers is not None:
            with timers("aggregate_metrics"):
                row = aggregate_area_metrics(
                    rollout.steps,
                    area_name,
                    metric_group,
                    route_key,
                    bag_name,
                    tl,
                    labeled_ranges=ep["labeled_ranges"],
                    video_start_idx=int(ep["video_start_idx"]),
                    video_end_idx=int(ep["video_end_idx"]),
                    span_index=span_index,
                    near_miss_thresh=near_miss_thresh,
                )
        else:
            row = aggregate_area_metrics(
                rollout.steps,
                area_name,
                metric_group,
                route_key,
                bag_name,
                tl,
                labeled_ranges=ep["labeled_ranges"],
                video_start_idx=int(ep["video_start_idx"]),
                video_end_idx=int(ep["video_end_idx"]),
                span_index=span_index,
                near_miss_thresh=near_miss_thresh,
            )
        if row is None:
            continue

        video_root = out_dir / "videos" / metric_group / area_name
        video_root.mkdir(parents=True, exist_ok=True)
        seg_tag = row["segment"].strip("[]").replace(",", "_").replace(" ", "")
        mp4 = video_root / f"{bag_name}_{span_index}_{seg_tag}.mp4"
        if timers is not None:
            with timers("build_videos"):
                built = build_area_video(
                    rollout,
                    mp4,
                    fps,
                    video_start_idx=int(ep["video_start_idx"]),
                    video_end_idx=int(ep["video_end_idx"]),
                )
        else:
            built = build_area_video(
                rollout,
                mp4,
                fps,
                video_start_idx=int(ep["video_start_idx"]),
                video_end_idx=int(ep["video_end_idx"]),
            )
        if built:
            row["video_path"] = str(mp4.relative_to(out_dir))
            video_mp4s.append(mp4)
        rows.append(row)
        if verbose:
            print(
                f"  [{area_name}] {bag_name} span#{span_index} {row['segment']} "
                f"steps={row['n_steps_run']} coll={row.get('collision_steps', 0)}"
            )
    return rows, video_mp4s


def process_grouped_bag(
    bag_name: str,
    entry: dict,
    npz_root: Path,
    out_dir: Path,
    model,
    model_args,
    *,
    area_to_metric: dict[str, str],
    allowed_areas: set[str] | None,
    rollout_kwargs: dict[str, Any],
    fps: float,
    verbose: bool,
) -> dict[str, Any] | None:
    """Run rollout + per-episode metrics/videos for one bag. Returns None if skipped."""
    seq_dir = npz_root / bag_name
    if not seq_dir.is_dir():
        if verbose:
            print(f"Skip bag {bag_name}: not under {npz_root}")
        return None
    paths = sorted(seq_dir.glob("*.npz"), key=_frame_index)
    if len(paths) < 2:
        if verbose:
            print(f"Skip bag {bag_name}: fewer than 2 NPZ frames")
        return None

    route_key = str(entry["route_key"])
    episodes = episodes_from_entry(entry)
    profile = bool(rollout_kwargs.get("profile"))
    bag_timers = Timers() if profile else None

    tl = RouteTimeline(paths, sidecar_dir=npz_root, timers=bag_timers)
    png_dir = out_dir / "rollouts" / bag_name
    if verbose:
        print(
            f"Full-route rollout: {bag_name} ({route_key}), "
            f"{len(paths)} frames, {len(episodes)} episodes..."
        )

    if bag_timers is not None:
        with bag_timers("rollout"):
            rollout = run_full_route_rollout(
                model,
                model_args,
                tl,
                route_key,
                bag_name,
                episodes,
                area_to_metric,
                png_dir,
                **rollout_kwargs,
            )
        if rollout.timers is not None:
            bag_timers.merge(rollout.timers)
    else:
        rollout = run_full_route_rollout(
            model,
            model_args,
            tl,
            route_key,
            bag_name,
            episodes,
            area_to_metric,
            png_dir,
            **rollout_kwargs,
        )

    if verbose:
        print(
            f"  done: steps={rollout.n_steps_run} terminated={rollout.terminated} "
            f"episodes={len(episodes)}"
        )

    rows, video_mp4s = _aggregate_bag_episodes(
        bag_name,
        episodes,
        rollout,
        tl,
        out_dir=out_dir,
        area_to_metric=area_to_metric,
        allowed_areas=allowed_areas,
        near_miss_thresh=float(rollout_kwargs["near_miss_thresh"]),
        fps=fps,
        verbose=verbose,
        timers=bag_timers,
    )

    bag_timing: dict[str, Any] = {}
    if bag_timers is not None:
        bag_timing = {
            "bag": bag_name,
            "n_steps_run": rollout.n_steps_run,
            "stages": bag_timers.as_dict(),
        }

    return {
        "bag_name": bag_name,
        "rollout": rollout,
        "rows": rows,
        "video_mp4s": video_mp4s,
        "bag_timing": bag_timing,
    }


def _finalize_grouped_summary(
    *,
    classification_json: Path | str,
    npz_root: Path,
    near_miss_thresh: float,
    all_rows: list[dict],
    video_mp4s: list[Path],
    rollouts: dict[str, RouteRolloutResult],
    elapsed_sec: float,
    timing: dict[str, Any] | None,
    out_dir: Path,
) -> dict:
    grouped_summary = aggregate_segment_rows(all_rows)
    area_names = sorted({r["area_name"] for r in all_rows})
    for area_name in area_names:
        area_rows = [r for r in all_rows if r["area_name"] == area_name]
        write_metrics_summary(
            aggregate_segment_rows(area_rows),
            out_dir / "by_area" / f"{area_name}_metrics_summary.json",
        )

    write_results_table(all_rows, out_dir / "results_table.csv")
    write_metrics_summary(grouped_summary, out_dir / "metrics_summary.json")

    summary: dict[str, Any] = {
        "mode": "grouped",
        "classification_json": str(Path(classification_json).resolve()),
        "npz_root": str(npz_root),
        "near_miss_thresh": near_miss_thresh,
        "n_episodes": len(all_rows),
        "n_bags": len(rollouts),
        "elapsed_sec": elapsed_sec,
        "grouped_summary": grouped_summary,
        "segments": all_rows,
        "video_mp4s": video_mp4s,
    }
    if timing is not None:
        summary["timing"] = timing
        with open(out_dir / "timing.json", "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                k: v
                for k, v in summary.items()
                if k not in ("segments", "video_mp4s", "grouped_summary", "timing")
            }
            | {"grouped_summary": grouped_summary}
            | ({"timing": timing} if timing else {}),
            f,
            indent=2,
            default=str,
        )
    return summary


def _build_timing_summary(
    global_timers: Timers,
    per_bag_timing: list[dict],
    total_steps: int,
    elapsed_sec: float,
) -> dict[str, Any]:
    stages = global_timers.as_dict()
    model_calls = stages.get("model_forward", {}).get("calls", 0)
    cached_calls = stages.get("model_forward_cached", {}).get("calls", 0)
    total_sim_steps = total_steps + cached_calls
    return {
        "elapsed_sec": elapsed_sec,
        "total_sim_steps": total_sim_steps,
        "model_forward_calls": model_calls,
        "model_forward_cached_calls": cached_calls,
        "model_forward_rate": model_calls / max(1, total_sim_steps),
        "stages": stages,
        "per_bag": per_bag_timing,
        "report": global_timers.report(total_sim_steps),
    }


@torch.no_grad()
def run_grouped_closed_loop_eval(
    model,
    model_args,
    npz_root: Path | str,
    classification_json: Path | str,
    out_dir: Path | str,
    *,
    device: str = "cuda",
    near_miss_thresh: float = 0.3,
    search_radius: float = 1.5,
    warmup_steps: int = 0,
    unstick_after: int = 300,
    unstick_advance_m: float = 2.5,
    draw_every: int = 8,
    fps: float = 10.0,
    areas: list[str] | None = None,
    replan_interval: int = 10,
    tracker_mode: str = "mpc",
    verbose: bool = True,
    profile: bool = False,
    profile_sync_gpu: bool = False,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    max_bags: int | None = None,
) -> dict:
    """Run grouped-by-area closed-loop eval; return summary for disk + wandb.

    One full-route rollout per bag in the classification JSON, then per-episode
    metrics (non-``is_skipped`` labeled frames) and continuous-span videos.

    When ``ddp_world_size > 1``, each rank processes ``bags[rank::world_size]`` and
    writes ``out_dir/ddp_shards/rank_XXX.json``. Call :func:`merge_ddp_grouped_shards`
    on rank 0 after ``torch.distributed.barrier()``.
    """
    npz_root = Path(npz_root).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = load_classification_json(Path(classification_json))
    time_series = time_series_from_doc(doc)
    if not time_series:
        raise ValueError(
            "classification JSON has no 'time_series' — re-run classify_scenario_corpus"
        )

    area_to_metric = load_area_metric_groups(doc)
    allowed_areas: set[str] | None = set(areas) if areas else None
    rollout_kwargs = _rollout_kwargs(
        device=device,
        near_miss_thresh=near_miss_thresh,
        search_radius=search_radius,
        warmup_steps=warmup_steps,
        unstick_after=unstick_after,
        unstick_advance_m=unstick_advance_m,
        draw_every=draw_every,
        replan_interval=replan_interval,
        tracker_mode=tracker_mode,
        profile=profile,
        profile_sync_gpu=profile_sync_gpu,
    )

    rollouts: dict[str, RouteRolloutResult] = {}
    all_rows: list[dict] = []
    video_mp4s: list[Path] = []
    per_bag_timing: list[dict] = []
    global_timers = Timers() if profile else None
    total_steps = 0

    t0 = time.perf_counter()

    for warning in validate_classification_npz_root(doc, npz_root):
        if verbose:
            print(f"Warning: {warning}")

    bag_items = sorted(time_series.items())
    if max_bags is not None:
        bag_items = bag_items[:max_bags]

    from scenario_generation.grouped_closed_loop_ddp import shard_bag_items, write_ddp_shard

    ddp_shard_mode = ddp_world_size > 1
    if ddp_shard_mode:
        bag_items = shard_bag_items(bag_items, ddp_rank, ddp_world_size)
        if verbose and bag_items:
            print(
                f"DDP rank {ddp_rank}/{ddp_world_size}: "
                f"{len(bag_items)} bag(s) -> {[b[0] for b in bag_items]}"
            )

    for bag_name, entry in bag_items:
        result = process_grouped_bag(
            bag_name,
            entry,
            npz_root,
            out_dir,
            model,
            model_args,
            area_to_metric=area_to_metric,
            allowed_areas=allowed_areas,
            rollout_kwargs=rollout_kwargs,
            fps=fps,
            verbose=verbose,
        )
        if result is None:
            continue
        rollouts[bag_name] = result["rollout"]
        all_rows.extend(result["rows"])
        video_mp4s.extend(result["video_mp4s"])
        total_steps += int(result["rollout"].n_steps_run)
        if result["bag_timing"]:
            per_bag_timing.append(result["bag_timing"])
            if global_timers is not None:
                for stage, info in result["bag_timing"]["stages"].items():
                    global_timers.add(stage, float(info["total_s"]), int(info["calls"]))

    elapsed_sec = time.perf_counter() - t0

    if ddp_shard_mode:
        write_ddp_shard(
            out_dir,
            ddp_rank,
            segments=all_rows,
            video_mp4s=video_mp4s,
            rollout_bags=sorted(rollouts.keys()),
            elapsed_sec=elapsed_sec,
            total_steps=total_steps,
            per_bag_timing=per_bag_timing,
            profile=profile,
        )
        return {
            "mode": "grouped",
            "ddp_shard": True,
            "ddp_rank": ddp_rank,
            "ddp_world_size": ddp_world_size,
            "n_bags": len(rollouts),
            "n_episodes": len(all_rows),
            "elapsed_sec": elapsed_sec,
            "classification_json": str(Path(classification_json).resolve()),
            "npz_root": str(npz_root),
            "near_miss_thresh": near_miss_thresh,
        }

    timing = None
    if profile and global_timers is not None:
        timing = _build_timing_summary(global_timers, per_bag_timing, total_steps, elapsed_sec)
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
    return summary
