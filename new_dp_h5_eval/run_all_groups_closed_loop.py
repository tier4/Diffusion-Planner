"""Run native-H5 closed-loop evaluation with the normal multi-input mode semantics.

This is the new-DP/ONNX counterpart of ``diffusion_planner/run_all_groups_closed_loop.py``.
It intentionally uses ``--closed_loop_h5_root`` rather than overloading the old NPZ option.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
from diffusion_planner.config.closed_loop_config import ClosedLoopConfig
from diffusion_planner.config.config_utils import save_config
from diffusion_planner.utils import ddp

from scenario_generation.closed_loop_ddp import shard_items
from scenario_generation.closed_loop_evaluation import ClosedLoopEvalConfig

from .closed_loop import (
    NativeH5FullRouteClosedLoopEvaluation,
    ReproducerOnnxModel,
    model_args,
    rollout_params_from_closed_loop_config,
)


def _write_groups_manifest(out_dir: Path, summaries: dict[str, dict]) -> None:
    """Write the same root/per-input aggregate fields as the standard runner."""
    n_segments = sum(int(summary.get("n_segments", 0) or 0) for summary in summaries.values())
    total_steps = sum(int(summary.get("total_steps", 0) or 0) for summary in summaries.values())
    route_completion = sum(
        float(summary.get("mean_route_completion", 0.0) or 0.0)
        * int(summary.get("n_segments", 0) or 0)
        for summary in summaries.values()
    )
    dev_num = dev_steps = 0
    for summary in summaries.values():
        value = summary.get("mean_gt_deviation_m")
        steps = int(summary.get("total_steps", 0) or 0)
        if value is not None and math.isfinite(float(value)) and steps:
            dev_num += float(value) * steps
            dev_steps += steps
    payload = {
        "n_groups": len(summaries),
        "n_segments": n_segments,
        "total_steps": total_steps,
        "mean_route_completion": route_completion / n_segments if n_segments else 0.0,
        "mean_gt_deviation_m": dev_num / dev_steps if dev_steps else float("inf"),
        "total_curb_hits": sum(
            int(s.get("road_border", {}).get("collision_count", 0) or 0) for s in summaries.values()
        ),
        "total_snaps": sum(
            int(s.get("reproducer", {}).get("snap_count", 0) or 0) for s in summaries.values()
        ),
        "total_red_light_violations": sum(
            int(s.get("red_light_violation", {}).get("count", 0) or 0) for s in summaries.values()
        ),
        "total_strong_brakes": sum(
            int(s.get("strong_brake", {}).get("count", 0) or 0) for s in summaries.values()
        ),
        "n_segments_diverged": sum(
            int(s.get("n_segments_diverged", 0) or 0) for s in summaries.values()
        ),
        "n_pass_segments": sum(int(s.get("pass_count", 0) or 0) for s in summaries.values()),
        "n_fail_segments": sum(int(s.get("fail_count", 0) or 0) for s in summaries.values()),
    }
    payload["pass_rate"] = payload["n_pass_segments"] / n_segments if n_segments else 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "groups.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_groups(manifest: Path) -> dict[str, list[dict]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest} must map group names to H5 route entries")
    groups: dict[str, list[dict]] = {}
    for group_name, entries in payload.items():
        if not isinstance(entries, list):
            raise ValueError(f"{manifest}: group {group_name!r} must be a list")
        routes = []
        for entry in entries:
            if isinstance(entry, str):
                route = {"h5_path": entry}
            elif isinstance(entry, dict) and set(entry).issubset(
                {"h5_path", "frame_start", "frame_stop"}
            ):
                route = dict(entry)
            else:
                raise ValueError(
                    f"{manifest}: {group_name!r} entries must be an H5 path string or "
                    "an object containing h5_path/frame_start/frame_stop"
                )
            if "h5_path" not in route:
                raise ValueError(f"{manifest}: {group_name!r} route object is missing h5_path")
            h5_path = Path(route["h5_path"])
            route["h5_path"] = str(h5_path if h5_path.is_absolute() else manifest.parent / h5_path)
            route["route_id"] = f"{group_name}__{Path(route['h5_path']).parent.name}"
            routes.append(route)
        groups[str(group_name)] = routes
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closed_loop_h5_root", type=Path, nargs="+", required=True)
    parser.add_argument("--closed_loop_object_modes", nargs="+", default=None)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--provider", action="append", dest="providers")
    parser.add_argument("--closed_loop_pass_conditions", default="")
    parser.add_argument("--closed_loop_draw_workers", type=int, default=4)
    args = parser.parse_args()

    modes = args.closed_loop_object_modes or ["objects"] * len(args.closed_loop_h5_root)
    if len(modes) != len(args.closed_loop_h5_root) or set(modes).difference({"objects", "noobj"}):
        parser.error(
            "--closed_loop_object_modes must provide one 'objects'/'noobj' mode per H5 manifest"
        )

    cfg = ClosedLoopConfig(
        device=args.device, closed_loop_pass_conditions=args.closed_loop_pass_conditions
    )
    cfg.closed_loop_draw_workers = args.closed_loop_draw_workers
    rank, local_rank, world_size = ddp.ddp_setup_universal(True, cfg)
    if cfg.device.startswith("cuda"):
        import torch

        torch.cuda.set_device(local_rank)
        cfg.device = f"cuda:{local_rank}"

    out_root = args.out_root / datetime.now().strftime("%Y%m%d_%H%M")
    out_root.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        save_config(cfg, out_root, "closed_loop_config.json")

    model = ReproducerOnnxModel(args.model_path, args.providers)
    evaluators: dict[str, NativeH5FullRouteClosedLoopEvaluation] = {}
    for manifest, mode in zip(args.closed_loop_h5_root, modes):
        label = manifest.stem if mode == "objects" else f"{manifest.stem}__noobj"
        for group_name, routes in _load_groups(manifest).items():
            group_out = out_root / label / group_name
            key = f"{label}/{group_name}"
            evaluators[key] = NativeH5FullRouteClosedLoopEvaluation(
                model,
                model_args(),
                config=ClosedLoopEvalConfig(
                    out_dir=group_out,
                    params=rollout_params_from_closed_loop_config(
                        cfg,
                        drop_objects=(mode == "noobj"),
                        draw_every=cfg.closed_loop_draw_every,
                    ),
                    fps=float(cfg.closed_loop_fps),
                    verbose=False,
                    profile=False,
                    pass_condition=cfg.pass_conditions.get_condition(group_name),
                ),
                routes=routes,
                seg_len=cfg.closed_loop_seg_len,
                ddp_rank=rank,
                ddp_world_size=world_size,
            )
    # Keep native-H5 evaluation on the same contract as the regular runner:
    # distribute all groups together, let every evaluator persist rank shards,
    # then have rank 0 merge them after a single barrier.
    assignments = {key: [] for key in evaluators}
    all_jobs = [
        (key, job) for key, evaluator in evaluators.items() for job in evaluator.discover_jobs()
    ]
    for key, job in shard_items(all_jobs, rank, world_size):
        assignments[key].append(job)

    started = time.perf_counter()
    partials = {key: evaluator.run(assignments[key]) for key, evaluator in evaluators.items()}
    if world_size > 1:
        torch.distributed.barrier()
        if rank == 0:
            elapsed_sec = time.perf_counter() - started
            for key, evaluator in evaluators.items():
                if partials[key].get("ddp_shard"):
                    partials[key] = evaluator.merge_ddp_shards(world_size, elapsed_sec=elapsed_sec)

    if rank == 0:
        summaries = partials
        _write_groups_manifest(out_root, summaries)
        for label in {key.split("/", 1)[0] for key in summaries}:
            scoped = {key: value for key, value in summaries.items() if key.startswith(f"{label}/")}
            _write_groups_manifest(out_root / label, scoped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
