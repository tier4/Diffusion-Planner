#!/usr/bin/env python3
"""Profile grouped closed-loop eval without training or wandb.

Usage (from repo root)::

    python -m scenario_generation.tools.profile_grouped_closed_loop \\
        --model_path best_model.pth \\
        --npz_root /path/to/valid/2026-01-15 \\
        --classification_json_root ../Diffusion-Planner-Meta-Repository/dataset/scenario_classification_json \\
        --out_dir /tmp/cl_profile \\
        --max_bags 1

Full 4-bag run with profiling::

    python -m scenario_generation.tools.profile_grouped_closed_loop \\
        --model_path best_model.pth \\
        --npz_root ... --classification_json_root ... --out_dir /tmp/cl_profile

Multi-GPU (server)::

    torchrun --nproc_per_node=4 -m scenario_generation.tools.profile_grouped_closed_loop \\
        --model_path best_model.pth \\
        --npz_root ... --classification_json_root ... --out_dir /tmp/cl_profile
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scenario_generation.closed_loop_cli import (  # noqa: E402
    add_grouped_args,
    add_output_args,
    add_rollout_args,
)
from scenario_generation.grouped_closed_loop_ddp import (  # noqa: E402
    ddp_device_for_rank,
    merge_ddp_grouped_shards,
)
from scenario_generation.grouped_closed_loop_eval import run_grouped_closed_loop_eval  # noqa: E402
from scenario_generation.scenario_classification import (  # noqa: E402
    discover_classification_eval_jobs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_rollout_args(p)
    add_output_args(p)
    add_grouped_args(p)
    p.add_argument(
        "--scenario_dataset_name",
        type=str,
        default="",
        help="dataset key when NPZ path cannot be parsed",
    )
    p.add_argument(
        "--max_bags",
        type=int,
        default=None,
        help="limit number of bags (quick smoke profile)",
    )
    p.add_argument(
        "--profile_sync_gpu",
        action="store_true",
        help="torch.cuda.synchronize() around model_forward for accurate GPU timing",
    )
    return p.parse_args()


def _init_ddp_if_needed() -> tuple[int, int]:
    """Initialize torch.distributed when launched via ``torchrun``."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1
    import torch.distributed as dist

    if dist.is_initialized():
        return int(os.environ.get("RANK", "0")), world_size

    from diffusion_planner.utils import ddp

    class _DdpArgs:
        ddp = True
        port = os.environ.get("MASTER_PORT", "29529")

    ddp.ddp_setup_universal(verbose=False, args=_DdpArgs())
    return ddp.get_rank(), ddp.get_world_size()


def main() -> int:
    args = parse_args()
    if args.out_dir is None:
        raise SystemExit("--out_dir is required")

    rank, world_size = _init_ddp_if_needed()
    ddp_active = world_size > 1
    cl_device = ddp_device_for_rank(args.device) if ddp_active else args.device

    cl_root = args.classification_json_root
    cl_explicit = args.classification_json
    if cl_explicit is not None and cl_explicit.is_dir():
        cl_root = cl_root or cl_explicit
        cl_explicit = None

    jobs = discover_classification_eval_jobs(
        args.npz_root,
        classification_root=cl_root,
        explicit_path=cl_explicit,
        dataset_name=args.scenario_dataset_name or None,
    )
    if not jobs:
        raise SystemExit("no classification JSON matched npz_root; set --classification_json_root")

    from scenario_generation.simulate import load_model

    model, model_args = load_model(args.model_path, cl_device)

    for job in jobs:
        job_out = args.out_dir if len(jobs) == 1 else args.out_dir / job.date
        if rank == 0:
            print(
                f"Profile job: {job.dataset_key} {job.date} "
                f"npz={job.npz_root} json={job.classification_json}"
                + (f" (DDP rank {rank}/{world_size})" if ddp_active else "")
            )
        summary = run_grouped_closed_loop_eval(
            model,
            model_args,
            job.npz_root,
            job.classification_json,
            job_out,
            device=cl_device,
            near_miss_thresh=args.near_miss_thresh,
            search_radius=args.search_radius,
            warmup_steps=args.warmup_steps,
            unstick_after=args.unstick_after,
            unstick_advance_m=args.unstick_advance_m,
            draw_every=args.draw_every,
            fps=float(args.fps),
            replan_interval=args.replan_interval,
            areas=args.areas,
            verbose=rank == 0,
            profile=True,
            profile_sync_gpu=args.profile_sync_gpu,
            max_bags=args.max_bags,
            ddp_rank=rank,
            ddp_world_size=world_size if ddp_active else 1,
        )

        if ddp_active:
            import torch.distributed as dist

            dist.barrier()
            if rank == 0:
                summary = merge_ddp_grouped_shards(
                    job_out,
                    world_size,
                    classification_json=job.classification_json,
                    npz_root=job.npz_root,
                    near_miss_thresh=args.near_miss_thresh,
                    profile=True,
                )

        if rank == 0:
            timing_path = Path(job_out) / "timing.json"
            print(
                f"\n=== profile done: {summary.get('n_bags', 0)} bags, "
                f"{summary.get('n_episodes', 0)} episodes in {summary.get('elapsed_sec', 0):.1f}s ==="
            )
            if timing_path.is_file():
                print(f"timing written -> {timing_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
