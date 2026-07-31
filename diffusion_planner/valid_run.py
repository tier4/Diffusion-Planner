#!/usr/bin/env python3
"""Run closed/open-loop L2 validation across all visible GPUs (Python replacement for valid_run.sh).

Thin launcher only: derives the checkpoint/config/output paths from --model_dir and runs
valid_predictor.py under torch.distributed.run. Standard validation renders the prediction PNGs
per-clip MP4s on rank 0 when --save_predictions_dir is set.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from diffusion_planner.utils.dist_init import dist_init_file_path
from run_utils import gpu_count, tee_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_dir", required=True, help="dir holding best_model.pth + args.json")
    p.add_argument(
        "--checkpoint",
        default="",
        help="checkpoint .pth to validate; defaults to <model_dir>/best_model.pth",
    )
    p.add_argument(
        "--output_dir",
        default="",
        help="validation output root; defaults to a timestamped directory under --model_dir",
    )
    p.add_argument("--valid_set_list", default="")
    p.add_argument("--scenario_based_open_loop_list", default="")
    p.add_argument("--scenario_based_open_loop_only", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.valid_set_list and not args.scenario_based_open_loop_only:
        raise ValueError(
            "--valid_set_list is required unless --scenario_based_open_loop_only is set"
        )
    if args.scenario_based_open_loop_only and not args.scenario_based_open_loop_list:
        raise ValueError("--scenario_based_open_loop_only requires Scenario-based Open-loop list")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")
    model_dir = Path(args.model_dir)
    model_path = Path(args.checkpoint) if args.checkpoint else model_dir / "best_model.pth"
    args_json_path = model_dir / "args.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Validation checkpoint not found: {model_path}")
    if not args_json_path.is_file():
        raise FileNotFoundError(f"Validation args.json not found: {args_json_path}")
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else model_dir / f"validation_result_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    save_dir = output_root / "predictions"

    here = Path(__file__).resolve().parent
    dist_init_file_path().unlink(missing_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        "1",
        "--nproc-per-node",
        str(gpu_count()),
        "--standalone",
        "valid_predictor.py",
        "--resume_model_path",
        str(model_path),
        "--args_json_path",
        str(args_json_path),
        "--batch_size",
        str(args.batch_size),
        "--save_predictions_dir",
        str(save_dir),
    ]
    if args.valid_set_list:
        cmd.extend(["--valid_set_list", str(Path(args.valid_set_list).resolve())])
    if args.scenario_based_open_loop_list:
        cmd.extend(
            [
                "--scenario_based_open_loop_list",
                str(Path(args.scenario_based_open_loop_list).resolve()),
            ]
        )
    if args.scenario_based_open_loop_only:
        cmd.append("--scenario_based_open_loop_only")
    rc = tee_run(cmd, cwd=here, log_path=output_root / "valid_log.txt")
    sys.exit(rc)


if __name__ == "__main__":
    main()
