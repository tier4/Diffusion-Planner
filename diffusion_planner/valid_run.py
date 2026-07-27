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

from run_utils import gpu_count, tee_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_dir", required=True, help="dir holding best_model.pth + args.json")
    p.add_argument(
        "--checkpoint",
        default="",
        help="checkpoint .pth to validate; defaults to <model_dir>/best_model.pth",
    )
    p.add_argument("--valid_set_list", default="")
    p.add_argument("--override_open_loop_list", default="")
    p.add_argument("--override_open_loop_config", default="")
    p.add_argument("--override_only", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.valid_set_list and not args.override_only:
        raise ValueError("--valid_set_list is required unless --override_only is set")
    if bool(args.override_open_loop_list) != bool(args.override_open_loop_config):
        raise ValueError(
            "--override_open_loop_list and --override_open_loop_config must be supplied together"
        )
    if args.override_only and not args.override_open_loop_list:
        raise ValueError("--override_only requires Override Open-loop list and config")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")

    model_dir = Path(args.model_dir)
    valid_set_list = args.valid_set_list
    model_path = Path(args.checkpoint) if args.checkpoint else model_dir / "best_model.pth"
    args_json_path = model_dir / "args.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Validation checkpoint not found: {model_path}")
    if not args_json_path.is_file():
        raise FileNotFoundError(f"Validation args.json not found: {args_json_path}")
    save_dir = model_dir / f"validation_result_{datetime.now():%Y%m%d_%H%M%S}" / "predictions"

    here = Path(__file__).resolve().parent
    Path("/tmp/tmp_dist_init").unlink(missing_ok=True)

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
        "--valid_set_list",
        valid_set_list,
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
        cmd.extend(["--valid_set_list", args.valid_set_list])
    if args.override_open_loop_list:
        cmd.extend(
            [
                "--override_open_loop_list",
                args.override_open_loop_list,
                "--override_open_loop_config",
                args.override_open_loop_config,
            ]
        )
    if args.override_only:
        cmd.append("--override_only")
    rc = tee_run(cmd, cwd=here, log_path=save_dir.parent / "valid_log.txt")
    sys.exit(rc)


if __name__ == "__main__":
    main()
