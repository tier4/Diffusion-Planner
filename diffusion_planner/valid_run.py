#!/usr/bin/env python3
"""Run closed/open-loop L2 validation across all visible GPUs (Python replacement for valid_run.sh).

Thin launcher only: derives the checkpoint/config/output paths from --model_dir and runs
valid_predictor.py under torch.distributed.run. valid_predictor.py itself is unchanged (it renders
the prediction PNGs + per-clip MP4s on rank 0 when --save_predictions_dir is set).
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def gpu_count() -> int:
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return len([ln for ln in out.splitlines() if ln.strip()])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_dir", required=True, help="dir holding best_model.pth + args.json")
    p.add_argument("--valid_set_list", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    valid_set_list = args.valid_set_list
    model_path = model_dir / "best_model.pth"
    args_json_path = model_dir / "args.json"
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
        "--save_predictions_dir",
        str(save_dir),
    ]
    sys.exit(subprocess.run(cmd, cwd=here).returncode)


if __name__ == "__main__":
    main()
