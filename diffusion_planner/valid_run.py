#!/usr/bin/env python3
"""Run closed/open-loop L2 validation across all visible GPUs (Python replacement for valid_run.sh).

Thin launcher only: derives the checkpoint/config/output paths from --model_dir and runs
valid_predictor.py under torch.distributed.run. valid_predictor.py itself is unchanged (it renders
the prediction PNGs + per-clip MP4s on rank 0 when --save_predictions_dir is set).
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from run_utils import gpu_count, tee_run


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
    rc = tee_run(cmd, cwd=here, log_path=save_dir.parent / "valid_log.txt")
    sys.exit(rc)


if __name__ == "__main__":
    main()
