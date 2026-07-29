#!/usr/bin/env python3
"""Launch pretrain/SFT training across all visible GPUs (Python replacement for train_run.sh).

Thin launcher only: it resolves the run dir, saves git info, sets NCCL env and runs
train_predictor.py under torch.distributed.run. train_predictor.py itself is unchanged.

--closed_loop_npz_root (optional) is forwarded to train_predictor.py's flag of the same name.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from diffusion_planner.scenario_based_open_loop.open_loop import (
    load_scenario_based_open_loop_settings,
)
from run_utils import NCCL_ENV, gpu_count, tee_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp_name", required=True)
    p.add_argument("--train_set_list", required=True)
    p.add_argument("--valid_set_list", required=True)
    p.add_argument("--output_root", default="/mnt/nvme/training_result")
    p.add_argument("--resume_model_path", default=None, help="optional: resume from this .pth")
    p.add_argument("--wandb_run_id", default=None, help="optional: existing wandb run id")
    p.add_argument("--wandb_project_name", default=None, help="optional: wandb project name")
    p.add_argument(
        "--closed_loop_npz_root",
        default="",
        help="optional: dir tree of route NPZ frames for closed-loop validation, OR a .json path "
        "list of such dirs (like --train_set_list). Empty = disabled.",
    )
    p.add_argument(
        "--scenario_based_open_loop_list",
        default="",
        help="optional JSON mapping Scenario-based Open-loop metric names to NPZ path lists. Empty = disabled.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    here = Path(__file__).resolve().parent
    if args.scenario_based_open_loop_list:
        load_scenario_based_open_loop_settings(args.scenario_based_open_loop_list)

    save_path = Path(args.output_root) / f"{datetime.now():%Y%m%d-%H%M%S}_{args.exp_name}"
    save_path.mkdir(parents=True, exist_ok=True)

    # Save git info next to the run.
    def git_output(cmd: list[str]) -> str:
        return subprocess.run(cmd, cwd=here, capture_output=True, text=True).stdout

    branch = git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    (save_path / "git_show.txt").write_text(
        f"branch: {branch}\n\n" + git_output(["git", "show", "-s", "--decorate"])
    )
    (save_path / "git_diff.txt").write_text(git_output(["git", "diff"]))

    optional: list[str] = []
    if args.resume_model_path:
        optional += ["--resume_model_path", str(Path(args.resume_model_path).resolve())]
    if args.wandb_run_id:
        optional += ["--wandb_run_id", args.wandb_run_id]
    if args.wandb_project_name:
        optional += ["--wandb_project_name", args.wandb_project_name]

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
        "train_predictor.py",
        "--exp_name",
        args.exp_name,
        "--train_set_list",
        str(Path(args.train_set_list).resolve()),
        "--valid_set_list",
        str(Path(args.valid_set_list).resolve()),
        "--use_wandb",
        "True",
        "--diffusion_model_type",
        "x_start",
        "--save_dir",
        str(save_path),
        "--train_epochs",
        "80",
        "--save_utd",
        "10",
        "--closed_loop_npz_root",
        str(Path(args.closed_loop_npz_root).resolve()) if args.closed_loop_npz_root else "",
        "--scenario_based_open_loop_list",
        str(Path(args.scenario_based_open_loop_list).resolve())
        if args.scenario_based_open_loop_list
        else "",
        *optional,
    ]
    rc = tee_run(
        cmd, cwd=here, env={**os.environ, **NCCL_ENV}, log_path=save_path / "train_log.txt"
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
