#!/usr/bin/env python3
"""Launch GRPO fine-tuning across all visible GPUs (Python replacement for grpo_run.sh).

Thin launcher only: resolves the run dir (``_grpo`` suffixed), saves git info, sets NCCL env and
runs train_grpo_predictor.py under torch.distributed.run. The trainer itself is unchanged.

--closed_loop_npz_root (optional) is forwarded to train_grpo_predictor.py's flag of the same name.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from run_utils import NCCL_ENV, gpu_count, tee_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--resume_model_path", required=True, help="pretrained/SFT checkpoint to start from"
    )
    p.add_argument("--exp_name", required=True, help="a '_grpo' suffix is appended automatically")
    p.add_argument("--train_set_list", required=True)
    p.add_argument("--valid_set_list", required=True)
    p.add_argument(
        "--closed_loop_npz_root",
        default="",
        help="optional: dir tree of route NPZ frames for closed-loop validation, OR a .json path "
        "list of such dirs (like --train_set_list). Empty = disabled.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    resume_model_path = str(Path(args.resume_model_path).resolve())
    exp_name = f"{args.exp_name}_grpo"
    train_set_list = str(Path(args.train_set_list).resolve())
    valid_set_list = str(Path(args.valid_set_list).resolve())
    closed_loop_npz_root = (
        str(Path(args.closed_loop_npz_root).resolve()) if args.closed_loop_npz_root else ""
    )

    here = Path(__file__).resolve().parent
    save_path = Path("/mnt/nvme/training_result") / f"{datetime.now():%Y%m%d-%H%M%S}_{exp_name}"
    save_path.mkdir(parents=True, exist_ok=True)

    def git_output(cmd: list[str]) -> str:
        return subprocess.run(cmd, cwd=here, capture_output=True, text=True).stdout

    branch = git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    (save_path / "git_show.txt").write_text(
        f"branch: {branch}\n\n" + git_output(["git", "show", "-s", "--decorate"])
    )
    (save_path / "git_diff.txt").write_text(git_output(["git", "diff"]))

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
        "train_grpo_predictor.py",
        "--exp_name",
        exp_name,
        "--train_set_list",
        train_set_list,
        "--valid_set_list",
        valid_set_list,
        "--resume_model_path",
        resume_model_path,
        "--save_dir",
        str(save_path),
        "--closed_loop_npz_root",
        closed_loop_npz_root,
    ]
    rc = tee_run(cmd, cwd=here, env={**os.environ, **NCCL_ENV}, log_path=save_path / "grpo_log.txt")
    sys.exit(rc)


if __name__ == "__main__":
    main()
