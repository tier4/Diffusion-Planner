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

from run_utils import NCCL_ENV, gpu_count, tee_run


def boolean(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args(args_list: list[str] | None = None) -> argparse.Namespace:
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
        "--cluster_json",
        default=None,
        help="optional: path to cluster assignment JSON for weighted sampling",
    )
    p.add_argument(
        "--cluster_weight_alpha",
        type=float,
        default=None,
        help="optional: exponent on inverse-frequency cluster weights "
        "(1.0 = full inverse, 0.0 = uniform)",
    )
    p.add_argument(
        "--force_aug_on_repeat",
        type=boolean,
        default=None,
        help="optional: force one augmentation on repeated weighted-sampler draws",
    )
    p.add_argument(
        "--repeat_aug_pool",
        default=None,
        help="optional: comma-separated augmentations eligible for repeat-draw forcing",
    )
    p.add_argument("--use_flip_augment", type=boolean, default=None)
    p.add_argument("--use_neighbor_dropout", type=boolean, default=None)
    p.add_argument("--use_neighbor_noise", type=boolean, default=None)
    p.add_argument("--use_turn_indicator_dropout", type=boolean, default=None)
    return p.parse_args(args_list)


def build_optional_args(args: argparse.Namespace) -> list[str]:
    optional: list[str] = []
    if args.resume_model_path:
        optional += ["--resume_model_path", str(Path(args.resume_model_path).resolve())]
    if args.wandb_run_id:
        optional += ["--wandb_run_id", args.wandb_run_id]
    if args.wandb_project_name:
        optional += ["--wandb_project_name", args.wandb_project_name]
    if args.cluster_json:
        # ``Path.resolve()`` is non-strict, so a typo would sail through and only
        # surface after every rank has loaded the (large) train/valid path lists
        # and built the model. Fail before launching torchrun.
        cluster_json = Path(args.cluster_json).resolve()
        if not cluster_json.is_file():
            sys.exit(f"--cluster_json not found: {cluster_json}")
        optional += ["--cluster_json", str(cluster_json)]
    if args.cluster_weight_alpha is not None:
        if not args.cluster_json:
            print(
                f"WARNING: --cluster_weight_alpha {args.cluster_weight_alpha} has no effect "
                "without --cluster_json; training will use the default DistributedSampler.",
                file=sys.stderr,
            )
        optional += ["--cluster_weight_alpha", str(args.cluster_weight_alpha)]
    if args.force_aug_on_repeat is not None:
        optional += ["--force_aug_on_repeat", str(args.force_aug_on_repeat)]
    if args.repeat_aug_pool is not None:
        optional += ["--repeat_aug_pool", args.repeat_aug_pool]
    for name in (
        "use_flip_augment",
        "use_neighbor_dropout",
        "use_neighbor_noise",
        "use_turn_indicator_dropout",
    ):
        value = getattr(args, name, None)
        if value is not None:
            optional += [f"--{name}", str(value)]
    return optional


def main() -> None:
    args = parse_args()

    here = Path(__file__).resolve().parent
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

    optional = build_optional_args(args)

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
        *optional,
    ]
    rc = tee_run(
        cmd, cwd=here, env={**os.environ, **NCCL_ENV}, log_path=save_path / "train_log.txt"
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
