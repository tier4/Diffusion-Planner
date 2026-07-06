#!/usr/bin/env python3
"""Launch GRPO fine-tuning across all visible GPUs (Python replacement for grpo_run.sh).

Thin launcher only: resolves the run dir (``_grpo`` suffixed), saves git info, sets NCCL env and
runs train_grpo_predictor.py under torch.distributed.run. The trainer itself is unchanged.

Env: CLOSED_LOOP_NPZ_ROOT (optional) is forwarded to --closed_loop_npz_root, as before.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

NCCL_ENV = {
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_P2P_DISABLE": "0",
    "NCCL_IB_DISABLE": "1",
    "NCCL_SOCKET_IFNAME": "lo",
    "NCCL_DEBUG": "INFO",
}


def gpu_count() -> int:
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return len([ln for ln in out.splitlines() if ln.strip()])


def tee_run(cmd: list[str], cwd: Path, env: dict, log_path: Path) -> int:
    """Run cmd streaming combined stdout/stderr to the console AND log_path (like `2>&1 | tee`)."""
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        for line in proc.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
            log.write(line)
            log.flush()
        return proc.wait()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--resume_model_path", required=True, help="pretrained/SFT checkpoint to start from"
    )
    p.add_argument("--exp_name", required=True, help="a '_grpo' suffix is appended automatically")
    p.add_argument("--train_set_list", required=True)
    p.add_argument("--valid_set_list", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    resume_model_path = str(Path(args.resume_model_path).resolve())
    exp_name = f"{args.exp_name}_grpo"
    train_set_list = str(Path(args.train_set_list).resolve())
    valid_set_list = str(Path(args.valid_set_list).resolve())

    here = Path(__file__).resolve().parent
    save_path = Path("/mnt/nvme/training_result") / f"{datetime.now():%Y%m%d-%H%M%S}_{exp_name}"
    save_path.mkdir(parents=True, exist_ok=True)

    for name, cmd in (("git_show.txt", ["git", "show", "-s"]), ("git_diff.txt", ["git", "diff"])):
        (save_path / name).write_text(
            subprocess.run(cmd, cwd=here, capture_output=True, text=True).stdout
        )

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
        os.environ.get("CLOSED_LOOP_NPZ_ROOT", ""),
    ]
    rc = tee_run(cmd, cwd=here, env={**os.environ, **NCCL_ENV}, log_path=save_path / "grpo_log.txt")
    sys.exit(rc)


if __name__ == "__main__":
    main()
