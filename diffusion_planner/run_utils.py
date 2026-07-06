"""Shared helpers for the train_run.py / grpo_run.py / valid_run.py launchers (thin Python
replacements for the old *_run.sh wrappers). Not used by the predictor scripts themselves."""

import subprocess
import sys
from pathlib import Path

# NCCL settings the training wrappers used to export before launching torch.distributed.run.
NCCL_ENV = {
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_P2P_DISABLE": "0",
    "NCCL_IB_DISABLE": "1",
    "NCCL_SOCKET_IFNAME": "lo",
    "NCCL_DEBUG": "INFO",
}


def gpu_count() -> int:
    """Number of visible GPUs (was ``nvidia-smi -L | wc -l``)."""
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return len([ln for ln in out.splitlines() if ln.strip()])


def tee_run(cmd: list[str], cwd: Path, log_path: Path, env: dict | None = None) -> int:
    """Run cmd, streaming combined stdout/stderr to the console AND log_path (like ``2>&1 | tee``).

    Returns the child's exit code. ``env=None`` inherits the current environment.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
