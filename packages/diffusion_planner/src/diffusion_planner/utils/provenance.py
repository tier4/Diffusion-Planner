"""Provenance recorded next to the weights in every checkpoint.

A checkpoint that cannot say how it was produced cannot be reproduced. Checkpoints used to
store only ``model_config``, which holds the architecture and nothing else, so the data
augmentation a run used had to be recovered from shell history afterwards. Everything here is
cheap to collect and is written once per checkpoint.

Collection never raises. A checkpoint must still be written if git is missing, the source is
not a repository, or the dataset path has gone away, so every probe returns ``None`` on
failure instead of propagating.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _run_git(repo: Path, *args: str) -> str | None:
    """Return stripped git output, or None if git cannot answer."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def find_repository(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` looking for a directory that contains ``.git``."""
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def git_state(repo: Path | None = None) -> dict[str, Any]:
    """Commit, branch and whether the working tree had uncommitted changes.

    ``dirty`` matters as much as the SHA: a run started from a modified tree is not described
    by its commit alone, and that needs to be visible in the checkpoint rather than inferred
    later.
    """
    # An explicitly given path goes through find_repository too, so that passing a directory
    # that is not a repository behaves the same as failing to detect one. Otherwise the two
    # paths disagree: the caller-supplied one would be reported as the repository while every
    # git probe under it returned None.
    repository = find_repository(repo)
    if repository is None:
        return {"repository": None}
    status = _run_git(repository, "status", "--porcelain")
    return {
        "repository": str(repository),
        "commit": _run_git(repository, "rev-parse", "HEAD"),
        "branch": _run_git(repository, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def dataset_fingerprint(path: str | Path | None) -> dict[str, Any]:
    """Identify the dataset index by path, size and modification time.

    Not a content hash: the index can be large and this runs on every checkpoint. Size and
    mtime are enough to notice that two runs read different data.
    """
    if path is None:
        return {"path": None}
    dataset_path = Path(str(path)).expanduser()
    try:
        stat = dataset_path.stat()
    except OSError:
        return {"path": str(dataset_path), "exists": False}
    return {
        "path": str(dataset_path),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def collect_provenance(
    run_config: Any = None,
    *,
    dataset_path: str | Path | None = None,
    repo: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble everything needed to reconstruct a run.

    ``run_config`` should be the fully resolved training config, already converted to plain
    containers. It is stored verbatim, so the augmentation settings that produced a model
    travel with it.
    """
    provenance: dict[str, Any] = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "git": git_state(repo),
        "dataset": dataset_fingerprint(dataset_path),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "run_config": run_config,
    }
    if extra:
        provenance.update(extra)
    return provenance
