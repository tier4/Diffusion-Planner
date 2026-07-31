"""The DDP rendezvous FileStore path must be unique per job.

Two jobs sharing one FileStore path with the same ``world_size`` rendezvous into each other's
process group. On this cluster ``/tmp`` is shared between jobs on a node
(``JobContainerType=(null)``), so the previously hardcoded ``/tmp/tmp_dist_init`` was reachable
by every concurrent run.
"""

from __future__ import annotations

import pytest
from diffusion_planner.utils.ddp import dist_init_file


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("DP_DDP_INIT_FILE", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)


def test_distinct_slurm_jobs_get_distinct_files(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1783")
    first = dist_init_file()
    monkeypatch.setenv("SLURM_JOB_ID", "1793")
    second = dist_init_file()
    assert first != second, "two slurm jobs would rendezvous into the same process group"
    assert "1783" in first and "1793" in second


def test_env_override_wins_over_slurm_job_id(monkeypatch):
    """Non-slurm launchers (and tests) need to set this explicitly."""
    monkeypatch.setenv("SLURM_JOB_ID", "1783")
    monkeypatch.setenv("DP_DDP_INIT_FILE", "/tmp/explicit_path")
    assert dist_init_file() == "/tmp/explicit_path"


def test_falls_back_when_not_under_slurm():
    assert dist_init_file() == "/tmp/tmp_dist_init"


def test_empty_override_is_ignored(monkeypatch):
    """An unset-but-exported shell variable must not produce an empty rendezvous path."""
    monkeypatch.setenv("DP_DDP_INIT_FILE", "")
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    assert dist_init_file() == "/tmp/tmp_dist_init_42"
