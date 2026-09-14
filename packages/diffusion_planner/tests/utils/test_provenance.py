"""Tests for the run provenance recorded in every checkpoint."""

from __future__ import annotations

import subprocess
from pathlib import Path

from diffusion_planner.utils.provenance import (
    collect_provenance,
    dataset_fingerprint,
    find_repository,
    git_state,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _make_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "a.txt").write_text("one")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "first")
    return repo


def test_git_state_reports_commit_branch_and_clean_tree(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    state = git_state(repo)
    assert state["commit"] is not None
    assert len(state["commit"]) == 40
    assert state["branch"] is not None
    assert state["dirty"] is False


def test_git_state_flags_a_dirty_tree(tmp_path: Path) -> None:
    """A run started from a modified tree is not described by its commit alone."""
    repo = _make_repo(tmp_path)
    (repo / "a.txt").write_text("changed")
    assert git_state(repo)["dirty"] is True


def test_git_state_survives_a_non_repository(tmp_path: Path) -> None:
    """Collection must never stop a checkpoint from being written."""
    assert git_state(tmp_path / "nowhere")["repository"] is None


def test_find_repository_walks_up_to_the_git_directory(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    nested = repo / "a" / "b"
    nested.mkdir(parents=True)
    assert find_repository(nested) == repo


def test_dataset_fingerprint_records_size_for_an_existing_file(tmp_path: Path) -> None:
    dataset = tmp_path / "train.parquet"
    dataset.write_bytes(b"0123456789")
    fingerprint = dataset_fingerprint(dataset)
    assert fingerprint["exists"] is True
    assert fingerprint["size_bytes"] == 10
    assert fingerprint["modified"].endswith("+00:00")


def test_dataset_fingerprint_handles_a_missing_file(tmp_path: Path) -> None:
    assert dataset_fingerprint(tmp_path / "gone.parquet")["exists"] is False
    assert dataset_fingerprint(None)["path"] is None


def test_collect_provenance_keeps_the_run_config_verbatim(tmp_path: Path) -> None:
    """The augmentation settings must travel with the weights."""
    config = {
        "transform_configs": {
            "unknown_label_augmentation": {
                "probability": 0.1,
                "probability_pedestrian": 0.25,
                "max_renamed_fraction": 0.5,
            }
        }
    }
    provenance = collect_provenance(
        config, dataset_path=None, repo=_make_repo(tmp_path)
    )
    augmentation = provenance["run_config"]["transform_configs"][
        "unknown_label_augmentation"
    ]
    assert augmentation["probability"] == 0.1
    assert augmentation["probability_pedestrian"] == 0.25
    assert provenance["torch_version"]
    assert provenance["saved_at"].endswith("+00:00")


def test_collect_provenance_merges_extra_fields(tmp_path: Path) -> None:
    provenance = collect_provenance(
        None, repo=_make_repo(tmp_path), extra={"run_name": "r"}
    )
    assert provenance["run_name"] == "r"
