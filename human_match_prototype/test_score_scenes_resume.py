"""Test --resume flag for score_scenes."""

import csv
import tempfile
from pathlib import Path


def _make_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_resume_skips_already_scored():
    """Resume mode should skip paths that already have rows in the CSV."""
    from human_match_prototype.score_scenes import _load_done_paths

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "scores.csv"
        _make_csv(
            csv_path,
            [
                {"npz_path": "/a.npz", "es_2s": 1.0},
                {"npz_path": "/b.npz", "es_2s": 2.0},
            ],
        )
        done = _load_done_paths(csv_path)
        assert done == {"/a.npz", "/b.npz"}


def test_resume_returns_empty_for_missing_csv():
    """If CSV doesn't exist, resume should return empty set."""
    from human_match_prototype.score_scenes import _load_done_paths

    done = _load_done_paths(Path("/nonexistent/scores.csv"))
    assert done == set()
