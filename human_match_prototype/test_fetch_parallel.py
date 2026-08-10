"""Test path list splitting logic for parallel fetch."""

import tempfile
from pathlib import Path

from human_match_prototype.fetch_parallel import split_paths_for_fetch


def test_split_excludes_existing_files():
    """Already-present files should be excluded from fetch lists."""
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        # Create one "existing" file
        existing = dest / "mnt" / "data" / "a.npz"
        existing.parent.mkdir(parents=True)
        existing.touch()

        all_paths = ["/mnt/data/a.npz", "/mnt/data/b.npz", "/mnt/data/c.npz"]
        chunks = split_paths_for_fetch(all_paths, str(dest), num_chunks=2)

        # a.npz should be excluded
        all_in_chunks = [p for chunk in chunks for p in chunk]
        assert "/mnt/data/a.npz" not in all_in_chunks
        assert len(all_in_chunks) == 2
        assert len(chunks) == 2


def test_split_round_robins_evenly():
    """Paths should be distributed roughly evenly across chunks."""
    paths = [f"/data/{i}.npz" for i in range(10)]
    chunks = split_paths_for_fetch(paths, "/nonexistent", num_chunks=3)
    sizes = [len(c) for c in chunks]
    assert sum(sizes) == 10
    assert max(sizes) - min(sizes) <= 1
