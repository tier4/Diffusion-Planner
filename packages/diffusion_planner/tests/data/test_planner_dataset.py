"""Focused tests for the H5-backed planner dataset."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from diffusion_planner.data import PlannerDataset


def write_shard(path: Path) -> None:
    """Write two minimal frames using the production H5 schema."""
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as file:
        file.attrs["format"] = "diffusion_planner_frame_dataset"
        file.attrs["format_version"] = 4
        file.attrs["num_frames"] = 2
        frames = file.create_group("frames")
        frames.create_dataset(
            "ego_agent_past", data=np.arange(24, dtype=np.float32).reshape(2, 2, 6)
        )
        metadata = file.create_group("metadata")
        metadata.create_dataset(
            "frame_time_ns", data=np.array([10, 20], dtype=np.int64)
        )


def write_index(path: Path, h5_path: Path) -> None:
    """Write the two rows addressing the test shard."""
    pq.write_table(
        pa.table(
            {
                "h5_path": [str(h5_path)] * 2,
                "frame_index": np.array([0, 1], dtype=np.int64),
                "frame_time_ns": np.array([10, 20], dtype=np.int64),
            }
        ),
        path,
    )


class PlannerDatasetTest(unittest.TestCase):
    """H5 frames are addressed by the lightweight Parquet index."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.index_path = self.root / "index.parquet"
        self.h5_path = self.root / "project/bag/frames.h5"
        write_shard(self.h5_path)
        write_index(self.index_path, Path("project/bag/frames.h5"))

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_loads_every_index_row(self) -> None:
        dataset = PlannerDataset(self.index_path)
        self.assertEqual(len(dataset), 2)
        np.testing.assert_array_equal(
            dataset[1]["ego_agent_past"].numpy(),
            np.arange(24, dtype=np.float32).reshape(2, 2, 6)[1],
        )

    def test_reports_the_h5_source(self) -> None:
        dataset = PlannerDataset(self.index_path)
        path, frame_time_ns = dataset.source(1)
        self.assertEqual(Path(path), self.h5_path)
        self.assertEqual(frame_time_ns, 20)

    def test_rejects_a_missing_index_column(self) -> None:
        bare_path = self.root / "bare.parquet"
        pq.write_table(pa.table({"h5_path": ["project/bag/frames.h5"]}), bare_path)
        with self.assertRaises(ValueError):
            PlannerDataset(bare_path)

    def test_resolves_h5_after_dataset_is_moved(self) -> None:
        moved_root = self.root.parent / f"{self.root.name}-moved"
        shutil.copytree(self.root, moved_root)
        self.addCleanup(shutil.rmtree, moved_root)
        dataset = PlannerDataset(moved_root / "index.parquet")
        path, _ = dataset.source(0)
        self.assertEqual(Path(path), moved_root / "project/bag/frames.h5")


if __name__ == "__main__":
    unittest.main()
