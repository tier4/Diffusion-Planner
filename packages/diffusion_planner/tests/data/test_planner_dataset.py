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


def write_shard(path: Path, offset: int = 0) -> None:
    """Write two minimal frames using the production H5 schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        file.attrs["format"] = "diffusion_planner_frame_dataset"
        file.attrs["format_version"] = 4
        file.attrs["num_frames"] = 2
        frames = file.create_group("frames")
        frames.create_dataset(
            "ego_agent_past",
            data=(offset + np.arange(24, dtype=np.float32)).reshape(2, 2, 6),
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


class PlannerDatasetMultiShardTest(unittest.TestCase):
    """Rows addressing several shards resolve to the right file and frame."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.index_path = self.root / "index.parquet"
        self.offsets = (0, 100)
        for shard, offset in enumerate(self.offsets):
            write_shard(self.root / f"project/bag{shard}/frames.h5", offset=offset)
        # Interleave the shards so consecutive rows alternate between files.
        pq.write_table(
            pa.table(
                {
                    "h5_path": [
                        "project/bag1/frames.h5",
                        "project/bag0/frames.h5",
                        "project/bag1/frames.h5",
                        "project/bag0/frames.h5",
                    ],
                    "frame_index": np.array([0, 1, 1, 0], dtype=np.int64),
                    "frame_time_ns": np.array([10, 20, 30, 40], dtype=np.int64),
                }
            ),
            self.index_path,
        )

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _expected(self, offset: int, frame_index: int) -> np.ndarray:
        return (offset + np.arange(24, dtype=np.float32)).reshape(2, 2, 6)[frame_index]

    def test_addresses_each_shard_and_frame(self) -> None:
        dataset = PlannerDataset(self.index_path)
        self.assertEqual(len(dataset._shard_paths), 2)
        for row, (shard, frame_index) in enumerate(((1, 0), (0, 1), (1, 1), (0, 0))):
            np.testing.assert_array_equal(
                dataset[row]["ego_agent_past"].numpy(),
                self._expected(self.offsets[shard], frame_index),
            )
            self.assertEqual(
                Path(dataset.source(row)[0]),
                self.root / f"project/bag{shard}/frames.h5",
            )

    def test_reopens_shards_evicted_by_the_file_cache(self) -> None:
        dataset = PlannerDataset(self.index_path, file_capacity=1)
        for row in range(4):
            dataset[row]
            self.assertEqual(len(dataset._shards), 1)
        np.testing.assert_array_equal(
            dataset[0]["ego_agent_past"].numpy(), self._expected(self.offsets[1], 0)
        )

    def test_rejects_a_frame_index_beyond_the_shard(self) -> None:
        pq.write_table(
            pa.table(
                {
                    "h5_path": ["project/bag0/frames.h5"],
                    "frame_index": np.array([7], dtype=np.int64),
                    "frame_time_ns": np.array([10], dtype=np.int64),
                }
            ),
            self.index_path,
        )
        dataset = PlannerDataset(self.index_path)
        with self.assertRaises(IndexError):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
