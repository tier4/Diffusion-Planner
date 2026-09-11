"""H5 -> shard conversion: chunked staging, packing, key-set, bit-exact verification."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from diffusion_planner.data.h5_to_shards import bag_relative, convert, plan_bags
from planner_shards.data_pipeline.versioning import DatasetRoot

SHAPES = {"ego_agent_past": (3, 2), "goal_pose": (4,), "lanes": (2, 2, 2)}


def _write_bag(path: Path, num_frames: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    arrays = {
        key: rng.standard_normal((num_frames, *shape)).astype(np.float32)
        for key, shape in SHAPES.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        file.attrs["format"] = "diffusion_planner_frame_dataset"
        file.attrs["format_version"] = 4
        file.attrs["num_frames"] = num_frames
        frames = file.create_group("frames")
        for key, values in arrays.items():
            frames.create_dataset(key, data=values, chunks=(1, *values.shape[1:]))
        file.create_group("metadata").create_dataset(
            "frame_time_ns", data=np.arange(num_frames, dtype=np.int64)
        )
    return arrays


@pytest.fixture
def h5_layout(tmp_path: Path) -> tuple[Path, Path, dict[str, tuple[Path, list[int]]]]:
    root = tmp_path / "h5"
    bags = {
        "site/bagA": (
            root / "site" / "bagA" / "frames.h5",
            list(range(0, 12, 2)),
        ),  # 1 Hz subset
        "site/bagB": (root / "site" / "bagB" / "frames.h5", list(range(5))),
        "other/bagC": (root / "other" / "bagC" / "frames.h5", list(range(7))),
    }
    for seed, (path, _) in enumerate(bags.values()):
        _write_bag(path, num_frames=12 if "bagA" in str(path) else 7, seed=seed)
    rows_h5, rows_index, rows_time = [], [], []
    index_dir = root / "indexes"
    index_dir.mkdir()
    for path, indices in bags.values():
        for index in indices:
            rows_h5.append(str(Path("..") / path.relative_to(root)))
            rows_index.append(index)
            rows_time.append(index * 1000)
    index = index_dir / "train.parquet"
    pq.write_table(
        pa.table(
            {
                "h5_path": rows_h5,
                "frame_index": pa.array(rows_index, pa.int64()),
                "frame_time_ns": pa.array(rows_time, pa.int64()),
            }
        ),
        index,
    )
    return root, index, bags


def test_plan_bags_groups_indexed_frames_per_bag(h5_layout) -> None:
    root, index, bags = h5_layout
    planned = plan_bags(index, root)
    assert [rel for _, rel, _ in planned] == sorted(bags)
    by_rel = {rel: indices for _, rel, indices in planned}
    assert by_rel["site/bagA"] == list(range(0, 12, 2))
    assert plan_bags(index, root, every=2, offset=1) == [planned[1]]
    assert len(plan_bags(index, root, max_bags=2)) == 2


def test_bag_relative_rejects_files_outside_the_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        bag_relative(tmp_path / "elsewhere" / "frames.h5", tmp_path / "root")


def test_convert_packs_every_indexed_frame_in_chunks_and_verifies(
    h5_layout, tmp_path: Path
) -> None:
    root, index, bags = h5_layout
    dest = tmp_path / "shards"
    staging = tmp_path / "staging"
    messages: list[str] = []
    result = convert(
        index,
        dest,
        tag="v1",
        staging_dir=staging,
        chunks=2,
        stage_workers=1,
        pack_workers=1,
        verify_samples=50,
        progress=messages.append,
    )
    expected_frames = sum(len(indices) for _, indices in bags.values())
    assert result["frames"] == result["staged_frames"] == expected_frames == 18
    assert result["bags"] == 3
    assert result["verify"] == {"members": 18, "checked": 18, "mismatches": 0}
    assert len(messages) == 2
    # one partition per bag, published under the final tag; chunk 0 left an intermediate version
    version = DatasetRoot(dest).read_version("v1")
    assert sorted(version.partitions) == sorted(bags)
    assert DatasetRoot(dest).latest() == "v1"
    assert DatasetRoot(dest).read_version("v1.part0")
    keyset = pq.read_table(result["keyset"])
    assert keyset.num_rows == 18
    # staged npz files are gone
    assert not list(staging.rglob("*.npz"))
