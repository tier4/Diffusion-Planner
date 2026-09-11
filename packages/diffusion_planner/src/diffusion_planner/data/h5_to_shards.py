"""Convert an H5 frame dataset (``scripts/dataset/create_h5_dataset.py`` output) into tar shards.

Every frame listed in the Parquet index becomes one shard member holding that frame's
``frames/*`` arrays, keyed ``<bag directory>/<frame_index:06d>``; one rosbag becomes one
partition. Frames are staged as per-frame ``.npz`` files one chunk of bags at a time and handed
to the unchanged ``planner_shards`` packer, so the result carries the same manifests, versions,
key-sets and scrub as any other shard dataset. Each chunk's staging files are deleted once packed,
so the staging directory only needs room for one chunk (about 90 KB per frame). All chunks
are staged through the same sub-directory, which keeps the packer's source namespace stable.
"""

from __future__ import annotations

import os
import random
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import numpy as np
import pyarrow.parquet as pq

from planner_shards.data_pipeline import encoding, tar_shards
from planner_shards.data_pipeline.manifest import read_manifest
from planner_shards.data_pipeline.pack_shards import main as pack_shards_main
from planner_shards.data_pipeline.versioning import DatasetRoot

hdf5plugin.register(filters="zstd")

PARTITION_REGEX = r"^(?P<partition>.+)/[0-9]{6}$"
_READ_BLOCK = 64


@dataclass(frozen=True)
class BagJob:
    """One rosbag's H5 file and the indexed frames to stage from it."""

    h5_path: str
    bag_rel: str
    frame_indices: tuple[int, ...]
    staging_root: str


def frame_datasets(file: h5py.File) -> dict[str, h5py.Dataset]:
    """Return the ``frames/*`` datasets of an H5 file keyed by tensor name."""
    frames = file["frames"]
    if not isinstance(frames, h5py.Group):
        raise ValueError(f"H5 'frames' must be a group: {file.filename}")
    datasets: dict[str, h5py.Dataset] = {}
    for key in frames.keys():  # noqa: SIM118 -- Group iteration is untyped; keys() is str
        dataset = frames[key]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"H5 'frames/{key}' must be a dataset: {file.filename}")
        datasets[str(key)] = dataset
    return datasets


def resolve_h5_path(index_path: Path, value: str) -> Path:
    """Resolve an index ``h5_path`` the way :class:`PlannerDataset` does (relative to the index)."""
    path = Path(value)
    return (path if path.is_absolute() else index_path.parent / path).resolve()


def bag_relative(h5_path: Path, h5_root: Path) -> str:
    """Return the bag directory of ``h5_path`` relative to ``h5_root``; it becomes the partition."""
    rel = os.path.relpath(h5_path.parent, h5_root)
    if rel == "." or rel.startswith(".."):
        raise ValueError(f"{h5_path} is not below the H5 root {h5_root}")
    return Path(rel).as_posix()


def plan_bags(
    index_path: Path,
    h5_root: Path,
    *,
    every: int = 1,
    offset: int = 0,
    max_bags: int | None = None,
) -> list[tuple[str, str, list[int]]]:
    """Group indexed frames by H5 file: ``(h5_path, bag_rel, sorted frame indices)`` per bag."""
    if every < 1 or offset < 0:
        raise ValueError(f"every must be >= 1 and offset >= 0: {every}, {offset}")
    table = pq.read_table(index_path, columns=["h5_path", "frame_index"])
    frames: dict[str, list[int]] = {}
    for value, index in zip(
        table.column("h5_path").to_pylist(),
        table.column("frame_index").to_pylist(),
        strict=True,
    ):
        frames.setdefault(str(value), []).append(int(index))
    bags = sorted(frames)[offset::every]
    if max_bags is not None:
        bags = bags[:max_bags]
    out = []
    for value in bags:
        h5_path = resolve_h5_path(index_path, value)
        out.append(
            (str(h5_path), bag_relative(h5_path, h5_root), sorted(frames[value]))
        )
    return out


def stage_bag(job: BagJob) -> tuple[int, int]:
    """Write one ``.npz`` per indexed frame of a bag; returns ``(frames, bytes)``."""
    out_dir = Path(job.staging_root) / job.bag_rel
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = np.asarray(job.frame_indices, dtype=np.int64)
    count = 0
    nbytes = 0
    with h5py.File(job.h5_path, "r") as file:
        datasets = frame_datasets(file)
        keys = sorted(datasets)
        for start in range(0, len(indices), _READ_BLOCK):
            block = indices[start : start + _READ_BLOCK]
            low, high = int(block[0]), int(block[-1]) + 1
            data = {key: np.asarray(datasets[key][low:high]) for key in keys}
            for frame_index in block:
                arrays = {
                    key: np.ascontiguousarray(data[key][int(frame_index) - low])
                    for key in keys
                }
                out = out_dir / f"{int(frame_index):06d}.npz"
                tmp = out.with_name(out.name + ".tmp")
                with open(tmp, "wb") as handle:
                    np.savez_compressed(handle, **arrays)  # pyright: ignore[reportArgumentType]
                os.replace(tmp, out)
                count += 1
                nbytes += out.stat().st_size
    return count, nbytes


def _run_cli(argv: list[str]) -> None:
    code = pack_shards_main(argv)
    if code != 0:
        raise RuntimeError(f"planner-shards {argv[0]} failed with exit code {code}")


def pack_staged(
    staging_root: Path,
    dest: Path,
    *,
    tag: str,
    base: str,
    workers: int,
    shard_size_gb: float,
) -> None:
    """Pack one staged chunk on top of ``base`` (``"none"`` for the first chunk)."""
    _run_cli(
        [
            "pack",
            "--source",
            str(staging_root),
            "--dest",
            str(dest),
            "--base",
            base,
            "--tag",
            tag,
            "--partition-regex",
            PARTITION_REGEX,
            "--workers",
            str(workers),
            "--shard-size-gb",
            str(shard_size_gb),
            "--quiet",
        ]
    )


def verify_members(
    dest: Path,
    tag: str,
    index_path: Path,
    h5_root: Path,
    *,
    samples: int,
    seed: int = 0,
) -> dict[str, int]:
    """Decode random members and compare them bit-exact with the source H5 frames."""
    root = DatasetRoot(dest)
    version = root.read_version(tag)
    h5_by_bag: dict[str, Path] = {}
    for value in set(
        pq.read_table(index_path, columns=["h5_path"]).column("h5_path").to_pylist()
    ):
        h5_path = resolve_h5_path(index_path, str(value))
        h5_by_bag[bag_relative(h5_path, h5_root)] = h5_path
    rows: list[tuple[Any, dict[str, Any]]] = []
    for _, entry in sorted(version.partitions.items()):
        table = read_manifest(
            root.manifest_path_for(entry.pid, entry.data_rev, entry.meta_rev),
            columns=["key", "shard_id", "sample_index_in_shard", "offset", "size"],
        )
        rows.extend((entry, row) for row in table.to_pylist())
    picked = random.Random(seed).sample(rows, min(samples, len(rows)))
    mismatches = 0
    for entry, row in picked:
        shard = (
            root.shards_dir_for(entry.pid, entry.data_rev)
            / entry.shards[int(row["shard_id"])]
        )
        with open(shard, "rb") as handle:
            payload = tar_shards.read_member(
                handle, int(row["offset"]), int(row["size"])
            )
        arrays = encoding.decode_for_training(payload)
        bag_rel, _, stem = str(row["key"]).rpartition("/")
        with h5py.File(h5_by_bag[bag_rel], "r") as file:
            reference = {
                key: np.asarray(dataset[int(stem)])
                for key, dataset in frame_datasets(file).items()
            }
        same = set(arrays) == set(reference) and all(
            arrays[key].dtype == reference[key].dtype
            and arrays[key].shape == reference[key].shape
            and np.array_equal(arrays[key], reference[key], equal_nan=True)
            for key in reference
        )
        mismatches += int(not same)
    return {"members": len(rows), "checked": len(picked), "mismatches": mismatches}


def convert(
    index_path: str | Path,
    dest: str | Path,
    *,
    tag: str,
    staging_dir: str | Path,
    h5_root: str | Path | None = None,
    chunks: int = 4,
    stage_workers: int = 32,
    pack_workers: int = 32,
    shard_size_gb: float = 1.0,
    every: int = 1,
    offset: int = 0,
    max_bags: int | None = None,
    verify_samples: int = 200,
    keyset_out: str | Path | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Convert every indexed frame into shards under ``dest`` and publish version ``tag``.

    ``h5_root`` defaults to the parent of the index directory, matching the
    ``<output_root>/<bags>/frames.h5`` + ``<output_root>/indexes/*.parquet`` layout.
    Intermediate chunk versions are published as ``<tag>.part<k>``; the final chunk publishes
    ``tag`` and ``latest``. A key-set selecting every frame is written to ``keyset_out``
    (default ``<dest>/keysets/<tag>.parquet``); the dataset is scrubbed and, when
    ``verify_samples`` > 0, random members are checked bit-exact against the H5 source.
    """
    index = Path(index_path).expanduser().resolve()
    dest_root = Path(dest).expanduser().resolve()
    staging = Path(staging_dir).expanduser().resolve()
    root = (
        index.parent.parent if h5_root is None else Path(h5_root).expanduser().resolve()
    )
    if chunks < 1:
        raise ValueError(f"chunks must be at least 1: {chunks}")
    bags = plan_bags(index, root, every=every, offset=offset, max_bags=max_bags)
    if not bags:
        raise ValueError(f"index selects no bags: {index}")
    parts = [part for part in (bags[k::chunks] for k in range(chunks)) if part]
    total_frames = sum(len(indices) for _, _, indices in bags)
    started = time.perf_counter()
    staged_frames = 0
    staged_bytes = 0
    base = "none"
    for number, part in enumerate(parts):
        chunk_tag = tag if number == len(parts) - 1 else f"{tag}.part{number}"
        # Every chunk is staged through the same directory: the packer records the resolved
        # source path as the dataset namespace and refuses to extend a version from another one.
        chunk_root = staging / "stage"
        if chunk_root.exists():
            shutil.rmtree(chunk_root)
        chunk_root.mkdir(parents=True)
        jobs = [
            BagJob(h5, rel, tuple(indices), str(chunk_root))
            for h5, rel, indices in part
        ]
        with ProcessPoolExecutor(max_workers=max(1, stage_workers)) as executor:
            for count, nbytes in executor.map(stage_bag, jobs):
                staged_frames += count
                staged_bytes += nbytes
        pack_staged(
            chunk_root,
            dest_root,
            tag=chunk_tag,
            base=base,
            workers=pack_workers,
            shard_size_gb=shard_size_gb,
        )
        shutil.rmtree(chunk_root)
        base = chunk_tag
        progress(
            f"chunk {number + 1}/{len(parts)}: {len(part)} bags packed as {chunk_tag}; "
            f"{staged_frames}/{total_frames} frames, {time.perf_counter() - started:.0f} s"
        )
    keyset = (
        dest_root / "keysets" / f"{tag}.parquet"
        if keyset_out is None
        else Path(keyset_out)
    )
    keyset.parent.mkdir(parents=True, exist_ok=True)
    _run_cli(
        [
            "keyset",
            "--dest",
            str(dest_root),
            "--tag",
            tag,
            "--where",
            "TRUE",
            "--out",
            str(keyset),
        ]
    )
    _run_cli(["scrub", "--dest", str(dest_root), "--tag", tag])
    result: dict[str, Any] = {
        "tag": tag,
        "dest": str(dest_root),
        "bags": len(bags),
        "frames": total_frames,
        "staged_frames": staged_frames,
        "staged_bytes": staged_bytes,
        "keyset": str(keyset),
        "wall_s": time.perf_counter() - started,
    }
    if verify_samples > 0:
        result["verify"] = verify_members(
            dest_root, tag, index, root, samples=verify_samples
        )
        if result["verify"]["mismatches"]:
            raise RuntimeError(
                f"{result['verify']['mismatches']} member(s) differ from the H5 source"
            )
    return result
