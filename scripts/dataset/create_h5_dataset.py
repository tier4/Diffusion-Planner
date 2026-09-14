"""Build preprocessed H5 shards and a split-level Parquet index."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import hydra
import ml_planner_data as mpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from autoware_msg_bag_converter.converter import convert_bag
from omegaconf import DictConfig
from tqdm import tqdm

from diffusion_planner.data.h5_writer import validate_h5_arrays, write_h5_shard

# ``override`` is a complete-route source split used for closed-loop
# evaluation.  The separately exported one-frame override H5 files remain
# open-loop inputs only.
SPLITS = ("train", "valid", "auto", "override")
FORMAT_NAME = "diffusion_planner_frame_dataset"
FORMAT_VERSION = 4


@dataclass(frozen=True)
class VehicleParameters:
    """Vehicle dimensions used while preprocessing source bags."""

    base_link_to_front: float
    vehicle_length: float
    vehicle_width: float


@dataclass(frozen=True)
class BagEntry:
    """One source bag and the metadata needed to generate its output."""

    bag_path: str
    relative_bag_path: str
    map_path: str
    project_id: str
    area_map_id: str
    area_map_version_id: str
    split: str


@dataclass(frozen=True)
class WorkerConfig:
    """Primitive generation settings safe to send to worker processes."""

    output_root: str
    converted_bag_root: str
    index_path: str
    frame_interval_s: float
    future_steps: int
    min_travel_distance: float
    split_routes: bool
    topic_drop_thresholds: dict[str, float]
    traffic_light_timeout_s: float
    neighbor_observation_timeout_s: float
    compression: str | None
    overwrite: bool
    resume: bool


@dataclass(frozen=True)
class BagResult:
    """Small result returned from a worker after its H5 file is complete."""

    table: pa.Table | None
    warnings: list[str]
    all_frames: int
    usable_frames: int
    created_frames: int
    failed_frames: int
    skipped: bool
    resumed: bool


def build_vehicles(config: DictConfig) -> dict[str, VehicleParameters]:
    """Build the project-to-vehicle mapping used by source bags."""
    return {
        str(project): VehicleParameters(
            base_link_to_front=float(node.base_link_to_front),
            vehicle_length=float(node.vehicle_length),
            vehicle_width=float(node.vehicle_width),
        )
        for project, node in config.items()
    }


def build_builder_param(config: WorkerConfig) -> Any:
    """Build the native generation parameters."""
    if not math.isfinite(config.frame_interval_s) or config.frame_interval_s <= 0.0:
        raise ValueError(
            f"frame_interval must be finite and positive: {config.frame_interval_s}"
        )
    if (
        not math.isfinite(config.min_travel_distance)
        or config.min_travel_distance < 0.0
    ):
        raise ValueError(
            f"min_travel_distance must be finite and non-negative: {config.min_travel_distance}"
        )

    thresholds = mpd.TopicDropThresholds()
    for topic, limit in config.topic_drop_thresholds.items():
        if not hasattr(thresholds, topic):
            raise ValueError(f"unknown topic in topic_drop_thresholds: {topic}")
        setattr(thresholds, topic, float(limit))

    param = mpd.DatasetBuilderParam()
    param.frame_interval_s = config.frame_interval_s
    param.num_future_steps = config.future_steps
    param.min_travel_distance = config.min_travel_distance
    param.split_routes = config.split_routes
    param.topic_drop_thresholds = thresholds
    param.traffic_light_timeout_s = config.traffic_light_timeout_s
    param.neighbor_observation_timeout_s = config.neighbor_observation_timeout_s
    return param


def discover_bags(root: Path, split: str) -> list[BagEntry]:
    """Discover bags and preserve their hierarchy relative to the requested root."""
    entries = []
    # Closed-loop selections are materialized as directory symlinks.  pathlib's
    # rglob() deliberately does not recurse into those directories, so walk the
    # tree with followlinks enabled to discover both regular and selected bags.
    info_paths = sorted(
        Path(directory) / "log_file_info.json"
        for directory, _, filenames in os.walk(root, followlinks=True)
        if "log_file_info.json" in filenames
    )
    for info_path in info_paths:
        bag_path = info_path.parent
        if (
            not (bag_path / "metadata.yaml").is_file()
            or bag_path.parents[1].name != split
        ):
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        map_version = str(info["area_map_version_id"])
        # Standard data is ``<project>/<split>/<route>/<bag>`` whereas the
        # override source has an extra ``override`` container:
        # ``<site>/override/<route>/<bag>``.  Its map lives under that
        # container, not directly below the site.
        map_root = bag_path.parents[1] if split == "override" else bag_path.parents[2]
        map_path = map_root / "map" / map_version / "lanelet2_map.osm"
        if not map_path.is_file():
            raise FileNotFoundError(f"map not found for {bag_path}: {map_path}")
        relative = bag_path.relative_to(root)
        if relative == Path("."):
            relative = Path(bag_path.name)
        entries.append(
            BagEntry(
                bag_path=str(bag_path),
                relative_bag_path=relative.as_posix(),
                map_path=str(map_path),
                project_id=str(info["project_id"]),
                area_map_id=str(info["area_map_id"]),
                area_map_version_id=map_version,
                split=split,
            )
        )
    return entries


def h5_relative_path(entry: BagEntry, route_group_id: int | None = None) -> Path:
    """Return the portable H5 path stored in the Parquet index."""
    relative = Path(entry.relative_bag_path)
    if route_group_id is not None:
        relative /= f"route_{route_group_id:08d}"
    return relative / "frames.h5"


def write_h5(
    path: Path,
    entry: BagEntry,
    result: Mapping[str, Any],
    config: WorkerConfig,
    route_group_id: int | None = None,
) -> None:
    """Write one complete shard through the shared H5 pipeline."""
    attributes: dict[str, Any] = {
        "source_bag_path": entry.relative_bag_path,
        "source_map_path": entry.map_path,
        "project_id": entry.project_id,
        "area_map_id": entry.area_map_id,
        "area_map_version_id": entry.area_map_version_id,
        "split": entry.split,
        "frame_interval_s": config.frame_interval_s,
        "traffic_light_timeout_s": config.traffic_light_timeout_s,
        "neighbor_observation_timeout_s": config.neighbor_observation_timeout_s,
    }
    if route_group_id is not None:
        attributes["route_group_id"] = route_group_id
    write_h5_shard(
        path,
        attributes=attributes,
        frames=result["frames"],
        metadata=result["metadata"],
        compression=config.compression,
    )


def read_existing_h5(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Validate an existing shard and load the metadata needed to rebuild its index."""
    with h5py.File(path, "r") as file:
        if file.attrs.get("format") != FORMAT_NAME:
            raise ValueError(f"unexpected H5 format: {path}")
        format_version = file.attrs.get("format_version")
        if not isinstance(format_version, (int, np.integer)):
            raise ValueError(f"invalid H5 format version: {path}")
        if int(format_version) != FORMAT_VERSION:
            raise ValueError(
                f"unsupported H5 format version: {path}; regenerate with overwrite=true"
            )
        if "frames" not in file or "metadata" not in file:
            raise ValueError(f"missing frames or metadata group: {path}")

        num_frames_value = file.attrs.get("num_frames")
        if not isinstance(num_frames_value, (int, np.integer)):
            raise ValueError(f"invalid or missing H5 num_frames: {path}")
        num_frames = int(num_frames_value)

        frames_object = file["frames"]
        metadata_object = file["metadata"]
        if not isinstance(frames_object, h5py.Group):
            raise ValueError(f"H5 frames must be a group: {path}")
        if not isinstance(metadata_object, h5py.Group):
            raise ValueError(f"H5 metadata must be a group: {path}")
        required_metadata = {"frame_time_ns", "ego_x", "ego_y", "ego_yaw"}
        missing_metadata = required_metadata.difference(metadata_object.keys())
        if missing_metadata:
            raise ValueError(
                f"H5 is missing required closed-loop pose metadata {sorted(missing_metadata)}: "
                f"{path}; regenerate with overwrite=true"
            )

        frame_lengths: set[int] = set()
        for key, values in frames_object.items():
            if not isinstance(values, h5py.Dataset):
                raise ValueError(f"H5 frames/{key} must be a dataset: {path}")
            if values.chunks is None or values.chunks[0] != 1:
                raise ValueError(
                    f"H5 frames/{key} is not chunked per frame: {path}; "
                    "regenerate with overwrite=true"
                )
            frame_lengths.add(len(values))

        metadata: dict[str, np.ndarray] = {}
        metadata_lengths: set[int] = set()
        for key, values in metadata_object.items():
            if not isinstance(values, h5py.Dataset):
                raise ValueError(f"H5 metadata/{key} must be a dataset: {path}")
            metadata[key] = np.asarray(values[...])
            metadata_lengths.add(len(values))
        if frame_lengths != {num_frames} or metadata_lengths != {num_frames}:
            raise ValueError(f"inconsistent first dimensions in {path}")
        return {"metadata": metadata}


def make_index_table(
    entry: BagEntry,
    h5_path: Path,
    index_path: Path,
    metadata: Mapping[str, np.ndarray],
) -> pa.Table:
    """Build the Parquet rows corresponding exactly to one H5 shard."""
    stored_h5_path = Path(os.path.relpath(h5_path, start=index_path.parent)).as_posix()
    num_frames = len(metadata["frame_time_ns"])
    columns = {
        "h5_path": pa.array([stored_h5_path] * num_frames, pa.string()),
        "frame_index": pa.array(np.arange(num_frames, dtype=np.int64)),
        "frame_time_ns": pa.array(metadata["frame_time_ns"], pa.int64()),
        "ego_x": pa.array(metadata["ego_x"], pa.float64()),
        "ego_y": pa.array(metadata["ego_y"], pa.float64()),
        "ego_yaw": pa.array(metadata["ego_yaw"], pa.float64()),
        "ego_speed_mps": pa.array(metadata["ego_speed_mps"], pa.float32()),
        "ego_yaw_rate_rps": pa.array(metadata["ego_yaw_rate_rps"], pa.float32()),
        "turn_indicator": pa.array(metadata["turn_indicator"], pa.uint8()),
        "num_objects": pa.array(metadata["num_objects"], pa.int32()),
        "project_id": pa.array([entry.project_id] * num_frames, pa.string()),
        "area_map_id": pa.array([entry.area_map_id] * num_frames, pa.string()),
        "area_map_version_id": pa.array(
            [entry.area_map_version_id] * num_frames, pa.string()
        ),
        "split": pa.array([entry.split] * num_frames, pa.string()),
    }
    if "route_group_id" in metadata:
        columns["route_group_id"] = pa.array(metadata["route_group_id"], pa.int64())
    return pa.table(columns)


def select_frames(
    result: Mapping[str, Any], selection: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    """Select one route group while preserving every frame and metadata field."""
    return {
        group_name: {
            key: np.asarray(values)[selection]
            for key, values in result[group_name].items()
        }
        for group_name in ("frames", "metadata")
    }


def normalized_bag_path(entry: BagEntry, config: WorkerConfig) -> Path:
    """Convert one source bag to current message schemas and return its path."""
    output = Path(config.converted_bag_root) / entry.relative_bag_path
    if (output / "metadata.yaml").is_file():
        return output

    incomplete = output.with_name(output.name + ".incomplete")
    if incomplete.exists():
        shutil.rmtree(incomplete)
    incomplete.parent.mkdir(parents=True, exist_ok=True)
    convert_bag(entry.bag_path, str(incomplete))
    if not (incomplete / "metadata.yaml").is_file():
        raise RuntimeError(f"converted bag has no metadata.yaml: {incomplete}")
    incomplete.replace(output)
    return output


def process_bag(
    packed: tuple[BagEntry, VehicleParameters, WorkerConfig],
) -> BagResult:
    """Generate or resume one H5 shard and return its index rows."""
    entry, vehicle, config = packed
    output_root = Path(config.output_root)
    output_path = output_root / h5_relative_path(entry)
    index_path = Path(config.index_path)
    if config.split_routes:
        existing_route_shards = sorted(
            (output_root / entry.relative_bag_path).glob("route_*/frames.h5")
        )
        if existing_route_shards:
            if not config.resume:
                raise FileExistsError(
                    f"route H5 output already exists for {entry.bag_path}; "
                    "use an empty output root"
                )
            tables = [
                make_index_table(
                    entry,
                    shard,
                    index_path,
                    read_existing_h5(shard)["metadata"],
                )
                for shard in existing_route_shards
            ]
            table = pa.concat_tables(tables)
            return BagResult(table, [], 0, 0, table.num_rows, 0, False, True)
    if output_path.exists():
        if config.overwrite:
            pass
        elif config.resume:
            existing = read_existing_h5(output_path)
            table = make_index_table(
                entry, output_path, index_path, existing["metadata"]
            )
            return BagResult(table, [], 0, 0, table.num_rows, 0, False, True)
        else:
            raise FileExistsError(f"H5 output already exists: {output_path}")

    spec = mpd.VehicleSpec(
        base_link_to_front=vehicle.base_link_to_front,
        vehicle_length=vehicle.vehicle_length,
        vehicle_width=vehicle.vehicle_width,
    )
    bag_path = normalized_bag_path(entry, config)
    result = mpd.create_bag_frame_data(
        bag_path=str(bag_path),
        map_path=entry.map_path,
        vehicle_spec=spec,
        param=build_builder_param(config),
    )
    stats = result["stats"]
    num_frames = validate_h5_arrays(result["frames"], result["metadata"])
    if num_frames == 0:
        return BagResult(
            None,
            list(result["warnings"]),
            int(stats["all_frames"]),
            int(stats["usable_frames"]),
            0,
            int(stats["failed_frames"]),
            bool(stats["skipped"]),
            False,
        )
    if config.split_routes:
        route_group_ids = np.asarray(result["metadata"]["route_group_id"])
        tables = []
        for route_group_id in np.unique(route_group_ids):
            group_id = int(route_group_id)
            route_result = select_frames(result, route_group_ids == route_group_id)
            route_path = output_root / h5_relative_path(entry, group_id)
            write_h5(route_path, entry, route_result, config, group_id)
            tables.append(
                make_index_table(
                    entry, route_path, index_path, route_result["metadata"]
                )
            )
        table = pa.concat_tables(tables)
    else:
        write_h5(output_path, entry, result, config)
        table = make_index_table(entry, output_path, index_path, result["metadata"])
    return BagResult(
        table,
        list(result["warnings"]),
        int(stats["all_frames"]),
        int(stats["usable_frames"]),
        int(stats["created_frames"]),
        int(stats["failed_frames"]),
        bool(stats["skipped"]),
        False,
    )


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="dataset/create_h5_dataset",
)
def main(config: DictConfig) -> None:
    """Build H5 shards and their split-level Parquet index."""
    if config.split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}: {config.split}")
    if config.jobs < 1:
        raise ValueError(f"jobs must be at least 1: {config.jobs}")

    root = Path(config.root).expanduser().resolve()
    output_root = Path(config.output_root).expanduser().resolve()
    index_output = Path(config.index_output).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    if index_output.exists() and not config.overwrite:
        raise FileExistsError(f"Parquet index already exists: {index_output}")

    entries = discover_bags(root, str(config.split))
    if config.limit is not None:
        entries = entries[: int(config.limit)]
    if not entries:
        raise FileNotFoundError(f"no {config.split} rosbags found under {root}")

    vehicles = build_vehicles(config.vehicles)
    unknown = sorted({entry.project_id for entry in entries} - set(vehicles))
    if unknown:
        raise ValueError(f"no vehicle configured for project(s): {', '.join(unknown)}")

    worker_config = WorkerConfig(
        output_root=str(output_root),
        converted_bag_root=str(Path(config.converted_bag_root).expanduser().resolve()),
        index_path=str(index_output),
        frame_interval_s=float(config.frame_interval),
        future_steps=int(config.future_steps),
        min_travel_distance=float(config.min_travel_distance),
        split_routes=bool(config.split_routes),
        topic_drop_thresholds={
            str(topic): float(limit)
            for topic, limit in config.topic_drop_thresholds.items()
        },
        traffic_light_timeout_s=float(config.traffic_light_timeout_s),
        neighbor_observation_timeout_s=float(config.neighbor_observation_timeout_s),
        compression=None if config.compression is None else str(config.compression),
        overwrite=bool(config.overwrite),
        resume=bool(config.resume),
    )
    packed = [(entry, vehicles[entry.project_id], worker_config) for entry in entries]

    output_root.mkdir(parents=True, exist_ok=True)
    index_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = index_output.with_suffix(index_output.suffix + ".incomplete")
    temporary_index.unlink(missing_ok=True)
    warning_log_path = index_output.parent / datetime.now().strftime(
        "%Y-%m-%d-%H-%M-%S.log"
    )
    writer: pq.ParquetWriter | None = None
    total_frames = failed_frames = skipped_bags = empty_bags = resumed_bags = 0
    try:
        with (
            warning_log_path.open("w", encoding="utf-8") as warning_log,
            ProcessPoolExecutor(max_workers=int(config.jobs)) as executor,
        ):
            progress = tqdm(
                zip(entries, executor.map(process_bag, packed), strict=True),
                total=len(entries),
                unit="bag",
                smoothing=0.0,
            )
            for entry, result in progress:
                for warning in result.warnings:
                    warning_log.write(f"{entry.bag_path}: {warning}\n")
                warning_log.flush()
                failed_frames += result.failed_frames
                skipped_bags += int(result.skipped)
                resumed_bags += int(result.resumed)
                if result.table is None or result.table.num_rows == 0:
                    empty_bags += int(not result.skipped)
                    continue
                if writer is None:
                    writer = pq.ParquetWriter(temporary_index, result.table.schema)
                writer.write_table(result.table)
                total_frames += result.table.num_rows
                progress.set_postfix(frames=total_frames, resumed=resumed_bags)
        if writer is None:
            raise RuntimeError("no bag produced any valid frame")
        writer.close()
        writer = None
        temporary_index.replace(index_output)
    except BaseException:
        if writer is not None:
            writer.close()
        temporary_index.unlink(missing_ok=True)
        raise

    print(
        f"wrote {index_output}: {total_frames} frames from {len(entries)} bag(s), "
        f"{resumed_bags} resumed, {skipped_bags} skipped, {empty_bags} empty, "
        f"{failed_frames} frame generation failure(s)"
    )
    print(f"H5 root: {output_root}")
    if warning_log_path.stat().st_size:
        print(f"wrote warnings to {warning_log_path}")
    else:
        warning_log_path.unlink()


if __name__ == "__main__":
    main()
