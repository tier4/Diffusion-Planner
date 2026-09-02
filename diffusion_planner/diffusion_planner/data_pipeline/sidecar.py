"""Sidecar (.json) parsing into the existing-field union schema (spec §3). No new fields."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa

from diffusion_planner.data_pipeline.errors import SidecarError

SIDECAR_FIELDS: list[tuple[str, pa.DataType]] = [
    ("is_skipped", pa.bool_()),
    ("skip_label", pa.int32()),
    ("timestamp", pa.int64()),
    ("x", pa.float64()),
    ("y", pa.float64()),
    ("z", pa.float64()),
    ("qx", pa.float64()),
    ("qy", pa.float64()),
    ("qz", pa.float64()),
    ("qw", pa.float64()),
    ("log_file_id", pa.string()),
    ("vehicle_id", pa.string()),
    ("project_id", pa.string()),
    ("map_version_id", pa.string()),
    ("date", pa.string()),
    ("bag_time", pa.string()),
    ("t4_dataset_id", pa.string()),
    ("t4_dataset_version_id", pa.string()),
    ("neighbor_count", pa.int32()),
]
_STRING_FIELDS = {
    "log_file_id",
    "vehicle_id",
    "project_id",
    "map_version_id",
    "date",
    "bag_time",
    "t4_dataset_id",
    "t4_dataset_version_id",
}
_FLOAT_FIELDS = {"x", "y", "z", "qx", "qy", "qz", "qw"}


def sidecar_path_for(npz_path: Path) -> Path:
    return Path(npz_path).with_suffix(".json")


def _load(raw: bytes) -> dict:
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SidecarError(f"invalid sidecar json: {e}") from e
    if not isinstance(obj, dict):
        raise SidecarError("sidecar json is not an object")
    return obj


def parse_sidecar(raw: bytes | None) -> dict[str, object]:
    fields: dict[str, object] = {name: None for name, _ in SIDECAR_FIELDS}
    if raw is None:
        return fields
    obj = _load(raw)
    if "is_skipped" in obj:
        if not isinstance(obj["is_skipped"], bool):
            raise SidecarError("is_skipped must be bool")
        fields["is_skipped"] = obj["is_skipped"]
    info = obj.get("skipping_info")
    if isinstance(info, dict) and isinstance(info.get("label"), int):
        fields["skip_label"] = int(info["label"])
    if "timestamp" in obj:
        if isinstance(obj["timestamp"], bool) or not isinstance(obj["timestamp"], int):
            raise SidecarError("timestamp must be int")
        fields["timestamp"] = int(obj["timestamp"])
    for k in _FLOAT_FIELDS:
        if k in obj:
            if isinstance(obj[k], bool) or not isinstance(obj[k], (int, float)):
                raise SidecarError(f"{k} must be numeric")
            fields[k] = float(obj[k])
    for k in _STRING_FIELDS:
        if k in obj:
            if not isinstance(obj[k], str):
                raise SidecarError(f"{k} must be string")
            fields[k] = obj[k]
    ids = obj.get("neighbor_ids")
    if ids is not None:
        if not isinstance(ids, list):
            raise SidecarError("neighbor_ids must be a list")
        fields["neighbor_count"] = len(ids)
    return fields


def neighbor_ids_of(raw: bytes | None) -> list[str] | None:
    if raw is None:
        return None
    ids = _load(raw).get("neighbor_ids")
    return [str(i) for i in ids] if isinstance(ids, list) else None


def is_rejected(fields: dict[str, object]) -> bool:
    return fields.get("is_skipped") is True
