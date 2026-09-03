from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

SIDECAR_VARIANTS = {
    "none": [],
    "psim": ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"],
    "skip": ["is_skipped", "skipping_info", "timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"],
    "neighbor": [
        "is_skipped",
        "neighbor_ids",
        "skipping_info",
        "timestamp",
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
    ],
    "full": [
        "bag_time",
        "date",
        "is_skipped",
        "log_file_id",
        "map_version_id",
        "neighbor_ids",
        "project_id",
        "qw",
        "qx",
        "qy",
        "qz",
        "skipping_info",
        "t4_dataset_id",
        "t4_dataset_version_id",
        "timestamp",
        "vehicle_id",
        "x",
        "y",
        "z",
    ],
}

_SHAPES = {  # (small shape, dtype) — same names/dtypes as the real npz, tiny sizes
    "version": ((1,), np.uint32),
    "ego_agent_past": ((31, 3), np.float32),
    "ego_current_state": ((10,), np.float32),
    "ego_agent_future": ((80, 3), np.float32),
    "neighbor_agents_past": ((4, 31, 11), np.float32),
    "neighbor_agents_future": ((4, 80, 4), np.float32),
    "static_objects": ((5, 10), np.float32),
    "ego_shape": ((3,), np.float32),
    "lanes": ((3, 20, 33), np.float32),
    "lanes_speed_limit": ((3, 1), np.float32),
    "lanes_has_speed_limit": ((3, 1), np.bool_),
    "route_lanes": ((2, 20, 33), np.float32),
    "route_lanes_speed_limit": ((2, 1), np.float32),
    "route_lanes_has_speed_limit": ((2, 1), np.bool_),
    "turn_indicators": ((31,), np.int32),
    "goal_pose": ((3,), np.float32),
    "polygons": ((2, 40, 3), np.float32),
    "line_strings": ((2, 20, 4), np.float32),
}


def make_arrays(rng: np.random.Generator, *, small: bool = True) -> dict[str, np.ndarray]:
    out = {}
    for name, (shape, dt) in _SHAPES.items():
        if dt == np.bool_:
            out[name] = rng.integers(0, 2, size=shape).astype(np.bool_)
        elif np.issubdtype(dt, np.integer):
            out[name] = rng.integers(0, 5, size=shape).astype(dt)
        else:
            arr = rng.standard_normal(shape).astype(dt)
            arr.reshape(-1)[: arr.size // 2] = 0.0  # padding-like zeros, as in real data
            out[name] = arr
    out["version"] = np.array([5], dtype=np.uint32)
    return out


def make_sidecar(variant: str, i: int, *, is_skipped: bool = False) -> dict | None:
    if variant == "none":
        return None
    d = {
        "timestamp": 1_700_000_000_000_000_000 + i * 300_000_000,
        "x": 1.0 * i,
        "y": 2.0,
        "z": 3.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.1,
        "qw": 0.99,
    }
    if variant in ("skip", "neighbor", "full"):
        d["is_skipped"] = is_skipped
        d["skipping_info"] = {"details": "Accepted", "incomplete_data_types": [], "label": 0}
    if variant in ("neighbor", "full"):
        d["neighbor_ids"] = [f"id{i}a", f"id{i}b"]
    if variant == "full":
        d.update(
            bag_time="10-00-00",
            date="2026-01-01",
            log_file_id="log",
            map_version_id="m-1",
            project_id="projA",
            t4_dataset_id="",
            t4_dataset_version_id="",
            vehicle_id="veh",
        )
    return d


def write_sample(dir_: Path, stem: str, arrays: dict, sidecar: dict | None, *, compressed=True):
    dir_.mkdir(parents=True, exist_ok=True)
    npz = dir_ / f"{stem}.npz"
    buf = io.BytesIO()
    (np.savez_compressed if compressed else np.savez)(buf, **arrays)
    npz.write_bytes(buf.getvalue())
    js = None
    if sidecar is not None:
        js = dir_ / f"{stem}.json"
        js.write_text(json.dumps(sidecar, indent=2))
    return npz, js


def make_tree(
    root: Path,
    layout: list[tuple[str, int, str]],
    seed: int = 0,
    skipped_every: int = 0,
    compressed: bool = True,
) -> list[str]:
    """Write frames under root/<rel_dir>/<basename>_<NNNNNNNN>.npz (+ .json). Returns keys."""
    rng = np.random.default_rng(seed)
    keys = []
    for rel_dir, n, variant in layout:
        d = Path(root) / rel_dir
        base = Path(rel_dir).name
        for i in range(n):
            stem = f"{base}_{i:08d}"
            skipped = bool(skipped_every) and (i % skipped_every == skipped_every - 1)
            write_sample(
                d,
                stem,
                make_arrays(rng),
                make_sidecar(variant, i, is_skipped=skipped),
                compressed=compressed,
            )
            keys.append(f"{rel_dir}/{stem}")
    return keys
