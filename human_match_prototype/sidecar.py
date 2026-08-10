"""Read NPZ JSON sidecars and build spatial indexes.

Inlined from diffusion_planner/util_scripts/search_scenes.py because
``util_scripts`` lives outside the installed ``diffusion_planner`` package
and is therefore not importable in the normal Python path.
"""

import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from tqdm import tqdm


def _quat_to_heading_deg(qz: float, qw: float) -> float:
    """Convert quaternion (z, w components) to heading in degrees [-180, 180)."""
    yaw_rad = 2.0 * math.atan2(qz, qw)
    deg = math.degrees(yaw_rad)
    return (deg + 180.0) % 360.0 - 180.0


def read_sidecar(npz_path: str) -> Optional[dict]:
    """Read JSON sidecar for an NPZ file.

    Returns dict with x, y, heading_deg, timestamp, map_version_id or None.
    """
    json_path = npz_path[:-4] + ".json"
    try:
        with open(json_path, "r") as f:
            j = json.load(f)
        return {
            "npz_path": npz_path,
            "x": j["x"],
            "y": j["y"],
            "heading_deg": _quat_to_heading_deg(j["qz"], j["qw"]),
            "timestamp": j.get("timestamp"),
            "map_version_id": j.get("map_version_id"),
        }
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def _read_sidecar_batch(npz_paths: list[str]) -> list[Optional[dict]]:
    """Read sidecars for a batch of NPZ paths (used by ProcessPoolExecutor)."""
    return [read_sidecar(p) for p in npz_paths]


def build_index(npz_paths: list[str], workers: int = 8, batch_size: int = 500) -> list[dict]:
    """Build spatial index from JSON sidecars using multiprocessing.

    Returns list of dicts with keys: npz_path, x, y, heading_deg, timestamp.
    Scenes without valid sidecars are silently skipped.
    """
    batches = [npz_paths[i : i + batch_size] for i in range(0, len(npz_paths), batch_size)]
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_read_sidecar_batch, batch): batch for batch in batches}
        with tqdm(total=len(npz_paths), desc="Reading sidecars", unit="scene") as pbar:
            for future in as_completed(futures):
                batch_results = future.result()
                for r in batch_results:
                    if r is not None:
                        results.append(r)
                pbar.update(len(batch_results))

    return results
