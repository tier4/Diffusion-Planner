"""vehicle_id -> ego (wheel_base, ego_length, ego_width).

The mapping (which contains vehicle_id) must not live in the public Diffusion-Planner repo, so it
is read from the private meta-repo at vehicle_shape/vehicle_shapes.json. The path is hardcoded
relative to this script: ros_scripts/ego_shapes.py -> parents[2] (meta-repo) -> vehicle_shape/.

JSON format:
    {"<vehicle_id>": {"wheel_base": <float>, "ego_length": <float>, "ego_width": <float>, "note": "..."}, ...}

Values come from dataset/generate_from_labeled.sh. Append a new vehicle_id to the JSON as needed.
"""

import json
from pathlib import Path

_SHAPES_PATH = Path(__file__).resolve().parents[2] / "vehicle_shape" / "vehicle_shapes.json"
_shapes_cache = None


def _load_shapes() -> dict:
    global _shapes_cache
    if _shapes_cache is not None:
        return _shapes_cache
    if not _SHAPES_PATH.is_file():
        raise FileNotFoundError(
            f"vehicle shapes JSON not found: {_SHAPES_PATH}\n"
            f"Create a JSON of {{vehicle_id: {{wheel_base, ego_length, ego_width}}}} "
            f"(see dataset/generate_from_labeled.sh for the values)."
        )
    _shapes_cache = json.loads(_SHAPES_PATH.read_text(encoding="utf-8"))
    return _shapes_cache


def lookup_ego_shape(vehicle_id: str):
    """Return (wheel_base, ego_length, ego_width) for vehicle_id; fail loudly if not registered."""
    shapes = _load_shapes()
    if vehicle_id not in shapes:
        raise KeyError(f"vehicle_id {vehicle_id} not registered in {_SHAPES_PATH}")
    entry = shapes[vehicle_id]
    return (entry["wheel_base"], entry["ego_length"], entry["ego_width"])


def read_vehicle_id(bag_path: Path) -> str:
    """Read vehicle_id from log_file_info.json located directly under the rosbag directory."""
    info_path = Path(bag_path) / "log_file_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"log_file_info.json not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if "vehicle_id" not in info:
        raise KeyError(f"vehicle_id not in {info_path}")
    return info["vehicle_id"]
