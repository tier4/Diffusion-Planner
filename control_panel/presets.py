"""Asset library for the control panel — a named registry of reusable assets.

Instead of one path per slot, the panel keeps **lists** of named entries per asset type
(models, datasets, reward configs, maps, run dirs) plus a few scalar settings. Every
consuming form field is a dropdown over these names, so you register a model/dataset once
and pick it anywhere.

Persisted to ``~/.diffusion_planner_presets.json`` (in $HOME, not the repo, so no internal
paths are committed). On first run it is seeded from the gitignored
``control_panel/_dev_presets.py`` (``DEV_LIBRARY``) if present — temporary testing defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

LIBRARY_PATH = Path.home() / ".diffusion_planner_presets.json"

# Asset types that are lists of {name, path, ...} entries.
LIST_TYPES = ("models", "loras", "policies", "datasets", "reward_configs", "maps", "run_dirs")
# Scalar shared settings.
SCALAR_KEYS = ("ego_shape", "output_dir", "ssd_root")

_EMPTY: dict = {
    "models": [],  # {name, path, args_json?, lora_dir?}
    "loras": [],  # {name, path}  LoRA adapter dirs (combine with any base model)
    "policies": [],  # {name, path}  exploration / guidance policy dirs
    "datasets": [],  # {name, path, role?}
    "reward_configs": [],  # {name, path}
    "maps": [],  # {name, path}
    "run_dirs": [],  # {name, path}
    "ego_shape": "4.76,7.24,2.29",
    "output_dir": "",
    "ssd_root": "",
    # Frozen baseline numbers (fill from memory; the eval table joins these as the baseline
    # column and computes Δ — never re-simulated/re-scored).
    "baseline_metrics": {
        "ego_l2": None,
        "neighbor_l2": None,
        "sc_min_dist_mean": None,
        "rb_crossings": None,
        "lane_departures": None,
        "centerline_mean": None,
    },
}


def _dev_library() -> dict | None:
    """Temporary testing defaults (gitignored). Absent in a clean checkout."""
    try:
        from ._dev_presets import DEV_LIBRARY  # type: ignore
    except Exception:
        return None
    return DEV_LIBRARY


def _normalize(data: dict) -> dict:
    """Backfill any missing top-level keys without dropping user content."""
    for k, v in _EMPTY.items():
        if k not in data:
            data[k] = json.loads(json.dumps(v))  # deep copy of the default
    for t in LIST_TYPES:
        if not isinstance(data.get(t), list):
            data[t] = []
    return data


def load_library() -> dict:
    """Load the library, seeding from DEV_LIBRARY on first run. Never raises on missing file."""
    if LIBRARY_PATH.exists():
        try:
            with open(LIBRARY_PATH) as f:
                return _normalize(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            raise RuntimeError(f"Could not read library at {LIBRARY_PATH}: {e}") from e
    seed = _dev_library()
    data = _normalize(json.loads(json.dumps(seed)) if seed else json.loads(json.dumps(_EMPTY)))
    save_library(data)
    return data


def save_library(data: dict) -> Path:
    with open(LIBRARY_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return LIBRARY_PATH


# --- query helpers (pure) -------------------------------------------------------------
def entry_names(library: dict, asset_type: str) -> list[str]:
    return [e.get("name", "") for e in library.get(asset_type, []) if e.get("name")]


def find_entry(library: dict, asset_type: str, name: str) -> dict | None:
    for e in library.get(asset_type, []):
        if e.get("name") == name:
            return e
    return None


def resolve_path(library: dict, asset_type: str, name: str) -> str:
    e = find_entry(library, asset_type, name)
    return e.get("path", "") if e else ""
