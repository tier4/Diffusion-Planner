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

# Asset types that are lists of {name, path, ...} entries. Datasets are split by form:
# scene_datasets = individual-scene list JSONs; route_datasets = contiguous per-frame NPZ dirs.
LIST_TYPES = (
    "models",
    "loras",
    "policies",
    "scene_datasets",
    "route_datasets",
    "grpo_configs",
    "reward_configs",
    "maps",
    "run_dirs",
)
# Scalar shared settings.
SCALAR_KEYS = ("ego_shape", "output_dir", "ssd_root", "workspace_root")

_EMPTY: dict = {
    "models": [],  # {name, path, args_json?, lora_dir?}
    "loras": [],  # {name, path}  LoRA adapter dirs (combine with any base model)
    "policies": [],  # {name, path}  exploration / guidance policy dirs
    "scene_datasets": [],  # {name, path}  individual-scene list JSON
    "route_datasets": [],  # {name, path}  contiguous per-frame NPZ dir
    "grpo_configs": [],  # {name, path}  GRPO / generation+training configs
    "reward_configs": [],  # {name, path}  reward / metrics-eval scoring configs
    "maps": [],  # {name, path}
    "run_dirs": [],  # {name, path}
    "ego_shape": "4.76,7.24,2.29",
    "output_dir": "",
    "ssd_root": "",
    "workspace_root": "",
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


def field_defaults() -> dict:
    """Pre-fill values for plain form fields, {workflow_key: {arg_name: value}} (gitignored)."""
    try:
        from ._dev_presets import DEV_FIELD_DEFAULTS  # type: ignore
    except Exception:
        return {}
    return DEV_FIELD_DEFAULTS


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
    if seed and seed.get("workspace_root") and not seed.get("models"):
        # Dev seed points at a workspace → scan it to auto-populate, keep seed scalars.
        data = scan_workspace(seed["workspace_root"])
        for k in ("ego_shape", "output_dir"):
            if seed.get(k):
                data[k] = seed[k]
        data = _normalize(data)
    else:
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


# --- workspace scan -------------------------------------------------------------------
# Standard workspace layout the scanner expects.
WORKSPACE_DIRS = {
    "models": "models",
    "loras": "loras",
    "policies": "policies",
    "grpo_configs": "configs/grpo",
    "reward_configs": "configs/reward",
    "maps": "maps",
    "scene_datasets": "datasets/scenes",
    "route_datasets": "datasets/routes",
}


def _is_route_dir(d: Path) -> bool:
    """A contiguous-route dir has per-frame NPZs named <prefix>_<frameidx>.npz."""
    import re

    pat = re.compile(r".+_\d+\.npz$")
    for f in d.glob("*.npz"):
        if pat.match(f.name):
            return True
    return False


def _scan_models(root: Path) -> list[dict]:
    out = []
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        pth = next(iter(d.glob("best_model.pth")), None) or next(
            iter(sorted(d.glob("*.pth"))), None
        )
        if not pth:
            continue
        entry = {"name": d.name, "path": str(pth)}
        for cand in (d / "args.json", d.parent / "args.json"):
            if cand.exists():
                entry["args_json"] = str(cand)
                break
        out.append(entry)
    return out


def _scan_dirs_with(root: Path, marker: str) -> list[dict]:
    """Subdirs containing a marker file (adapter_config.json / exploration_policy_config.json)."""
    if not root.is_dir():
        return []
    return [
        {"name": d.name, "path": str(d)}
        for d in sorted(p for p in root.iterdir() if p.is_dir())
        if (d / marker).exists()
    ]


def _scan_files(root: Path, glob: str) -> list[dict]:
    if not root.is_dir():
        return []
    return [{"name": f.stem, "path": str(f)} for f in sorted(root.glob(glob))]


def _scan_run_loras(runs_root: Path) -> list[dict]:
    """LoRA epoch dirs produced by training runs anywhere under runs/.

    A finished RSFT/GRPO run writes ``lora_epoch_NNN/`` (and ``lora_latest/``) inside its run
    folder, NOT in the top-level ``loras/`` dir — so they are auto-discovered here and named
    ``<run>_epNNN`` (timestamp prefix stripped) so a completed experiment's checkpoints appear
    as selectable LoRAs without any manual registration.
    """
    import re

    if not runs_root.is_dir():
        return []
    ts_re = re.compile(r"^\d{8}-\d{6}_")  # strip the "20260625-120000_" run-folder prefix
    out: list[dict] = []
    seen: set[str] = set()
    dirs = sorted(runs_root.rglob("lora_epoch_*")) + sorted(runs_root.rglob("lora_latest"))
    for d in dirs:
        if not d.is_dir() or not (d / "adapter_config.json").exists():
            continue
        run_name = ts_re.sub("", d.parent.name)
        ep = d.name.replace("lora_epoch_", "ep").replace("lora_latest", "latest")
        base = f"{run_name}_{ep}"
        name, n = base, 2
        while name in seen:
            name, n = f"{base}-{n}", n + 1
        seen.add(name)
        out.append({"name": name, "path": str(d)})
    return out


def _scan_scene_datasets(root: Path) -> list[dict]:
    """Scene datasets = datasets/scenes/*.json lists, PLUS any subfolder of NPZs (e.g. mined
    collision windows or editor saves) — for which we auto-materialize a <name>.json list so it
    is usable by every tool that takes a --scenes JSON.
    """
    if not root.is_dir():
        return []
    out = [{"name": f.stem, "path": str(f)} for f in sorted(root.glob("*.json"))]
    have = {e["name"] for e in out}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name in have:
            continue
        npzs = sorted(str(p) for p in d.rglob("*.npz"))
        if not npzs:
            continue
        listf = root / f"{d.name}.json"
        if not listf.exists():
            with open(listf, "w") as fh:
                json.dump(npzs, fh, indent=2)
        out.append({"name": d.name, "path": str(listf)})
    return out


def create_workspace(root: str | Path) -> str:
    """Create the standard (empty) workspace folder structure at ``root``."""
    root = Path(root).expanduser()
    for sub in WORKSPACE_DIRS.values():
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    return str(root)


def scan_workspace(root: str | Path) -> dict:
    """Walk the standard workspace layout and return a library dict (auto-detected assets).

    Detection by on-disk signature: model = .pth + args.json; LoRA = dir w/ adapter_config.json;
    policy = dir w/ exploration_policy_config.json; reward config = configs/*.json; map = maps/*.osm;
    scene dataset = datasets/scenes/*.json; route dataset = datasets/routes/<dir of *_<idx>.npz>.
    Follows symlinks (so scattered assets can be linked into the workspace).
    """
    root = Path(root).expanduser()
    lib = json.loads(json.dumps(_EMPTY))
    lib["workspace_root"] = str(root)
    if not root.is_dir():
        return lib
    lib["models"] = _scan_models(root / WORKSPACE_DIRS["models"])
    # LoRAs come from the top-level loras/ dir AND from any training run under runs/ (the place
    # RSFT/GRPO actually writes lora_epoch_NNN/). De-dup names across both sources.
    loras = _scan_dirs_with(root / WORKSPACE_DIRS["loras"], "adapter_config.json")
    taken = {e["name"] for e in loras}
    for e in _scan_run_loras(root / "runs"):
        name, n = e["name"], 2
        while name in taken:
            name, n = f"{e['name']}-{n}", n + 1
        e["name"] = name
        taken.add(name)
        loras.append(e)
    lib["loras"] = loras
    lib["policies"] = _scan_dirs_with(
        root / WORKSPACE_DIRS["policies"], "exploration_policy_config.json"
    )
    lib["grpo_configs"] = _scan_files(root / WORKSPACE_DIRS["grpo_configs"], "*.json")
    lib["reward_configs"] = _scan_files(root / WORKSPACE_DIRS["reward_configs"], "*.json")
    lib["maps"] = _scan_files(root / WORKSPACE_DIRS["maps"], "*.osm")
    lib["scene_datasets"] = _scan_scene_datasets(root / WORKSPACE_DIRS["scene_datasets"])
    routes_root = root / WORKSPACE_DIRS["route_datasets"]
    if routes_root.is_dir():
        lib["route_datasets"] = [
            {
                "name": str(d.relative_to(routes_root)),
                "path": str(d),
            }
            for d in sorted(p for p in routes_root.rglob("*") if p.is_dir())
            if _is_route_dir(d)
        ]
    return lib
