"""Ego vehicle dimensions for scoring native H5 with the shared NPZ-era code.

Native H5 frames store ``ego_shape`` as ``[base_link_to_front, length, width]``: the
ML-Planner converter writes ``ego_wheel_base + ego_front_overhang`` there, and the planner
node builds the same value from the vehicle info. The code shared with the NPZ evaluation
reads ``ego_shape`` as ``[wheelbase, length, width]``, the NPZ converter's layout: in the
closed-loop reproducer the MPC tracker, the road-border and object boxes and the renderer,
and in open-loop the ego box of ``planner_metrics`` (object avoidance). Passing the native
value through would simulate a longer wheelbase and shift the footprint forward (by about
1.3 m for Erga), so those consumers get ``sim_ego_shape`` instead. The model input keeps the
native value.

The wheelbase cannot be recovered from an H5 shard, so it is read from the per-project
table that both converters are built from (``dataset/data_converter_param.json`` of the
meta repository).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# new_dp_h5_eval/ -> Diffusion-Planner/ -> meta repository root
DEFAULT_VEHICLE_PARAMS = (
    Path(__file__).resolve().parents[2] / "dataset" / "data_converter_param.json"
)
DEFAULT_PROJECT_VEHICLE_MAP = DEFAULT_VEHICLE_PARAMS.with_name("project_vehicle_map.json")
_REQUIRED_KEYS = ("ego_wheel_base", "ego_front_overhang", "ego_length", "ego_width")
# float32 storage in the H5 rounds the table values by well under a millimetre.
_SHAPE_ATOL_M = 1e-3


@dataclass(frozen=True)
class VehicleDimensions:
    wheel_base: float
    front_overhang: float
    length: float
    width: float

    @property
    def base_link_to_front(self) -> float:
        return self.wheel_base + self.front_overhang


def load_vehicle_params(path: str | Path) -> dict[str, VehicleDimensions]:
    """Read ``{project_id: VehicleDimensions}`` from a converter parameter json.

    Entries without all four ego keys (for example the conversion defaults) are skipped.
    """
    path = Path(path).expanduser()
    entries = json.loads(path.read_text())
    if not isinstance(entries, dict):
        raise ValueError(f"{path}: expected a json object of project ids")
    vehicles = {}
    for project, values in entries.items():
        if not isinstance(values, dict) or not all(key in values for key in _REQUIRED_KEYS):
            continue
        try:
            vehicles[str(project)] = VehicleDimensions(
                wheel_base=float(values["ego_wheel_base"]),
                front_overhang=float(values["ego_front_overhang"]),
                length=float(values["ego_length"]),
                width=float(values["ego_width"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: project {project!r} has a non-numeric ego dimension"
            ) from exc
    if not vehicles:
        raise ValueError(f"{path}: no project entry has all of {', '.join(_REQUIRED_KEYS)}")
    return vehicles


def add_vehicle_params_argument(parser) -> None:
    parser.add_argument(
        "--vehicle_params",
        type=Path,
        default=DEFAULT_VEHICLE_PARAMS,
        help="Per-project ego dimensions (ego_wheel_base, ego_front_overhang, ego_length, "
        "ego_width), the json both dataset converters read. Ego geometry is scored with the "
        "wheelbase, which native H5 does not store (default: the meta repository's "
        "dataset/data_converter_param.json).",
    )


def vehicle_params_from_args(parser, args) -> dict[str, VehicleDimensions]:
    """Load ``--vehicle_params`` or exit through ``parser.error`` with the reason."""
    if not args.vehicle_params.is_file():
        parser.error(
            f"--vehicle_params {args.vehicle_params} does not exist; pass the meta repository's "
            "dataset/data_converter_param.json"
        )
    try:
        return load_vehicle_params(args.vehicle_params)
    except (OSError, ValueError) as exc:
        parser.error(f"cannot read --vehicle_params {args.vehicle_params}: {exc}")


def add_project_vehicle_map_argument(parser) -> None:
    parser.add_argument(
        "--project_vehicle_map",
        type=Path,
        default=DEFAULT_PROJECT_VEHICLE_MAP,
        help="{project_id: vehicle}, which picks each route's directory when "
        "--plan_optimizer_config_dir holds one per vehicle (default: the meta repository's "
        "dataset/project_vehicle_map.json).",
    )


def load_project_vehicle_map(path: str | Path) -> dict[str, str]:
    path = Path(path).expanduser()
    entries = json.loads(path.read_text())
    if not isinstance(entries, dict) or not all(isinstance(v, str) for v in entries.values()):
        raise ValueError(f"{path}: expected a json object of project id -> vehicle name")
    return {str(project): vehicle for project, vehicle in entries.items()}


def sim_ego_shape(
    native_shapes: np.ndarray,
    project_id: str,
    vehicles: dict[str, VehicleDimensions],
    *,
    source: str,
) -> np.ndarray:
    """Return ``[wheelbase, length, width]`` for native ``[base_link_to_front, length, width]``.

    ``native_shapes`` holds the shard's per-frame ``ego_shape`` rows. Every row must match the
    project's table entry, so a shard converted with other dimensions fails here instead of
    being scored with a wrong footprint.
    """
    vehicle = vehicles.get(project_id)
    if vehicle is None:
        raise ValueError(
            f"{source}: project_id {project_id!r} has no ego dimensions in the vehicle params "
            f"(known: {', '.join(sorted(vehicles))}); pass --vehicle_params with an entry for it"
        )
    rows = np.asarray(native_shapes, dtype=np.float64).reshape(-1, 3)
    expected = np.array([vehicle.base_link_to_front, vehicle.length, vehicle.width])
    bad = ~np.all(np.abs(rows - expected) <= _SHAPE_ATOL_M, axis=1)
    if bad.any():
        first = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"{source}: ego_shape row {first} is {rows[first].tolist()}, but project "
            f"{project_id!r} has [base_link_to_front, length, width] = {expected.tolist()} in the "
            f"vehicle params; the shard and the vehicle params disagree"
        )
    return np.array([vehicle.wheel_base, vehicle.length, vehicle.width], dtype=np.float32)
