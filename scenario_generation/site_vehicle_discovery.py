"""Group a curated flat JSON path list into site and vehicle groups for closed-loop evaluation.

A "site" is inferred from the path component immediately before the first recognized split
dir (``_SPLIT_DIR_NAMES``) in each JSON entry, following the existing
``{project}/{area_map_id}_{area_map_name}/{split}/...`` layout convention. Each site's routes
are evaluated independently — callers must never merge multiple sites' npz under one
``--npz_root``/``rglob``, since route grouping is filename-only and ignores directory
structure (see route_timeline.route_prefix); mixing sites risks silently merging unrelated
routes that happen to share a bag-name prefix.

``split`` is ``valid`` for the label-based downloader's original layout, or ``manual``/``auto``
for the PR253 ros_scripts reorg (train/valid -> manual, auto stays auto) — both conventions
coexist across existing datasets, so all three are checked.

The path component one level above the site is the ``project`` -- a data-collection
campaign name, not a vehicle model (several projects can share one vehicle). This module
has no built-in project->vehicle mapping; pass one via ``project_vehicle_map`` in
:func:`discover_sites_with_vehicles_from_json` if needed. The JSON file it is loaded from is
supplied at runtime and is not tracked in this repository (``*project_vehicle_map*.json`` is
gitignored).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

_SPLIT_DIR_NAMES = ("valid", "manual", "auto", "train", "test", "override")


@dataclass
class NPZPathInfo:
    """Structured fields parsed from a single NPZ path or route directory.

    ``split`` falls back to ``"unknown"`` when no recognised split dir is found, so the caller
    can still write the tag rather than silently dropping the frame.
    """

    project: str
    site: str
    split: str
    date: str
    bag_time: str

    def route_key(self, include_split: bool = False) -> str:
        """Route key matching the format used by ``split_labels.json``.

        The ``include_split`` variant matches labels files that embed the split token
        (e.g. ``manual``) in the key; the plain variant is preferred when available.
        """
        if include_split:
            return f"{self.project}/{self.site}/{self.split}/{self.date}/{self.bag_time}"
        return f"{self.project}/{self.site}/{self.date}/{self.bag_time}"


def parse_npz_path(path: str | Path) -> NPZPathInfo | None:
    """Parse a single NPZ path or route directory into structured fields.

    Supported input forms::

        .../{project}/{site}/{split}/{date}/{bag_time}/routes/frame.npz
        .../{project}/{site}/{split}/{date}/{bag_time}/routes          ← routes/ dir
        .../{project}/{site}/{split}/{date}/{bag_time}                ← bag_time (no routes/)

    Returns ``None`` if the path has no recognised split dir; the caller decides
    whether to skip or to record ``"unknown"`` for the missing fields.
    """
    p = Path(path)
    if p.is_file() and p.suffix == ".npz":
        p = p.parent  # .../routes
    if p.name == "routes":
        p = p.parent  # .../bag_time

    parts = p.parts
    split_idx = None
    for i, part in enumerate(parts):
        if part in _SPLIT_DIR_NAMES:
            split_idx = i
            break

    if split_idx is None:
        return None

    split = parts[split_idx]
    site = parts[split_idx - 1] if split_idx > 0 else "unknown"
    project = parts[split_idx - 2] if split_idx > 1 else "unknown"

    # The date dir is one level below split; the bag_time dir is two levels below split.
    date = parts[split_idx + 1] if split_idx + 1 < len(parts) else "unknown"
    bag_time = parts[split_idx + 2] if split_idx + 2 < len(parts) else "unknown"

    return NPZPathInfo(
        project=project,
        site=site,
        split=split,
        date=date,
        bag_time=bag_time,
    )


def discover_sites_from_json(json_path: str | Path) -> dict[str, list[Path]]:
    """Group a curated flat JSON path list (the ``--closed_loop_npz_root`` JSON-list
    convention, e.g. ``path_list_closed_loop.json``) into per-site npz roots.

    Site name is the path component immediately before the first recognised split
    dir (see ``_SPLIT_DIR_NAMES``) in each entry. Entries with no recognised split dir are
    skipped. Grouping is keyed on ``(project, site)`` -- never on vehicle type -- so entries
    merge only within one project and a site name shared across two projects stays split
    (see :func:`discover_sites_with_vehicles_from_json`).

    Thin wrapper around :func:`discover_sites_with_vehicles_from_json` that drops the
    vehicle_type/project info.
    """
    sites = discover_sites_with_vehicles_from_json(json_path)
    return {name: info["npz_roots"] for name, info in sites.items()}


def discover_sites_with_vehicles_from_json(
    json_path: str | Path,
    project_vehicle_map: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Like :func:`discover_sites_from_json`, but also resolves each site's ``project`` and,
    via ``project_vehicle_map`` (``{project_code_name: vehicle_type_label}``), its
    ``vehicle_type``.

    With no map at all, ``vehicle_type`` is left ``""`` -- a project is a data-collection
    campaign name and not a vehicle model, so labelling sites with it would report something
    the caller never asked for. Only a project missing from a map that *was* supplied falls
    back to the raw project name, with a warning. Downstream consumers already treat ``""``
    as "unlabelled" (the report hides the column, the W&B rollup omits the per-vehicle keys).

    Returns ``{site_key: {"npz_roots": [Path, ...], "vehicle_type": str, "project": str}}``.

    Grouping is keyed on ``(project, site)``; ``vehicle_type`` is a label only and never
    merges roots. Several projects can share one vehicle, so keying on the vehicle would pool
    two projects' routes under one site key and leave ``project`` describing only whichever
    was seen first. If one site name appears under N projects, all N are kept as separate
    entries (``f"{project}__{site}"``) instead of merging their routes.
    """
    labelling = project_vehicle_map is not None
    project_vehicle_map = project_vehicle_map or {}
    entries = json.loads(Path(json_path).read_text())

    # Pass 1: resolve (site, project, vehicle_type) per entry.
    # Projects missing from the map are collected rather than warned about here --
    # a curated manifest holds one entry per route dir, so warning inline would repeat
    # the same line thousands of times.
    parsed: list[tuple[str, str, str, Path]] = []
    unmapped_projects: set[str] = set()
    for entry in entries:
        info = parse_npz_path(entry)
        if info is None:
            continue
        project = info.project
        vehicle_type = project_vehicle_map.get(project)
        if vehicle_type is None:
            # No map -> unlabelled. Map supplied but this project is not in it ->
            # fall back to the project name so the site is still distinguishable.
            vehicle_type = project if labelling else ""
            if labelling:
                unmapped_projects.add(project)
        parsed.append((info.site, project, vehicle_type, Path(entry)))
    if unmapped_projects:
        print(
            f"unrecognized project(s) {sorted(unmapped_projects)}, using them as vehicle_type",
            file=sys.stderr,
        )

    # Pass 2: group by site, then by project -- never by vehicle_type, since several projects
    # can map to one vehicle and pooling their roots would put both projects' routes under a
    # single ``project``. Grouping in two steps (rather than straight on the pair) means a
    # collision between any number of projects is seen in full before deciding site keys.
    by_site: dict[str, dict[str, dict]] = {}
    for site, project, vehicle_type, original_path in parsed:
        group = by_site.setdefault(site, {}).setdefault(
            project, {"npz_roots": [], "project": project, "vehicle_type": vehicle_type}
        )
        group["npz_roots"].append(original_path)

    sites: dict[str, dict] = {}
    for site, groups in by_site.items():
        if len(groups) == 1:
            sites[site] = next(iter(groups.values()))
        else:
            print(f"site {site!r} spans projects {sorted(groups)}, splitting", file=sys.stderr)
            for project, group in groups.items():
                sites[f"{project}__{site}"] = group
    return sites


if __name__ == "__main__":
    import argparse
    from collections import defaultdict

    parser = argparse.ArgumentParser(
        description="Discover site and vehicle groups from raw data. "
        "Outputs sites.json by default; also outputs vehicles.json when --project_vehicle_map is given."
    )
    parser.add_argument("--input", "-i", required=True, help="Flat JSON path list input")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for JSON files")
    parser.add_argument(
        "--project_vehicle_map",
        default=None,
        help="Path to JSON file mapping project names to vehicle types "
        "(e.g. {'project_a': 'vehicle_a'}). When provided, also outputs vehicles.json.",
    )
    args = parser.parse_args()

    # Load project->vehicle map if provided
    project_vehicle_map = None
    if args.project_vehicle_map:
        project_vehicle_map = json.loads(Path(args.project_vehicle_map).read_text())

    # Always output sites.json
    sites = discover_sites_with_vehicles_from_json(args.input, project_vehicle_map)
    sites_out = {name: [str(p) for p in info["npz_roots"]] for name, info in sites.items()}
    sites_path = Path(args.output_dir) / "sites.json"
    sites_path.parent.mkdir(parents=True, exist_ok=True)
    sites_path.write_text(json.dumps(sites_out, indent=2))
    print(f"Wrote {len(sites_out)} sites to {sites_path}")

    # Output vehicles.json only when project_vehicle_map is provided
    if project_vehicle_map:
        by_vehicle: dict[str, list[str]] = defaultdict(list)
        for name, info in sites.items():
            vehicle = info["vehicle_type"] or "unknown"
            by_vehicle[vehicle].extend([str(p) for p in info["npz_roots"]])
        vehicles_out = dict(by_vehicle)
        vehicles_path = Path(args.output_dir) / "vehicles.json"
        vehicles_path.write_text(json.dumps(vehicles_out, indent=2))
        print(f"Wrote {len(vehicles_out)} vehicles to {vehicles_path}")
