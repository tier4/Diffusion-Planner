"""Group a curated flat JSON path list into per-site npz roots.

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
from pathlib import Path

_SPLIT_DIR_NAMES = ("valid", "manual", "auto")


def discover_sites_from_json(json_path: str | Path) -> dict[str, list[Path]]:
    """Group a curated flat JSON path list (the ``--closed_loop_npz_root`` JSON-list
    convention, e.g. ``path_list_closed_loop.json``) into per-site npz roots.

    Site name is the path component immediately before the first recognized split
    dir (see ``_SPLIT_DIR_NAMES``) in each entry. Entries with no recognized split dir are
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

    # Pass 1: resolve (site, project, vehicle_type) per entry. Projects missing from the map
    # are collected rather than warned about here -- a curated manifest holds one entry per
    # route dir, so warning inline would repeat the same line thousands of times.
    parsed: list[tuple[str, str, str, Path]] = []
    unmapped_projects: set[str] = set()
    for entry in entries:
        path = Path(entry)
        parts = path.parts
        for i, part in enumerate(parts):
            if i > 0 and part in _SPLIT_DIR_NAMES:
                site = parts[i - 1]
                project = parts[i - 2] if i >= 2 else "unknown"
                vehicle_type = project_vehicle_map.get(project)
                if vehicle_type is None:
                    # No map -> unlabelled. Map supplied but this project is not in it ->
                    # fall back to the project name so the site is still distinguishable.
                    vehicle_type = project if labelling else ""
                    if labelling:
                        unmapped_projects.add(project)
                parsed.append((site, project, vehicle_type, path))
                break
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
    for site, project, vehicle_type, path in parsed:
        group = by_site.setdefault(site, {}).setdefault(
            project, {"npz_roots": [], "project": project, "vehicle_type": vehicle_type}
        )
        group["npz_roots"].append(path)

    sites: dict[str, dict] = {}
    for site, groups in by_site.items():
        if len(groups) == 1:
            sites[site] = next(iter(groups.values()))
        else:
            print(f"site {site!r} spans projects {sorted(groups)}, splitting", file=sys.stderr)
            for project, group in groups.items():
                sites[f"{project}__{site}"] = group
    return sites
