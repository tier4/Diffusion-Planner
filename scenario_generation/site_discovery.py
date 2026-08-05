"""Group a curated flat JSON path list into per-site npz roots.

A "site" is inferred from the path component immediately before the first recognized split
dir (``_SPLIT_DIR_NAMES``) in each JSON entry, following the existing
``{project}/{area_map_id}_{area_map_name}/{split}/...`` layout convention (e.g.
``x2_dev/2231_odaiba_shinagawa_copied_from_xx1``). Each site's routes are evaluated
independently — callers must never merge multiple sites' npz under one
``--npz_root``/``rglob``, since route grouping is filename-only and ignores directory
structure (see route_timeline.route_prefix); mixing sites risks silently merging unrelated
routes that happen to share a bag-name prefix.

``split`` is ``valid`` for the label-based downloader's original layout, or ``manual``/``auto``
for the PR253 ros_scripts reorg (train/valid -> manual, auto stays auto) — both conventions
coexist across existing datasets, so all three are checked.

The path component one level above the site is the ``project`` -- a data-collection
campaign name, not a vehicle model (several projects can share one vehicle). This module
has no built-in project->vehicle mapping; pass one via ``project_vehicle_map`` in
:func:`discover_sites_with_vehicles_from_json` if needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SPLIT_DIR_NAMES = ("valid", "manual", "auto")


def discover_sites_from_json(json_path: str | Path) -> dict[str, list[Path]]:
    """Group a curated flat JSON path list (the ``--closed_loop_npz_root`` JSON-list
    convention, e.g. ``path_list_closed_loop_x2.json``) into per-site npz roots.

    Site name is the path component immediately before the first recognized split
    dir (see ``_SPLIT_DIR_NAMES``) in each entry. Entries with no recognized split dir are
    skipped. Multiple entries resolving to the same site (e.g. several curated
    date/time roots under one site) are merged into that site's list of roots.

    Thin wrapper around :func:`discover_sites_with_vehicles_from_json` for callers that
    don't need vehicle type.
    """
    sites = discover_sites_with_vehicles_from_json(json_path)
    return {name: info["npz_roots"] for name, info in sites.items()}


def discover_sites_with_vehicles_from_json(
    json_path: str | Path,
    project_vehicle_map: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Like :func:`discover_sites_from_json`, but also resolves each site's ``project`` and,
    via ``project_vehicle_map`` (``{project_code_name: vehicle_type_label}``), its
    ``vehicle_type``. Falls back to the raw project name if the map is omitted or missing
    an entry.

    Returns ``{site_key: {"npz_roots": [Path, ...], "vehicle_type": str, "project": str}}``.
    If one site name appears under two vehicle types, both are kept as separate entries
    (``f"{vehicle_type}__{site}"``) instead of merging their routes.
    """
    project_vehicle_map = project_vehicle_map or {}
    entries = json.loads(Path(json_path).read_text())
    sites: dict[str, dict] = {}
    for entry in entries:
        path = Path(entry)
        parts = path.parts
        for i, part in enumerate(parts):
            if i > 0 and part in _SPLIT_DIR_NAMES:
                site = parts[i - 1]
                project = parts[i - 2] if i >= 2 else "unknown"
                vehicle_type = project_vehicle_map.get(project)
                if vehicle_type is None:
                    vehicle_type = project
                    if project_vehicle_map:
                        print(f"unrecognized project {project!r}, using it as vehicle_type", file=sys.stderr)

                key = site
                existing = sites.get(key)
                if existing is not None and existing["vehicle_type"] != vehicle_type:
                    print(f"site {site!r} has two vehicle types, splitting", file=sys.stderr)
                    sites[f"{existing['vehicle_type']}__{site}"] = sites.pop(key)
                    key = f"{vehicle_type}__{site}"
                elif f"{vehicle_type}__{site}" in sites:
                    key = f"{vehicle_type}__{site}"

                info = sites.setdefault(
                    key,
                    {"npz_roots": [], "vehicle_type": vehicle_type, "project": project},
                )
                info["npz_roots"].append(path)
                break
    return sites
