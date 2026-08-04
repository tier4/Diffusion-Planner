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
"""

from __future__ import annotations

import json
from pathlib import Path

_SPLIT_DIR_NAMES = ("valid", "manual", "auto")


def discover_sites_from_json(json_path: str | Path) -> dict[str, list[Path]]:
    """Group a curated flat JSON path list (the ``--closed_loop_npz_root`` JSON-list
    convention, e.g. ``path_list_closed_loop_x2.json``) into per-site npz roots.

    Site name is the path component immediately before the first recognized split
    dir (see ``_SPLIT_DIR_NAMES``) in each entry. Entries with no recognized split dir are
    skipped. Multiple entries resolving to the same site (e.g. several curated
    date/time roots under one site) are merged into that site's list of roots.
    """
    entries = json.loads(Path(json_path).read_text())
    sites: dict[str, list[Path]] = {}
    for entry in entries:
        path = Path(entry)
        parts = path.parts
        for i, part in enumerate(parts):
            if i > 0 and part in _SPLIT_DIR_NAMES:
                sites.setdefault(parts[i - 1], []).append(path)
                break
    return sites
