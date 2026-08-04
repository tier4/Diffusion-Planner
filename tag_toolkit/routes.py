"""Map NPZ paths to route directories.

`dataset/generate_from_labeled.sh` is the golden standard for dataset layout.
Its converter writes each bag under:

- ``…/<project>/<map_id>/{manual|auto}/<date>/<bag_time>/routes/<file>.npz``

So a *route* is the bag directory that owns the frames:

- ``…/<bag_time>/routes/<file>.npz`` → ``…/<bag_time>``
- ``…/<bag_time>/<file>.npz`` (flat layout) → ``…/<bag_time>``

Closed-loop path lists already use these route directories as entries.

Also exposes a small helper for extracting the frame number from a filename
matching the convention ``<bag_time>_<prefix>_<8 digits>.npz``. NPZs that
don't match the convention return ``None``.
"""

from __future__ import annotations

import re
from pathlib import Path

ROUTES_DIR_NAME = "routes"
_FRAME_NUMBER_RE = re.compile(r"_(\d{8})\.npz$")


def route_of(path: str | Path) -> Path:
    """Return the bag/route directory for an NPZ path or route directory itself."""
    p = Path(path)
    if p.suffix == ".npz":
        parent = p.parent
        if parent.name == ROUTES_DIR_NAME:
            return parent.parent
        return parent
    # Already a directory (typical closed-loop / bag path).
    if p.name == ROUTES_DIR_NAME:
        return p.parent
    return p


def extract_frame_number(path: str | Path) -> int | None:
    """Extract frame number from an NPZ filename.

    Filenames follow the pattern: ``<bag_time>_<prefix>_<frame_number>.npz``
    where ``frame_number`` is an 8-digit zero-padded integer.

    Returns the frame number as ``int``, or ``None`` if the filename does
    not match the convention.
    """
    match = _FRAME_NUMBER_RE.search(Path(path).name)
    return int(match.group(1)) if match else None
