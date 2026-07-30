"""Map NPZ paths to route directories.

`dataset/generate_from_labeled.sh` is the golden standard for dataset layout.
Its converter writes each bag under:

- ``…/<project>/<map_id>/{manual|auto}/<date>/<bag_time>/routes/<file>.npz``

So a *route* is the bag directory that owns the frames:

- ``…/<bag_time>/routes/<file>.npz`` → ``…/<bag_time>``
- ``…/<bag_time>/<file>.npz`` (flat layout) → ``…/<bag_time>``

Closed-loop path lists already use these route directories as entries.
"""

from __future__ import annotations

from pathlib import Path

ROUTES_DIR_NAME = "routes"


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
