"""Map NPZ paths to route directories.

The dataset has two layouts:

- Closed-loop: ``…/<bag_time>/routes/<file>.npz``
- Flat: ``…/<bag_time>/<file>.npz``

A *route* is the parent directory of the NPZ file:

- ``…/<bag_time>/routes/<file>.npz`` → ``…/<bag_time>/routes``
- ``…/<bag_time>/<file>.npz`` (flat layout) → ``…/<bag_time>``

Also exposes a helper for extracting the frame number from a filename
matching the convention ``<prefix>_<8 digits>.npz``.
"""

from __future__ import annotations

import re
from pathlib import Path

_FRAME_NUMBER_RE = re.compile(r"_(\d{8})\.npz$")


def route_of(path: str | Path) -> Path:
    """Return the directory containing the given path.

    - If input is an NPZ file, returns its parent directory.
    - If input is a directory, returns the directory itself.
    - If input is neither, raises ValueError.
    """
    p = Path(path)
    if p.suffix.lower() == ".npz":
        return p.parent
    if p.is_dir():
        return p
    raise ValueError(f"Expected an NPZ file or directory, got: {path}")


def extract_frame_number(path: str | Path) -> int | None:
    """Extract frame number from an NPZ filename.

    Filenames follow the pattern: ``<prefix>_<frame_number>.npz``
    where ``frame_number`` is an 8-digit zero-padded integer.

    Returns the frame number as ``int``, or ``None`` if the filename does
    not match the convention.
    """
    match = _FRAME_NUMBER_RE.search(Path(path).name)
    return int(match.group(1)) if match else None
