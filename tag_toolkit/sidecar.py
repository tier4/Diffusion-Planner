"""Parse and rewrite the ``tags`` field on NPZ sidecar JSON files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

TAG_RE = re.compile(r"^([a-z0-9_]+):([a-z0-9_]+)$")


def parse_tag(tag: str) -> tuple[str, str]:
    """``dimension:value`` -> ``(dimension, value)``."""
    match = TAG_RE.fullmatch(tag)
    if not match:
        raise ValueError(
            f"bad tag {tag!r}: expected dimension:value with [a-z0-9_]+ on each side"
        )
    return match.group(1), match.group(2)


def format_tag(dimension: str, value: str) -> str:
    tag = f"{dimension}:{value}"
    parse_tag(tag)
    return tag


def normalize_tags(tags: Iterable[str]) -> list[str]:
    """Validate, dedupe, and sort tag strings."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in tags:
        parse_tag(raw)  # validate
        if raw not in seen:
            seen.add(raw)
            out.append(raw)
    out.sort()
    return out


def sidecar_path(npz: str | Path) -> Path:
    path = Path(npz)
    if path.suffix != ".npz":
        raise ValueError(f"expected .npz path, got {path}")
    return path.with_suffix(".json")


def read_sidecar(npz: str | Path) -> dict:
    """Load sidecar JSON. Missing file -> empty dict."""
    side = sidecar_path(npz)
    if not side.is_file():
        return {}
    data = json.loads(side.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"sidecar must be a JSON object: {side}")
    return data


def read_tags(npz: str | Path) -> list[str]:
    """Return the ``tags`` list for *npz* (missing ≡ ``[]``)."""
    data = read_sidecar(npz)
    tags = data.get("tags", [])
    if tags is None:
        return []
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise ValueError(f"sidecar 'tags' must be a list of strings: {sidecar_path(npz)}")
    return list(tags)


def write_tags(npz: str | Path, tags: Iterable[str], *, create: bool = False) -> list[str]:
    """Write normalized ``tags`` into the sidecar, preserving other fields.

    If the sidecar is missing and *create* is false, raises ``FileNotFoundError``.
    """
    side = sidecar_path(npz)
    if side.is_file():
        data = read_sidecar(npz)
    elif create:
        data = {}
    else:
        raise FileNotFoundError(f"sidecar not found: {side}")
    normalized = normalize_tags(tags)
    data["tags"] = normalized
    side.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return normalized


def tag_values_for(tags: Iterable[str], dimension: str) -> list[str]:
    """Values for *dimension* present in *tags*, sorted unique."""
    values = {parse_tag(t)[1] for t in tags if parse_tag(t)[0] == dimension}
    return sorted(values)


def drop_dimension(tags: Iterable[str], dimension: str) -> list[str]:
    return [t for t in tags if parse_tag(t)[0] != dimension]
