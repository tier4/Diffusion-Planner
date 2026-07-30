"""Helpers that load ``tag_taxonomy.yaml`` (not used by query/mutate)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml

_TAXONOMY_WARNING = (
    "tag_taxonomy.yaml is documentation only; listed tags may not match tags "
    "actually present on a given source. Query/mutate APIs do not use this file."
)


def package_docs_dir() -> Path:
    return Path(__file__).resolve().parent / "docs"


def default_taxonomy_path() -> Path:
    return package_docs_dir() / "tag_taxonomy.yaml"


def load_taxonomy(path: str | Path | None = None) -> dict[str, Any]:
    """Load taxonomy YAML and emit a warning that it may not match reality."""
    warnings.warn(_TAXONOMY_WARNING, UserWarning, stacklevel=2)
    file_path = Path(path) if path else default_taxonomy_path()
    data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"taxonomy must be a mapping: {file_path}")
    return data


def list_known_tags(path: str | Path | None = None) -> list[str]:
    """Return ``dimension:value`` strings documented in the taxonomy (open dims omitted)."""
    data = load_taxonomy(path)
    dims = data.get("dimensions") or {}
    out: list[str] = []
    for dim_name, dim_body in dims.items():
        if not isinstance(dim_body, dict):
            continue
        values = dim_body.get("values") or []
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, dict) and "name" in item:
                out.append(f"{dim_name}:{item['name']}")
            elif isinstance(item, str):
                out.append(f"{dim_name}:{item}")
    return sorted(out)
