"""Helpers for loading ``tag_taxonomy.yaml``.

The taxonomy YAML is documentation only; query/mutate APIs never read
it. Helpers here are for CLI / docs workflows and tolerate malformed
entries silently — the YAML is a human reference, not a contract.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml

from .sidecar import parse_tag

_TAXONOMY_WARNING = (
    "tag_taxonomy.yaml is documentation only; listed tags may not match tags "
    "actually present on a given source. Query/mutate APIs do not use this file."
)


def package_docs_dir() -> Path:
    return Path(__file__).resolve().parent / "docs"


def default_taxonomy_path() -> Path:
    return package_docs_dir() / "tag_taxonomy.yaml"


def load_taxonomy(
    path: str | Path | None = None,
    *,
    warn: bool = True,
) -> dict[str, Any]:
    """Load taxonomy YAML. Malformed entries are dropped silently.

    Args:
        path: Path to the taxonomy YAML. ``None`` (default) loads the
            package-bundled ``docs/tag_taxonomy.yaml``.
        warn: If ``True`` (default) emit a one-time reminder that the
            YAML is informational only. Set to ``False`` for programmatic
            callers that don't want the noise.
    """
    if warn:
        warnings.warn(_TAXONOMY_WARNING, UserWarning, stacklevel=2)
    file_path = Path(path) if path else default_taxonomy_path()
    try:
        data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"taxonomy YAML failed to parse: {file_path}") from exc
    if not isinstance(data, dict):
        if data is None:
            return {}
        raise ValueError(f"taxonomy must be a mapping: {file_path}")
    return data


def list_known_tags(path: str | Path | None = None) -> list[str]:
    """Return ``dimension:value`` strings documented in the taxonomy.

    Open dimensions (those without an explicit ``values`` list) are
    emitted as a placeholder ``<dim>:<open>`` so callers can see which
    dimensions were declared even when their values are not enumerated.
    """
    data = load_taxonomy(path, warn=False)
    dims = data.get("dimensions") or {}
    if not isinstance(dims, dict):
        return []
    out: list[str] = []
    for dim_name, dim_body in dims.items():
        if dim_body is None:
            # Open dimension declared without a body, e.g. ``notes:`` in YAML.
            try:
                parse_tag(f"{dim_name}:placeholder")
            except ValueError:
                continue
            out.append(f"{dim_name}:<open>")
            continue
        if not isinstance(dim_body, dict):
            continue
        try:
            parse_tag(f"{dim_name}:placeholder")
        except ValueError:
            continue
        values = dim_body.get("values")
        if not isinstance(values, list):
            out.append(f"{dim_name}:<open>")
            continue
        for item in values:
            if isinstance(item, dict):
                name = item.get("name")
                if not isinstance(name, str):
                    continue
                tag = f"{dim_name}:{name}"
            elif isinstance(item, str):
                tag = f"{dim_name}:{item}"
            else:
                continue
            try:
                parse_tag(tag)
            except ValueError:
                continue
            out.append(tag)
    return sorted(set(out))
