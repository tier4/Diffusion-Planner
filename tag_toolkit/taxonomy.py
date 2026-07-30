"""Helpers that load ``tag_taxonomy.yaml`` (not used by query/mutate).

The taxonomy YAML is documentation; query/mutate APIs never read it. Helpers
in this module exist only for CLI / docs workflows, so they should never
crash a user because one entry is malformed — they warn-skip and continue.
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
    """Load taxonomy YAML and (by default) emit a documentation warning.

    Malformed entries (non-mapping dimension bodies, non-list values, items
    that are neither ``str`` nor ``{name: str}``) are skipped with a warning
    rather than raising.

    Args:
        path: Path to the taxonomy YAML. ``None`` (default) loads the
            package-bundled ``docs/tag_taxonomy.yaml``.
        warn: If ``True`` (default) emit a one-time ``UserWarning`` that
            documents the YAML is informational only — handy for CLI users
            who want the reminder, noisy for programmatic callers. Set to
            ``False`` to suppress it (e.g. inside a loop).
    """
    if warn:
        warnings.warn(_TAXONOMY_WARNING, UserWarning, stacklevel=2)
    file_path = Path(path) if path else default_taxonomy_path()
    try:
        data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"taxonomy YAML failed to parse: {file_path}") from exc
    if not isinstance(data, dict):
        # An empty file is acceptable and means "no dimensions documented".
        if data is None:
            return {}
        raise ValueError(f"taxonomy must be a mapping: {file_path}")
    return data


def list_known_tags(path: str | Path | None = None) -> list[str]:
    """Return ``dimension:value`` strings documented in the taxonomy.

    Open dimensions (those without an explicit ``values`` list) are emitted
    as a placeholder ``<dim>:<open>`` so callers can see which dimensions
    were declared even when their values are not enumerated. Malformed
    entries are skipped with a warning.

    Suppresses the package-wide "taxonomy is docs only" warning via
    ``load_taxonomy(..., warn=False)`` because this function is a
    programmatic API and the warning is intended for one-shot CLI use.
    """
    data = load_taxonomy(path, warn=False)
    dims = data.get("dimensions") or {}
    if not isinstance(dims, dict):
        warnings.warn("taxonomy 'dimensions' is not a mapping; returning []", stacklevel=2)
        return []
    out: list[str] = []
    for dim_name, dim_body in dims.items():
        if dim_body is None:
            # Open dimension declared without a body, e.g. ``notes:`` in YAML.
            try:
                parse_tag(f"{dim_name}:placeholder")
            except ValueError:
                warnings.warn(
                    f"taxonomy: skipping open dimension with invalid name {dim_name!r}",
                    stacklevel=2,
                )
                continue
            out.append(f"{dim_name}:<open>")
            continue
        if not isinstance(dim_body, dict):
            warnings.warn(f"taxonomy: skipping non-mapping dimension {dim_name!r}", stacklevel=2)
            continue
        # Skip if the dimension name itself is not a valid tag dimension.
        try:
            parse_tag(f"{dim_name}:placeholder")
        except ValueError:
            warnings.warn(f"taxonomy: skipping dimension with invalid name {dim_name!r}", stacklevel=2)
            continue
        values = dim_body.get("values")
        if values is None:
            # Open dimension: no enumerated values, emit a placeholder.
            out.append(f"{dim_name}:<open>")
            continue
        if not isinstance(values, list):
            warnings.warn(f"taxonomy: 'values' for {dim_name!r} is not a list; skipping", stacklevel=2)
            continue
        for item in values:
            if isinstance(item, dict):
                name = item.get("name")
                if not isinstance(name, str):
                    warnings.warn(
                        f"taxonomy: skipping {dim_name!r} value without string 'name': {item!r}",
                        stacklevel=2,
                    )
                    continue
                tag = f"{dim_name}:{name}"
            elif isinstance(item, str):
                tag = f"{dim_name}:{item}"
            else:
                warnings.warn(
                    f"taxonomy: skipping {dim_name!r} value of unexpected type {type(item).__name__}",
                    stacklevel=2,
                )
                continue
            # Final validation — the doc string itself must round-trip parse_tag.
            try:
                parse_tag(tag)
            except ValueError:
                warnings.warn(f"taxonomy: skipping value that fails parse_tag: {tag!r}", stacklevel=2)
                continue
            out.append(tag)
    return sorted(set(out))