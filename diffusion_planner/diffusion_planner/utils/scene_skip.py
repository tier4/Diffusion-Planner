"""Skip-for-training filter for the unified NPZ corpus.

The converter can emit *every* 10 Hz frame (``--write_skipped_npz=1``), including the
ones the production filter would normally drop (red/yellow-light stop, no-future-
progress, collision, off-lane, stale data). Each frame's JSON **sidecar** carries
``is_skipped`` (the ``.npz`` itself has no such field). Those frames exist only so the
closed-loop perception reproducer has a gap-free timeline — they are NOT valid
supervision, so training / eval / data-gen must drop them.

This module is the one shared place that resolves a frame's sidecar and reads the
flag, so every consumer filters identically. It is intentionally dependency-light
(stdlib only) and lives at the lowest package layer so the ``DiffusionPlannerData``
loader and everything above it can import it without a circular dependency.

Backward-compatible: a frame with no resolvable sidecar (older corpora, or padded
NPZs whose sidecars live elsewhere and no ``sidecar_root`` was given) is treated as
NOT skipped; ``filter_scene_list`` loudly logs how many were unresolved so a silent
no-op is visible.
"""

from __future__ import annotations

import json
from pathlib import Path

# stem -> sidecar path index per sidecar_root, built once (one rglob) and reused for
# every scene, so a large scene list doesn't pay an O(files) rglob per entry.
_SIDECAR_INDEX_CACHE: dict[Path, dict[str, Path]] = {}


def _sidecar_index(sidecar_root: Path) -> dict[str, Path]:
    key = sidecar_root.resolve()
    idx = _SIDECAR_INDEX_CACHE.get(key)
    if idx is None:
        idx = {p.stem: p for p in sidecar_root.rglob("*.json")}
        _SIDECAR_INDEX_CACHE[key] = idx
    return idx


def scene_entry_path(entry) -> str:
    """The npz path of a scene-list entry: a bare string, or a dict keyed
    path / npz / scene_path (the forms used across the scene-list tools)."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("path") or entry.get("npz") or entry.get("scene_path") or ""
    return ""


def resolve_sidecar(npz_path: str | Path, sidecar_root: str | Path | None = None) -> Path | None:
    """Locate a frame's pose/skip JSON sidecar: sibling ``.json`` first, then by stem
    under ``sidecar_root`` (one cached rglob). ``None`` if not found (no raise)."""
    npz_path = Path(npz_path)
    sib = npz_path.with_suffix(".json")
    if sib.is_file():
        return sib
    if sidecar_root is not None:
        sidecar_root = Path(sidecar_root)
        cand = sidecar_root / f"{npz_path.stem}.json"
        if cand.is_file():
            return cand
        return _sidecar_index(sidecar_root).get(npz_path.stem)
    return None


def _sidecar_state(npz_path: str | Path, sidecar_root) -> tuple[bool, bool]:
    """(is_skipped, sidecar_resolved). Missing sidecar/field => (False, resolved?)."""
    sc = resolve_sidecar(npz_path, sidecar_root)
    if sc is None:
        return False, False
    try:
        d = json.loads(sc.read_text())
    except (OSError, json.JSONDecodeError):
        return False, False
    return bool(d.get("is_skipped", False)), True


def is_skipped(npz_path: str | Path, sidecar_root: str | Path | None = None) -> bool:
    """True iff the frame's sidecar marks it skip_for_training (missing => False)."""
    return _sidecar_state(npz_path, sidecar_root)[0]


def filter_scene_list(
    scenes: list,
    sidecar_root: str | Path | None = None,
    enabled: bool = True,
    label: str = "",
) -> list:
    """Drop scenes flagged ``is_skipped`` in their sidecar. ``enabled=False`` returns the
    input unchanged. Entry types (str / dict) are preserved. Frames with no resolvable
    sidecar are kept (treated as not-skipped) and counted in a single loud log line."""
    if not enabled or not scenes:
        return scenes
    kept, dropped, no_sidecar = [], 0, 0
    for entry in scenes:
        skipped, resolved = _sidecar_state(scene_entry_path(entry), sidecar_root)
        if not resolved:
            no_sidecar += 1
        if skipped:
            dropped += 1
        else:
            kept.append(entry)
    tag = f"{label}: " if label else ""
    print(
        f"[skip-filter] {tag}kept {len(kept)}/{len(scenes)} "
        f"(dropped {dropped} skip_for_training; {no_sidecar} no-sidecar->kept)"
    )
    return kept


def load_scene_list(
    path_or_dir: str | Path,
    sidecar_root: str | Path | None = None,
    skip_filter: bool = True,
) -> list:
    """Read a scene-list JSON (list, or ``{"files": [...]}``) or glob a dir of ``*.npz``,
    then drop skip_for_training frames by default. Set ``skip_filter=False`` to keep them
    (the reproducer's opt-out)."""
    p = Path(path_or_dir)
    if p.is_dir():
        scenes: list = sorted(str(f) for f in p.rglob("*.npz"))
    else:
        data = json.loads(p.read_text())
        scenes = data["files"] if isinstance(data, dict) else data
    return filter_scene_list(scenes, sidecar_root, enabled=skip_filter, label=str(path_or_dir))
