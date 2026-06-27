"""Skip-for-training filter for the unified NPZ corpus.

The converter can emit *every* 10 Hz frame (``--write_skipped_npz=1``), including the
ones the production filter would normally drop (red/yellow-light stop, no-future-
progress, collision, off-lane, stale data). Each frame's JSON **sidecar** carries
``is_skipped`` (the ``.npz`` itself has no such field). Those frames exist only so the
closed-loop perception reproducer has a gap-free timeline — they are NOT valid
supervision, so training / eval / data-gen must drop them.

This module is the one shared place that resolves a frame's sidecar and reads the
flag, so every consumer filters identically. It is intentionally dependency-light
(stdlib + tqdm) and lives at the lowest package layer so importers above it have no
circular dependency.

Backward-compatible: a frame with no resolvable sidecar (older corpora, or padded
NPZs whose sidecars live elsewhere and no ``sidecar_root`` was given) is treated as
NOT skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

# stem -> sidecar path index per sidecar_root, built once (one rglob) and reused for every
# scene, so a large scene list doesn't pay an O(files) rglob per entry. Cached for the
# process lifetime (never invalidated) — correct for the batch train/eval CLIs here; a
# long-lived process that writes new sidecars after the first lookup won't see them.
_SIDECAR_INDEX_CACHE: dict[Path, dict[str, Path]] = {}


def is_skipped(npz_path: str | Path, sidecar_root: str | Path | None = None) -> bool:
    """True iff the frame's JSON sidecar marks it skip_for_training.

    Resolves the sidecar and reads the flag in one self-contained pass: sibling
    ``<stem>.json`` first, then ``<stem>.json`` directly under ``sidecar_root``, then by
    stem via one cached ``rglob`` of ``sidecar_root``. A missing/empty path, an unresolved
    or unreadable sidecar, or a sidecar without the field all mean *not skipped* (False),
    which keeps older corpora backward compatible.
    """
    if not npz_path or str(npz_path) in ("", "."):
        return False
    npz_path = Path(npz_path)

    sidecar: Path | None = npz_path.with_suffix(".json")
    if not sidecar.is_file():
        sidecar = None
        if sidecar_root is not None:
            sidecar_root = Path(sidecar_root)
            cand = sidecar_root / f"{npz_path.stem}.json"
            if cand.is_file():
                sidecar = cand
            else:
                key = sidecar_root.resolve()
                index = _SIDECAR_INDEX_CACHE.get(key)
                if index is None:
                    index = {p.stem: p for p in sidecar_root.rglob("*.json")}
                    _SIDECAR_INDEX_CACHE[key] = index
                sidecar = index.get(npz_path.stem)
    if sidecar is None:
        return False
    try:
        return bool(json.loads(sidecar.read_text()).get("is_skipped", False))
    except (OSError, json.JSONDecodeError):
        return False


def filter_scene_list(
    scenes: list,
    sidecar_root: str | Path | None = None,
    enabled: bool = True,
    label: str = "",
) -> list:
    """Drop scenes flagged ``is_skipped`` in their sidecar. ``enabled=False`` returns the
    input unchanged. Entry types are preserved: an entry is a bare npz-path string, or a
    dict keyed ``path`` / ``npz`` / ``scene_path``. Frames with no resolvable sidecar are
    kept (treated as not-skipped)."""
    if not enabled or not scenes:
        return scenes
    kept, dropped = [], 0
    for entry in tqdm(scenes, desc="[skip-filter]", unit="scene"):
        if isinstance(entry, dict):
            path = entry.get("path") or entry.get("npz") or entry.get("scene_path") or ""
        else:
            path = entry if isinstance(entry, str) else ""
        if is_skipped(path, sidecar_root):
            dropped += 1
        else:
            kept.append(entry)
    tag = f"{label}: " if label else ""
    print(f"[skip-filter] {tag}kept {len(kept)}/{len(scenes)} (dropped {dropped} skip_for_training)")
    return kept
