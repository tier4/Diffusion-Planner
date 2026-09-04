# -*- coding: utf-8 -*-
"""Type definitions for the TagStore package."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Literal, Sequence

from ..routes import extract_frame_number

Clause = str | dict[str, Any]
Granularity = Literal["route", "frame"]
FrameFilter = tuple[int, int] | str | None
Scope = "None | TagStore | Path | Sequence[None | TagStore | Path | Sequence]"


def match_frame_filter(path: Path, frame_filter: FrameFilter) -> bool:
    """Check if a path matches the given frame filter."""
    if frame_filter is None:
        return True
    if isinstance(frame_filter, str):
        return fnmatch(path.name, frame_filter)
    frame_num = extract_frame_number(path)
    if frame_num is None:
        return False
    return frame_filter[0] <= frame_num <= frame_filter[1]


@dataclass
class MutationResult:
    """Result of a mutation operation (add_tags / remove_tags / etc.)."""

    changed: int
    skipped: int
    failed: list[str] = field(default_factory=list)
    first_error: BaseException | None = field(default=None, repr=False)

    def __bool__(self) -> bool:
        """True if any frame was changed."""
        return self.changed > 0


@dataclass
class Bucket:
    """One cell of :meth:`TagStore.group_by`."""

    values: dict[str, str | None]
    members: list[Path] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Number of unique paths in this cell (== len(members))."""
        return len(self.members)

    def label(self, sep: str = " | ") -> str:
        """Join dimension values with *sep*; ``None`` renders as ``-``."""
        return sep.join("-" if v is None else v for v in self.values.values())


@dataclass
class FrameTagDiff:
    """Per-frame tag drift between the index and a sidecar on disk."""

    npz: Path
    index_tags: frozenset[str]
    disk_tags: frozenset[str]


@dataclass
class IndexDiff:
    """Structured report comparing the index to on-disk sidecars."""

    frames_checked: int
    frames_with_tag_diff: int
    orphan_frames: list[Path]
    tags_added: Counter[str]
    tags_removed: Counter[str]
    per_frame: list[FrameTagDiff]

    @property
    def is_consistent(self) -> bool:
        """True when the index exactly matches every existing sidecar."""
        return self.frames_with_tag_diff == 0 and not self.orphan_frames
