"""tag_toolkit — tags live on each NPZ sidecar; query and mutate via SQLite.

Quick start::

    # Open or create a SQLite index
    from tag_toolkit import TagStore
    store = TagStore("/path/to/dataset.tags.db")   # persisted
    # or:
    store = TagStore("/path/to/dataset/")     # in-memory, mutations stay in memory

    # Build index from sidecars
    store.rebuild_index("/path/to/dataset/")

    # Query
    store.query("split:auto")                              # routes with split:auto
    store.add_tags(["override_metric:centerline"])          # add tags
    store.tags_of(granularity="frame")                    # union of all tags

For more examples, see docs/usage.md and docs/design.md.
"""

from .routes import extract_frame_number, route_of
from .sidecar import StaleIndexError, format_tag, normalize_tags, parse_tag, read_tags, write_tags
from .source import expand_source, load_json
from .store import (
    Bucket,
    FrameTagDiff,
    IndexDiff,
    MutationResult,
    TagStore,
    format_buckets,
)
from .taxonomy import list_known_tags, load_taxonomy

__version__ = "0.1.0"

__all__ = [
    "Bucket",
    "FrameTagDiff",
    "IndexDiff",
    "MutationResult",
    "StaleIndexError",
    "TagStore",
    "expand_source",
    "extract_frame_number",
    "format_buckets",
    "format_tag",
    "list_known_tags",
    "load_json",
    "load_taxonomy",
    "normalize_tags",
    "parse_tag",
    "read_tags",
    "route_of",
    "write_tags",
]
