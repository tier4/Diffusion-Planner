"""tag_toolkit — tags live on each NPZ sidecar; query and mutate from an index file.

Quick start::

    # Build index once (slow, do once after updating tags)
    from tag_toolkit import TagStore
    TagStore.build_index("/path/to/dataset", "/path/to/tags.tag")

    # Use index (fast, no NPZ reading)
    store = TagStore("/path/to/tags.tag")
    store.query("split:auto")                                # routes with split:auto
    store.add_tags(["override_metric:centerline"])           # add tags
    store.tags_of(granularity="frame")                       # union of all tags

For more examples, see docs/usage.md and docs/design.md.

Note: ``TagStore.add_tags`` and friends update the in-memory index
immediately, but a previously-saved ``.tag`` file on disk is **not**
rewritten. After on-the-fly mutations, call :meth:`TagStore.build_index`
again to re-persist the index — otherwise later sessions will see the
old, pre-mutation state.
"""

from .routes import route_of
from .sidecar import format_tag, normalize_tags, parse_tag, read_tags, write_tags
from .source import expand_source, load_json
from .store import (
    Bucket,
    Clause,
    FrameFilter,
    Granularity,
    Scope,
    ScopeItem,
    TagStore,
    format_buckets,
)
from .taxonomy import list_known_tags, load_taxonomy

__version__ = "0.1.0"

__all__ = [
    "Bucket",
    "Clause",
    "FrameFilter",
    "Granularity",
    "Scope",
    "ScopeItem",
    "TagStore",
    "expand_source",
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
