"""tag_toolkit — tags live on each NPZ sidecar; query and mutate by source.
"""

from .routes import route_of
from .sidecar import format_tag, normalize_tags, parse_tag, read_tags, write_tags
from .source import expand_source, load_json
from .store import Bucket, TagStore, format_buckets
from .taxonomy import list_known_tags, load_taxonomy

__version__ = "0.1.0"

__all__ = [
    "Bucket",
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
