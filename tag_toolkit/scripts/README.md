# `tag_toolkit/scripts/` — CLI scripts

Standalone scripts that wrap `tag_toolkit` for common workflows. Each one
runs directly with `python <script>` and is documented in its own top-level
docstring. This README is a one-line index.

## `tag_management/` — write tags / maintain the index

| Script | Purpose |
|---|---|
| [`incremental_index.py`](tag_management/incremental_index.py) | Append new frames to an existing `.db` index without rescanning old frames; pre-filters duplicates via the `frames` PK. |
| [`write_route_tags_from_csv.py`](tag_management/write_route_tags_from_csv.py) | Apply route-level tags from a CSV column-to-dimension mapping (e.g. `devops_site`, `devops_override_label`) to one dataset tree, batched `sync=False`. |
| [`write_site_split_tags.py`](tag_management/write_site_split_tags.py) | Walk one dataset tree, infer `site:*` and `split:{manual\|auto\|train\|valid}` tags from the directory layout and an optional `split_labels.json`. |

## `tag_usage/` — read tags / build datasets

| Script | Purpose |
|---|---|
| [`export_dataset.py`](tag_usage/export_dataset.py) | Run a tag query, then organise matching frames (or close-loop segments with margin context) into a symlinked output tree; supports `--dimension` sub-dir grouping. |

All scripts adjust `sys.path` so they run without an editable install; see
the `_REPO_ROOT = Path(__file__).resolve().parents[4]` boilerplate at the top
of each file.