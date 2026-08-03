#!/usr/bin/env python3
"""Write route-level tags from a CSV file using TagStore.

This script reads a CSV file containing tag data and applies tags to routes
in a dataset. Tags are written to NPZ sidecar JSON files.

IMPORTANT: This script operates at the ROUTE level. Each route's tags are
stored in the NPZ sidecars belonging to that route.

Usage::

    python write_route_tags_from_csv.py \\
        /path/to/dataset \\
        /path/to/mapping.csv \\
        --match-col t4_dataset_id \\
        --tag-dimensions devops_site devops_override_label \\
        [--dry-run] \\
        [--index /path/to/output.tag]

Example CSV (scenario_t4_mapping.csv)::

    t4_dataset_id,devops_override_label,devops_site
    ba98f1fc-8adb-491d-b0e6-314b4a00849d,obstacle_stop,odaiba
    cf60a9fa-050e-4f07-85a6-69b34a60c548,traffic_light,odaiba
    ...

Supported --match-col values:
    - t4_dataset_id: matches against the t4_dataset_id field in route sidecar JSON

The --tag-dimensions specifies which columns from the CSV to apply as tags.
Each specified column becomes a tag dimension with its cell value as the
tag value (format: "dimension:value").
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

# Add Diffusion-Planner to path for tag_toolkit
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TAG_TOOLKIT_PATH = _REPO_ROOT / "Diffusion-Planner" / "tag_toolkit"
if _TAG_TOOLKIT_PATH.exists():
    sys.path.insert(0, str(_TAG_TOOLKIT_PATH.parent))

from tag_toolkit import TagStore
from tag_toolkit.sidecar import format_tag, is_valid_dimension, read_sidecar


# =============================================================================
# match_col dispatch mechanism
# =============================================================================

# Supported match_col values
SUPPORTED_MATCH_COLS = {"t4_dataset_id"}


def _normalize_for_tag(value: str) -> str:
    """Normalize a value for use in tags (dimension or value).

    Ensures lowercase and no spaces, as required by format_tag.
    """
    return value.strip().lower().replace(" ", "_")


def load_csv_index(csv_path: Path, match_col: str) -> dict[str, dict]:
    """Load CSV and index by match_col for O(1) lookup.

    Args:
        csv_path: Path to the CSV file
        match_col: Column name to use as the index key

    Returns:
        Dict mapping match_col value -> CSV row dict

    Raises:
        FileNotFoundError: if csv_path does not exist
        ValueError: if match_col is not in CSV header
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    # Validate match_col exists
    if match_col not in rows[0]:
        available = list(rows[0].keys())
        raise ValueError(
            f"match_col '{match_col}' not found in CSV. Available columns: {available}"
        )

    index: dict[str, dict] = {}
    for row in rows:
        key = row.get(match_col, "").strip()
        if key:
            index[key] = row

    return index


# Supported match_col values
SUPPORTED_MATCH_COLS = {"t4_dataset_id"}


def _get_field_from_sidecar(
    store: TagStore, route: Path, field_name: str
) -> str | None:
    """Extract a field from a route's NPZ sidecar.

    Reads the first NPZ in the route and extracts the specified field.

    Args:
        store: TagStore instance
        route: Route path
        field_name: Field name to extract from sidecar

    Returns:
        Field value, or None if not found
    """
    idx = store._require_index()
    frames = idx.frames_of_route.get(route)
    if not frames:
        return None
    first_npz = frames[0]
    sidecar_data = read_sidecar(first_npz)
    return sidecar_data.get(field_name)


# Registry of match value getters
# To add a new match_col, implement a getter function and register it here
MATCH_VALUE_GETTERS = {
    "t4_dataset_id": lambda store, route: _get_field_from_sidecar(store, route, "t4_dataset_id"),
}


def _get_match_value(store: TagStore, route: Path, match_col: str) -> str | None:
    """Dispatch to the appropriate getter based on match_col.

    Args:
        store: TagStore instance
        route: Route path to extract value from
        match_col: The match column type (e.g., 't4_dataset_id')

    Returns:
        The match value, or None if not found

    Raises:
        ValueError: If match_col is not supported
    """
    if match_col not in SUPPORTED_MATCH_COLS:
        raise ValueError(
            f"Unsupported match_col: '{match_col}'. Supported: {sorted(SUPPORTED_MATCH_COLS)}"
        )
    getter = MATCH_VALUE_GETTERS[match_col]
    return getter(store, route)


def apply_csv_tags_to_routes(
    store: TagStore,
    csv_path: Path,
    *,
    match_col: str,
    tag_dimensions: Sequence[str],
    dry_run: bool = False,
) -> dict[str, int]:
    """Apply tags from CSV to routes in a TagStore.

    Matches routes by reading match_col from their NPZ sidecar JSON files,
    then applies tags from the matching CSV row.

    Args:
        store: TagStore instance with an initialized index
        csv_path: Path to the CSV file containing tag data
        match_col: Column name in CSV to match against (read from route sidecar)
        tag_dimensions: Column names from CSV to apply as tag dimensions
        dry_run: If True, only print what would be done without making changes

    Returns:
        Dict with statistics:
            - total_routes: total routes in the store
            - matched_routes: routes that matched a CSV row
            - tagged_files: files actually written (excludes dry_run)
            - unmatched_csv_rows: CSV rows with no matching route
            - errors: number of errors encountered

    Raises:
        FileNotFoundError: if csv_path does not exist
        ValueError: if match_col or any tag_dimension is not in CSV header
    """
    csv_index = load_csv_index(csv_path, match_col)

    # Validate tag_dimensions against CSV
    if not csv_index:
        return {
            "total_routes": 0,
            "matched_routes": 0,
            "tagged_files": 0,
            "unmatched_csv_rows": 0,
            "errors": 0,
        }

    sample_row = next(iter(csv_index.values()))
    for dim in tag_dimensions:
        if dim not in sample_row:
            available = list(sample_row.keys())
            raise ValueError(
                f"tag_dimension '{dim}' not found in CSV. Available columns: {available}"
            )

    idx = store._require_index()
    routes = idx.routes
    total_routes = len(routes)
    matched_routes = 0
    tagged_files = 0
    unmatched_csv_keys = set(csv_index.keys())
    errors = 0

    for route in routes:
        # Get match value from route's sidecar
        match_value = _get_match_value(store, route, match_col)
        if not match_value:
            continue

        # Look up in CSV index
        csv_row = csv_index.get(match_value)
        if csv_row is None:
            continue

        matched_routes += 1
        unmatched_csv_keys.discard(match_value)

        # Build tag list from CSV
        tags_to_add: list[str] = []
        for dim in tag_dimensions:
            raw_value = csv_row.get(dim, "").strip()
            if not raw_value:
                continue

            # Normalize dimension and value
            norm_dim = _normalize_for_tag(dim)
            norm_value = _normalize_for_tag(raw_value)

            # Validate dimension format
            if not is_valid_dimension(norm_dim):
                print(f"  Warning: skipping invalid dimension '{norm_dim}' from column '{dim}'")
                continue

            try:
                tags_to_add.append(format_tag(norm_dim, norm_value))
            except ValueError as e:
                print(f"  Warning: skipping invalid tag '{norm_dim}:{norm_value}': {e}")
                continue

        if not tags_to_add:
            continue

        # Apply tags
        if dry_run:
            tags_str = ", ".join(tags_to_add)
            print(f"  [DRY-RUN] Would add [{tags_str}] to {route.name}")
        else:
            try:
                store.add_tags_to_route(tags_to_add, route)
                tagged_files += 1
            except Exception as e:
                print(f"  Error: Failed to update {route}: {e}")
                errors += 1

    return {
        "total_routes": total_routes,
        "matched_routes": matched_routes,
        "tagged_files": tagged_files,
        "unmatched_csv_rows": len(unmatched_csv_keys),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write route-level tags from a CSV file using TagStore.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Data source: directory, .tag index, NPZ file, or path-list JSON",
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="CSV file containing tag data",
    )
    parser.add_argument(
        "--match-col",
        type=str,
        required=True,
        help="Column name in CSV to match against (read from route sidecar)",
    )
    parser.add_argument(
        "--tag-dimensions",
        type=str,
        nargs="+",
        required=True,
        help="Column names from CSV to apply as tag dimensions",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        dest="index_path",
        help="Output path for the .tag index file (optional)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Validate CSV exists
    if not args.csv_path.is_file():
        print(f"Error: CSV file not found: {args.csv_path}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("Write Route Tags from CSV")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"CSV: {args.csv_path}")
    print(f"Match column: {args.match_col}")
    print(f"Tag dimensions: {args.tag_dimensions}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Build or load index
    print("[1/3] Loading data source...")
    try:
        store = TagStore(args.source)
        if not store.has_index():
            print(f"Error: No index available for source: {args.source}", file=sys.stderr)
            return 1
        print(f"  Loaded {len(store.route_paths())} routes")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Apply tags
    print("[2/3] Applying tags...")
    stats = apply_csv_tags_to_routes(
        store,
        args.csv_path,
        match_col=args.match_col,
        tag_dimensions=args.tag_dimensions,
        dry_run=args.dry_run,
    )

    # Save index if requested
    if args.index_path and not args.dry_run:
        print(f"[3/3] Saving index to {args.index_path}...")
        TagStore.build_index(store, args.index_path)
        print(f"  Index saved to {args.index_path}")
    else:
        print("[3/3] Skipping index save (use --index to save)")

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total routes in store: {stats['total_routes']}")
    print(f"  Routes matched from CSV: {stats['matched_routes']}")
    print(f"  Files tagged: {stats['tagged_files']}")
    print(f"  Unmatched CSV rows: {stats['unmatched_csv_rows']}")
    print(f"  Errors: {stats['errors']}")
    print()

    if not args.dry_run and stats['unmatched_csv_rows'] > 0:
        print("Note: Some CSV rows had no matching route. "
              "This is normal if the CSV contains routes not in the dataset.")

    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
