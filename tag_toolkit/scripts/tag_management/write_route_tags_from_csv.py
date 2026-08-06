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
        [--output-index /path/to/output.db]

All file writes are batched (per-file fsync deferred) and synced once at
the end for maximum performance.

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
from collections import defaultdict
from pathlib import Path
from typing import Sequence

# Add Diffusion-Planner to path for tag_toolkit
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TAG_TOOLKIT_PATH = _REPO_ROOT / "Diffusion-Planner" / "tag_toolkit"
if _TAG_TOOLKIT_PATH.exists():
    sys.path.insert(0, str(_TAG_TOOLKIT_PATH.parent))

from tqdm import tqdm

from tag_toolkit import TagStore
from tag_toolkit.routes import route_of
from tag_toolkit.sidecar import format_tag, is_valid_dimension, read_sidecar

# =============================================================================
# match_col dispatch mechanism
# =============================================================================


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


def _get_field_from_sidecar(
    by_route: dict[Path, list[Path]],
    route: Path,
    field_name: str,
) -> str | None:
    """Extract a field from a route's first NPZ sidecar.

    Uses a pre-built route -> [npz] index built once per call site to
    avoid an O(N_routes × N_npz) scan inside the per-route loop.

    Args:
        by_route: Pre-built mapping from route path to its NPZ paths.
        route: Route path whose first NPZ sidecar should be read.
        field_name: Field name to extract from the sidecar.

    Returns:
        Field value, or None if the route has no NPZ in the index.
    """
    route_npz_paths = by_route.get(route)
    if not route_npz_paths:
        return None

    first_npz = route_npz_paths[0]
    sidecar_data = read_sidecar(first_npz)
    return sidecar_data.get(field_name)


# Registry of match value getters. To add a new ``--match-col`` value,
# add an entry mapping the column name to a callable
# ``(by_route, route) -> match_value | None``. ``_get_match_value`` validates
# the key against this mapping, so we don't need a separate
# ``SUPPORTED_MATCH_COLS`` set.
MATCH_VALUE_GETTERS = {
    "t4_dataset_id": lambda by_route, route: _get_field_from_sidecar(
        by_route, route, "t4_dataset_id"
    ),
}


def _get_match_value(
    by_route: dict[Path, list[Path]],
    route: Path,
    match_col: str,
) -> str | None:
    """Dispatch to the per-match_col getter registered in ``MATCH_VALUE_GETTERS``.

    Args:
        by_route: Pre-built mapping from route path to NPZ paths.
        route: Route path to extract the match value from.
        match_col: The match column type (e.g., 't4_dataset_id').

    Returns:
        The match value, or None if not found.

    Raises:
        ValueError: If match_col is not registered.
    """
    getter = MATCH_VALUE_GETTERS.get(match_col)
    if getter is None:
        raise ValueError(
            f"Unsupported match_col: '{match_col}'. Supported: {sorted(MATCH_VALUE_GETTERS)}"
        )
    return getter(by_route, route)


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
            - csv_matched_routes: routes found in the CSV
            - store_unmatched_routes: routes in store but not in CSV
            - dim_tagged_counts: dict of {dimension: count} for tagged routes
            - tagged_files: number of routes successfully tagged
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
            "csv_matched_routes": 0,
            "store_unmatched_routes": [],
            "dim_tagged_counts": {dim: 0 for dim in tag_dimensions},
            "tagged_files": 0,
            "errors": 0,
        }

    sample_row = next(iter(csv_index.values()))
    for dim in tag_dimensions:
        if dim not in sample_row:
            available = list(sample_row.keys())
            raise ValueError(
                f"tag_dimension '{dim}' not found in CSV. Available columns: {available}"
            )

    # Pre-build route -> [npz] index so the per-route lookup is O(1)
    # instead of O(N_npz). Without this the loop is O(N_routes × N_npz).
    by_route: dict[Path, list[Path]] = defaultdict(list)
    for npz in store.npz_paths():
        by_route[route_of(npz)].append(npz)

    # Get all routes from the store
    routes = store.route_paths()
    total_routes = len(routes)
    csv_matched_routes = 0
    store_unmatched_routes: list[str] = []
    tagged_files = 0
    dim_tagged_counts: dict[str, int] = {dim: 0 for dim in tag_dimensions}
    errors = 0

    # Collect all route-tag pairs first (for dry-run or batch processing)
    route_tags_to_apply: list[tuple[Path, list[str]]] = []

    for route in routes:
        # Get match value from route's sidecar
        match_value = _get_match_value(by_route, route, match_col)
        if not match_value:
            continue

        # Look up in CSV index
        csv_row = csv_index.get(match_value)
        if csv_row is None:
            store_unmatched_routes.append(match_value)
            continue

        csv_matched_routes += 1

        # Build tag list from CSV. Record which raw columns actually produced
        # a tag so per-dimension counts are accurate without re-normalising.
        tags_to_add: list[str] = []
        tagged_dims: list[str] = []
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
                tagged_dims.append(dim)
            except ValueError as e:
                print(f"  Warning: skipping invalid tag '{norm_dim}:{norm_value}': {e}")
                continue

        if not tags_to_add:
            continue

        # Track per-dimension counts (one increment per raw column that yielded a tag)
        for dim in tagged_dims:
            dim_tagged_counts[dim] += 1

        route_tags_to_apply.append((route, tags_to_add))

    # Apply tags (batch or individual)
    if dry_run:
        for route, tags_to_add in route_tags_to_apply:
            tags_str = ", ".join(tags_to_add)
            print(f"  [DRY-RUN] Would add [{tags_str}] to {route.name}")
    else:
        with tqdm(total=len(route_tags_to_apply), desc="Tagging routes", unit="route") as pbar:
            for route, tags_to_add in route_tags_to_apply:
                try:
                    # Pass the precomputed NPZ list as scope so add_tags hits
                    # the SQL frames table directly — avoids the os.walk that
                    # add_tags_to_route triggers through scope resolution.
                    result = store.add_tags(tags_to_add, scope=by_route[route], sync=False)
                    if result.failed:
                        errors += len(result.failed)
                        print(
                            f"\n  Error: {len(result.failed)}/{len(by_route[route])} "
                            f"frame(s) failed in {route}: {result.first_error}"
                        )
                    tagged_files += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"\n  Error: Failed to update {route}: {e}")
                    errors += 1
                    pbar.update(1)

    return {
        "total_routes": total_routes,
        "csv_matched_routes": csv_matched_routes,
        "store_unmatched_routes": store_unmatched_routes,
        "dim_tagged_counts": dim_tagged_counts,
        "tagged_files": tagged_files,
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
        help="Data source: directory, .db index, NPZ file, or path-list JSON",
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
        "--output-index",
        type=Path,
        default=None,
        dest="output_index",
        help="Output path for the .db index file (optional)",
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
    if args.output_index and not args.dry_run:
        print(f"[3/3] Saving index to {args.output_index}...")
        store.export_index(args.output_index)
        print(f"  Index saved to {args.output_index}")
    else:
        print("[3/3] Skipping index save (use --output-index to save)")

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total routes in store: {stats['total_routes']}")
    print(f"  Routes found in CSV: {stats['csv_matched_routes']}")
    print(f"  Errors: {stats['errors']}")
    print()
    print("  Per-dimension tagged routes:")
    for dim, count in stats["dim_tagged_counts"].items():
        print(f"    {dim}: {count}")

    unmatched_routes = stats.get("store_unmatched_routes", [])
    if unmatched_routes:
        print()
        print("  Routes in store but not in CSV:")
        for key in sorted(unmatched_routes):
            print(f"    {key}")

    print()
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
