"""Search and filter NPZ scenes by world-frame position, heading, and trajectory properties.

Reads the JSON sidecar (same stem as .npz) to get world-frame ego pose (MGRS),
then applies spatial, heading, and trajectory-based filters. Outputs a filtered
JSON list compatible with path_list.json format.

Examples:
    # Find scenes in a bounding box
    python -m diffusion_planner.util_scripts.search_scenes path_list.json \
        --bbox 89120,42430,89140,42450

    # Find scenes near a point within 50m
    python -m diffusion_planner.util_scripts.search_scenes path_list.json \
        --center 89135,42440 --radius 50

    # Filter by heading (ego facing roughly east, handles wraparound)
    python -m diffusion_planner.util_scripts.search_scenes path_list.json \
        --bbox 89120,42430,89140,42450 --heading 80,120

    # Filter by trajectory travel distance (moving scenes only)
    python -m diffusion_planner.util_scripts.search_scenes path_list.json \
        --center 89135,42440 --radius 50 --min-travel 5.0

    # Filter by trajectory endpoint direction (ego-frame, degrees CCW from +X)
    python -m diffusion_planner.util_scripts.search_scenes path_list.json \
        --center 89135,42440 --radius 50 --trajectory-heading -30,30

    # Build a spatial index cache for fast repeated queries
    python -m diffusion_planner.util_scripts.search_scenes path_list.json \
        --build-index index.parquet

    # Use cached index for instant queries
    python -m diffusion_planner.util_scripts.search_scenes path_list.json \
        --index index.parquet --center 89135,42440 --radius 50

    # Show statistics instead of saving
    python -m diffusion_planner.util_scripts.search_scenes path_list.json \
        --bbox 89120,42430,89140,42450 --stats

    # Group results by driving sequence (consecutive timestamps from same bag)
    python -m diffusion_planner.util_scripts.search_scenes path_list.json \
        --bbox 89120,42430,89140,42450 --group-sequences
"""

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

def _quat_to_heading_deg(qz: float, qw: float) -> float:
    """Convert quaternion (z, w components) to heading in degrees [-180, 180).

    Convention: radians CCW from +X axis, converted to degrees.
    """
    yaw_rad = 2.0 * math.atan2(qz, qw)
    deg = math.degrees(yaw_rad)
    # Normalize to [-180, 180)
    deg = (deg + 180.0) % 360.0 - 180.0
    return deg


def read_sidecar(npz_path: str) -> Optional[dict]:
    """Read JSON sidecar for an NPZ file. Returns dict with x, y, heading_deg, timestamp or None."""
    json_path = npz_path[:-4] + ".json"  # .npz -> .json
    try:
        with open(json_path, "r") as f:
            j = json.load(f)
        return {
            "npz_path": npz_path,
            "x": j["x"],
            "y": j["y"],
            "heading_deg": _quat_to_heading_deg(j["qz"], j["qw"]),
            "timestamp": j.get("timestamp"),
        }
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def _read_sidecar_batch(npz_paths: list[str]) -> list[Optional[dict]]:
    """Read sidecars for a batch of NPZ paths (used by ProcessPoolExecutor)."""
    return [read_sidecar(p) for p in npz_paths]


def build_index(npz_paths: list[str], workers: int = 8, batch_size: int = 500) -> list[dict]:
    """Build spatial index from JSON sidecars using multiprocessing.

    Returns list of dicts with keys: npz_path, x, y, heading_deg, timestamp.
    Scenes without valid sidecars are silently skipped.
    """
    # Split into batches for multiprocessing
    batches = [npz_paths[i:i + batch_size] for i in range(0, len(npz_paths), batch_size)]
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_read_sidecar_batch, batch): batch for batch in batches}
        with tqdm(total=len(npz_paths), desc="Reading sidecars", unit="scene") as pbar:
            for future in as_completed(futures):
                batch_results = future.result()
                for r in batch_results:
                    if r is not None:
                        results.append(r)
                pbar.update(len(batch_results))

    return results


def save_index_parquet(index: list[dict], path: str) -> None:
    """Save index to parquet for fast reloading. Requires pyarrow."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "npz_path": [r["npz_path"] for r in index],
        "x": [r["x"] for r in index],
        "y": [r["y"] for r in index],
        "heading_deg": [r["heading_deg"] for r in index],
        "timestamp": [r["timestamp"] for r in index],
    })
    pq.write_table(table, path)


def load_index_parquet(path: str) -> list[dict]:
    """Load index from parquet."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    df = table.to_pydict()
    return [
        {
            "npz_path": df["npz_path"][i],
            "x": df["x"][i],
            "y": df["y"][i],
            "heading_deg": df["heading_deg"][i],
            "timestamp": df["timestamp"][i],
        }
        for i in range(len(df["npz_path"]))
    ]


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def filter_bbox(index: list[dict], xmin: float, ymin: float, xmax: float, ymax: float) -> list[dict]:
    """Filter scenes by axis-aligned bounding box in world coordinates."""
    return [r for r in index if xmin <= r["x"] <= xmax and ymin <= r["y"] <= ymax]


def filter_radius(index: list[dict], cx: float, cy: float, radius: float) -> list[dict]:
    """Filter scenes within a radius of a center point."""
    r2 = radius * radius
    return [r for r in index if (r["x"] - cx) ** 2 + (r["y"] - cy) ** 2 <= r2]


def _heading_in_range(heading: float, hmin: float, hmax: float) -> bool:
    """Check if heading (degrees) is in [hmin, hmax], handling wraparound.

    All values in [-180, 180). If hmin <= hmax, it's a simple range check.
    If hmin > hmax (e.g., 170 to -170), it wraps around ±180.
    """
    if hmin <= hmax:
        return hmin <= heading <= hmax
    else:
        # Wraparound: e.g., heading in [170, 180) or [-180, -170]
        return heading >= hmin or heading <= hmax


def filter_heading(index: list[dict], hmin: float, hmax: float) -> list[dict]:
    """Filter scenes by ego heading range (degrees, handles ±180 wraparound)."""
    return [r for r in index if _heading_in_range(r["heading_deg"], hmin, hmax)]


def _compute_trajectory_props(npz_path: str) -> Optional[dict]:
    """Load NPZ and compute trajectory properties from ego_agent_future.

    Returns dict with travel_dist, endpoint_x, endpoint_y, trajectory_heading_deg.
    trajectory_heading_deg is the direction from origin to trajectory endpoint (ego frame).
    """
    try:
        d = np.load(npz_path)
        fut = d["ego_agent_future"]  # (80, 3) = [x, y, yaw_rad]
        # Total travel distance along the trajectory
        dx = np.diff(fut[:, 0])
        dy = np.diff(fut[:, 1])
        travel_dist = float(np.sqrt(dx ** 2 + dy ** 2).sum())
        # Endpoint in ego frame
        ex, ey = float(fut[-1, 0]), float(fut[-1, 1])
        # Direction from ego to endpoint
        traj_heading_deg = math.degrees(math.atan2(ey, ex))
        return {
            "travel_dist": travel_dist,
            "endpoint_x": ex,
            "endpoint_y": ey,
            "trajectory_heading_deg": traj_heading_deg,
        }
    except Exception:
        return None


def _compute_trajectory_batch(npz_paths: list[str]) -> list[tuple[str, Optional[dict]]]:
    """Compute trajectory properties for a batch (used by ProcessPoolExecutor)."""
    return [(p, _compute_trajectory_props(p)) for p in npz_paths]


def enrich_with_trajectory(index: list[dict], workers: int = 8, batch_size: int = 200) -> list[dict]:
    """Add trajectory properties (travel_dist, endpoint, trajectory_heading) to index entries.

    Loads the NPZ files to read ego_agent_future. Entries where NPZ can't be read are dropped.
    """
    path_to_entry = {r["npz_path"]: r for r in index}
    paths = [r["npz_path"] for r in index]
    batches = [paths[i:i + batch_size] for i in range(0, len(paths), batch_size)]
    enriched = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_compute_trajectory_batch, batch): batch for batch in batches}
        with tqdm(total=len(paths), desc="Reading trajectories", unit="scene") as pbar:
            for future in as_completed(futures):
                for npz_path, props in future.result():
                    if props is not None:
                        entry = dict(path_to_entry[npz_path])
                        entry.update(props)
                        enriched.append(entry)
                pbar.update(len(futures[future]))
                del futures[future]  # free reference
                break  # re-check as_completed after deleting

    # Simpler fallback: just iterate
    if not enriched:
        for npz_path, entry in path_to_entry.items():
            props = _compute_trajectory_props(npz_path)
            if props:
                e = dict(entry)
                e.update(props)
                enriched.append(e)

    return enriched


def enrich_with_trajectory_simple(index: list[dict], workers: int = 8, batch_size: int = 200) -> list[dict]:
    """Add trajectory properties to index entries using multiprocessing."""
    path_to_entry = {r["npz_path"]: r for r in index}
    paths = [r["npz_path"] for r in index]
    batches = [paths[i:i + batch_size] for i in range(0, len(paths), batch_size)]
    enriched = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for batch in batches:
            futures.append(executor.submit(_compute_trajectory_batch, batch))

        with tqdm(total=len(paths), desc="Reading trajectories", unit="scene") as pbar:
            for future in as_completed(futures):
                for npz_path, props in future.result():
                    if props is not None:
                        entry = dict(path_to_entry[npz_path])
                        entry.update(props)
                        enriched.append(entry)
                pbar.update(batch_size)

    return enriched


def filter_travel_distance(index: list[dict], min_dist: Optional[float] = None, max_dist: Optional[float] = None) -> list[dict]:
    """Filter by total GT trajectory travel distance (meters)."""
    result = index
    if min_dist is not None:
        result = [r for r in result if r.get("travel_dist", 0) >= min_dist]
    if max_dist is not None:
        result = [r for r in result if r.get("travel_dist", float("inf")) <= max_dist]
    return result


def filter_trajectory_heading(index: list[dict], hmin: float, hmax: float) -> list[dict]:
    """Filter by trajectory endpoint direction in ego frame (degrees, handles wraparound)."""
    return [r for r in index if "trajectory_heading_deg" in r and _heading_in_range(r["trajectory_heading_deg"], hmin, hmax)]


# ---------------------------------------------------------------------------
# Sequence grouping
# ---------------------------------------------------------------------------

def _bag_prefix(npz_path: str) -> str:
    """Extract bag/sequence prefix from NPZ path (everything before the last _NNNNN.npz)."""
    return npz_path.rsplit("_", 1)[0]


def group_sequences(entries: list[dict], max_gap_frames: int = 5) -> list[list[dict]]:
    """Group entries into continuous driving sequences.

    Entries from the same bag prefix with frame numbers within max_gap_frames
    of each other are grouped together. Returns list of groups, sorted by frame number.
    """
    from collections import defaultdict
    import re

    bags = defaultdict(list)
    for entry in entries:
        prefix = _bag_prefix(entry["npz_path"])
        # Extract frame number from path: ..._0000000000000431.npz
        match = re.search(r"_(\d+)\.npz$", entry["npz_path"])
        frame = int(match.group(1)) if match else 0
        bags[prefix].append((frame, entry))

    groups = []
    for prefix, items in bags.items():
        items.sort(key=lambda x: x[0])
        current_group = [items[0]]
        for i in range(1, len(items)):
            if items[i][0] - current_group[-1][0] <= max_gap_frames:
                current_group.append(items[i])
            else:
                groups.append([e for _, e in current_group])
                current_group = [items[i]]
        groups.append([e for _, e in current_group])

    groups.sort(key=lambda g: g[0]["npz_path"])
    return groups


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(entries: list[dict]) -> None:
    """Print summary statistics for a set of scene entries."""
    if not entries:
        print("No scenes matched.")
        return

    xs = [r["x"] for r in entries]
    ys = [r["y"] for r in entries]
    headings = [r["heading_deg"] for r in entries]

    print(f"Matched scenes: {len(entries)}")
    print(f"  X range: {min(xs):.1f} - {max(xs):.1f} (span: {max(xs)-min(xs):.1f}m)")
    print(f"  Y range: {min(ys):.1f} - {max(ys):.1f} (span: {max(ys)-min(ys):.1f}m)")
    print(f"  Heading range: {min(headings):.1f} - {max(headings):.1f} deg")

    # Sequence info
    prefixes = set(_bag_prefix(r["npz_path"]) for r in entries)
    print(f"  Unique bag sequences: {len(prefixes)}")

    if "travel_dist" in entries[0]:
        dists = [r["travel_dist"] for r in entries]
        print(f"  Travel distance: {min(dists):.1f} - {max(dists):.1f}m (mean: {sum(dists)/len(dists):.1f}m)")
    if "trajectory_heading_deg" in entries[0]:
        th = [r["trajectory_heading_deg"] for r in entries]
        print(f"  Trajectory heading: {min(th):.1f} - {max(th):.1f} deg")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search and filter NPZ scenes by position, heading, and trajectory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to path_list.json or directory containing NPZ files",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON path (default: auto-generated from input name + filters)",
    )

    # Spatial filters
    spatial = parser.add_argument_group("spatial filters")
    spatial.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Bounding box: xmin,ymin,xmax,ymax",
    )
    spatial.add_argument(
        "--center",
        type=str,
        default=None,
        help="Center point for radius search: x,y",
    )
    spatial.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Radius in meters (requires --center)",
    )

    # Heading filter
    heading = parser.add_argument_group("heading filter")
    heading.add_argument(
        "--heading",
        type=str,
        default=None,
        help="Ego heading range in degrees: min,max (handles ±180 wraparound)",
    )

    # Trajectory filters (require reading NPZ files)
    traj = parser.add_argument_group("trajectory filters (reads NPZ files, slower)")
    traj.add_argument(
        "--min-travel",
        type=float,
        default=None,
        help="Minimum GT trajectory travel distance in meters",
    )
    traj.add_argument(
        "--max-travel",
        type=float,
        default=None,
        help="Maximum GT trajectory travel distance in meters",
    )
    traj.add_argument(
        "--trajectory-heading",
        type=str,
        default=None,
        help="Trajectory endpoint direction range (ego frame, degrees): min,max",
    )

    # Index caching
    cache = parser.add_argument_group("index caching")
    cache.add_argument(
        "--build-index",
        type=str,
        default=None,
        help="Build and save spatial index to this parquet file (then exit)",
    )
    cache.add_argument(
        "--index",
        type=str,
        default=None,
        help="Load pre-built spatial index from parquet (skips sidecar reading)",
    )

    # Output options
    output = parser.add_argument_group("output options")
    output.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics instead of saving JSON",
    )
    output.add_argument(
        "--group-sequences",
        action="store_true",
        help="Group results by continuous driving sequence and show summary",
    )
    output.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )

    return parser.parse_args()


def load_npz_paths(input_path: str) -> list[str]:
    """Load NPZ paths from a JSON list or by globbing a directory."""
    p = Path(input_path)
    if p.is_file() and p.suffix == ".json":
        with open(p) as f:
            return json.load(f)
    elif p.is_dir():
        return sorted(str(f) for f in p.rglob("*.npz"))
    else:
        print(f"Error: {input_path} is not a .json file or directory", file=sys.stderr)
        sys.exit(1)


def auto_output_name(input_path: str, filters: dict) -> str:
    """Generate output filename from input path and applied filters."""
    stem = Path(input_path).stem
    parts = [stem]
    if filters.get("bbox"):
        parts.append("bbox")
    if filters.get("center"):
        parts.append(f"r{filters['radius']:.0f}")
    if filters.get("heading"):
        parts.append(f"h{filters['heading']}")
    if filters.get("min_travel") or filters.get("max_travel"):
        parts.append("traj")
    parent = Path(input_path).parent
    return str(parent / ("_".join(parts) + ".json"))


def main():
    args = parse_args()

    # Load scene paths
    npz_paths = load_npz_paths(args.input)
    print(f"Loaded {len(npz_paths)} scene paths from {args.input}")

    # Build or load index
    if args.build_index:
        index = build_index(npz_paths, workers=args.workers)
        save_index_parquet(index, args.build_index)
        print(f"Saved index ({len(index)} scenes) to {args.build_index}")
        return

    if args.index:
        print(f"Loading cached index from {args.index}")
        index = load_index_parquet(args.index)
        print(f"Loaded {len(index)} scenes from index")
    else:
        index = build_index(npz_paths, workers=args.workers)

    print(f"Indexed {len(index)} scenes with valid sidecars")

    # Apply spatial filters
    if args.bbox:
        xmin, ymin, xmax, ymax = [float(v) for v in args.bbox.split(",")]
        index = filter_bbox(index, xmin, ymin, xmax, ymax)
        print(f"After bbox filter: {len(index)} scenes")

    if args.center:
        if args.radius is None:
            print("Error: --center requires --radius", file=sys.stderr)
            sys.exit(1)
        cx, cy = [float(v) for v in args.center.split(",")]
        index = filter_radius(index, cx, cy, args.radius)
        print(f"After radius filter: {len(index)} scenes")

    if args.heading:
        hmin, hmax = [float(v) for v in args.heading.split(",")]
        index = filter_heading(index, hmin, hmax)
        print(f"After heading filter: {len(index)} scenes")

    # Trajectory filters (require NPZ loading)
    needs_traj = any([args.min_travel, args.max_travel, args.trajectory_heading])
    if needs_traj:
        index = enrich_with_trajectory_simple(index, workers=args.workers)
        print(f"Enriched {len(index)} scenes with trajectory data")

        if args.min_travel or args.max_travel:
            index = filter_travel_distance(index, args.min_travel, args.max_travel)
            print(f"After travel distance filter: {len(index)} scenes")

        if args.trajectory_heading:
            thmin, thmax = [float(v) for v in args.trajectory_heading.split(",")]
            index = filter_trajectory_heading(index, thmin, thmax)
            print(f"After trajectory heading filter: {len(index)} scenes")

    # Output
    if args.stats or args.group_sequences:
        # Enrich with trajectory for stats if not already done
        if not needs_traj and index:
            print("Loading trajectory data for statistics...")
            index = enrich_with_trajectory_simple(index, workers=args.workers)

        print_stats(index)

        if args.group_sequences:
            groups = group_sequences(index)
            print(f"\n{'='*60}")
            print(f"Sequences: {len(groups)}")
            for i, group in enumerate(groups):
                prefix = _bag_prefix(group[0]["npz_path"]).split("/")[-1]
                dists = [r.get("travel_dist", 0) for r in group]
                print(f"  [{i}] {prefix}: {len(group)} scenes, "
                      f"travel {min(dists):.1f}-{max(dists):.1f}m")

    if not args.stats:
        # Sort by path for deterministic output
        index.sort(key=lambda r: r["npz_path"])
        output_paths = [r["npz_path"] for r in index]

        if args.output:
            out_path = args.output
        else:
            filters = {
                "bbox": args.bbox,
                "center": args.center,
                "radius": args.radius,
                "heading": args.heading,
                "min_travel": args.min_travel,
                "max_travel": args.max_travel,
            }
            out_path = auto_output_name(args.input, filters)

        with open(out_path, "w") as f:
            json.dump(output_paths, f, indent=4)
        print(f"\nSaved {len(output_paths)} scenes to {out_path}")


if __name__ == "__main__":
    main()
