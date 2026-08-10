"""Build a lanelet-annotated spatial index from training NPZ sidecars.

Runs on the training server. Reads only JSON sidecars (~200B each), never NPZ files.

Usage:
  ionice -c3 python -m human_match_prototype.build_lanelet_index \
    --data_root /path/to/training_data \
    --map /path/to/lanelet2_map.osm \
    --output lanelet_index.parquet \
    [--workers 8] [--rate_limit 0.001]
"""

import argparse
import hashlib
import time
import warnings
from pathlib import Path

from tqdm import tqdm


def _find_nearest_lanelet(lanelet_layer, point, count):
    """Find the nearest lanelet(s) to a point. Separated for testability (lanelet2 not in test env)."""
    import lanelet2

    return lanelet2.geometry.findNearest(lanelet_layer, point, count)


def _make_point(x: float, y: float):
    """Create a lanelet2 BasicPoint2d. Separated for testability."""
    import lanelet2

    return lanelet2.core.BasicPoint2d(x, y)


def _compute_map_hash(map_path: str) -> str:
    h = hashlib.sha256()
    with open(map_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def add_lanelet_ids(index: list[dict], lanelet_layer, rate_limit: float = 0.0) -> list[dict]:
    result = []
    for entry in tqdm(index, desc="Assigning lanelets"):
        point = _make_point(entry["x"], entry["y"])
        matches = _find_nearest_lanelet(lanelet_layer, point, 1)
        if matches:
            _, ll = matches[0]
            entry_copy = dict(entry)
            entry_copy["lanelet_id"] = ll.id
            result.append(entry_copy)
        else:
            warnings.warn(
                f"No lanelet match for {entry.get('npz_path', '?')} "
                f"at ({entry['x']}, {entry['y']}); entry dropped",
                stacklevel=2,
            )
        if rate_limit > 0:
            time.sleep(rate_limit)
    return result


def save_lanelet_index(index: list[dict], path: str, map_hash: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = {
        "npz_path": [r["npz_path"] for r in index],
        "x": [r["x"] for r in index],
        "y": [r["y"] for r in index],
        "heading_deg": [r["heading_deg"] for r in index],
        "timestamp": [r["timestamp"] for r in index],
        "lanelet_id": [r["lanelet_id"] for r in index],
    }
    if index and "map_version_id" in index[0]:
        columns["map_version_id"] = [r.get("map_version_id") for r in index]
    table = pa.table(columns)
    metadata = table.schema.metadata or {}
    metadata[b"map_hash"] = map_hash.encode()
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path)


def load_lanelet_index(path: str) -> tuple[list[dict], str]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    map_hash = (table.schema.metadata or {}).get(b"map_hash", b"").decode()
    df = table.to_pydict()
    has_map_version = "map_version_id" in df
    index = [
        {
            "npz_path": df["npz_path"][i],
            "x": df["x"][i],
            "y": df["y"][i],
            "heading_deg": df["heading_deg"][i],
            "timestamp": df["timestamp"][i],
            "lanelet_id": df["lanelet_id"][i],
            **({"map_version_id": df["map_version_id"][i]} if has_map_version else {}),
        }
        for i in range(len(df["npz_path"]))
    ]
    return index, map_hash


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data_root", required=True, help="Training data root (contains *.npz + *.json)"
    )
    p.add_argument("--map", required=True, help="Lanelet2 .osm map file")
    p.add_argument("--output", required=True, help="Output .parquet file")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--rate_limit", type=float, default=0.001, help="Sleep between lanelet lookups (seconds)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    import lanelet2

    from human_match_prototype.sidecar import build_index

    npz_paths = sorted(str(p) for p in Path(args.data_root).rglob("*.npz"))
    print(f"Found {len(npz_paths)} NPZ files under {args.data_root}")

    index = build_index(npz_paths, workers=args.workers)
    print(f"Built spatial index: {len(index)} scenes with valid sidecars")

    projection = lanelet2.projection.UtmProjector(lanelet2.io.Origin(35.6491, 139.753))
    lanelet_map, _ = lanelet2.io.loadRobust(args.map, projection)
    print(f"Loaded lanelet2 map: {len(list(lanelet_map.laneletLayer))} lanelets")

    index = add_lanelet_ids(index, lanelet_map.laneletLayer, rate_limit=args.rate_limit)
    print(f"Assigned lanelet IDs to {len(index)} scenes")

    map_hash = _compute_map_hash(args.map)
    save_lanelet_index(index, args.output, map_hash)
    print(f"Saved lanelet index to {args.output} (map hash: {map_hash[:12]}...)")


if __name__ == "__main__":
    main()
