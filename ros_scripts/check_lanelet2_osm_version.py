import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("map_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    map_dir = args.map_dir

    date_dir_list = sorted([d for d in map_dir.iterdir() if d.is_dir()])
    for date_dir in date_dir_list:
        print(f"Checking {date_dir}")
        time_dir_list = sorted([d for d in date_dir.iterdir() if d.is_dir()])
        version_set = set()
        for time_dir in time_dir_list:
            if time_dir.name == "pointcloud_map.pcd":
                continue
            cache_file = time_dir / ".osm_caches"
            if not cache_file.exists():
                print(f"  {time_dir}: No cache file")
                continue
            f = cache_file.open("r")
            line = f.readline().strip()
            osm_version = line.split("/")[-2]
            version_set.add(osm_version)

        print(f"{len(version_set)=}")
