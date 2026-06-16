import argparse
import json
import logging
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

from parse_rosbag_by_cpp import main as parse_rosbag_main_cpp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CPP_BINARY = (
    PROJECT_ROOT / "cpp_tools" / "build" / "autoware_diffusion_planner_tools" / "data_converter"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir_list", type=Path, nargs="+")
    parser.add_argument("--save_root", type=Path, required=True)
    parser.add_argument("--cpp_binary_path", type=Path, default=DEFAULT_CPP_BINARY)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--min_frames", type=int, default=0)
    parser.add_argument("--min_distance", type=float, default=50.0)
    parser.add_argument("--search_nearest_route", type=int, default=1)
    parser.add_argument("--convert_yellow", type=int, default=0)
    parser.add_argument("--convert_red", type=int, default=0)
    parser.add_argument("--interpolation", type=int, default=1)
    parser.add_argument("--ego_wheel_base", type=float, default=2.75)
    parser.add_argument("--ego_length", type=float, default=4.34)
    parser.add_argument("--ego_width", type=float, default=1.70)
    parser.add_argument("--static_object_margin", type=float, default=0.0)
    parser.add_argument("--neighbor_margin", type=float, default=0.0)
    parser.add_argument("--road_border_margin", type=float, default=0.0)
    parser.add_argument("--collision_time_stride", type=int, default=5)
    parser.add_argument("--offlane_max_score", type=float, default=6.0)
    parser.add_argument("--offlane_time_stride", type=int, default=1)
    parser.add_argument("--write_skipped_npz", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=32)
    return parser.parse_args()


def _resolve_vector_map_path(bag_path: Path) -> Path:
    # area_map_version_id は log_file_info.json から読む（metadata.yaml は参照しない）
    info_path = bag_path / "log_file_info.json"
    date = bag_path.parent.name
    bag_time = bag_path.name

    map_version_id = None
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if "area_map_version_id" in info:
            map_version_id = info["area_map_version_id"]

    # Search from near bag path to upper directories to support multiple layouts.
    candidate_bases = []
    max_levels = min(len(bag_path.parents), 6)
    for i in range(1, max_levels):
        base = bag_path.parents[i]
        if base not in candidate_bases:
            candidate_bases.append(base)

    candidate_paths = []
    for base in candidate_bases:
        map_dir = base / "map"
        if not map_dir.is_dir():
            continue

        if map_version_id:
            candidate_paths.append(map_dir / map_version_id / "lanelet2_map.osm")

        # Legacy layouts.
        candidate_paths.append(map_dir / date / bag_time / "lanelet2_map.osm")
        candidate_paths.append(map_dir / date / "lanelet2_map.osm")
        candidate_paths.append(map_dir / bag_time / "lanelet2_map.osm")
        candidate_paths.append(map_dir / "lanelet2_map.osm")

    for path in candidate_paths:
        if path.is_file():
            return path

    searched = (
        "\n".join(str(path) for path in candidate_paths)
        if candidate_paths
        else "(no map dir found)"
    )
    raise FileNotFoundError(
        f"lanelet2_map.osm was not found for bag: {bag_path}\n"
        f"log_file_info: {info_path}\n"
        f"area_map_version_id: {map_version_id}\n"
        f"searched:\n{searched}"
    )


def process_single_bag(args_tuple):
    (
        cpp_binary_path,
        bag_path,
        save_root,
        step,
        limit,
        min_frames,
        min_distance,
        search_nearest_route,
        convert_yellow,
        convert_red,
        interpolation,
        ego_wheel_base,
        ego_length,
        ego_width,
        static_object_margin,
        neighbor_margin,
        road_border_margin,
        collision_time_stride,
        offlane_max_score,
        offlane_time_stride,
        write_skipped_npz,
    ) = args_tuple

    logging.info(f"Processing bag: {bag_path}")

    project_name = bag_path.parents[3].name
    map_name = bag_path.parents[2].name
    train_or_val = bag_path.parents[1].name
    date = bag_path.parent.name
    time = bag_path.name

    vector_map_path = _resolve_vector_map_path(bag_path)

    save_dir = (save_root / project_name / map_name / train_or_val / date / time).resolve()
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    if save_dir.is_dir():
        logging.info(f"Already exists: {save_dir}")
        return f"Skipped (already exists): {save_dir}"

    try:
        parse_rosbag_main_cpp(
            cpp_binary_path,
            rosbag_path=bag_path,
            vector_map_path=vector_map_path,
            save_dir=save_dir,
            step=step,
            limit=limit,
            min_frames=min_frames,
            min_distance=min_distance,
            search_nearest_route=search_nearest_route,
            convert_yellow=convert_yellow,
            convert_red=convert_red,
            interpolation=interpolation,
            ego_wheel_base=ego_wheel_base,
            ego_length=ego_length,
            ego_width=ego_width,
            static_object_margin=static_object_margin,
            neighbor_margin=neighbor_margin,
            road_border_margin=road_border_margin,
            collision_time_stride=collision_time_stride,
            offlane_max_score=offlane_max_score,
            offlane_time_stride=offlane_time_stride,
            write_skipped_npz=write_skipped_npz,
        )
        logging.info(f"Completed: {save_dir}")
    except Exception as e:
        error_msg = f"Error processing {bag_path}: {str(e)}"
        logging.error(error_msg)


if __name__ == "__main__":
    start_time = time.perf_counter()
    args = parse_args()
    target_dir_list = args.target_dir_list
    save_root = args.save_root
    cpp_binary_path = args.cpp_binary_path
    step = args.step
    limit = args.limit
    min_frames = args.min_frames
    min_distance = args.min_distance
    search_nearest_route = args.search_nearest_route
    convert_yellow = args.convert_yellow
    convert_red = args.convert_red
    interpolation = args.interpolation
    ego_wheel_base = args.ego_wheel_base
    ego_length = args.ego_length
    ego_width = args.ego_width
    static_object_margin = args.static_object_margin
    neighbor_margin = args.neighbor_margin
    road_border_margin = args.road_border_margin
    collision_time_stride = args.collision_time_stride
    offlane_max_score = args.offlane_max_score
    offlane_time_stride = args.offlane_time_stride
    write_skipped_npz = args.write_skipped_npz
    num_workers = args.num_workers or cpu_count()

    save_root = save_root.resolve()
    save_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(save_root / "log.txt", mode="w"), logging.StreamHandler()],
    )

    metadata_list = []
    for target_dir in target_dir_list:
        metadata_list.extend(list(target_dir.glob("**/metadata.yaml")))
    bag_dir_list = [
        metadata_path.parent for metadata_path in metadata_list if metadata_path.is_file()
    ]
    bag_dir_list = list(set(bag_dir_list))
    bag_dir_list.sort()

    logging.info(f"Found {len(bag_dir_list)} bag directories to process")
    logging.info(f"Using {num_workers} parallel workers")

    process_args = []
    for bag_path in bag_dir_list:
        process_args.append(
            (
                cpp_binary_path,
                bag_path,
                save_root,
                step,
                limit,
                min_frames,
                min_distance,
                search_nearest_route,
                convert_yellow,
                convert_red,
                interpolation,
                ego_wheel_base,
                ego_length,
                ego_width,
                static_object_margin,
                neighbor_margin,
                road_border_margin,
                collision_time_stride,
                offlane_max_score,
                offlane_time_stride,
                write_skipped_npz,
            )
        )

    with Pool(processes=num_workers) as pool:
        pool.map(process_single_bag, process_args)

    elapsed_seconds = int(time.perf_counter() - start_time)
    hours = elapsed_seconds // 3600
    minutes = (elapsed_seconds % 3600) // 60
    seconds = elapsed_seconds % 60
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    print(f"Total elapsed time: {time_str}")

    with open(save_root / "processing_time.txt", "w") as summary_file:
        summary_file.write(f"Total elapsed time: {time_str}\n")
