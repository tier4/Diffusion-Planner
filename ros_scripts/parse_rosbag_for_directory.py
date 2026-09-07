import argparse
import logging
import os
import shutil
import sys
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path

from parse_rosbag_by_cpp import main as parse_rosbag_main_cpp
from parse_rosbag_by_cpp import write_conversion_manifest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CPP_BINARY = (
    PROJECT_ROOT / "cpp_tools" / "build" / "autoware_diffusion_planner_tools" / "data_converter"
)

# train/valid are human-driven -> manual, auto stays auto
SPLIT_TO_MODE = {"train": "manual", "valid": "manual", "auto": "auto"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir_list", type=Path, nargs="+")
    parser.add_argument("--save_root", type=Path, required=True)
    parser.add_argument("--cpp_binary_path", type=Path, default=DEFAULT_CPP_BINARY)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--min_frames", type=int, default=1700)
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
    parser.add_argument("--write_skipped_npz", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() // 2)
    parser.add_argument(
        "--conversion_manifest_path",
        type=Path,
        help="Write one record per attempted ROSBAG conversion as JSON.",
    )
    return parser.parse_args()


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

    # bag_path layout: <proj_id>/<map_id>/<split>/<date>/<time>
    date = bag_path.parent.name
    time = bag_path.name
    split = bag_path.parent.parent.name
    map_id = bag_path.parent.parent.parent.name
    proj_id = bag_path.parent.parent.parent.parent.name

    if not map_id or not proj_id:
        logging.warning(
            f"Unexpected bag path layout for {bag_path}: expected "
            f"<proj_id>/<map_id>/<split>/<date>/<time>. Skipping."
        )
        return {
            "status": "skipped",
            "mode": None,
            "bag_path": str(bag_path),
            "output_dir": None,
            "reason": "unexpected_layout",
            "error": None,
        }

    # split が train/valid/auto のいずれでもない場合 (例: psim) は manual にフォールバック
    mode = SPLIT_TO_MODE.get(split, "manual")

    map_dir = bag_path.parent.parent.parent / "map" / date
    vector_map_path = map_dir / "lanelet2_map.osm"

    # if there is map/$date/$time, use it
    if (map_dir / time).is_dir():
        vector_map_path = map_dir / time / "lanelet2_map.osm"

    save_dir = (save_root / proj_id / map_id / mode / date / time).resolve()

    has_npz = save_dir.is_dir() and any(save_dir.rglob("*.npz"))
    has_override_json = mode != "auto" or (save_dir / "control_mode_4_intervals.json").is_file()
    if has_npz and has_override_json:
        logging.info(f"Already exists: {save_dir}")
        return {
            "status": "skipped",
            "mode": mode,
            "reason": "already_exists",
            "bag_path": str(bag_path),
            "output_dir": str(save_dir),
            "error": None,
        }

    save_dir.mkdir(parents=True, exist_ok=True)

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
            extract_override_segments=(mode == "auto"),
        )
    except Exception as e:
        error_msg = f"Error processing {bag_path}: {str(e)}"
        logging.error(error_msg)
        return {
            "status": "failed",
            "mode": mode,
            "bag_path": str(bag_path),
            "output_dir": str(save_dir),
            "reason": "conversion_failed",
            "error": str(e),
        }

    # The C++ converter writes the per-frame npz/json directly under save_dir but
    # emits the per-sequence route json into a nested "routes" subdir. Flatten it
    # so save_dir does not end up with a redundant routes/routes level. This runs
    # only after a successful conversion, and its own failures are reported
    # separately so a completed conversion is not mislabeled as a failure.
    try:
        nested_routes = save_dir / "routes"
        if nested_routes.is_dir():
            for entry in nested_routes.iterdir():
                shutil.move(str(entry), str(save_dir / entry.name))
            nested_routes.rmdir()
    except Exception as e:
        error_msg = f"Error flattening routes for {save_dir}: {str(e)}"
        logging.error(error_msg)
        return {
            "status": "failed",
            "mode": mode,
            "bag_path": str(bag_path),
            "output_dir": str(save_dir),
            "reason": "postprocess_failed",
            "error": str(e),
        }

    logging.info(f"Completed: {save_dir}")
    return {
        "status": "converted",
        "mode": mode,
        "bag_path": str(bag_path),
        "output_dir": str(save_dir),
        "reason": None,
        "error": None,
    }


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

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(save_root / "log.txt", mode="w"), logging.StreamHandler()],
    )

    # search "metadata.yaml"
    metadata_list = []
    for target_dir in target_dir_list:
        metadata_list.extend(list(target_dir.glob("**/metadata.yaml")))
    bag_dir_list = [
        metadata_path.parent for metadata_path in metadata_list if metadata_path.is_file()
    ]
    bag_dir_list = list(set(bag_dir_list))  # Remove duplicates
    bag_dir_list.sort()

    logging.info(f"Found {len(bag_dir_list)} bag directories to process")
    logging.info(f"Using {num_workers} parallel workers")

    # Prepare arguments for parallel processing
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

    # Process bags in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_bag, process_args)

    status_counts = Counter(result["status"] for result in results)
    converted_count = status_counts["converted"]
    skipped_count = status_counts["skipped"]
    failed_count = status_counts["failed"]
    logging.info(
        "Conversion summary: converted=%d, skipped=%d, failed=%d",
        converted_count,
        skipped_count,
        failed_count,
    )

    if args.conversion_manifest_path is not None:
        write_conversion_manifest(
            args.conversion_manifest_path,
            results,
            converted_count,
            skipped_count,
            failed_count,
        )

    elapsed_seconds = int(time.perf_counter() - start_time)
    hours = elapsed_seconds // 3600
    minutes = (elapsed_seconds % 3600) // 60
    seconds = elapsed_seconds % 60
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    print(f"Total elapsed time: {time_str}")

    with open(save_root / "processing_time.txt", "w") as summary_file:
        summary_file.write(f"Total elapsed time: {time_str}\n")

    if converted_count == 0 and failed_count > 0:
        logging.error("No bags were converted successfully.")
        sys.exit(1)
