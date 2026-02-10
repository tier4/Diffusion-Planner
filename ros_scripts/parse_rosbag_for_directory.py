import argparse
import logging
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

from parse_rosbag_by_cpp import main as parse_rosbag_main_cpp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir_list", type=Path, nargs="+")
    parser.add_argument("--save_root", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--min_frames", type=int, default=1700)
    parser.add_argument("--min_distance", type=float, default=50.0)
    parser.add_argument("--search_nearest_route", type=int, default=1)
    parser.add_argument("--convert_yellow", type=int, default=0)
    parser.add_argument("--convert_red", type=int, default=0)
    parser.add_argument("--ego_wheel_base", type=float, default=2.75)
    parser.add_argument("--ego_length", type=float, default=4.34)
    parser.add_argument("--ego_width", type=float, default=1.70)
    parser.add_argument("--num_workers", type=int, default=32)
    return parser.parse_args()


def process_single_bag(args_tuple):
    (
        bag_path,
        save_root,
        step,
        limit,
        min_frames,
        min_distance,
        search_nearest_route,
        convert_yellow,
        convert_red,
        ego_wheel_base,
        ego_length,
        ego_width,
    ) = args_tuple

    logging.info(f"Processing bag: {bag_path}")

    date = bag_path.parent.name
    time = bag_path.name

    map_dir = bag_path.parent.parent.parent / "map" / date
    vector_map_path = map_dir / "lanelet2_map.osm"

    # if there is map/$date/$time, use it
    if (map_dir / time).is_dir():
        vector_map_path = map_dir / time / "lanelet2_map.osm"

    (save_root / date).mkdir(parents=True, exist_ok=True)
    save_dir = (save_root / date / time).resolve()

    if save_dir.is_dir():
        logging.info(f"Already exists: {save_dir}")
        return f"Skipped (already exists): {save_dir}"

    try:
        parse_rosbag_main_cpp(
            Path("~/autoware/build/autoware_diffusion_planner/data_converter").expanduser(),
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
            ego_wheel_base=ego_wheel_base,
            ego_length=ego_length,
            ego_width=ego_width,
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
    step = args.step
    limit = args.limit
    min_frames = args.min_frames
    min_distance = args.min_distance
    search_nearest_route = args.search_nearest_route
    convert_yellow = args.convert_yellow
    convert_red = args.convert_red
    ego_wheel_base = args.ego_wheel_base
    ego_length = args.ego_length
    ego_width = args.ego_width
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
                bag_path,
                save_root,
                step,
                limit,
                min_frames,
                min_distance,
                search_nearest_route,
                convert_yellow,
                convert_red,
                ego_wheel_base,
                ego_length,
                ego_width,
            )
        )

    # Process bags in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_bag, process_args)

    elapsed_seconds = int(time.perf_counter() - start_time)
    hours = elapsed_seconds // 3600
    minutes = (elapsed_seconds % 3600) // 60
    seconds = elapsed_seconds % 60
    print(f"Total elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")
