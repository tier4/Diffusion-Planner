import argparse
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path

from parse_rosbag import main as parse_rosbag_main
from parse_rosbag_by_cpp import main as parse_rosbag_main_cpp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir_list", type=Path, nargs="+")
    parser.add_argument("--save_root", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--min_frames", type=int, default=1700)
    parser.add_argument("--search_nearest_route", type=int, default=1)
    parser.add_argument("--convert_yellow", type=int, default=0)
    parser.add_argument("--convert_red", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=32)
    return parser.parse_args()


def process_single_bag(args_tuple):
    (
        bag_path,
        save_root,
        step,
        limit,
        min_frames,
        search_nearest_route,
        convert_yellow,
        convert_red,
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
        use_cpp = True
        if use_cpp:
            parse_rosbag_main_cpp(
                Path("/home/ubuntu/autoware/build/autoware_diffusion_planner/data_converter"),
                rosbag_path=bag_path,
                vector_map_path=vector_map_path,
                save_dir=save_dir,
                step=step,
                limit=limit,
                min_frames=min_frames,
                search_nearest_route=search_nearest_route,
                convert_yellow=convert_yellow,
                convert_red=convert_red,
            )
        else:
            parse_rosbag_main(
                rosbag_path=bag_path,
                vector_map_path=vector_map_path,
                save_dir=save_dir,
                step=step,
                limit=limit,
                min_frames=min_frames,
                search_nearest_route=search_nearest_route,
            )
        logging.info(f"Completed: {save_dir}")
    except Exception as e:
        error_msg = f"Error processing {bag_path}: {str(e)}"
        logging.error(error_msg)


if __name__ == "__main__":
    args = parse_args()
    target_dir_list = args.target_dir_list
    save_root = args.save_root
    step = args.step
    limit = args.limit
    min_frames = args.min_frames
    search_nearest_route = args.search_nearest_route
    convert_yellow = args.convert_yellow
    convert_red = args.convert_red
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
                search_nearest_route,
                convert_yellow,
                convert_red,
            )
        )

    # Process bags in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_bag, process_args)
