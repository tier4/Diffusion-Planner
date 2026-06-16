import argparse
import json
from multiprocessing import Pool
from pathlib import Path
from shutil import rmtree

import numpy as np
import torch
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.visualize_input import visualize_inputs
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("save_path", type=Path)
    return parser.parse_args()


def build_annotation(npz_path: Path) -> tuple[str, str]:
    """Build the top banner for a frame as (text, color).

    The banner is always two lines (status + npz path) drawn at a fixed size so the
    figure height stays constant across frames (avoids the mp4 "jumping" when only
    some frames carried a banner). Kept frames show a faint grey "OK"; frames that
    the C++ data_converter would drop show the reason in red. The filter result is
    read from the sibling <token>.json (npz keys are left unchanged).
    """
    status, color = "OK", "0.7"  # faint grey for kept frames
    json_path = npz_path.with_suffix(".json")
    if json_path.is_file():
        try:
            with open(json_path, "r") as f:
                info = json.load(f)
        except (OSError, json.JSONDecodeError):
            info = {}
        if info.get("is_skipped"):
            details = info.get("skipping_info", {}).get("details", "skipped")
            status = f"WOULD BE DROPPED IN REAL GENERATION — {details}"
            color = "red"
    return f"{status}\n{npz_path}", color


if __name__ == "__main__":
    args = parse_args()
    input_path = args.input_path
    save_path = args.save_path

    ext = input_path.suffix

    def process_one_data(input_path: Path, save_path: Path):
        loaded = np.load(input_path)
        data = {}
        for key, value in loaded.items():
            if key == "map_name" or key == "token":
                continue
            # add batch size axis
            data[key] = torch.tensor(np.expand_dims(value, axis=0))
        data["goal_pose"] = heading_to_cos_sin(data["goal_pose"])
        data["ego_agent_past"] = heading_to_cos_sin(data["ego_agent_past"])

        annotation, annotation_color = build_annotation(input_path)
        visualize_inputs(
            data,
            save_path,
            view_ranges=[60, 150],
            annotation=annotation,
            annotation_color=annotation_color,
        )

    if ext == ".npz":
        process_one_data(input_path, save_path)
        print(f"Saved visualization to {save_path}")
    elif ext == ".json":
        with open(input_path, "r") as f:
            path_list = json.load(f)
        path_list = sorted(path_list)
        save_path.mkdir(parents=True, exist_ok=True)
        assert save_path.is_dir()
        rmtree(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        def process_path(path: str):
            path = Path(path)
            dirname = path.parent.name
            basename = path.stem
            curr_save_path = save_path / f"{dirname}_{basename}.png"
            if curr_save_path.exists():
                return
            process_one_data(path, curr_save_path)

        pool = Pool(8)
        with tqdm(total=len(path_list)) as pbar:
            for _ in pool.imap_unordered(process_path, path_list):
                pbar.update(1)
