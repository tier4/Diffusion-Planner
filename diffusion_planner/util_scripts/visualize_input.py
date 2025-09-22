import argparse
import json
from multiprocessing import Pool
from pathlib import Path
from shutil import rmtree

import numpy as np
import torch
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.config import Config
from diffusion_planner.utils.visualize_input import visualize_inputs
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("args_json", type=Path)
    parser.add_argument("--save_path", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_path = args.input_path
    args_json = args.args_json
    save_path = args.save_path

    ext = input_path.suffix
    config_obj = Config(args_json)

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
        data = config_obj.observation_normalizer(data)

        visualize_inputs(data, config_obj.observation_normalizer, save_path)

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
