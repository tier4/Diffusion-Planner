import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=Path)
    parser.add_argument("config_json_path", type=Path)
    parser.add_argument("batch_size", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = args.ckpt_path
    config_json_path = args.config_json_path
    batch_size = args.batch_size

    with open(config_json_path, "r") as f:
        config_json = json.load(f)
    config_obj = Config(config_json_path)

    diffusion_planner = Diffusion_Planner(config_obj)
    diffusion_planner.eval()
    diffusion_planner.cuda()
    diffusion_planner.decoder.decoder.training = False

    ckpt = torch.load(ckpt_path)
    state_dict = ckpt["model"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    diffusion_planner.load_state_dict(new_state_dict)

    dev = diffusion_planner.parameters().__next__().device

    input_dict = {
        "ego_current_state": torch.zeros((batch_size, 10), device=dev),
        "neighbor_agents_past": torch.zeros((batch_size, 32, 21, 11), device=dev),
        "lanes": torch.zeros((batch_size, 70, 20, 13), dtype=torch.float32, device=dev),
        "lanes_speed_limit": torch.zeros((batch_size, 70, 1), device=dev),
        "lanes_has_speed_limit": torch.zeros((batch_size, 70, 1), dtype=torch.bool, device=dev),
        "route_lanes": torch.zeros((batch_size, 25, 20, 13), dtype=torch.float32, device="cuda"),
        "route_lanes_speed_limit": torch.zeros((batch_size, 25, 1), device=dev),
        "route_lanes_has_speed_limit": torch.zeros(
            (batch_size, 25, 1), dtype=torch.bool, device=dev
        ),
        "static_objects": torch.zeros((batch_size, 5, 10), device=dev),
    }

    NUM = 100
    elapsed_list = []
    with torch.no_grad():
        for i in range(NUM):
            torch.cuda.synchronize()
            start = time.time()
            diffusion_planner(input_dict)
            torch.cuda.synchronize()
            end = time.time()
            elapsed = end - start
            elapsed_list.append(elapsed)

    # 上下5個を除外して平均を計算
    elapsed_list = np.array(elapsed_list) * 1000
    elapsed_list = sorted(elapsed_list)
    top_5 = elapsed_list[-5:]
    bottom_5 = elapsed_list[:5]
    elapsed_list = elapsed_list[5:-5]
    elapsed_mean = np.mean(elapsed_list)
    elapsed_std = np.std(elapsed_list)
    print(f"{batch_size},{elapsed_mean:.4f},{elapsed_std:.4f}")
    print(f"top_5: {top_5}")
    print(f"bottom_5: {bottom_5}")
