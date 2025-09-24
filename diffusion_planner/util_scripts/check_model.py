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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    args.hidden_dim = 256
    args.use_ego_history = False
    args.agent_num = 32
    args.static_objects_num = 5
    args.lane_num = 70
    args.lane_len = 20
    args.route_num = 25
    args.route_len = 20
    args.time_len = 21
    args.encoder_drop_path_rate = 0.1
    args.decoder_drop_path_rate = 0.1
    args.encoder_mixer_depth = 3
    args.encoder_fusion_depth = 3
    args.decoder_depth = 3
    args.static_objects_state_dim = 10
    args.num_heads = 8
    args.predicted_neighbor_num = 32
    args.future_len = 80
    args.diffusion_model_type = "x_start"
    args.state_normalizer = None
    args.observation_normalizer = None
    args.ego_agent_past = None

    diffusion_planner = Diffusion_Planner(args)
    print(diffusion_planner)
    diffusion_planner.eval()
    diffusion_planner.cuda()
    diffusion_planner.decoder.decoder.training = True

    dev = diffusion_planner.parameters().__next__().device

    batch_size = 2

    input_dict = {
        "diffusion_time": torch.zeros((batch_size,), device=dev),
        "sampled_trajectories": torch.zeros((batch_size, 32 + 1, 80 + 1, 4), device=dev),
        "gt_trajectories": torch.zeros((batch_size, 32 + 1, 80 + 1, 4), device=dev),
        "ego_agent_past": torch.zeros((batch_size, 21, 4), device=dev),
        "ego_current_state": torch.zeros((batch_size, 10), device=dev),
        "neighbor_agents_past": torch.zeros((batch_size, 32, 21, 11), device=dev),
        "lanes": torch.zeros((batch_size, 70, 20, 8 + 5 + 20), dtype=torch.float32, device=dev),
        "lanes_speed_limit": torch.zeros((batch_size, 70, 1), device=dev),
        "lanes_has_speed_limit": torch.zeros((batch_size, 70, 1), dtype=torch.bool, device=dev),
        "route_lanes": torch.zeros(
            (batch_size, 25, 20, 8 + 5 + 20), dtype=torch.float32, device="cuda"
        ),
        "route_lanes_speed_limit": torch.zeros((batch_size, 25, 1), device=dev),
        "route_lanes_has_speed_limit": torch.zeros(
            (batch_size, 25, 1), dtype=torch.bool, device=dev
        ),
        "static_objects": torch.zeros((batch_size, 5, 10), device=dev),
        "goal_pose": torch.zeros((batch_size, 4), device=dev),
        "ego_shape": torch.zeros((batch_size, 3), device=dev),
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
