import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.dimensions import OUTPUT_MODE_TRAJECTORY, output_dim_for_mode
from diffusion_planner.loss import (
    compute_ego_edge_points,
    compute_neighbor_collision_penalty,
    compute_road_border_penalty,
    loss_func,
    make_turn_indicator_gt,
)
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils import ddp
from diffusion_planner.utils.config import Config
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.train_utils import resume_model, set_seed
from timm.utils import ModelEma
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


@torch.no_grad()
def validate_model(model, val_loader, args, return_pred=False) -> tuple[float, float]:
    """return: ave_loss_ego, ave_loss_neighbor"""
    device = args.device
    model.eval()
    total_loss_ego = 0.0
    total_loss_neighbor = 0.0
    total_samples_ego = 0
    total_samples_neighbor = 0

    predictions = []
    turn_indicators = []
    loss_ego_list = []

    total_result_dict = defaultdict(list)
    turn_indicator_correct = 0.0
    turn_indicator_total = 0
    turn_indicator_change_correct = 0.0
    turn_indicator_change_total = 0

    delay = 0

    for inputs in val_loader:
        inputs = {key: value.to(device) for key, value in inputs.items()}
        B = inputs["ego_current_state"].shape[0]

        turn_indicator_seq = inputs["turn_indicators"]

        output_mode = args.output_mode
        D = output_dim_for_mode(output_mode)
        inputs["sampled_trajectories"] = torch.zeros(B, 33, 81, D, dtype=torch.float32)
        inputs["delay"] = torch.full((B,), delay, dtype=torch.float32, device=device)

        inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
        inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])

        ego_future = inputs["ego_agent_future"]
        ego_future = heading_to_cos_sin(ego_future)  # (B, T, 4)
        neighbors_future = inputs["neighbor_agents_future"]
        neighbor_future_mask = (
            torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
        )  # (B, Pn, T)
        neighbors_future = heading_to_cos_sin(neighbors_future)  # (B, Pn, T, 4)
        neighbors_future[neighbor_future_mask] = 0.0

        B, Pn, T, _ = neighbors_future.shape
        ego_current, neighbors_current = (
            inputs["ego_current_state"][:, :4],
            inputs["neighbor_agents_past"][:, :Pn, -1, :4],
        )
        inputs = args.observation_normalizer(inputs)

        _, outputs = model(inputs)

        neighbor_current_mask = (
            torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        )  # (B, Pn)
        neighbor_mask = torch.concat(
            (neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1
        )  # (B, Pn, T + 1)

        gt_future = torch.cat(
            [ego_future[:, None, :, :], neighbors_future[..., :]], dim=1
        )  # (B, Pn + 1, T, 4)
        current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)
        # (B, Pn + 1, 4)

        all_gt = torch.cat(
            [current_states[:, :, None, :], gt_future], dim=2
        )  # (B, Pn + 1, T + 1, 4)
        all_gt[:, 1:][neighbor_mask] = 0.0

        prediction = outputs["prediction"]
        turn_indicator_logit = outputs["turn_indicator_logit"]
        turn_indicator = turn_indicator_logit.argmax(dim=-1)
        turn_indicator_gt = make_turn_indicator_gt(turn_indicator_seq)
        correct = (turn_indicator == turn_indicator_gt).long()
        turn_indicator_correct += correct.sum().item()
        turn_indicator_total += correct.numel()
        change_mask = turn_indicator_seq[:, -1] != turn_indicator_seq[:, -2]
        change_count = change_mask.sum().item()
        if change_count > 0:
            turn_indicator_change_correct += correct[change_mask].sum().item()
            turn_indicator_change_total += change_count
        if return_pred:
            predictions.append(prediction)
            turn_indicators.append(turn_indicator)

        neighbors_future_valid = ~neighbor_future_mask
        all_gt = all_gt[:, :, 1:, :]  # (B, Pn + 1, T, 4)
        loss_tensor = (prediction - all_gt) ** 2
        loss_ego = loss_tensor[:, 0, :]
        loss_ego_list.append(loss_ego)
        loss_nei = loss_tensor[:, 1:, :]
        loss_nei = loss_nei[neighbors_future_valid]
        total_loss_ego += loss_ego.mean().item() * B
        total_samples_ego += B
        if loss_nei.shape[0] > 0:
            nei_B = loss_nei.shape[0]
            total_loss_neighbor += loss_nei.mean().item() * nei_B
            total_samples_neighbor += nei_B

        loss_dict = loss_func(prediction, all_gt)
        for key, val in loss_dict.items():
            # val : (B, Pn + 1, T)
            total_result_dict[f"ego_{key}"].append(val[:, 0, :])  # (B, T)

        # Compute ego edge points for penalty metrics
        ego_edge_points = compute_ego_edge_points(prediction[:, 0], inputs["ego_shape"], n_interp=args.road_border_n_interp)

        denorm_inputs = args.observation_normalizer.inverse(inputs)
        neighbor_penalty = compute_neighbor_collision_penalty(
            ego_edge_points,
            neighbors_future,
            neighbors_future_valid,
            denorm_inputs["neighbor_agents_past"],
            margin=args.neighbor_collision_margin,
        )
        total_result_dict["ego_neighbor_margin_loss"].append(neighbor_penalty)

        # Road border collision metric
        rb_penalty = compute_road_border_penalty(
            ego_edge_points,
            denorm_inputs["line_strings"],
            margin=args.road_border_margin,
        )
        total_result_dict["ego_road_border_loss"].append(rb_penalty)

    avg_loss_ego = total_loss_ego / total_samples_ego
    avg_loss_neighbor = total_loss_neighbor / max(total_samples_neighbor, 1)
    loss_ego = torch.cat(loss_ego_list, dim=0)

    for key, val in total_result_dict.items():
        total_result_dict[key] = torch.cat(val, dim=0)  # (total_samples, T)

    if return_pred:
        predictions = torch.cat(predictions, dim=0)
        turn_indicators = torch.cat(turn_indicators, dim=0)
    turn_indicator_accuracy = (
        turn_indicator_correct / turn_indicator_total if turn_indicator_total > 0 else 0.0
    )
    turn_indicator_change_accuracy = (
        turn_indicator_change_correct / turn_indicator_change_total
        if turn_indicator_change_total > 0
        else 0.0
    )
    return {
        "avg_loss_ego": avg_loss_ego,
        "avg_loss_neighbor": avg_loss_neighbor,
        "loss_ego": loss_ego,
        "predictions": predictions,
        "turn_indicators": turn_indicators,
        "turn_indicator_accuracy": turn_indicator_accuracy,
        "turn_indicator_change_accuracy": turn_indicator_change_accuracy,
        "turn_indicator_change_total": turn_indicator_change_total,
        **total_result_dict,
    }


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    # Arguments
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--valid_set_list", type=str, help="data list of train data", default=None)

    parser.add_argument("--future_len", type=int, help="number of time point", default=80)
    parser.add_argument("--agent_num", type=int, help="number of agents", default=32)

    # DataLoader parameters
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # Training
    parser.add_argument("--seed", type=int, help="fix random seed", default=3407)
    parser.add_argument("--train_epochs", type=int, help="epochs of training", default=500)
    parser.add_argument("--batch_size", type=int, help="batch size (default: 2048)", default=32)

    parser.add_argument(
        "--device", type=str, help="run on which device (default: cuda)", default="cuda"
    )

    # decoder
    parser.add_argument(
        "--predicted_neighbor_num",
        type=int,
        help="number of neighbor agents to predict",
        default=32,
    )
    parser.add_argument("--resume_model_path", type=str, help="path to resume model", required=True)
    parser.add_argument("--args_json_path", type=str, help="path to resume model", required=True)
    parser.add_argument(
        "--save_predictions_dir", type=str, help="path to save prediction", default=None
    )

    # distributed training parameters
    parser.add_argument("--ddp", default=True, type=boolean, help="use ddp or not")
    parser.add_argument("--port", default="22323", type=str, help="port")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    config_json_path = args.args_json_path

    with open(config_json_path, "r") as f:
        config_json = json.load(f)
    config_obj = Config(config_json_path)

    # init ddp
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
    print(f"{global_rank=}, {rank=}")

    if global_rank == 0:
        # Logging
        print("Batch size: {}".format(args.batch_size))
        print("Use device: {}".format(args.device))

    else:
        save_path = None

    # set seed
    set_seed(args.seed + global_rank)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size

    # set up data loaders
    valid_set = DiffusionPlannerData(args.valid_set_list)
    valid_sampler = DistributedSampler(
        valid_set, num_replicas=ddp.get_world_size(), rank=global_rank, shuffle=False
    )
    valid_loader = DataLoader(
        valid_set,
        sampler=valid_sampler,
        batch_size=batch_size // ddp.get_world_size(),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    if global_rank == 0:
        print("Dataset Prepared: {} valid data\n".format(len(valid_set)))

    if args.ddp:
        torch.distributed.barrier()

    # set up model
    diffusion_planner = Diffusion_Planner(config_obj)
    diffusion_planner = diffusion_planner.to(rank if args.device == "cuda" else args.device)

    if args.ddp:
        diffusion_planner = DDP(diffusion_planner, device_ids=[rank], find_unused_parameters=True)

    if global_rank == 0:
        print(
            "Model Params: {}".format(
                sum(p.numel() for p in ddp.get_model(diffusion_planner, args.ddp).parameters())
            )
        )

    # optimizer
    params = [
        {
            "params": ddp.get_model(diffusion_planner, args.ddp).parameters(),
            "lr": 0.0,
        }
    ]

    optimizer = optim.AdamW(params)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, train_epochs, 0.0)

    if args.resume_model_path is not None:
        print(f"Model loaded from {args.resume_model_path}")
        model_ema = ModelEma(
            diffusion_planner,
            decay=0.999,
            device=args.device,
        )
        diffusion_planner, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_model(
            args.resume_model_path,
            diffusion_planner,
            optimizer,
            scheduler,
            model_ema,
            args.device,
        )
    else:
        init_epoch = 0
        wandb_id = None

    if args.ddp:
        torch.distributed.barrier()

    valid_dict = validate_model(diffusion_planner, valid_loader, config_obj, return_pred=True)
    loss_ego = valid_dict["loss_ego"]
    avg_loss_ego = valid_dict["avg_loss_ego"]
    avg_loss_neighbor = valid_dict["avg_loss_neighbor"]
    predictions = valid_dict["predictions"]
    turn_indicators = valid_dict["turn_indicators"]
    turn_indicator_accuracy = valid_dict["turn_indicator_accuracy"]
    turn_indicator_change_accuracy = valid_dict["turn_indicator_change_accuracy"]
    turn_indicator_change_total = valid_dict["turn_indicator_change_total"]
    print(f"{avg_loss_ego=:.4f} {avg_loss_neighbor=:.4f}")
    print(f"{predictions.shape=}")
    print(f"{turn_indicators.shape=}")
    print(f"{turn_indicator_accuracy=:.4f}")
    if turn_indicator_change_total > 0:
        print(f"{turn_indicator_change_accuracy=:.4f} ({turn_indicator_change_total=:d})")
    else:
        print("turn_indicator_change_accuracy=0.0000 (num_samples=0)")
    if "ego_neighbor_margin_loss" in valid_dict:
        print(
            "ego_neighbor_margin_loss_mean="
            f"{valid_dict['ego_neighbor_margin_loss'].mean().item():.4f}"
        )
    if "ego_road_border_loss" in valid_dict:
        print(
            "ego_road_border_loss_mean="
            f"{valid_dict['ego_road_border_loss'].mean().item():.4f}"
        )

    valid_dict_to_save = {
        "avg_loss_ego": avg_loss_ego,
        "avg_loss_neighbor": avg_loss_neighbor,
        "turn_indicator_accuracy": turn_indicator_accuracy,
        "turn_indicator_change_accuracy": turn_indicator_change_accuracy,
        "turn_indicator_change_total": turn_indicator_change_total,
    }
    for key, val in valid_dict.items():
        if key.startswith("ego_"):
            valid_dict_to_save[f"{key}"] = val.mean().item()

    if args.save_predictions_dir is None:
        exit(0)

    save_predictions_dir = Path(args.save_predictions_dir)
    save_predictions_dir.mkdir(parents=True, exist_ok=True)

    with open(save_predictions_dir.parent / "valid_dict.json", "w") as f:
        json.dump(valid_dict_to_save, f, indent=4)

    for i in range(predictions.shape[0]):
        np.savez(
            save_predictions_dir / f"prediction{i:08d}.npz",
            prediction=predictions[i].cpu().numpy(),
            turn_indicator=turn_indicators[i].cpu().numpy(),
        )
        loss_dict = {
            "loss_ego_total": loss_ego[i].mean().item(),
            "loss_ego_3sec": torch.sqrt(loss_ego[i, 30 - 1, :2].sum()).item(),
            "loss_ego_5sec": torch.sqrt(loss_ego[i, 50 - 1, :2].sum()).item(),
            "loss_ego_8sec": torch.sqrt(loss_ego[i, 80 - 1, :2].sum()).item(),
        }
        for key, val in valid_dict.items():
            if not key.startswith("ego_"):
                continue
            loss_dict[key] = val[i].mean().item()
        with open(save_predictions_dir / f"loss{i:08d}.json", "w") as f:
            json.dump(loss_dict, f, indent=4)
