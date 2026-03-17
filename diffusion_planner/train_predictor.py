import argparse
import json
import os
import sys

import pandas as pd
import torch
import wandb
from diffusion_planner.dimensions import *
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train_epoch import train_epoch
from diffusion_planner.utils import ddp
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.train_utils import resume_model, set_seed
from timm.utils import ModelEma
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from valid_predictor import validate_model


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
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, help="save dir for model ckpt", default=".")

    # Data
    parser.add_argument("--train_set_list", type=str, required=True)
    parser.add_argument("--valid_set_list", type=str, required=True)

    parser.add_argument("--future_len", type=int, default=OUTPUT_T)
    parser.add_argument("--time_len", type=int, default=INPUT_T + 1)
    parser.add_argument("--ego_prediction_horizon", type=int, default=OUTPUT_T)

    parser.add_argument("--agent_state_dim", type=int, help="past state dim for agents", default=11)
    parser.add_argument("--agent_num", type=int, default=MAX_NUM_NEIGHBORS)

    parser.add_argument("--static_objects_state_dim", type=int, default=10)
    parser.add_argument("--static_objects_num", type=int, default=5)

    parser.add_argument("--lane_num", type=int, default=NUM_SEGMENTS_IN_LANE)
    parser.add_argument("--lane_len", type=int, default=POINTS_PER_LANELET)

    parser.add_argument("--route_num", type=int, default=NUM_SEGMENTS_IN_ROUTE)
    parser.add_argument("--route_len", type=int, default=POINTS_PER_LANELET)

    parser.add_argument("--polygon_num", type=int, default=NUM_POLYGONS)
    parser.add_argument("--polygon_len", type=int, default=POINTS_PER_POLYGON)

    parser.add_argument("--line_string_num", type=int, default=NUM_LINE_STRINGS)
    parser.add_argument("--line_string_len", type=int, default=POINTS_PER_LINE_STRING)

    # DataLoader parameters
    parser.add_argument("--use_data_augment", default=True, type=boolean)
    parser.add_argument("--augment_prob", type=float, help="augmentation probability", default=0.5)
    parser.add_argument("--normalization_file_path", default="normalization.json", type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin-mem", action="store_true", help="Pin CPU memory in DataLoader")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Training
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--save_utd", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warm_up_epoch", type=int, default=5)
    parser.add_argument("--encoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--decoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--use_ego_history", type=boolean, default=True)
    parser.add_argument("--ego_history_dropout_rate", type=float, default=0.6)
    parser.add_argument("--use_turn_indicators", type=boolean, default=True)

    parser.add_argument("--coeff_position_lat_loss", type=float, default=1.0)
    parser.add_argument("--coeff_position_lon_loss", type=float, default=1.0)
    parser.add_argument("--coeff_heading_l2_loss", type=float, default=1.0)
    parser.add_argument("--coeff_velocity", type=float, default=1.0)
    parser.add_argument(
        "--coeff_timestep",
        type=list,
        default=[1.0, 1.0, 1.0, 1.0],
        help="Set for 4 sections [0,20), [20, 40), [40, 60), [60, 80)",
    )

    parser.add_argument("--coeff_road_border_loss", type=float, default=1.0)
    parser.add_argument("--road_border_margin", type=float, default=0.25)
    parser.add_argument("--road_border_n_interp", type=int, default=2)

    parser.add_argument("--coeff_neighbor_collision_loss", type=float, default=0.0)
    parser.add_argument("--neighbor_collision_margin", type=float, default=2.0)

    parser.add_argument("--alpha_planning_loss", type=float, default=1.0)
    parser.add_argument("--alpha_neighbor_loss", type=float, default=0.1)

    # Velocity representation & hybrid loss (HDP paper, Section IV-B)
    parser.add_argument(
        "--use_velocity_representation",
        type=boolean,
        default=False,
        help="Output trajectory as per-frame displacement instead of absolute waypoints",
    )
    parser.add_argument(
        "--hybrid_loss_omega",
        type=float,
        default=0.1,
        help="Weight for waypoint loss term in hybrid loss (omega in the paper)",
    )
    parser.add_argument(
        "--hybrid_loss_window",
        type=int,
        default=10,
        help="Gradient detach window size W for the waypoint loss term",
    )

    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument("--device", type=str, help="run on which device", default="cuda")

    parser.add_argument("--use_ema", default=True, type=boolean)

    # Model
    parser.add_argument("--encoder_mixer_depth", type=int, default=6)
    parser.add_argument("--encoder_fusion_depth", type=int, default=6)
    parser.add_argument("--decoder_depth", type=int, help="number of decoding layers", default=3)
    parser.add_argument("--num_heads", type=int, help="number of multi-head", default=8)
    parser.add_argument("--hidden_dim", type=int, help="hidden dimension", default=256)
    parser.add_argument(
        "--diffusion_model_type",
        type=str,
        choices=["x_start", "flow_matching"],
        default="x_start",
    )
    parser.add_argument("--predicted_neighbor_num", type=int, default=32)

    parser.add_argument("--resume_model_path", type=str, help="path to resume model", default=None)

    parser.add_argument("--use_wandb", default=False, type=boolean)
    parser.add_argument("--notes", default="", type=str)

    # distributed training parameters
    parser.add_argument("--ddp", default=True, type=boolean, help="use ddp or not")
    parser.add_argument("--port", default="22323", type=str, help="port")

    args = parser.parse_args()

    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)

    return args


def mean_ego_loss(loss_dict):
    result = {}
    for key, val in loss_dict.items():
        if key.startswith("ego_"):
            result[f"valid_loss/{key}"] = val.mean().item()
    return result


def model_training(args):
    # init ddp
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
    print(f"{global_rank=}, {rank=}")

    if global_rank == 0:
        # Logging
        print("------------- {} -------------".format(args.exp_name))
        print("Batch size: {}".format(args.batch_size))
        print("Learning rate: {}".format(args.learning_rate))
        print("Use device: {}".format(args.device))

        if args.resume_model_path is not None:
            save_path = os.path.dirname(args.resume_model_path)
        else:
            from datetime import datetime

            time = datetime.now()
            time = time.strftime("%Y%m%d-%H%M%S")

            save_path = f"{args.save_dir}/{time}_{args.exp_name}/"
            os.makedirs(save_path, exist_ok=True)

        # Save args
        args_dict = vars(args)
        args_dict = {
            k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict()
            for k, v in args_dict.items()
        }
        args_dict["major_version"] = 3

        with open(os.path.join(save_path, "args.json"), "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=4)

    else:
        save_path = None

    # set seed
    set_seed(args.seed + global_rank)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    save_utd = args.save_utd

    # set up data loaders
    aug = (
        StatePerturbation(augment_prob=args.augment_prob, device=args.device)
        if args.use_data_augment
        else None
    )

    # prepare dataset
    train_set = DiffusionPlannerData(args.train_set_list)
    valid_set = DiffusionPlannerData(args.valid_set_list)

    train_sampler = DistributedSampler(
        train_set, num_replicas=ddp.get_world_size(), rank=global_rank, shuffle=True
    )
    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=batch_size // ddp.get_world_size(),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # Validation is only performed on rank 0 with full dataset
    # Other ranks will get a dummy loader (not used)
    if global_rank == 0:
        valid_loader = DataLoader(
            valid_set,
            batch_size=batch_size // 2,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False,
        )
    else:
        # Dummy loader for non-main processes (won't be used)
        valid_loader = None

    if global_rank == 0:
        print("Dataset Prepared: {} train data\n".format(len(train_set)))

    if args.ddp:
        torch.distributed.barrier()

    # set up model
    diffusion_planner = Diffusion_Planner(args)
    diffusion_planner = diffusion_planner.to(rank if args.device == "cuda" else args.device)

    if args.ddp:
        diffusion_planner = DDP(diffusion_planner, device_ids=[rank], find_unused_parameters=True)

    if args.use_ema:
        model_ema = ModelEma(
            diffusion_planner,
            decay=0.999,
            device=args.device,
        )

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
            "lr": args.learning_rate,
        }
    ]

    optimizer = optim.AdamW(params)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, train_epochs, args.warm_up_epoch)

    if args.resume_model_path is not None:
        print(f"Model loaded from {args.resume_model_path}")
        diffusion_planner, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_model(
            args.resume_model_path, diffusion_planner, optimizer, scheduler, model_ema, args.device
        )

        # Override learning rate with the new value
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate
        print(f"Learning rate reset to {args.learning_rate}")

    else:
        init_epoch = 0
        wandb_id = None

    # logger
    if global_rank == 0:
        os.environ["WANDB_MODE"] = "online" if args.use_wandb else "offline"
        wandb.init(
            project="Diffusion-Planner",
            name=args.exp_name,
            notes=args.notes,
            resume="allow",
            id=wandb_id,
            dir=f"{save_path}",
        )
        wandb.config.update(args)

    if args.ddp:
        torch.distributed.barrier()

    data_list = []
    best_loss = float("inf")

    if global_rank == 0:
        valid_dict = validate_model(diffusion_planner, valid_loader, args)
        valid_loss_ego = valid_dict["avg_loss_ego"]
        valid_loss_neighbor = valid_dict["avg_loss_neighbor"]
        mean_ego_loss_dict = mean_ego_loss(valid_dict)
        valid_loss_ego_position_lat_loss = mean_ego_loss_dict.get(
            "valid_loss/ego_position_lat_loss", 0.0
        )
        valid_loss_ego_position_lon_loss = mean_ego_loss_dict.get(
            "valid_loss/ego_position_lon_loss", 0.0
        )
        turn_indicator_accuracy = valid_dict["turn_indicator_accuracy"]
        turn_indicator_change_accuracy = valid_dict["turn_indicator_change_accuracy"]
        turn_indicator_change_total = valid_dict["turn_indicator_change_total"]
        print(
            f"{valid_loss_ego=:.3f}\n"
            f"{valid_loss_neighbor=:.3f}\n"
            f"{valid_loss_ego_position_lat_loss=:.3f}\n"
            f"{valid_loss_ego_position_lon_loss=:.3f}\n"
            f"{turn_indicator_accuracy=:.3f}\n"
            f"{turn_indicator_change_accuracy=:.3f}\n"
            f"{turn_indicator_change_total=:.3f}"
        )

    # begin training
    for epoch in range(init_epoch, train_epochs):
        # Synchronize all processes before training
        if args.ddp:
            torch.distributed.barrier()

        # Adjust learning rate for final 10 epochs
        final_epoch_count = 10
        if epoch >= train_epochs - final_epoch_count:
            base_lr = args.learning_rate
            if epoch >= train_epochs - final_epoch_count // 2:  # Last 5 epochs: LR * 1/100
                adjusted_lr = base_lr * 0.01
            else:  # First 5 of final 10 epochs: LR * 1/10
                adjusted_lr = base_lr * 0.1
            for param_group in optimizer.param_groups:
                param_group["lr"] = adjusted_lr
            if global_rank == 0:
                print(f"Final phase: Epoch {epoch + 1}, LR adjusted to {adjusted_lr}")

        # training step
        train_loss, train_total_loss = train_epoch(
            train_loader, diffusion_planner, optimizer, args, model_ema, aug
        )

        if global_rank == 0:
            valid_dict = validate_model(diffusion_planner, valid_loader, args)
            valid_loss_ego = valid_dict["avg_loss_ego"]
            valid_loss_neighbor = valid_dict["avg_loss_neighbor"]
            mean_ego_loss_dict = mean_ego_loss(valid_dict)
            valid_loss_ego_position_lat_loss = mean_ego_loss_dict.get(
                "valid_loss/ego_position_lat_loss", 0.0
            )
            valid_loss_ego_position_lon_loss = mean_ego_loss_dict.get(
                "valid_loss/ego_position_lon_loss", 0.0
            )
            turn_indicator_accuracy = valid_dict["turn_indicator_accuracy"]
            turn_indicator_change_accuracy = valid_dict["turn_indicator_change_accuracy"]
            turn_indicator_change_total = valid_dict["turn_indicator_change_total"]
            print(
                f"Epoch {epoch + 1}/{train_epochs}\n"
                f"{valid_loss_ego=:.3f}\n"
                f"{valid_loss_neighbor=:.3f}\n"
                f"{valid_loss_ego_position_lat_loss=:.3f}\n"
                f"{valid_loss_ego_position_lon_loss=:.3f}\n"
                f"{turn_indicator_accuracy=:.3f}\n"
                f"{turn_indicator_change_accuracy=:.3f}\n"
                f"{turn_indicator_change_total=:.3f}"
            )

            lr_dict = {"lr": optimizer.param_groups[0]["lr"]}
            wandb.log(
                {
                    **{f"train_loss/{k}": v for k, v in train_loss.items()},
                    **{f"lr/{k}": v for k, v in lr_dict.items()},
                    "valid_loss/ego": valid_loss_ego,
                    "valid_loss/neighbors": valid_loss_neighbor,
                    "valid_loss/turn_indicator_accuracy": turn_indicator_accuracy,
                    "valid_loss/turn_indicator_change_accuracy": turn_indicator_change_accuracy,
                    **mean_ego_loss_dict,
                },
                step=epoch + 1,
            )

            curr_data = {
                "epoch": epoch + 1,
                "train_loss": train_total_loss,
                "valid_loss_ego": valid_loss_ego,
                "valid_loss_neighbor": valid_loss_neighbor,
                "valid_loss_ego_position_lat_loss": valid_loss_ego_position_lat_loss,
                "valid_loss_ego_position_lon_loss": valid_loss_ego_position_lon_loss,
            }
            data_list.append(curr_data)
            df = pd.DataFrame(data_list)
            df.to_csv(os.path.join(save_path, "train_log.tsv"), index=False, sep="\t")

            model_dict = {
                "epoch": epoch + 1,
                "model": diffusion_planner.state_dict(),
                "ema_state_dict": model_ema.ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "schedule": scheduler.state_dict(),
                "loss": valid_loss_ego,
                "wandb_id": wandb_id,
            }
            torch.save(model_dict, f"{save_path}/latest.pth")

            if (epoch + 1 - init_epoch) % save_utd == 0:
                curr_dir = os.path.join(save_path, f"epoch{epoch + 1:04d}")
                os.makedirs(curr_dir, exist_ok=True)
                torch.save(model_dict, f"{curr_dir}/best_model.pth")
                with open(os.path.join(curr_dir, "best_model_info.json"), "w") as f:
                    json.dump(curr_data, f, indent=4)
                with open(os.path.join(curr_dir, "args.json"), "w", encoding="utf-8") as f:
                    json.dump(args_dict, f, indent=4)

            if valid_loss_ego_position_lat_loss < best_loss:
                curr_dir = os.path.join(save_path, "best_model")
                os.makedirs(curr_dir, exist_ok=True)
                torch.save(model_dict, f"{curr_dir}/best_model.pth")
                best_loss = valid_loss_ego_position_lat_loss
                curr_data["best_loss"] = best_loss
                with open(os.path.join(curr_dir, "best_model_info.json"), "w") as f:
                    json.dump(curr_data, f, indent=4)
                with open(os.path.join(curr_dir, "args.json"), "w", encoding="utf-8") as f:
                    json.dump(args_dict, f, indent=4)

        scheduler.step()
        train_sampler.set_epoch(epoch + 1)


if __name__ == "__main__":
    args = get_args()

    assert len(args.coeff_timestep) == 4

    # Run
    model_training(args)
