"""GRPO fine-tuning entrypoint, placed alongside ``train_predictor.py``.

This mirrors the supervised trainer (same DDP setup, optimizer/scheduler, EMA, checkpointing
and wandb logging) but swaps the per-epoch training step for ``train_grpo_epoch``: it samples
groups of trajectories, scores them with a collision-based reward, and takes group-relative
policy-gradient steps. It is intended to be run starting from a pretrained checkpoint
(``--resume_model_path``).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import wandb
from diffusion_planner.dimensions import *
from diffusion_planner.grpo_epoch import train_grpo_epoch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils import ddp
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.synthetic_neighbors import SyntheticColliderInjector
from diffusion_planner.utils.neighbor_db import NeighborPatternDB
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
    parser = argparse.ArgumentParser(description="GRPO Training")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, help="save path for model ckpt", required=True)

    # Data
    parser.add_argument("--train_set_list", type=str, required=True)
    parser.add_argument("--valid_set_list", type=str, required=True)
    parser.add_argument("--train_subsample_step", type=int, default=10,
                        help="keep every Nth training sample (data_list[::N]); 1 = use all, "
                             "10 = use 1/10 for faster iteration")

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

    # DataLoader
    parser.add_argument("--normalization_file_path", default="normalization.json", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Training
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64, help="number of scenes per step")
    parser.add_argument("--save_utd", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warm_up_epoch", type=int, default=2)
    parser.add_argument("--encoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--decoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--use_ego_history", type=boolean, default=True)
    parser.add_argument("--ego_history_dropout_rate", type=float, default=0.5)
    parser.add_argument("--use_turn_indicators", type=boolean, default=True)

    # ----- GRPO-specific -----
    parser.add_argument("--num_generations", type=int, default=8,
                        help="N: trajectories sampled per scene (the GRPO group size)")
    parser.add_argument("--grpo_noise_scale", type=float, default=3.0,
                        help="multiplier on the initial diffusion noise during sampling")
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--w_collision", type=float, default=1.0,
                        help="weight on the neighbor-collision penalty in the reward")
    parser.add_argument("--w_road_border", type=float, default=1.0,
                        help="weight on the road-border penalty in the reward (0 disables)")
    parser.add_argument("--w_gt_l2", type=float, default=0.03,
                        help="weight on the realism penalty: ADE (mean L2) between the generated "
                             "ego trajectory and the scene's own GT ego future (0 disables)")
    parser.add_argument("--sft_prob", type=float, default=0.5,
                        help="probability of running a normal supervised step instead of a "
                             "GRPO step on a given batch (0 = pure GRPO, 1 = pure supervised)")

    # Synthetic adversarial neighbor augmentation (see utils/synthetic_neighbors.py):
    # spawn constant-acceleration neighbors that are guaranteed to collide with the ego GT
    # (but avoidable -- they keep clear of the ego's t=0 pose), to drive the collision reward.
    parser.add_argument("--neighbor_inject_max", type=int, default=1,
                        help="max synthetic colliders injected per scene (count ~ U[1, max])")
    parser.add_argument("--neighbor_inject_prob", type=float, default=0.5,
                        help="per-scene probability of injecting any synthetic colliders")
    parser.add_argument("--pedestrian_prob", type=float, default=0.3,
                        help="fraction of injected colliders that are pedestrians")
    parser.add_argument("--bicycle_prob", type=float, default=0.2,
                        help="fraction of injected colliders that are bicycles (rest: vehicles)")
    parser.add_argument("--collider_keep_clear_radius", type=float, default=3.0,
                        help="min distance the collider path keeps from the ego t=0 pose "
                             "(guarantees the forced collision is avoidable)")
    parser.add_argument("--collider_straight_line", type=boolean, default=True,
                        help="colliders drive at constant velocity straight at the collision "
                             "point (easy, history-predictable). False = random-heading "
                             "constant-accel (curved) colliders")

    # Real-neighbor DB collision-search augmentation (utils/neighbor_db.py). When
    # --neighbor_db_path is set, real neighbor tracks that already collide with the scene's ego
    # GT are searched and pasted verbatim, instead of the synthetic colliders above.
    parser.add_argument("--neighbor_db_path", type=str, default="",
                        help="path to a neighbor-pattern DB (built by neighbor_db.py); "
                             "empty = use the synthetic collider generator instead")
    parser.add_argument("--neighbor_db_collision_margin", type=float, default=2.0,
                        help="(DB) max distance [m] from an ego GT waypoint to count as a "
                             "colliding track during the DB search")
    parser.add_argument("--neighbor_min_collision_time", type=float, default=0.8,
                        help="(DB) earliest future time [s] a collision may occur at")
    parser.add_argument("--neighbor_search_subsample", type=int, default=0,
                        help="(DB) cap the per-scene search to this many random patterns "
                             "(0 = search the whole DB)")

    # Loss coefficients (shared with the supervised trainer / loss machinery)
    parser.add_argument("--coeff_position_lat_loss", type=float, default=1.0)
    parser.add_argument("--coeff_position_lon_loss", type=float, default=1.0)
    parser.add_argument("--coeff_heading_l2_loss", type=float, default=1.0)
    parser.add_argument("--coeff_velocity", type=float, default=1.0)
    parser.add_argument("--coeff_timestep", type=list, default=[1.0, 1.0, 1.0, 1.0])

    parser.add_argument("--coeff_road_border_loss", type=float, default=1.0)
    parser.add_argument("--road_border_margin", type=float, default=0.25)
    parser.add_argument("--road_border_n_interp", type=int, default=2)

    parser.add_argument("--coeff_neighbor_collision_loss", type=float, default=0.0)
    parser.add_argument("--neighbor_collision_margin", type=float, default=0.25)

    parser.add_argument("--alpha_planning_loss", type=float, default=1.0)
    parser.add_argument("--alpha_neighbor_loss", type=float, default=0.1)

    parser.add_argument("--use_velocity_representation", type=boolean, default=False)
    parser.add_argument("--hybrid_loss_omega", type=float, default=0.1)
    parser.add_argument("--hybrid_loss_window", type=int, default=10)

    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--use_ema", default=True, type=boolean)

    # Model
    parser.add_argument("--encoder_mixer_depth", type=int, default=6)
    parser.add_argument("--encoder_fusion_depth", type=int, default=6)
    parser.add_argument("--decoder_depth", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_model_type", type=str,
                        choices=["x_start", "flow_matching"], default="x_start")
    parser.add_argument("--predicted_neighbor_num", type=int, default=MAX_NUM_NEIGHBORS)

    parser.add_argument("--resume_model_path", type=str, default=None,
                        help="pretrained checkpoint to start GRPO from (recommended)")

    parser.add_argument("--use_wandb", default=True, type=boolean)
    parser.add_argument("--notes", default="", type=str)

    parser.add_argument("--ddp", default=True, type=boolean)
    parser.add_argument("--port", default="22323", type=str)

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
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
    print(f"{global_rank=}, {rank=}")

    if global_rank == 0:
        print("------------- {} -------------".format(args.exp_name))
        print("Scenes per step (batch_size): {}".format(args.batch_size))
        print("Group size (num_generations): {}".format(args.num_generations))
        print("Learning rate: {}".format(args.learning_rate))

        if args.resume_model_path is not None:
            save_path = args.save_dir
        else:
            save_path = args.save_dir
        os.makedirs(save_path, exist_ok=True)

        args_dict = vars(args)
        args_dict = {
            k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict()
            for k, v in args_dict.items()
        }
        args_dict["major_version"] = 4

        with open(os.path.join(save_path, "args.json"), "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=4)
    else:
        save_path = None

    set_seed(args.seed + global_rank)

    train_epochs = args.train_epochs
    batch_size = args.batch_size
    save_utd = args.save_utd

    # Adversarial neighbor generator for GRPO augmentation: either real tracks searched from a
    # DB (collision-search) or synthetic constant-velocity/accel colliders.
    if args.neighbor_db_path:
        collider_injector = NeighborPatternDB(
            db_path=args.neighbor_db_path,
            collision_margin=args.neighbor_db_collision_margin,
            keep_clear_radius=args.collider_keep_clear_radius,
            min_collision_time=args.neighbor_min_collision_time,
            search_subsample=args.neighbor_search_subsample,
        )
        if global_rank == 0:
            print(f"Neighbor DB collision-search augmentation: "
                  f"{collider_injector.num_patterns} patterns, "
                  f"margin={args.neighbor_db_collision_margin}m "
                  f"keep_clear={args.collider_keep_clear_radius}m")
    else:
        collider_injector = SyntheticColliderInjector(
            pedestrian_prob=args.pedestrian_prob,
            bicycle_prob=args.bicycle_prob,
            keep_clear_radius=args.collider_keep_clear_radius,
            straight_line=args.collider_straight_line,
        )
        if global_rank == 0:
            print(f"Synthetic collider augmentation: ped={args.pedestrian_prob} "
                  f"bike={args.bicycle_prob} keep_clear={args.collider_keep_clear_radius}m")

    if global_rank == 0 and args.w_gt_l2 > 0.0:
        print(f"GT-L2 realism reward enabled: w_gt_l2={args.w_gt_l2}")

    train_set = DiffusionPlannerData(args.train_set_list)
    valid_set = DiffusionPlannerData(args.valid_set_list)

    train_set.data_list = train_set.data_list[:: args.train_subsample_step]

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

    if global_rank == 0:
        valid_loader = DataLoader(
            valid_set,
            batch_size=128,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False,
        )
        print("Dataset Prepared: {} train data\n".format(len(train_set)))
    else:
        valid_loader = None

    if args.ddp:
        torch.distributed.barrier()

    diffusion_planner = Diffusion_Planner(args)
    diffusion_planner = diffusion_planner.to(rank if args.device == "cuda" else args.device)

    if args.ddp:
        diffusion_planner = DDP(diffusion_planner, device_ids=[rank], find_unused_parameters=True)

    model_ema = ModelEma(diffusion_planner, decay=0.999, device=args.device)

    if global_rank == 0:
        print("Model Params: {}".format(
            sum(p.numel() for p in ddp.get_model(diffusion_planner, args.ddp).parameters())
        ))

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
        # GRPO restarts the LR schedule from the configured base rate.
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate
        init_epoch = 0
        print(f"Learning rate set to {args.learning_rate}")
    else:
        init_epoch = 0
        wandb_id = None

    if global_rank == 0:
        os.environ["WANDB_MODE"] = "online" if args.use_wandb else "offline"
        wandb.init(
            project="Diffusion-Planner-GRPO",
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
    best_reward = -float("inf")

    for epoch in range(init_epoch, train_epochs):
        if args.ddp:
            torch.distributed.barrier()

        train_loss, train_total_loss = train_grpo_epoch(
            train_loader, diffusion_planner, optimizer, args, model_ema, collider_injector,
        )

        if global_rank == 0:
            valid_dict = validate_model(diffusion_planner, valid_loader, args)
            valid_loss_ego = valid_dict["avg_loss_ego"]
            valid_neighbor_margin = valid_dict["ego_neighbor_margin_loss"].mean().item()
            valid_road_border = valid_dict["ego_road_border_loss"].mean().item()
            train_reward = train_loss["reward_mean"]
            print(
                f"Epoch {epoch + 1}/{train_epochs}\n"
                f"{train_reward=:.4f}\n"
                f"{valid_loss_ego=:.4f}\n"
                f"{valid_neighbor_margin=:.4f}\n"
                f"{valid_road_border=:.4f}"
            )

            wandb.log(
                {
                    **{f"train/{k}": v for k, v in train_loss.items()},
                    "lr": optimizer.param_groups[0]["lr"],
                    "valid/ego": valid_loss_ego,
                    "valid/neighbor_margin": valid_neighbor_margin,
                    "valid/road_border": valid_road_border,
                },
                step=epoch + 1,
            )

            curr_data = {
                "epoch": epoch + 1,
                "train_reward_mean": train_reward,
                "train_loss": train_total_loss,
                "valid_loss_ego": valid_loss_ego,
                "valid_neighbor_margin": valid_neighbor_margin,
                "valid_road_border": valid_road_border,
            }
            data_list.append(curr_data)
            pd.DataFrame(data_list).to_csv(
                os.path.join(save_path, "train_log.tsv"), index=False, sep="\t"
            )

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
                with open(os.path.join(curr_dir, "args.json"), "w", encoding="utf-8") as f:
                    json.dump(args_dict, f, indent=4)

            if train_reward > best_reward:
                curr_dir = os.path.join(save_path, "best_model")
                os.makedirs(curr_dir, exist_ok=True)
                torch.save(model_dict, f"{curr_dir}/best_model.pth")
                best_reward = train_reward
                curr_data["best_reward"] = best_reward
                with open(os.path.join(curr_dir, "best_model_info.json"), "w") as f:
                    json.dump(curr_data, f, indent=4)

        scheduler.step()
        train_sampler.set_epoch(epoch + 1)


if __name__ == "__main__":
    args = get_args()

    assert len(args.coeff_timestep) == 4

    model_training(args)
