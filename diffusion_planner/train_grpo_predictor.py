"""GRPO fine-tuning entrypoint.

Mirrors the supervised trainer (same DDP setup, optimizer/scheduler, EMA, checkpointing
and wandb logging) but swaps the per-epoch training step for ``train_grpo_epoch``.
Run starting from a pretrained checkpoint (``--resume_model_path``).
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import wandb
from diffusion_planner.config import GRPOConfig, build_config, build_parser
from diffusion_planner.dimensions import *
from diffusion_planner.grpo_epoch import train_grpo_epoch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train import closed_loop_validate
from diffusion_planner.utils import ddp
from diffusion_planner.utils.augmenter_factory import augmenter_from_args
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.neighbor_db import NeighborPatternDB
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.onnx_export import export_checkpoint_onnx_guarded
from diffusion_planner.utils.synthetic_neighbors import SyntheticColliderInjector
from diffusion_planner.utils.train_utils import resume_model, set_seed
from timm.utils import ModelEma
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from valid_predictor import aggregate_valid_metrics, validate_model


def get_args() -> GRPOConfig:
    args = build_parser(GRPOConfig, description="GRPO Training").parse_args()
    cfg = build_config(GRPOConfig, args)
    cfg.state_normalizer = StateNormalizer.from_json(cfg)
    cfg.observation_normalizer = ObservationNormalizer.from_json(cfg)
    return cfg


def mean_ego_loss(loss_dict):
    result = {}
    for key, val in loss_dict.items():
        if key.startswith("ego_"):
            result[f"valid_loss/{key}"] = val.mean().item()
    return result


def model_training(args):
    global_rank, rank, world_size = ddp.ddp_setup_universal(True, args)
    print(f"{global_rank=}, {rank=}")

    save_path = args.save_dir
    if global_rank == 0:
        print("------------- {} -------------".format(args.exp_name))
        print("Scenes per step (batch_size): {}".format(args.batch_size))
        print("Group size (num_generations): {}".format(args.num_generations))
        print("Learning rate: {}".format(args.learning_rate))

        os.makedirs(save_path, exist_ok=True)

        args_dict = vars(args)
        args_dict = {
            k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict()
            for k, v in args_dict.items()
        }
        args_dict["major_version"] = 4

        with open(os.path.join(save_path, "args.json"), "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=4)

    set_seed(args.seed + global_rank)

    train_epochs = args.train_epochs
    batch_size = args.batch_size
    save_utd = args.save_utd

    if args.neighbor_db_path:
        collider_injector = NeighborPatternDB(
            db_path=args.neighbor_db_path,
            collision_margin=args.neighbor_db_collision_margin,
            keep_clear_radius=args.collider_keep_clear_radius,
            min_collision_time=args.neighbor_min_collision_time,
            search_subsample=args.neighbor_search_subsample,
        )
        if global_rank == 0:
            print(
                f"Neighbor DB collision-search augmentation: "
                f"{collider_injector.num_patterns} patterns, "
                f"margin={args.neighbor_db_collision_margin}m "
                f"keep_clear={args.collider_keep_clear_radius}m"
            )
    else:
        collider_injector = SyntheticColliderInjector(
            pedestrian_prob=args.pedestrian_prob,
            bicycle_prob=args.bicycle_prob,
            keep_clear_radius=args.collider_keep_clear_radius,
            straight_line=args.collider_straight_line,
        )
        if global_rank == 0:
            print(
                f"Synthetic collider augmentation: ped={args.pedestrian_prob} "
                f"bike={args.bicycle_prob} keep_clear={args.collider_keep_clear_radius}m"
            )

    if global_rank == 0 and args.w_gt_l2 > 0.0:
        print(f"GT-L2 realism reward enabled: w_gt_l2={args.w_gt_l2}")

    if global_rank == 0 and args.w_kinematic > 0.0:
        print(f"Kinematic-feasibility reward enabled: w_kinematic={args.w_kinematic}")

    aug = augmenter_from_args(args)
    if aug is not None and global_rank == 0:
        print(f"Data augmentation enabled: type={args.augment_type} prob={args.augment_prob}")

    train_set = DiffusionPlannerData(args.train_set_list)
    valid_set = DiffusionPlannerData(args.valid_set_list)

    train_set.data_list = train_set.data_list[:: args.train_subsample_step]

    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=global_rank, shuffle=True
    )
    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=batch_size // world_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    valid_sampler = DistributedSampler(
        valid_set, num_replicas=world_size, rank=global_rank, shuffle=False
    )
    valid_loader = DataLoader(
        valid_set,
        sampler=valid_sampler,
        batch_size=max(128 // world_size, 1),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    if global_rank == 0:
        print("Dataset Prepared: {} train data\n".format(len(train_set)))

    if args.ddp:
        torch.distributed.barrier()

    diffusion_planner = Diffusion_Planner(args)
    diffusion_planner = diffusion_planner.to(rank if args.device == "cuda" else args.device)

    if args.ddp:
        diffusion_planner = DDP(diffusion_planner, device_ids=[rank], find_unused_parameters=True)

    model_ema = ModelEma(diffusion_planner, decay=0.999, device=args.device)

    if global_rank == 0:
        print(
            "Model Params: {}".format(
                sum(p.numel() for p in ddp.get_model(diffusion_planner, args.ddp).parameters())
            )
        )

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
            args.resume_model_path,
            diffusion_planner,
            optimizer,
            scheduler,
            model_ema,
            args.device,
            use_ddp=args.ddp,
        )
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
            train_loader,
            diffusion_planner,
            optimizer,
            args,
            model_ema,
            collider_injector,
            aug,
        )

        valid_dict = validate_model(diffusion_planner, valid_loader, args)
        agg = aggregate_valid_metrics(valid_dict, args.device)
        if global_rank == 0:
            valid_loss_ego = agg["avg_loss_ego"]
            mean_ego_loss_dict = agg["ego_means"]
            valid_neighbor_margin = mean_ego_loss_dict.get("ego_neighbor_margin_loss", 0.0)
            valid_road_border = mean_ego_loss_dict.get("ego_road_border_loss", 0.0)
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
                export_checkpoint_onnx_guarded(
                    config_json_path=os.path.join(save_path, "args.json"),
                    ckpt_path=f"{curr_dir}/best_model.pth",
                    output_dir=curr_dir,
                    output_prefix="diffusion_planner",
                    use_ema=False,
                    use_simplify=False,
                    opset_version=20,
                    external_data=False,
                )

            if train_reward > best_reward:
                curr_dir = os.path.join(save_path, "best_model")
                os.makedirs(curr_dir, exist_ok=True)
                torch.save(model_dict, f"{curr_dir}/best_model.pth")
                best_reward = train_reward
                curr_data["best_reward"] = best_reward
                with open(os.path.join(curr_dir, "best_model_info.json"), "w") as f:
                    json.dump(curr_data, f, indent=4)
                export_checkpoint_onnx_guarded(
                    config_json_path=os.path.join(save_path, "args.json"),
                    ckpt_path=f"{curr_dir}/best_model.pth",
                    output_dir=curr_dir,
                    output_prefix="diffusion_planner",
                    use_ema=False,
                    use_simplify=False,
                    opset_version=20,
                    external_data=False,
                )

        if (epoch + 1 - init_epoch) // save_utd == (train_epochs - init_epoch) // save_utd:
            curr_dir = os.path.join(save_path, f"epoch{epoch + 1:04d}")
            os.makedirs(curr_dir, exist_ok=True)
            closed_loop_validate(
                diffusion_planner,
                args,
                epoch,
                os.path.join(curr_dir, "closed_loop"),
            )

        scheduler.step()
        train_sampler.set_epoch(epoch + 1)


if __name__ == "__main__":
    args = get_args()
    assert len(args.coeff_timestep) == 4
    model_training(args)
