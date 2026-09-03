import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
import wandb
from timm.utils import ModelEma
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from diffusion_planner.config import TrainConfig
from diffusion_planner.dimensions import *
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.scenario_based_open_loop.validate import scenario_based_open_loop_validate
from diffusion_planner.train_epoch import train_epoch
from diffusion_planner.utils import ddp, shard_ddp
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.data_augmentation_bridge import (
    StatePerturbation as BridgeStatePerturbation,
)
from diffusion_planner.utils.data_augmentation_frenet import (
    frenet_augmenter_from_args,
)
from diffusion_planner.utils.dataset import DiffusionPlannerData, DiffusionPlannerPairData
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts, final_phase_lr
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.onnx_export import export_checkpoint_onnx_guarded
from diffusion_planner.utils.train_utils import resume_model, set_seed
from diffusion_planner.validate_model import (
    aggregate_replan_consistency_metrics,
    aggregate_valid_metrics,
    validate_model,
    validate_replan_consistency,
)


def find_upward(start_file: str, target_name: str) -> Path:
    directory = Path(start_file).resolve().parent
    for candidate_dir in [directory, *directory.parents]:
        candidate = candidate_dir / target_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{target_name} up {directory}")


def _add_path_list_compressed(artifact: wandb.Artifact, path: Path, tmp_dir: Path) -> None:
    """Attach a path-list json to the artifact as zstd-compressed ``<name>.zst``.

    Raw path lists are multi-GB of near-identical absolute paths; zstd shrinks them
    ~100x in a few seconds. openjson() reads ``*.zst`` transparently on download.
    """
    try:
        import zstandard
    except ImportError:
        print(f"zstandard not installed; uploading {path.name} uncompressed.")
        artifact.add_file(str(path), name=path.name)
        return
    dst = tmp_dir / f"{path.name}.zst"
    cctx = zstandard.ZstdCompressor(level=3, threads=-1)
    with open(path, "rb") as fin, open(dst, "wb") as fout:
        cctx.copy_stream(fin, fout)
    artifact.add_file(str(dst), name=dst.name)


def log_dataset_artifact(
    run: wandb.sdk.wandb_run.Run, exp_name: str, train_set_list: str, valid_set_list: str
) -> None:
    artifact = wandb.Artifact(
        name=f"dataset_{exp_name}",
        type="dataset",
        metadata={"train_set_list": train_set_list, "valid_set_list": valid_set_list},
    )
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        _add_path_list_compressed(artifact, Path(train_set_list), tmp_dir)
        _add_path_list_compressed(artifact, Path(valid_set_list), tmp_dir)
        try:
            summary_csv = find_upward(train_set_list, "summary.csv")
            artifact.add_file(str(summary_csv), name="summary.csv")
        except FileNotFoundError:
            print("summary.csv not found, skipping.")
        try:
            rosbag_summary_csv = find_upward(train_set_list, "rosbag_summary.csv")
            artifact.add_file(str(rosbag_summary_csv), name="rosbag_summary.csv")
        except FileNotFoundError:
            print("rosbag_summary.csv not found, skipping.")
        # add_file() copies into wandb's staging cache, so the temp dir can be
        # removed as soon as use_artifact() returns.
        run.use_artifact(artifact)


def mean_ego_loss(loss_dict):
    result = {}
    for key, val in loss_dict.items():
        if key.startswith("ego_"):
            result[f"valid_loss/{key}"] = val.mean().item()
    return result


def mean_epdms_metric(loss_dict):
    result = {}
    for key, val in loss_dict.items():
        if not key.startswith("epdms_"):
            continue
        metric = key.removeprefix("epdms_")
        tensor = val.float()
        if metric.endswith("_available"):
            result[f"valid_epdms/{metric}"] = tensor.mean().item()
            continue
        available = loss_dict.get(f"{key}_available")
        if available is None:
            result[f"valid_epdms/{metric}"] = tensor.mean().item()
            continue
        mask = available.float() > 0.5
        result[f"valid_epdms/{metric}_coverage"] = mask.float().mean().item()
        result[f"valid_epdms/{metric}"] = tensor[mask].mean().item() if mask.any() else float("nan")
    return result


def wandb_epdms_metrics(epdms_means):
    return {
        f"valid_epdms/{key}": value
        for key, value in epdms_means.items()
        if not key.endswith("_coverage")
    }


def closed_loop_validate(model, args: TrainConfig, epoch: int, out_dir: str) -> None:
    """Closed-loop rendered rollout; logs metrics + videos to wandb."""
    import os

    if not args.closed_loop_npz_root:
        return

    from run_all_groups_closed_loop import _load_group_results, run_closed_loop_main

    net = ddp.get_model(model, args.ddp)
    was_training = net.training
    net.eval()

    try:
        run_closed_loop_main(
            model=net,
            model_args=args,
            cfg=args,
            out_root=out_dir,
            wandb_run=wandb.run,
            only_json=None,
            render_media=args.render_media,
        )

        if ddp.get_rank() == 0:
            for group_key, summary in _load_group_results(out_dir).items():
                print(
                    f"closed-loop [{group_key}] @epoch {epoch + 1}: {summary.get('n_segments', 0)} seg in "
                    f"{summary.get('elapsed_sec', 0):.1f}s  route_completion={summary.get('mean_route_completion', 0.0):.3f}  "
                    f"collisions={summary.get('object', {}).get('collision_count', 0)}  "
                    f"curb_hits={summary.get('road_border', {}).get('collision_count', 0)}  "
                    f"snaps={summary.get('reproducer', {}).get('snap_count', 0)}  -> "
                    f"{len(summary.get('video_mp4s', []))} video(s)"
                )

    finally:
        net.train(was_training)


def scenario_sim_validate(args, epoch: int, ckpt_path: str, out_dir: str) -> None:
    """Evaluate a just-saved checkpoint against the OpenSCENARIO suite, out of process.

    Rank 0 only, on the checkpoint-save cadence. The other ranks wait at the next epoch's
    ``torch.distributed.barrier()``, which inherits the process group's timeout.

    The driver fans out one process per scenario, which is the configuration the suite's
    throughput was measured with: the rollout saturates the GPUs in time only near 96-way
    concurrency, so sharding one scenario per rank inside this process would leave most of that
    throughput unused. It also keeps the ROS overlay out of the training process -- the driver
    sources it for its own children, so nothing here imports the interpreter.

    A failed evaluation is reported and training continues: losing a data point costs less than
    losing the run.
    """
    if not args.scenario_sim_driver:
        return

    started = time.perf_counter()
    rc = subprocess.run(
        ["bash", args.scenario_sim_driver],
        env={**os.environ, "CKPT": ckpt_path, "OUT": out_dir},
    ).returncode
    elapsed = time.perf_counter() - started
    status = "ok" if rc == 0 else f"FAILED rc={rc}"
    print(f"scenario_sim @epoch {epoch + 1}: {status} in {elapsed:.1f}s -> {out_dir}", flush=True)

    if rc != 0 or not args.use_wandb or wandb.run is None:
        return

    try:
        from scenario_generation.wandb_scenario_sim import (
            build_scenario_sim_wandb_payload,
            load_case_rows,
        )

        out_p = Path(out_dir)
        payload = build_scenario_sim_wandb_payload(load_case_rows(out_p), media_root=out_p)
        wandb.run.log(payload, step=epoch + 1)
        print(
            f"wandb: logged scenario_sim @epoch {epoch + 1} "
            f"(pass rate {payload.get('scenario_sim/pass_rate')}%)",
            flush=True,
        )
    except Exception as exc:
        print(f"Warning: Failed to log scenario_sim to wandb: {exc}", flush=True)


def model_training(args: TrainConfig):
    save_path = args.save_dir
    assert len(args.coeff_timestep) == 4, "coeff_timestep must be a list of 4 elements"

    # init ddp
    global_rank, rank, world_size = ddp.ddp_setup_universal(True, args)
    print(f"{global_rank=}, {rank=}")

    if global_rank == 0:
        # Logging
        print("------------- {} -------------".format(args.exp_name))
        print("Batch size: {}".format(args.batch_size))
        print("Learning rate: {}".format(args.learning_rate))
        print("Use device: {}".format(args.device))
        print("Deterministic mode: {}".format(args.deterministic))

        # Save args
        args_dict = vars(args)
        args_dict = {
            k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict()
            for k, v in args_dict.items()
        }
        args_dict["major_version"] = 5

        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "args.json"), "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=4)

    # set seed
    set_seed(args.seed + global_rank)

    # Deterministic
    if args.deterministic:
        # Set CUBLAS_WORKSPACE_CONFIG to ensure deterministic behavior for cuBLAS operations.
        # 4096:8 means 24 MiB workspace with more memory, faster
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    save_utd = args.save_utd

    # set up data loaders
    if args.use_data_augment:
        if args.augment_type == "bridge":
            aug = BridgeStatePerturbation(augment_prob=args.augment_prob, device=args.device)
        elif args.augment_type == "frenet":
            aug = frenet_augmenter_from_args(args)
        else:
            aug = StatePerturbation(
                augment_prob=args.augment_prob,
                num_refine=args.num_refine,
                device=args.device,
                ego_past_noise_std=args.ego_past_noise_std,
                use_smoothing_future_trajectory=args.use_smoothing_future_trajectory,
            )
    else:
        aug = None

    # prepare dataset
    mode = shard_ddp.validate_args(args)
    train_sampler = None
    train_shard_ds = None
    if mode == "npz":
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

        # Validation is sharded across all ranks (DistributedSampler); each rank computes
        # metrics on its shard and they are all-reduced via aggregate_valid_metrics.
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
    else:
        train_loader, valid_loader, train_shard_ds, _ = shard_ddp.build_loaders(
            args, global_rank, ddp.get_world_size(), batch_size // ddp.get_world_size(), save_path
        )

    valid_pair_loader = None
    if args.enable_replan_consistency_eval:
        if mode == "shards":
            raise ValueError("--enable_replan_consistency_eval requires the npz path")
        expected_gap = args.replan_consistency_expected_gap or None
        valid_pair_set = DiffusionPlannerPairData(args.valid_set_list, expected_gap=expected_gap)
        if len(valid_pair_set) > 0:
            valid_pair_sampler = DistributedSampler(
                valid_pair_set,
                num_replicas=ddp.get_world_size(),
                rank=global_rank,
                shuffle=False,
            )
            valid_pair_loader = DataLoader(
                valid_pair_set,
                sampler=valid_pair_sampler,
                batch_size=batch_size // ddp.get_world_size(),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )

    if global_rank == 0:
        if mode == "npz":
            print("Dataset Prepared: {} train data\n".format(len(train_set)))
        else:
            print("Dataset Prepared: {} train data (shard mode)\n".format(len(train_shard_ds)))
        if args.enable_replan_consistency_eval:
            print(
                "Replan consistency validation pairs: {}".format(
                    0 if valid_pair_loader is None else len(valid_pair_loader.dataset)
                )
            )

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
            decay=getattr(args, "ema_decay", 0.999),
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
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer,
        train_epochs,
        args.warm_up_epoch,
        lr_schedule=args.lr_schedule,
    )

    if args.resume_model_path is not None:
        print(f"Model loaded from {args.resume_model_path}")
        # We always use new wandb run for each training session, so we don't need to load the wandb_id from the model_dict.
        diffusion_planner, optimizer, scheduler, init_epoch, _, model_ema = resume_model(
            args.resume_model_path,
            diffusion_planner,
            optimizer,
            scheduler,
            model_ema,
            args.device,
            use_ddp=args.ddp,
        )

        # Override learning rate with the new value
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate
        print(f"Learning rate reset to {args.learning_rate}")

    else:
        init_epoch = 0
    # logger
    if global_rank == 0:
        os.environ["WANDB_MODE"] = "online" if args.use_wandb else "offline"

        # if wandb_run_id is given, the training will be logged to the existing run instead of creating a new one.
        wandb.init(
            project=args.wandb_project_name,
            name=args.exp_name,
            notes=args.notes,
            resume="allow",
            id=args.wandb_run_id,
            dir=f"{save_path}",
        )

        wandb.config.update(args_dict)

        # this function creates dataset artifacts and associate them with wandb run
        # if wandb_run_id is given, the input artifact is assumed to be created externally and will not be executed
        if args.use_wandb and args.wandb_run_id is None:
            log_dataset_artifact(wandb.run, args.exp_name, args.train_set_list, args.valid_set_list)

    if args.ddp:
        torch.distributed.barrier()

    data_list = []
    best_loss = float("inf")

    valid_dict = validate_model(diffusion_planner, valid_loader, args)
    agg = aggregate_valid_metrics(valid_dict, args.device)
    replan_agg = {}
    if valid_pair_loader is not None:
        replan_dict = validate_replan_consistency(diffusion_planner, valid_pair_loader, args)
        replan_agg = aggregate_replan_consistency_metrics(replan_dict, args.device)
    if global_rank == 0:
        valid_loss_ego = agg["avg_loss_ego"]
        valid_loss_neighbor = agg["avg_loss_neighbor"]
        mean_ego_loss_dict = {f"valid_loss/{k}": v for k, v in agg["ego_means"].items()}
        mean_epdms_dict = wandb_epdms_metrics(agg["epdms_means"])
        valid_loss_ego_position_lat_loss = mean_ego_loss_dict.get(
            "valid_loss/ego_position_lat_loss", 0.0
        )
        valid_loss_ego_position_lon_loss = mean_ego_loss_dict.get(
            "valid_loss/ego_position_lon_loss", 0.0
        )
        turn_indicator_accuracy = agg["turn_indicator_accuracy"]
        turn_indicator_change_accuracy = agg["turn_indicator_change_accuracy"]
        turn_indicator_change_total = agg["turn_indicator_change_total"]
        print(
            f"{valid_loss_ego=:.3f}\n"
            f"{valid_loss_neighbor=:.3f}\n"
            f"{valid_loss_ego_position_lat_loss=:.3f}\n"
            f"{valid_loss_ego_position_lon_loss=:.3f}\n"
            f"{turn_indicator_accuracy=:.3f}\n"
            f"{turn_indicator_change_accuracy=:.3f}\n"
            f"{turn_indicator_change_total=:.3f}"
        )
        if replan_agg.get("replan_consistency_count", 0) > 0:
            print(
                "replan_position_consistency={:.3f}\n"
                "replan_heading_consistency={:.3f}\n"
                "replan_consistency_count={:d}".format(
                    replan_agg["replan_position_consistency"],
                    replan_agg["replan_heading_consistency"],
                    replan_agg["replan_consistency_count"],
                )
            )

    # begin training
    # Timing reference for the ETA: wall clock from the first epoch's start, so the average
    # epoch cost it is divided by includes checkpoint save / ONNX export / closed-loop epochs.
    training_start_time = time.perf_counter()
    for epoch in range(init_epoch, train_epochs):
        epoch_start_time = time.perf_counter()
        # Synchronize all processes before training
        if args.ddp:
            torch.distributed.barrier()

        # Adjust learning rate for the final 10 epochs (constant schedule only —
        # the cosine schedule already anneals to 0 and must not be overridden)
        adjusted_lr = final_phase_lr(args.learning_rate, epoch, train_epochs, args.lr_schedule)
        if adjusted_lr is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = adjusted_lr
            if global_rank == 0:
                print(f"Final phase: Epoch {epoch + 1}, LR adjusted to {adjusted_lr}")

        # training step
        train_start_time = time.perf_counter()
        try:
            train_loss, train_total_loss = train_epoch(
                train_loader, diffusion_planner, optimizer, args, model_ema, aug
            )
        except Exception as e:
            if mode == "shards":
                shard_ddp.coordinated_abort(e)
            raise
        train_sec = time.perf_counter() - train_start_time

        valid_start_time = time.perf_counter()
        valid_dict = validate_model(diffusion_planner, valid_loader, args)
        agg = aggregate_valid_metrics(valid_dict, args.device)
        replan_agg = {}
        if valid_pair_loader is not None:
            replan_dict = validate_replan_consistency(diffusion_planner, valid_pair_loader, args)
            replan_agg = aggregate_replan_consistency_metrics(replan_dict, args.device)
        valid_sec = time.perf_counter() - valid_start_time
        if global_rank == 0:
            valid_loss_ego = agg["avg_loss_ego"]
            valid_loss_neighbor = agg["avg_loss_neighbor"]
            mean_ego_loss_dict = {f"valid_loss/{k}": v for k, v in agg["ego_means"].items()}
            replan_loss_dict = {f"valid_loss/{k}": v for k, v in replan_agg.items()}
            mean_epdms_dict = wandb_epdms_metrics(agg["epdms_means"])
            valid_loss_ego_position_lat_loss = mean_ego_loss_dict.get(
                "valid_loss/ego_position_lat_loss", 0.0
            )
            valid_loss_ego_position_lon_loss = mean_ego_loss_dict.get(
                "valid_loss/ego_position_lon_loss", 0.0
            )
            turn_indicator_accuracy = agg["turn_indicator_accuracy"]
            turn_indicator_change_accuracy = agg["turn_indicator_change_accuracy"]
            turn_indicator_change_total = agg["turn_indicator_change_total"]
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
            if replan_agg.get("replan_consistency_count", 0) > 0:
                print(
                    "replan_position_consistency={:.3f}\n"
                    "replan_heading_consistency={:.3f}\n"
                    "replan_consistency_count={:d}".format(
                        replan_agg["replan_position_consistency"],
                        replan_agg["replan_heading_consistency"],
                        replan_agg["replan_consistency_count"],
                    )
                )

            # Timing, reported in hours (a single epoch is far too long for seconds to read
            # well). epoch_hour covers train + validation; the ETA is based on the average
            # wall-clock epoch of this run so far, which also absorbs the save/export epochs.
            epoch_sec = time.perf_counter() - epoch_start_time
            num_train_steps = len(train_loader)
            train_step_sec = train_sec / num_train_steps if num_train_steps > 0 else float("nan")
            elapsed_sec = time.perf_counter() - training_start_time
            epochs_done = epoch + 1 - init_epoch
            remaining_epochs = train_epochs - (epoch + 1)
            train_hour = train_sec / 3600.0
            valid_hour = valid_sec / 3600.0
            epoch_hour = epoch_sec / 3600.0
            elapsed_hour = elapsed_sec / 3600.0
            eta_hour = (elapsed_hour / epochs_done) * remaining_epochs
            time_dict = {
                "time/train_hour": train_hour,
                "time/valid_hour": valid_hour,
                "time/epoch_hour": epoch_hour,
                "time/train_step_sec": train_step_sec,
                "time/elapsed_hour": elapsed_hour,
                "time/eta_hour": eta_hour,
            }
            print(
                f"time: train={train_hour:.3f}h "
                f"(x{num_train_steps} steps, {train_step_sec:.3f}s/step), "
                f"valid={valid_hour:.3f}h, epoch={epoch_hour:.3f}h, "
                f"elapsed={elapsed_hour:.2f}h, eta={eta_hour:.2f}h"
            )

            lr_dict = {"lr": optimizer.param_groups[0]["lr"]}
            wandb.log(
                {
                    **{f"train_loss/{k}": v for k, v in train_loss.items()},
                    **{f"lr/{k}": v for k, v in lr_dict.items()},
                    **time_dict,
                    "valid_loss/ego": valid_loss_ego,
                    "valid_loss/neighbors": valid_loss_neighbor,
                    "valid_loss/turn_indicator_accuracy": turn_indicator_accuracy,
                    "valid_loss/turn_indicator_change_accuracy": turn_indicator_change_accuracy,
                    **mean_ego_loss_dict,
                    **replan_loss_dict,
                    **mean_epdms_dict,
                },
                step=epoch + 1,
            )

            scenario_output_dir = None
            if (epoch + 1 - init_epoch) % save_utd == 0:
                scenario_output_dir = os.path.join(
                    save_path,
                    f"epoch{epoch + 1:04d}",
                    "open_loop_override",
                )
            scenario_based_open_loop_validate(
                diffusion_planner,
                args,
                epoch,
                output_dir=scenario_output_dir,
            )

            curr_data = {
                "epoch": epoch + 1,
                "train_loss": train_total_loss,
                "valid_loss_ego": valid_loss_ego,
                "valid_loss_neighbor": valid_loss_neighbor,
                "valid_loss_ego_position_lat_loss": valid_loss_ego_position_lat_loss,
                "valid_loss_ego_position_lon_loss": valid_loss_ego_position_lon_loss,
                "train_hour": train_hour,
                "valid_hour": valid_hour,
                "epoch_hour": epoch_hour,
                **replan_agg,
                **{k.replace("/", "_"): v for k, v in mean_epdms_dict.items()},
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
                # We always use new wandb run for each training session, so we don't need to save the wandb_id in the model_dict.
                "wandb_id": None,
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
                # Export ONNX next to the checkpoint (regular weights, ORT validation skipped).
                export_checkpoint_onnx_guarded(
                    config_json_path=os.path.join(curr_dir, "args.json"),
                    ckpt_path=f"{curr_dir}/best_model.pth",
                    output_dir=Path(curr_dir),
                    output_prefix="diffusion_planner",
                    use_ema=False,
                    use_simplify=False,
                    opset_version=20,
                    external_data=False,
                )
                scenario_sim_validate(
                    args,
                    epoch,
                    f"{curr_dir}/best_model.pth",
                    os.path.join(curr_dir, "scenario_sim"),
                )

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
                # Export ONNX next to the checkpoint (regular weights, ORT validation skipped).
                export_checkpoint_onnx_guarded(
                    config_json_path=os.path.join(curr_dir, "args.json"),
                    ckpt_path=f"{curr_dir}/best_model.pth",
                    output_dir=Path(curr_dir),
                    output_prefix="diffusion_planner",
                    use_ema=False,
                    use_simplify=False,
                    opset_version=20,
                    external_data=False,
                )

        if epoch + 1 == train_epochs:
            # closed-loop validation runs on all ranks, only at the final epoch
            curr_dir = os.path.join(save_path, f"epoch{epoch + 1:04d}")
            os.makedirs(curr_dir, exist_ok=True)
            closed_loop_validate(
                diffusion_planner,
                args,
                epoch,
                os.path.join(curr_dir, "closed_loop"),
            )

        scheduler.step()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch + 1)
        else:
            train_shard_ds.set_epoch(epoch + 1)

    if global_rank == 0 and wandb.run is not None:
        wandb.finish()

    # Tear down the DDP process group explicitly: without this, NCCL's heartbeat +
    # IB event threads can intermittently deadlock interpreter shutdown and the
    # training process never exits (observed hanging a full R2LPL round).
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
