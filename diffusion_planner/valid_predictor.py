import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils import ddp
from diffusion_planner.utils.config import Config
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.path_key import data_path_to_rel
from diffusion_planner.utils.train_utils import resume_model, set_seed
from diffusion_planner.valid_config import ValidConfig
from diffusion_planner.validate_model import aggregate_valid_metrics, validate_model
from timm.utils import ModelEma
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args(args_list=None):
    """Parse command line arguments and return Namespace."""
    parser = argparse.ArgumentParser(description="Validation Entrypoint")

    parser.add_argument("--valid_set_list", type=str, default=None)
    parser.add_argument("--future_len", type=int, default=80)
    parser.add_argument("--agent_num", type=int, default=32)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--predicted_neighbor_num", type=int, default=32)
    parser.add_argument("--resume_model_path", type=str, required=True)
    parser.add_argument("--args_json_path", type=str, required=True)
    parser.add_argument("--save_predictions_dir", type=str, default=None)
    parser.add_argument("--ddp", default=True, type=boolean)
    parser.add_argument("--port", default="22323", type=str)

    return parser.parse_args(args_list)


def run_validation(valid_cfg: ValidConfig):
    """Core logic for validation."""

    # 1. Restore model configuration from training settings (args.json)
    config_obj = Config(valid_cfg.args_json_path)

    # Override and synchronize batch size and DDP settings for validation execution
    # (Note: Depending on the Config class implementation, it is safer to pass DDP and device info here)
    config_obj.device = valid_cfg.device
    config_obj.ddp = valid_cfg.ddp

    # init ddp
    global_rank, rank, _ = ddp.ddp_setup_universal(True, valid_cfg)
    print(f"{global_rank=}, {rank=}")

    if global_rank == 0:
        print(f"Batch size: {valid_cfg.batch_size}")
        print(f"Use device: {valid_cfg.device}")

    # set seed
    set_seed(valid_cfg.seed + global_rank)

    # set up data loaders
    valid_set = DiffusionPlannerData(valid_cfg.valid_set_list)
    valid_sampler = DistributedSampler(
        valid_set, num_replicas=ddp.get_world_size(), rank=global_rank, shuffle=False
    )
    valid_loader = DataLoader(
        valid_set,
        sampler=valid_sampler,
        batch_size=valid_cfg.batch_size // ddp.get_world_size(),
        num_workers=valid_cfg.num_workers,
        pin_memory=valid_cfg.pin_mem,
        drop_last=False,
    )

    if global_rank == 0:
        print("Dataset Prepared: {} valid data\n".format(len(valid_set)))

    if valid_cfg.ddp:
        torch.distributed.barrier()

    # set up model (restore structure using training config_obj)
    diffusion_planner = Diffusion_Planner(config_obj)
    diffusion_planner = diffusion_planner.to(
        rank if valid_cfg.device == "cuda" else valid_cfg.device
    )

    if valid_cfg.ddp:
        diffusion_planner = DDP(diffusion_planner, device_ids=[rank], find_unused_parameters=True)

    if global_rank == 0:
        print(
            "Model Params: {}".format(
                sum(p.numel() for p in ddp.get_model(diffusion_planner, valid_cfg.ddp).parameters())
            )
        )

    # optimizer (dummy)
    params = [{"params": ddp.get_model(diffusion_planner, valid_cfg.ddp).parameters(), "lr": 0.0}]
    optimizer = optim.AdamW(params)

    # load weights
    print(f"Model loaded from {valid_cfg.resume_model_path}")
    model_ema = ModelEma(diffusion_planner, decay=0.999, device=valid_cfg.device)

    diffusion_planner, _, _, _, _, _ = resume_model(
        valid_cfg.resume_model_path,
        diffusion_planner,
        optimizer,
        None,  # scheduler is not needed
        model_ema,
        valid_cfg.device,
    )

    if valid_cfg.ddp:
        torch.distributed.barrier()

    valid_dict = validate_model(diffusion_planner, valid_loader, config_obj, return_pred=True)

    # Per-rank tensors (this rank's DistributedSampler shard, in loader order).
    loss_ego = valid_dict["loss_ego"]
    predictions = valid_dict["predictions"]
    turn_indicators = valid_dict["turn_indicators"]

    # Aggregate the scalar metrics across all ranks (collective; every rank calls
    # validate_model above, so every rank reaches here).
    agg = aggregate_valid_metrics(valid_dict, valid_cfg.device)
    avg_loss_ego = agg["avg_loss_ego"]
    avg_loss_neighbor = agg["avg_loss_neighbor"]
    turn_indicator_accuracy = agg["turn_indicator_accuracy"]
    turn_indicator_change_accuracy = agg["turn_indicator_change_accuracy"]
    turn_indicator_change_total = agg["turn_indicator_change_total"]

    if global_rank == 0:
        print(f"{avg_loss_ego=:.4f} {avg_loss_neighbor=:.4f}")
        print(f"{turn_indicator_accuracy=:.4f}")
        if turn_indicator_change_total > 0:
            print(f"{turn_indicator_change_accuracy=:.4f} ({turn_indicator_change_total=:d})")
        else:
            print("turn_indicator_change_accuracy=0.0000 (num_samples=0)")
        if "ego_neighbor_margin_loss" in agg["ego_means"]:
            print(
                f"ego_neighbor_margin_loss_mean={agg['ego_means']['ego_neighbor_margin_loss']:.4f}"
            )
        if "ego_road_border_loss" in agg["ego_means"]:
            print(f"ego_road_border_loss_mean={agg['ego_means']['ego_road_border_loss']:.4f}")

    # Save results
    if valid_cfg.save_predictions_dir is None:
        return

    save_predictions_dir = Path(valid_cfg.save_predictions_dir)
    save_predictions_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate metrics JSON is global; write once from rank 0.
    if global_rank == 0:
        valid_dict_to_save = {
            "avg_loss_ego": avg_loss_ego,
            "avg_loss_neighbor": avg_loss_neighbor,
            "turn_indicator_accuracy": turn_indicator_accuracy,
            "turn_indicator_change_accuracy": turn_indicator_change_accuracy,
            "turn_indicator_change_total": turn_indicator_change_total,
            **agg["ego_means"],
        }
        with open(save_predictions_dir.parent / "valid_dict.json", "w") as f:
            json.dump(valid_dict_to_save, f, indent=4)

    # Map each prediction (loader order) back to its source data path, and save under a
    # path that mirrors the input's directory hierarchy. The relative path is unique per
    # data point, so ranks never collide and the local index is irrelevant.
    sampler_indices = list(valid_sampler)
    assert len(sampler_indices) == predictions.shape[0]

    # Progress is driven by the SLOWEST rank (see validate_model): every `save_sync_every`
    # files all ranks rendezvous on all-reduce(MIN) and rank 0 displays that minimum, so
    # the bar reaches 100% only when every rank has finished saving its shard.
    save_sync_every = 200
    n_save = predictions.shape[0]
    pbar = tqdm(total=n_save, desc="save (slowest rank)", disable=global_rank != 0)
    for i in range(n_save):
        rel = data_path_to_rel(valid_set.data_list[sampler_indices[i]])
        out_base = save_predictions_dir / rel
        out_base.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_base.with_suffix(".npz"),
            prediction=predictions[i].cpu().numpy(),
            turn_indicator=turn_indicators[i].cpu().numpy(),
        )
        loss_dict = {
            "loss_ego_total": loss_ego[i].mean().item(),
            "loss_ego_3sec": torch.sqrt(loss_ego[i, 30 - 1, :2].sum()).item(),
            "loss_ego_5sec": torch.sqrt(loss_ego[i, 50 - 1, :2].sum()).item(),
            "loss_ego_8sec": torch.sqrt(loss_ego[i, 80 - 1, :2].sum()).item(),
        }
        for key_metric, val in valid_dict.items():
            if not key_metric.startswith("ego_"):
                continue
            loss_dict[key_metric] = val[i].mean().item()
        with open(out_base.with_suffix(".json"), "w") as f:
            json.dump(loss_dict, f, indent=4)

        if (i + 1) % save_sync_every == 0 or (i + 1) == n_save:
            min_done = int(ddp.all_reduce_min(i + 1, valid_cfg.device))
            if global_rank == 0:
                pbar.n = min_done
                pbar.refresh()
    pbar.close()


def main():
    # 1. Parse command line arguments
    args = get_args()
    args_dict = vars(args)

    # 2. Convert Namespace to Dataclass
    valid_cfg = ValidConfig(**args_dict)

    # 3. Execute validation logic
    run_validation(valid_cfg)


if __name__ == "__main__":
    main()
