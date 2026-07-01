import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from diffusion_planner.dimensions import MAX_NUM_AGENTS, OUTPUT_T, POSE_DIM
from diffusion_planner.loss import (
    compute_ego_edge_points,
    compute_neighbor_collision_penalty,
    compute_road_border_penalty,
    loss_func,
    make_turn_indicator_gt,
)
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils import ddp


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

    # Progress is driven by the SLOWEST rank: every `progress_sync_every` batches all
    # ranks rendezvous on an all-reduce(MIN) of their completed-batch count and rank 0
    # displays that minimum, so the bar reaches 100% only when every rank is done. The
    # rendezvous also keeps a fast rank from racing far ahead (bounding memory imbalance).
    progress_sync_every = 20
    total_batches = len(val_loader)
    pbar = tqdm(total=total_batches, desc="validate (slowest rank)", disable=ddp.get_rank() != 0)
    for step, inputs in enumerate(val_loader):
        inputs = {key: value.to(device) for key, value in inputs.items()}
        B = inputs["ego_current_state"].shape[0]

        turn_indicator_seq = inputs["turn_indicators"]

        inputs["sampled_trajectories"] = torch.zeros(
            B, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM, dtype=torch.float32
        )
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
            predictions.append(prediction.cpu())
            turn_indicators.append(turn_indicator.cpu())

        neighbors_future_valid = ~neighbor_future_mask
        all_gt = all_gt[:, :, 1:, :]  # (B, Pn + 1, T, 4)
        loss_tensor = (prediction - all_gt) ** 2
        loss_ego = loss_tensor[:, 0, :]
        loss_ego_list.append(loss_ego.cpu())
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
            total_result_dict[f"ego_{key}"].append(val[:, 0, :].cpu())  # (B, T)

        # Compute ego edge points for penalty metrics
        ego_edge_points = compute_ego_edge_points(
            prediction[:, 0], inputs["ego_shape"], n_interp=args.road_border_n_interp
        )

        denorm_inputs = args.observation_normalizer.inverse(inputs)
        neighbor_penalty = compute_neighbor_collision_penalty(
            ego_edge_points,
            neighbors_future,
            neighbors_future_valid,
            denorm_inputs["neighbor_agents_past"],
            margin_vehicle=args.neighbor_collision_margin_vehicle,
            margin_pedestrian=args.neighbor_collision_margin_pedestrian,
            margin_bicycle=args.neighbor_collision_margin_bicycle,
        )
        total_result_dict["ego_neighbor_margin_loss"].append(neighbor_penalty.cpu())

        # Road border collision metric
        rb_penalty = compute_road_border_penalty(
            ego_edge_points,
            denorm_inputs["line_strings"],
            margin=args.road_border_margin,
        )
        total_result_dict["ego_road_border_loss"].append(rb_penalty.cpu())

        if (step + 1) % progress_sync_every == 0 or (step + 1) == total_batches:
            min_done = int(ddp.all_reduce_min(step + 1, device))
            if ddp.get_rank() == 0:
                pbar.n = min_done
                pbar.refresh()
    pbar.close()

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
        # Raw per-rank accumulators, kept so callers that run validation on ALL ranks
        # (DistributedSampler shards) can all-reduce them into globally-correct metrics
        # via aggregate_valid_metrics(). validate_model itself stays collective-free so
        # it remains safe to call on rank 0 only (as train.py / grpo do at some sites).
        "_loss_ego_sum": total_loss_ego,
        "_samples_ego": total_samples_ego,
        "_loss_neighbor_sum": total_loss_neighbor,
        "_samples_neighbor": total_samples_neighbor,
        "_turn_correct": turn_indicator_correct,
        "_turn_total": turn_indicator_total,
        "_turn_change_correct": turn_indicator_change_correct,
        **total_result_dict,
    }


def aggregate_valid_metrics(valid_dict, device):
    """All-reduce the scalar validation metrics across DDP ranks.

    COLLECTIVE: must be called by every rank that participated in the matching
    ``validate_model`` call. Returns a dict of globally-aggregated scalars; the
    per-sample tensors in ``valid_dict`` are left untouched (callers still use them
    to save per-data-point files). In single-process runs this is a no-op pass-through.

    Note: with ``DistributedSampler`` the dataset is padded to be divisible by the world
    size, so up to ``world_size - 1`` samples are duplicated across ranks. These averages
    therefore carry a negligible padding bias; the per-data-point files are exact.
    """
    loss_ego_sum = ddp.all_reduce_sum(valid_dict["_loss_ego_sum"], device)
    samples_ego = ddp.all_reduce_sum(valid_dict["_samples_ego"], device)
    loss_nei_sum = ddp.all_reduce_sum(valid_dict["_loss_neighbor_sum"], device)
    samples_nei = ddp.all_reduce_sum(valid_dict["_samples_neighbor"], device)
    turn_correct = ddp.all_reduce_sum(valid_dict["_turn_correct"], device)
    turn_total = ddp.all_reduce_sum(valid_dict["_turn_total"], device)
    turn_change_correct = ddp.all_reduce_sum(valid_dict["_turn_change_correct"], device)
    turn_change_total = ddp.all_reduce_sum(valid_dict["turn_indicator_change_total"], device)

    ego_means = {}
    for key, val in valid_dict.items():
        if key.startswith("ego_"):
            local_sum = ddp.all_reduce_sum(val.sum().item(), device)
            local_cnt = ddp.all_reduce_sum(val.numel(), device)
            ego_means[key] = local_sum / max(local_cnt, 1)

    return {
        "avg_loss_ego": loss_ego_sum / max(samples_ego, 1),
        "avg_loss_neighbor": loss_nei_sum / max(samples_nei, 1),
        "turn_indicator_accuracy": (turn_correct / turn_total) if turn_total > 0 else 0.0,
        "turn_indicator_change_accuracy": (
            (turn_change_correct / turn_change_total) if turn_change_total > 0 else 0.0
        ),
        "turn_indicator_change_total": int(turn_change_total),
        "ego_means": ego_means,
    }
