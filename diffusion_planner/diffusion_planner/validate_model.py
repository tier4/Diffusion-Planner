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
from planner_metrics import (
    EPDMSLikeConfig,
    RewardConfig,
    compute_subscores_batch,
    epdms_like_aggregate,
    gt_path_length,
)

# Reward subscores logged as additive validation metrics (valid_loss/ego_subscore_*).
# These are the RLVR reward's subscores (EPDMS-INSPIRED — custom thresholds,
# goal-based ego-progress, penalty signs), NOT a faithful EPDMS port (that lives
# in OnePlanner; see issue #142). Logging them lets best-model selection see the
# same physical quantities the reward is built from. Best-model selection itself
# is unchanged.
_VAL_SUBSCORE_KEYS = (
    "safety",
    "ttc",
    "progress",
    "comfort",
    "centerline",
    "red_light",
    "feasibility",
)
_VAL_SUBSCORE_CFG = RewardConfig()
_VAL_EPDMS_LIKE_CFG = EPDMSLikeConfig()
# Components returned by epdms_like_aggregate, logged as ``ego_subscore_<key>``.
# ``epdms_like`` is the single [0,1] EPDMS-structured proxy score (NOT a faithful
# NAVSIM EPDMS; see #142); the rest are its binary gates and normalized quality
# terms, kept for debugging why a checkpoint scores the way it does.
_VAL_EPDMS_LIKE_KEYS = (
    "epdms_like",
    "gate_nc",
    "gate_dac",
    "gate_tlc",
    "gate_kin",
    "q_ttc",
    "q_progress",
    "q_comfort",
    "q_lane",
    "quality",
)


@torch.no_grad()
def _reward_subscores_per_scene(
    ego_pred, data_batched, config, keys, gt_progress=None, epdms_cfg=None
):
    """Per-scene reward subscores for a validation batch.

    ``compute_subscores_batch`` is single-scene / N-trajectory (its map + neighbor
    terms use one scene's tensors), so this loops over the ``B`` scenes — each with
    one ego prediction — and stacks the requested subscores.

    Args:
        ego_pred: ``(B, T, 4)`` ego predictions, metre ego-frame.
        data_batched: dict of ``(B, ...)`` tensors (ego_shape / neighbors / map /
            goal); each is sliced ``[b:b+1]`` per scene.
        config: RewardConfig (thresholds; weights are irrelevant to raw subscores).
        keys: subscore names to return.

    Returns:
        ``{name: (B,) tensor}`` for each requested key. When ``gt_progress`` is
        provided, the dict also includes the EPDMS-like component keys (``epdms_like``
        plus its gates / normalized quality terms; see ``_VAL_EPDMS_LIKE_KEYS``).
    """
    n = ego_pred.shape[0]
    acc = {name: [] for name in keys}
    want_epdms = gt_progress is not None
    epdms_acc = {k: [] for k in _VAL_EPDMS_LIKE_KEYS} if want_epdms else {}
    for b in range(n):
        data_b = {k: v[b : b + 1] for k, v in data_batched.items()}
        subs_b = compute_subscores_batch(ego_pred[b : b + 1], data_b, config)
        for name in keys:
            acc[name].append(subs_b[name])
        if want_epdms:
            _, comp_b = epdms_like_aggregate(subs_b, gt_progress[b : b + 1], epdms_cfg)
            for k in _VAL_EPDMS_LIKE_KEYS:
                epdms_acc[k].append(comp_b[k])
    out = {name: torch.cat(vals) for name, vals in acc.items()}
    out.update({k: torch.cat(vals) for k, vals in epdms_acc.items()})
    return out


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

    for inputs in tqdm(val_loader, desc="validate", disable=ddp.get_rank() != 0):
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
            margin=args.neighbor_collision_margin,
        )
        total_result_dict["ego_neighbor_margin_loss"].append(neighbor_penalty.cpu())

        # Road border collision metric
        rb_penalty = compute_road_border_penalty(
            ego_edge_points,
            denorm_inputs["line_strings"],
            margin=args.road_border_margin,
        )
        total_result_dict["ego_road_border_loss"].append(rb_penalty.cpu())

        # Reward subscores as additive validation metrics (logged via the ego_*
        # machinery as valid_loss/ego_subscore_*). Same metre ego-frame tensors
        # the rb / neighbor metrics above use; one scene per prediction.
        data_batched = {
            "ego_shape": inputs["ego_shape"],
            "neighbor_agents_future": neighbors_future,
            "neighbor_agents_past": denorm_inputs["neighbor_agents_past"],
        }
        for k in ("lanes", "route_lanes", "line_strings", "polygons", "goal_pose"):
            if k in denorm_inputs:
                data_batched[k] = denorm_inputs[k]
        gt_progress = gt_path_length(ego_future)  # (B,) expert path length, metres
        subscores = _reward_subscores_per_scene(
            prediction[:, 0],
            data_batched,
            _VAL_SUBSCORE_CFG,
            _VAL_SUBSCORE_KEYS,
            gt_progress=gt_progress,
            epdms_cfg=_VAL_EPDMS_LIKE_CFG,
        )
        for name, val in subscores.items():
            total_result_dict[f"ego_subscore_{name}"].append(val.cpu())  # (B,)

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
