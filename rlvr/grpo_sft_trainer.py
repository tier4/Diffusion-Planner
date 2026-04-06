"""GRPO-ranked SFT trainer: generate N trajectories, pick best by reward, SFT on it.

This hybrid approach combines GRPO's trajectory generation and reward scoring
with standard SFT diffusion training. For each scene:
  1. Generate N trajectories using the batched sampler
  2. Score all trajectories with the reward function
  3. Select the best-reward trajectory
  4. Apply Savitzky-Golay filter to smooth it
  5. Train LoRA using standard diffusion SFT loss (MSE at random timestep t)

Two neighbor modes:
  - "gt_neighbor": use real GT neighbor trajectories from NPZ data
  - "baseline_neighbor": use baseline (no-LoRA) model prediction as neighbor target
"""

from __future__ import annotations

import contextlib
import gc
import random as _random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import savgol_filter
from torch import nn
from tqdm import tqdm

from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.module.decoder import generate_prefix_mask
from preference_optimization.utils import load_npz_data
from rlvr.grpo_config import GRPOConfig
from rlvr.grpo_trainer_batched import (
    _normalize_batch,
    _stack_scene_data,
    generate_all_scenes_batched,
)
from rlvr.reward import RewardConfig, compute_reward_batch


def _smooth_trajectory(traj: np.ndarray, window: int, order: int) -> np.ndarray:
    """Apply Savitzky-Golay filter to smooth a trajectory.

    Args:
        traj: [T, 4] trajectory (x, y, cos_heading, sin_heading).
        window: SG filter window length (must be odd).
        order: SG filter polynomial order.

    Returns:
        [T, 4] smoothed trajectory.
    """
    T = traj.shape[0]
    # Window must be odd and <= T
    w = min(window, T)
    if w % 2 == 0:
        w -= 1
    if w < order + 2:
        return traj  # too short to filter
    smoothed = np.copy(traj)
    # Smooth x, y
    smoothed[:, 0] = savgol_filter(traj[:, 0], w, order)
    smoothed[:, 1] = savgol_filter(traj[:, 1], w, order)
    # Smooth heading: filter cos/sin then renormalize
    smoothed[:, 2] = savgol_filter(traj[:, 2], w, order)
    smoothed[:, 3] = savgol_filter(traj[:, 3], w, order)
    # Renormalize cos/sin to unit circle
    norm = np.sqrt(smoothed[:, 2] ** 2 + smoothed[:, 3] ** 2).clip(min=1e-6)
    smoothed[:, 2] /= norm
    smoothed[:, 3] /= norm
    return smoothed


def _get_baseline_neighbor_prediction(
    model: nn.Module,
    model_args,
    norm_data: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Run the model with LoRA disabled to get baseline neighbor predictions.

    Args:
        model: LoRA-wrapped model.
        model_args: Config from load_model.
        norm_data: Normalized observation dict (B=1).
        device: Torch device.

    Returns:
        [Pn, T, 4] baseline neighbor predictions (denormalized).
    """
    inner = model.module if hasattr(model, "module") else model
    use_lora_disable = hasattr(inner, "disable_adapter")

    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len
    B = norm_data["ego_current_state"].shape[0]

    ego_current = norm_data["ego_current_state"][:, :4]
    neighbors_current = norm_data["neighbor_agents_past"][:, :P - 1, -1, :4]
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)

    xT = current_states[:, :, None, :].expand(-1, -1, future_len + 1, -1).clone()
    xT[:, :, 1:, :] = 0.0  # deterministic

    data_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v
                 for k, v in norm_data.items()}
    data_copy["sampled_trajectories"] = xT

    ctx = inner.disable_adapter() if use_lora_disable else contextlib.nullcontext()
    with ctx, torch.no_grad():
        _, decoder_output = model(data_copy)
        # [B, P, T+1, 4] -> neighbor predictions [B, Pn, T, 4]
        if "prediction" in decoder_output:
            full_pred = decoder_output["prediction"]  # [B, P, T, 4]
            neighbor_pred = full_pred[:, 1:, :, :]  # [B, Pn, T, 4]
        elif "model_output" in decoder_output:
            full_pred = decoder_output["model_output"][:, :, 1:, :]  # [B, P, T, 4]
            neighbor_pred = full_pred[:, 1:, :, :]  # [B, Pn, T, 4]
        else:
            raise KeyError("Model output missing 'prediction' and 'model_output'")

    return neighbor_pred[0].detach()  # [Pn, T, 4]


def _compute_sft_diffusion_loss(
    model: nn.Module,
    model_args,
    data: dict[str, torch.Tensor],
    ego_gt: torch.Tensor,
    neighbor_gt: torch.Tensor,
    neighbor_mask: torch.Tensor,
    device: torch.device,
    K: int = 8,
    neighbor_reg_weight: float = 0.0,
    neighbor_reg_only: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute standard SFT diffusion loss with ego + neighbor targets.

    Matches the SFT training procedure from decoder.py:
    - Sample random timestep t ~ U[eps, 1)
    - Add noise via VPSDE marginal_prob
    - Model predicts x_0 from x_t
    - MSE loss against GT

    When neighbor_reg_weight > 0, adds a regularization term that penalizes
    the LoRA model's neighbor predictions from diverging from the base model's
    neighbor predictions at the same (noise, timestep) inputs.

    Args:
        model: Policy model (LoRA-wrapped).
        model_args: Config from load_model.
        data: Raw observation dict (NOT normalized). B=1.
        ego_gt: [B, T, 4] ego ground truth trajectory (x, y, cos, sin).
        neighbor_gt: [B, Pn, T, 4] neighbor ground truth trajectories.
        neighbor_mask: [B, Pn, T] boolean mask (True = invalid/padded).
        device: Torch device.
        K: Number of (noise, timestep) samples to average over.
        neighbor_reg_weight: Weight for neighbor regularization loss (0=disabled).

    Returns:
        (loss, metrics_dict)
    """
    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    Pn = P - 1
    future_len = model_args.future_len
    eps = 1e-3

    # Normalize the ego GT and neighbor GT using the state normalizer
    norm = model_args.state_normalizer
    ego_mean = norm.mean[0].to(device)
    ego_std = norm.std[0].to(device)
    ego_gt_norm = (ego_gt - ego_mean) / ego_std  # [B, T, 4]

    # For neighbors, use the same normalizer
    neighbor_gt_norm = (neighbor_gt - ego_mean) / ego_std  # [B, Pn, T, 4]
    # Zero out invalid neighbors
    neighbor_gt_norm[neighbor_mask] = 0.0

    # Build current states
    ego_current = data["ego_current_state"][:, :4]
    neighbors_current = data["neighbor_agents_past"][:, :Pn, -1, :4]
    ego_current_norm = (ego_current - ego_mean) / ego_std
    neighbors_current_norm = (neighbors_current - ego_mean) / ego_std
    current_states = torch.cat([ego_current_norm[:, None], neighbors_current_norm], dim=1)  # [B, P, 4]

    # Neighbor current validity mask
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0  # [B, Pn]
    # Full neighbor mask: [B, Pn, T+1] (current + future)
    full_neighbor_mask = torch.cat(
        (neighbor_current_mask.unsqueeze(-1), neighbor_mask), dim=-1
    )  # [B, Pn, T+1]

    # Build all_gt: [B, P, T+1, 4]
    gt_future = torch.cat([ego_gt_norm[:, None, :, :], neighbor_gt_norm], dim=1)  # [B, P, T, 4]
    all_gt = torch.cat([current_states[:, :, None, :], gt_future], dim=2)  # [B, P, T+1, 4]
    # Zero out invalid neighbors
    all_gt[:, 1:][full_neighbor_mask] = 0.0

    # Normalize observation data
    data_normalized = model_args.observation_normalizer(
        {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    )

    total_ego_loss = 0.0
    total_neighbor_loss = 0.0
    total_neighbor_reg_loss = 0.0
    # Combine future padding mask with current-timestep validity:
    # a neighbor absent at the current timestep should not contribute to loss.
    neighbors_future_valid = ~neighbor_mask  # [B, Pn, T]
    neighbors_future_valid = neighbors_future_valid & (~neighbor_current_mask.unsqueeze(-1))  # [B, Pn, T]

    # Check if model supports LoRA disable (needed for neighbor regularization)
    inner = model.module if hasattr(model, "module") else model
    use_neighbor_reg = neighbor_reg_weight > 0.0 and Pn > 0 and hasattr(inner, "disable_adapter")

    for _ in range(K):
        # Sample random timestep
        t = torch.rand(B, device=device) * (1 - eps) + eps
        t_4d = t.view(B, 1, 1, 1).expand(B, P, future_len + 1, 1).clone()

        # Prefix mask with random delay
        max_delay = 5
        delay = torch.randint(0, max_delay + 1, (B,), device=device)
        prefix_mask = generate_prefix_mask(delay, P, future_len + 1)
        mask_coeff = _random.uniform(0.0, 1.0)
        curr_mask_time = torch.maximum(t_4d * mask_coeff, torch.tensor(eps, device=device))
        t_4d = torch.where(prefix_mask, curr_mask_time, t_4d)

        # Noise and diffusion
        z = torch.randn(B, P, future_len, 4, device=device)
        mean, std = VPSDE_linear().marginal_prob(all_gt[..., 1:, :], t_4d[..., 1:, :])
        xT = mean + std * z
        xT_full = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
        xT_full = torch.where(prefix_mask, all_gt, xT_full)

        # Forward pass
        merged_inputs = {**data_normalized}
        merged_inputs["gt_trajectories"] = all_gt
        merged_inputs["sampled_trajectories"] = xT_full
        merged_inputs["diffusion_time"] = t_4d
        merged_inputs["prefix_mask"] = prefix_mask
        if "delay" not in merged_inputs:
            merged_inputs["delay"] = delay

        _, outputs = model(merged_inputs)

        if "model_output" in outputs:
            model_output = outputs["model_output"][:, :, 1:, :]  # [B, P, T, 4]
        else:
            raise KeyError("Model output missing 'model_output'")

        gt_target = all_gt[:, :, 1:, :]  # [B, P, T, 4]

        # Ego loss: MSE over all timesteps
        ego_loss = F.mse_loss(model_output[:, 0], gt_target[:, 0])
        total_ego_loss += ego_loss

        # Neighbor loss: MSE over valid timesteps only (skipped when neighbor_reg_only)
        if Pn > 0 and neighbors_future_valid.any() and not neighbor_reg_only:
            neighbor_pred = model_output[:, 1:]  # [B, Pn, T, 4]
            neighbor_target = gt_target[:, 1:]  # [B, Pn, T, 4]
            # Per-element MSE, then mask
            neighbor_mse = ((neighbor_pred - neighbor_target) ** 2).mean(dim=-1)  # [B, Pn, T]
            masked_loss = neighbor_mse[neighbors_future_valid]
            if masked_loss.numel() > 0:
                total_neighbor_loss += masked_loss.mean()

        # Neighbor regularization: MSE(lora_neighbor, base_neighbor) at same inputs
        if use_neighbor_reg and neighbors_future_valid.any():
            with inner.disable_adapter(), torch.no_grad():
                _, base_outputs = model(merged_inputs)
            base_neighbor = base_outputs["model_output"][:, 1:, 1:, :]  # [B, Pn, T, 4]
            lora_neighbor = model_output[:, 1:]  # [B, Pn, T, 4]
            reg_mse = ((lora_neighbor - base_neighbor.detach()) ** 2).mean(dim=-1)  # [B, Pn, T]
            masked_reg = reg_mse[neighbors_future_valid]
            if masked_reg.numel() > 0:
                total_neighbor_reg_loss += masked_reg.mean()

    ego_loss_avg = total_ego_loss / K
    neighbor_loss_avg = total_neighbor_loss / K if isinstance(total_neighbor_loss, torch.Tensor) else torch.tensor(0.0, device=device)
    neighbor_reg_avg = total_neighbor_reg_loss / K if isinstance(total_neighbor_reg_loss, torch.Tensor) else torch.tensor(0.0, device=device)

    # Combined loss: ego + neighbor (neighbor weight = 1.0 to match SFT) + neighbor reg
    loss = ego_loss_avg + neighbor_loss_avg
    if use_neighbor_reg:
        loss = loss + neighbor_reg_weight * neighbor_reg_avg

    metrics = {
        "sft_ego_loss": ego_loss_avg.item(),
        "sft_neighbor_loss": neighbor_loss_avg.item() if isinstance(neighbor_loss_avg, torch.Tensor) else 0.0,
        "sft_total_loss": loss.item(),
        "sft_neighbor_reg_loss": neighbor_reg_avg.item() if isinstance(neighbor_reg_avg, torch.Tensor) else 0.0,
    }
    return loss, metrics


def train_epoch_ranked_sft(
    model: nn.Module,
    model_args,
    optimizer: torch.optim.Optimizer,
    scene_paths: list[str],
    config: GRPOConfig,
    reward_config: RewardConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """GRPO-ranked SFT epoch: generate, rank, filter, train with SFT loss.

    Steps:
      1. Load scenes, generate K trajectories per scene (batched)
      2. Score all trajectories with reward function
      3. For each scene, select the best-reward trajectory
      4. Apply Savitzky-Golay filter to smooth it
      5. Train LoRA using standard SFT diffusion loss

    Args:
        model: LoRA-wrapped policy model.
        model_args: Config from load_model.
        optimizer: AdamW optimizer for LoRA parameters.
        scene_paths: List of NPZ file paths.
        config: GRPOConfig with ranked_sft_mode, sg_filter_*, etc.
        reward_config: Reward configuration for scoring.
        device: Torch device.
        epoch: Current epoch number (1-indexed).

    Returns:
        Dict of averaged training metrics.
    """
    torch.cuda.empty_cache()
    gc.collect()

    K = config.num_generations
    mode = config.ranked_sft_mode
    assert mode in ("gt_neighbor", "baseline_neighbor"), (
        f"ranked_sft_mode must be 'gt_neighbor' or 'baseline_neighbor', got {mode!r}"
    )

    if epoch == 1:
        print(f"  [ranked-sft] mode={mode}, K={K}, "
              f"sg_window={config.sg_filter_window}, sg_order={config.sg_filter_order}, "
              f"neighbor_reg={config.neighbor_reg_weight}, reg_only={config.neighbor_reg_only}")

    # 1. Load all scenes
    print(f"  Loading {len(scene_paths)} scenes...")
    all_data = []
    valid_paths = []
    for path in scene_paths:
        try:
            data = load_npz_data(path, device)
            all_data.append(data)
            valid_paths.append(path)
        except Exception as e:
            from pathlib import Path as _Path
            print(f"  [skip] {_Path(path).name}: {e}")

    N = len(all_data)
    if N == 0:
        return {}

    # 2. Stack and normalize for batched generation
    print(f"  Stacking {N} scenes into batch...")
    batch_data = _stack_scene_data(all_data, device)
    norm_batch = _normalize_batch(batch_data, model_args)

    # Compute GT max speed for speed guidance
    gt_speeds_list = []
    for d in all_data:
        gt = d.get("ego_agent_future")
        if gt is not None:
            if gt.dim() == 3:
                gt = gt[0]
            gt_np = gt.cpu().numpy()
            gt_valid = ~((gt_np[:, 0] == 0) & (gt_np[:, 1] == 0))
            if gt_valid.sum() >= 5:
                vel = np.diff(gt_np[gt_valid][:, :2], axis=0) / 0.1
                gt_speeds_list.append(float(np.linalg.norm(vel, axis=-1).max()))
            else:
                gt_speeds_list.append(3.0)
        else:
            gt_speeds_list.append(3.0)
    median_gt_speed = float(np.median(gt_speeds_list))

    # 3. Generate K trajectories for all scenes (batched)
    print(f"  Generating {K} trajectories x {N} scenes (batched)...")
    model.eval()
    with torch.no_grad():
        all_trajs = generate_all_scenes_batched(
            model, model_args, norm_batch, K, config.noise_scale_range, device,
            gt_max_speed=median_gt_speed,
        )  # [N, K, T, 4]

    torch.cuda.empty_cache()
    gc.collect()

    # 4. Score and select best trajectory per scene
    print(f"  Scoring and selecting best trajectories...")
    best_ego_trajs = []  # [T, 4] numpy arrays
    scene_norm_data = []  # per-scene normalized data dicts
    scene_raw_data = []  # per-scene raw data dicts
    best_rewards_list = []

    for i in tqdm(range(N), desc="Scoring"):
        traj_K = all_trajs[i]  # [K, T, 4]
        data_i = all_data[i]

        rewards = compute_reward_batch(traj_K, data_i, reward_config)
        reward_vals = np.array([r.total for r in rewards])
        best_idx = int(np.argmax(reward_vals))
        best_reward = reward_vals[best_idx]

        # Get best trajectory and smooth it
        best_traj = traj_K[best_idx].cpu().numpy()  # [T, 4]
        best_traj_smooth = _smooth_trajectory(
            best_traj, config.sg_filter_window, config.sg_filter_order
        )

        best_ego_trajs.append(best_traj_smooth)
        best_rewards_list.append(best_reward)

        # Per-scene normalized data (B=1 slice)
        norm_i = {}
        for k, v in norm_batch.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == N:
                norm_i[k] = v[i:i + 1]
            else:
                norm_i[k] = v
        scene_norm_data.append(norm_i)
        scene_raw_data.append(data_i)

    mean_best_reward = float(np.mean(best_rewards_list))
    print(f"  Mean best-of-{K} reward: {mean_best_reward:.2f}")

    # Free generation tensors
    del all_trajs
    torch.cuda.empty_cache()
    gc.collect()

    # 5. Optionally compute baseline neighbor predictions (once, before training)
    baseline_neighbor_preds = []
    if mode == "baseline_neighbor":
        print(f"  Computing baseline neighbor predictions...")
        model.eval()
        for i in tqdm(range(N), desc="Baseline neighbors"):
            neighbor_pred = _get_baseline_neighbor_prediction(
                model, model_args, scene_norm_data[i], device
            )  # [Pn, T, 4]
            baseline_neighbor_preds.append(neighbor_pred)
        torch.cuda.empty_cache()

    # 6. Train with SFT diffusion loss
    print(f"  Training on {N} scenes (ranked SFT, mode={mode})...")
    model.train()
    optimizer.zero_grad()

    all_metrics = {}
    n_scenes = 0
    accum_count = 0
    accum_target = config.grad_accum_groups

    Pn = model_args.predicted_neighbor_num
    future_len = model_args.future_len

    for i in tqdm(range(N), desc="Training"):
        data_i = scene_raw_data[i]

        # Ego GT: the filtered best trajectory
        ego_gt_np = best_ego_trajs[i]  # [T, 4]
        ego_gt = torch.tensor(ego_gt_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, 4]
        # Truncate or pad to future_len
        T_actual = ego_gt.shape[1]
        if T_actual > future_len:
            ego_gt = ego_gt[:, :future_len, :]
        elif T_actual < future_len:
            pad = torch.zeros(1, future_len - T_actual, 4, device=device)
            ego_gt = torch.cat([ego_gt, pad], dim=1)

        # Neighbor GT
        if mode == "gt_neighbor":
            # Use real GT from NPZ data
            neighbors_future = data_i.get("neighbor_agents_future")
            if neighbors_future is not None:
                if neighbors_future.dim() == 3:
                    neighbors_future = neighbors_future.unsqueeze(0)
                # [B, Pn_raw, T, D] -> truncate to predicted_neighbor_num
                neighbors_future = neighbors_future[:, :Pn, :, :]
                # Convert heading to cos/sin if needed (3D -> 4D)
                if neighbors_future.shape[-1] == 3:
                    neighbors_future = torch.cat([
                        neighbors_future[..., :2],
                        neighbors_future[..., 2:3].cos(),
                        neighbors_future[..., 2:3].sin(),
                    ], dim=-1)
                # Mask invalid timesteps
                neighbor_mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0  # [B, Pn, T]
                # Pad to future_len if needed
                T_n = neighbors_future.shape[2]
                if T_n < future_len:
                    pad_n = torch.zeros(1, Pn, future_len - T_n, 4, device=device)
                    neighbors_future = torch.cat([neighbors_future, pad_n], dim=2)
                    pad_mask = torch.ones(1, Pn, future_len - T_n, dtype=torch.bool, device=device)
                    neighbor_mask = torch.cat([neighbor_mask, pad_mask], dim=2)
                elif T_n > future_len:
                    neighbors_future = neighbors_future[:, :, :future_len, :]
                    neighbor_mask = neighbor_mask[:, :, :future_len]
                # Pad Pn dimension if needed
                actual_pn = neighbors_future.shape[1]
                if actual_pn < Pn:
                    pad_pn = torch.zeros(1, Pn - actual_pn, future_len, 4, device=device)
                    neighbors_future = torch.cat([neighbors_future, pad_pn], dim=1)
                    pad_mask_pn = torch.ones(1, Pn - actual_pn, future_len, dtype=torch.bool, device=device)
                    neighbor_mask = torch.cat([neighbor_mask, pad_mask_pn], dim=1)
                neighbors_future = neighbors_future[:, :Pn, :future_len, :4]
                neighbor_mask = neighbor_mask[:, :Pn, :future_len]
            else:
                neighbors_future = torch.zeros(1, Pn, future_len, 4, device=device)
                neighbor_mask = torch.ones(1, Pn, future_len, dtype=torch.bool, device=device)
        else:
            # baseline_neighbor mode
            neighbor_pred = baseline_neighbor_preds[i]  # [Pn, T, 4]
            # The baseline predictions are already denormalized by the decoder
            # (inference mode applies state_normalizer.inverse). No further
            # denormalization needed — the loss function normalizes internally.
            neighbors_future = neighbor_pred.unsqueeze(0)  # [1, Pn, T, 4]
            # Truncate/pad to future_len
            T_n = neighbors_future.shape[2]
            if T_n > future_len:
                neighbors_future = neighbors_future[:, :, :future_len, :]
            elif T_n < future_len:
                pad_n = torch.zeros(1, Pn, future_len - T_n, 4, device=device)
                neighbors_future = torch.cat([neighbors_future, pad_n], dim=2)
            # Mask: non-zero positions are valid
            neighbor_mask = torch.sum(torch.ne(neighbors_future[..., :2], 0), dim=-1) == 0  # [1, Pn, T]

        # Compute SFT diffusion loss
        loss, metrics = _compute_sft_diffusion_loss(
            model=model,
            model_args=model_args,
            data=data_i,
            ego_gt=ego_gt,
            neighbor_gt=neighbors_future,
            neighbor_mask=neighbor_mask,
            device=device,
            K=config.diffusion_k_steps,
            neighbor_reg_weight=config.neighbor_reg_weight,
            neighbor_reg_only=config.neighbor_reg_only,
        )

        scaled_loss = loss / accum_target
        scaled_loss.backward()
        accum_count += 1

        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0.0) + v
        n_scenes += 1

        if accum_count >= accum_target:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=5.0,
            )
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

    # Flush remaining gradients
    if accum_count > 0:
        if accum_count < accum_target:
            scale_fix = accum_target / accum_count
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.mul_(scale_fix)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=5.0,
        )
        optimizer.step()
        optimizer.zero_grad()

    avg_metrics = {k: v / max(n_scenes, 1) for k, v in all_metrics.items()}
    avg_metrics["mean_best_reward"] = mean_best_reward
    return avg_metrics
