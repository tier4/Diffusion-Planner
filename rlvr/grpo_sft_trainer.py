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
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.module.decoder import generate_prefix_mask
from scipy.signal import savgol_filter
from torch import nn
from tqdm import tqdm

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
    ego_il_weight: float = 0.0,
    ego_il_mode: str = "gt",
    ego_gt_real: torch.Tensor | None = None,
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

    When ego_il_weight > 0, adds MSE(model_ego, real_GT_ego) to anchor the
    model's ego predictions near ground truth while learning from ranked trajs.

    Args:
        model: Policy model (LoRA-wrapped).
        model_args: Config from load_model.
        data: Raw observation dict (NOT normalized). B=1.
        ego_gt: [B, T, 4] ego ground truth trajectory (x, y, cos, sin).
            In ranked SFT, this is the best-of-K ranked trajectory.
        neighbor_gt: [B, Pn, T, 4] neighbor ground truth trajectories.
        neighbor_mask: [B, Pn, T] boolean mask (True = invalid/padded).
        device: Torch device.
        K: Number of (noise, timestep) samples to average over.
        neighbor_reg_weight: Weight for neighbor regularization loss (0=disabled).
        neighbor_reg_only: If True, drop the neighbor SFT loss and only use the
            reg term. Only takes effect when neighbor reg is active (model has
            disable_adapter and neighbor_reg_weight > 0).
        ego_il_weight: Weight for ego IL regularization (0=disabled).
        ego_gt_real: [B, T, 4] real GT ego trajectory for IL regularization.
            Required when ego_il_weight > 0.

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

    # Normalize real GT ego for IL loss (if provided)
    use_ego_il = ego_il_weight > 0.0
    if use_ego_il and ego_il_mode == "gt" and ego_gt_real is None:
        use_ego_il = False  # GT mode requires ego_gt_real
    if use_ego_il and ego_il_mode == "gt":
        ego_gt_real_norm = (ego_gt_real - ego_mean) / ego_std  # [B, T, 4]

    total_ego_loss = 0.0
    total_neighbor_loss = 0.0
    total_neighbor_reg_loss = 0.0
    total_ego_il_loss = 0.0
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

        # Ego loss: MSE over all timesteps (against ranked trajectory)
        ego_loss = F.mse_loss(model_output[:, 0], gt_target[:, 0])
        total_ego_loss += ego_loss

        # Ego IL loss (GT mode): MSE against real GT ego trajectory
        if use_ego_il and ego_il_mode == "gt":
            ego_il_loss = F.mse_loss(model_output[:, 0], ego_gt_real_norm)
            total_ego_il_loss += ego_il_loss

        # Neighbor loss: MSE over valid timesteps only.
        # Skipped when neighbor_reg_only=True AND reg is actually active.
        skip_neighbor_sft = neighbor_reg_only and use_neighbor_reg
        if Pn > 0 and neighbors_future_valid.any() and not skip_neighbor_sft:
            neighbor_pred = model_output[:, 1:]  # [B, Pn, T, 4]
            neighbor_target = gt_target[:, 1:]  # [B, Pn, T, 4]
            # Per-element MSE, then mask
            neighbor_mse = ((neighbor_pred - neighbor_target) ** 2).mean(dim=-1)  # [B, Pn, T]
            masked_loss = neighbor_mse[neighbors_future_valid]
            if masked_loss.numel() > 0:
                total_neighbor_loss += masked_loss.mean()

        # Neighbor regularization + ego IL (baseline mode): reuse base model forward pass
        need_base_pass = use_neighbor_reg or (use_ego_il and ego_il_mode == "baseline")
        if need_base_pass and hasattr(inner, "disable_adapter"):
            with inner.disable_adapter(), torch.no_grad():
                _, base_outputs = model(merged_inputs)
            base_output = base_outputs["model_output"][:, :, 1:, :]  # [B, P, T, 4]

            # Neighbor reg
            if use_neighbor_reg and neighbors_future_valid.any():
                base_neighbor = base_output[:, 1:]  # [B, Pn, T, 4]
                lora_neighbor = model_output[:, 1:]  # [B, Pn, T, 4]
                reg_mse = ((lora_neighbor - base_neighbor.detach()) ** 2).mean(dim=-1)
                masked_reg = reg_mse[neighbors_future_valid]
                if masked_reg.numel() > 0:
                    total_neighbor_reg_loss += masked_reg.mean()

            # Ego IL (baseline mode): MSE(lora_ego, base_ego)
            if use_ego_il and ego_il_mode == "baseline":
                base_ego = base_output[:, 0]  # [B, T, 4]
                ego_il_loss = F.mse_loss(model_output[:, 0], base_ego.detach())
                total_ego_il_loss += ego_il_loss

    ego_loss_avg = total_ego_loss / K
    neighbor_loss_avg = total_neighbor_loss / K if isinstance(total_neighbor_loss, torch.Tensor) else torch.tensor(0.0, device=device)
    neighbor_reg_avg = total_neighbor_reg_loss / K if isinstance(total_neighbor_reg_loss, torch.Tensor) else torch.tensor(0.0, device=device)
    ego_il_avg = total_ego_il_loss / K if isinstance(total_ego_il_loss, torch.Tensor) else torch.tensor(0.0, device=device)

    # Combined loss: ego + neighbor (neighbor weight = 1.0 to match SFT) + neighbor reg + ego IL
    loss = ego_loss_avg + neighbor_loss_avg
    if use_neighbor_reg:
        loss = loss + neighbor_reg_weight * neighbor_reg_avg
    if use_ego_il:
        loss = loss + ego_il_weight * ego_il_avg

    metrics = {
        "sft_ego_loss": ego_loss_avg.item(),
        "sft_neighbor_loss": neighbor_loss_avg.item() if isinstance(neighbor_loss_avg, torch.Tensor) else 0.0,
        "sft_total_loss": loss.item(),
        "sft_neighbor_reg_loss": neighbor_reg_avg.item() if isinstance(neighbor_reg_avg, torch.Tensor) else 0.0,
        "sft_ego_il_loss": ego_il_avg.item() if isinstance(ego_il_avg, torch.Tensor) else 0.0,
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
    exploration_policy=None,
    exploration_optimizer=None,
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
              f"neighbor_reg={config.neighbor_reg_weight}, reg_only={config.neighbor_reg_only}, "
              f"ego_il={config.ego_il_weight}")

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

    # 2b. Apply per-epoch schedules to reward weights and guidance params
    scheduled = config.get_all_scheduled_values(epoch, config.train_epochs)
    reward_weight_names = {
        "w_progress", "w_safety", "w_smooth", "w_feasibility", "w_centerline",
        "stopped_penalty", "underprogress_penalty", "progress_norm_scale",
    }
    for name, value in scheduled.items():
        if name in reward_weight_names and hasattr(reward_config, name):
            setattr(reward_config, name, value)
    if scheduled:
        sched_str = ", ".join(f"{k}={v:.3f}" for k, v in scheduled.items())
        print(f"  [schedule] epoch {epoch}: {sched_str}")

    # Extract longitudinal guidance params from schedule (default: off)
    lon_eta = scheduled.get("longitudinal_eta", 0.0)
    lon_lambda = scheduled.get("longitudinal_lambda", config.lambda_lon)
    lon_scale = scheduled.get("longitudinal_scale", 10.0)

    # Extract lateral guidance params from schedule (default: off)
    lat_eta = scheduled.get("lateral_eta", 0.0)
    lat_lambda = scheduled.get("lateral_lambda", config.lambda_lat)
    lat_scale = scheduled.get("lateral_scale", 5.0)

    # Extract speed stretch from schedule (default: 1.0 = no stretch)
    spd_stretch = scheduled.get("speed_stretch", 1.0)

    # 2c. Optionally use exploration policy to generate K diverse trajectories per scene
    if exploration_policy is not None:
        from diffusion_planner.model.guidance.composer import GuidanceComposer
        from diffusion_planner.model.guidance.config import GuidanceConfig as _GC
        from diffusion_planner.model.guidance.config import GuidanceSetConfig

        from exploration_policy.utils import generate_reference_trajectory, run_frozen_encoder
        from rlvr.closed_loop.batched_rollout import _batched_generate_varied_noise
        # NOTE: per-scene loop matches grpo_exploration_trainer's generate_policy_guided_group.
        # Batching across scenes would require handling per-scene Beta distributions in a single
        # forward pass, which is complex. For 50-500 scenes this takes ~3 min, acceptable.
        print(f"  Explorer-guided generation: {K} samples from Beta distribution per scene...")
        exploration_policy.eval()
        model.eval()

        _lat_lambda = config.exploration_lambda_lat
        _lon_lambda = config.exploration_lambda_lon
        _guide_scale = config.exploration_guidance_scale
        noise_min, noise_max = config.noise_scale_range
        _train_explorer = exploration_optimizer is not None

        all_scene_trajs = []  # will be [N, K, T, 4]
        # Store per-scene explorer data for training
        _explorer_scenes = []  # list of dicts with distributions and sampled etas

        for i in range(N):
            norm_i = {k: v[i:i+1] if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == N else v
                      for k, v in norm_batch.items()}
            with torch.no_grad():
                scene_enc = run_frozen_encoder(model, norm_i)
                x_ref_np = generate_reference_trajectory(model, model_args, norm_i, device)
                x_ref = torch.from_numpy(x_ref_np).unsqueeze(0).to(device=device, dtype=torch.float32)
                norm_i["reference_trajectory"] = x_ref

                # Get Beta distributions and sample K etas
                output = exploration_policy(scene_enc, x_ref, deterministic=False)
                eta_lat_01 = output.lat_dist.rsample((K,)).squeeze(-1)  # [K]
                eta_lon_01 = output.lon_dist.rsample((K,)).squeeze(-1)  # [K]
                eta_lat_vals = 2.0 * eta_lat_01 - 1.0  # map to [-1, 1]
                eta_lon_vals = 2.0 * eta_lon_01 - 1.0

                if _train_explorer:
                    _explorer_scenes.append({
                        "scene_enc": scene_enc.detach(),
                        "x_ref": x_ref.detach(),
                        "eta_lat_01": eta_lat_01.detach(),
                        "eta_lon_01": eta_lon_01.detach(),
                    })

                # Expand scene data from B=1 to B=K
                K_data = {}
                for k_key, v in norm_i.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] == 1:
                        K_data[k_key] = v.expand(K, *v.shape[1:]).contiguous()
                    else:
                        K_data[k_key] = v

                # Build batched guidance with K different etas
                guidance_fns = [
                    _GC("lateral", enabled=True, scale=1.0,
                         params={"lambda_lat": _lat_lambda, "eta_lat": eta_lat_vals}),
                    _GC("longitudinal", enabled=True, scale=1.0,
                         params={"lambda_lon": _lon_lambda, "eta_lon": eta_lon_vals}),
                ]
                composer = GuidanceComposer(GuidanceSetConfig(
                    functions=guidance_fns, global_scale=_guide_scale))

                # Generate K trajectories (first deterministic, rest with varied noise)
                traj_tensor = _batched_generate_varied_noise(
                    model, model_args, K_data,
                    noise_min=noise_min, noise_max=noise_max,
                    first_deterministic=True,
                    composer=composer, device=device,
                )  # [K, T, 4]
                all_scene_trajs.append(traj_tensor)

        all_trajs = torch.stack(all_scene_trajs)  # [N, K, T, 4]
        print(f"  Explorer: K={K}, guide_scale={_guide_scale}, noise=[{noise_min},{noise_max}]")
        torch.cuda.empty_cache()
        gc.collect()
    else:
        # 3. Standard generation: K trajectories for all scenes (batched)
        print(f"  Generating {K} trajectories x {N} scenes (batched)...")
        model.eval()
        with torch.no_grad():
            all_trajs = generate_all_scenes_batched(
                model, model_args, norm_batch, K, config.noise_scale_range, device,
                gt_max_speed=median_gt_speed,
                longitudinal_eta=lon_eta,
                longitudinal_lambda=lon_lambda,
                longitudinal_scale=lon_scale,
                lateral_eta=lat_eta,
                lateral_lambda=lat_lambda,
                lateral_scale=lat_scale,
                speed_stretch=spd_stretch,
            )  # [N, K, T, 4]

    torch.cuda.empty_cache()
    gc.collect()

    # 4. Score and select best trajectory per scene
    # Selective training: only SFT on scenes where best-of-K improves significantly
    # over the deterministic (index 0) trajectory. Controlled by config.selective_threshold.
    # 0 = train all scenes (default), >0 = skip scenes with best-det < threshold.
    selective_thresh = getattr(config, "selective_threshold", 0.0)
    # Allow scheduling of selective_threshold (e.g., 0 for first epochs, then 3.0)
    sched_thresh = scheduled.get("selective_threshold")
    if sched_thresh is not None:
        selective_thresh = sched_thresh
    print(f"  Scoring and selecting best trajectories...")
    best_ego_trajs = []  # [T, 4] numpy arrays
    best_rewards_list = []
    scene_train_mask = []  # True = train on this scene, False = skip

    for i in tqdm(range(N), desc="Scoring"):
        traj_K = all_trajs[i]  # [K, T, 4]
        data_i = all_data[i]

        rewards = compute_reward_batch(traj_K, data_i, reward_config)
        reward_vals = np.array([r.total for r in rewards])
        best_idx = int(np.argmax(reward_vals))
        best_reward = reward_vals[best_idx]
        det_reward = reward_vals[0]  # deterministic trajectory is always index 0

        # Selective: skip scene if improvement is below threshold
        improvement = best_reward - det_reward
        should_train = selective_thresh <= 0 or improvement >= selective_thresh
        scene_train_mask.append(should_train)
        if selective_thresh > 0 and epoch <= 2 and should_train:
            from pathlib import Path as _P
            print(f"    SEL [{_P(valid_paths[i]).stem[:30]}] det={det_reward:.1f} best={best_reward:.1f} imp={improvement:.1f}")

        # Get best trajectory and smooth it
        best_traj = traj_K[best_idx].cpu().numpy()  # [T, 4]
        best_traj_smooth = _smooth_trajectory(
            best_traj, config.sg_filter_window, config.sg_filter_order
        )

        best_ego_trajs.append(best_traj_smooth)
        best_rewards_list.append(best_reward)

    mean_best_reward = float(np.mean(best_rewards_list))
    n_selected = sum(scene_train_mask)
    print(f"  Mean best-of-{K} reward: {mean_best_reward:.2f}")
    if selective_thresh > 0:
        print(f"  Selective training: {n_selected}/{N} scenes selected "
              f"(threshold={selective_thresh:.1f}, skipped {N - n_selected})")

    # --- Train explorer on trajectory rewards (if optimizer provided) ---
    explorer_metrics = {}
    if exploration_policy is not None and exploration_optimizer is not None and _explorer_scenes:
        from exploration_policy.loss import compute_exploration_loss
        from rlvr.reward import compute_group_advantages
        print(f"  Training explorer on {len(_explorer_scenes)} scenes...")
        exploration_policy.train()
        exploration_optimizer.zero_grad()
        total_policy_loss = 0.0
        n_explorer = 0

        for i in range(N):
            if i >= len(_explorer_scenes):
                break
            es = _explorer_scenes[i]
            traj_K = all_trajs[i]  # [K, T, 4]
            data_i = all_data[i]

            # Compute rewards for this scene's K trajectories
            rewards = compute_reward_batch(traj_K, data_i, reward_config)
            advantages = compute_group_advantages(rewards)

            if np.all(advantages == 0):
                continue

            # Recompute explorer distributions (with grad)
            policy_output = exploration_policy(es["scene_enc"], es["x_ref"], deterministic=True)
            log_probs = (policy_output.lat_dist.log_prob(es["eta_lat_01"])
                         + policy_output.lon_dist.log_prob(es["eta_lon_01"]))
            if log_probs.dim() > 1:
                log_probs = log_probs.squeeze(-1)

            advantages_t = torch.tensor(advantages, device=device, dtype=torch.float32)

            if config.exploration_loss_type == "best_sample_mse":
                best_idx = advantages_t.argmax()
                pred_lat = policy_output.lat_dist.mean.squeeze()
                pred_lon = policy_output.lon_dist.mean.squeeze()
                policy_loss = (pred_lat - es["eta_lat_01"][best_idx].detach()) ** 2 \
                            + (pred_lon - es["eta_lon_01"][best_idx].detach()) ** 2
            else:
                policy_loss, _ = compute_exploration_loss(
                    advantages=advantages_t, log_probs=log_probs,
                    lat_dist=policy_output.lat_dist, lon_dist=policy_output.lon_dist,
                    entropy_coef=config.exploration_entropy_coef,
                    kl_coef=config.exploration_kl_coef,
                )

            (policy_loss / N).backward()
            total_policy_loss += policy_loss.item()
            n_explorer += 1

        if n_explorer > 0:
            torch.nn.utils.clip_grad_norm_(exploration_policy.parameters(), max_norm=1.0)
            exploration_optimizer.step()
            exploration_optimizer.zero_grad()
            explorer_metrics["explorer_loss"] = total_policy_loss / n_explorer
            print(f"  Explorer loss: {explorer_metrics['explorer_loss']:.4f} ({n_explorer} scenes)")

        exploration_policy.eval()
        del _explorer_scenes

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
            norm_i = {
                k: v[i:i + 1] if isinstance(v, torch.Tensor) and v.shape[0] == N else v
                for k, v in norm_batch.items()
            }
            neighbor_pred = _get_baseline_neighbor_prediction(
                model, model_args, norm_i, device
            )  # [Pn, T, 4]
            baseline_neighbor_preds.append(neighbor_pred)
        torch.cuda.empty_cache()

    # 6. Prepare all training targets (ego GT + neighbor GT) upfront
    Pn = model_args.predicted_neighbor_num
    future_len = model_args.future_len

    use_ego_il = config.ego_il_weight > 0.0 and config.ego_il_mode == "gt"
    print(f"  Preparing training targets for {N} scenes...")
    all_ego_gt = []       # list of [1, T, 4]
    all_ego_gt_real = []  # list of [1, T, 4] — real GT for IL reg
    all_neighbor_gt = []  # list of [1, Pn, T, 4]
    all_neighbor_mask = []  # list of [1, Pn, T]

    for i in range(N):
        # Ego GT: the filtered best trajectory
        ego_gt_np = best_ego_trajs[i]  # [T, 4]
        ego_gt = torch.tensor(ego_gt_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, 4]
        T_actual = ego_gt.shape[1]
        if T_actual > future_len:
            ego_gt = ego_gt[:, :future_len, :]
        elif T_actual < future_len:
            pad = torch.zeros(1, future_len - T_actual, 4, device=device)
            ego_gt = torch.cat([ego_gt, pad], dim=1)
        all_ego_gt.append(ego_gt)

        # Real GT ego trajectory for IL regularization
        if use_ego_il:
            data_i = all_data[i]
            real_gt = data_i.get("ego_agent_future")
            if real_gt is not None:
                if real_gt.dim() == 3:
                    real_gt = real_gt[:1]  # [1, T, C]
                elif real_gt.dim() == 2:
                    real_gt = real_gt.unsqueeze(0)  # [1, T, C]
                # Convert heading if needed (angle -> cos/sin)
                if real_gt.shape[-1] == 3:
                    real_gt = torch.cat([
                        real_gt[..., :2],
                        real_gt[..., 2:3].cos(),
                        real_gt[..., 2:3].sin(),
                    ], dim=-1)
                real_gt = real_gt[..., :4]
                T_r = real_gt.shape[1]
                if T_r > future_len:
                    real_gt = real_gt[:, :future_len, :]
                elif T_r < future_len:
                    pad_r = torch.zeros(1, future_len - T_r, 4, device=device)
                    real_gt = torch.cat([real_gt, pad_r], dim=1)
            else:
                # Fallback: use ranked traj as IL target (no effect)
                real_gt = ego_gt.clone()
            all_ego_gt_real.append(real_gt)

        # Neighbor GT
        if mode == "gt_neighbor":
            data_i = all_data[i]
            neighbors_future = data_i.get("neighbor_agents_future")
            if neighbors_future is not None:
                if neighbors_future.dim() == 3:
                    neighbors_future = neighbors_future.unsqueeze(0)
                neighbors_future = neighbors_future[:, :Pn, :, :]
                if neighbors_future.shape[-1] == 3:
                    neighbors_future = torch.cat([
                        neighbors_future[..., :2],
                        neighbors_future[..., 2:3].cos(),
                        neighbors_future[..., 2:3].sin(),
                    ], dim=-1)
                neighbor_mask = torch.sum(torch.ne(neighbors_future[..., :2], 0), dim=-1) == 0
                T_n = neighbors_future.shape[2]
                if T_n < future_len:
                    pad_n = torch.zeros(1, Pn, future_len - T_n, 4, device=device)
                    neighbors_future = torch.cat([neighbors_future, pad_n], dim=2)
                    pad_mask = torch.ones(1, Pn, future_len - T_n, dtype=torch.bool, device=device)
                    neighbor_mask = torch.cat([neighbor_mask, pad_mask], dim=2)
                elif T_n > future_len:
                    neighbors_future = neighbors_future[:, :, :future_len, :]
                    neighbor_mask = neighbor_mask[:, :, :future_len]
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
            neighbor_pred = baseline_neighbor_preds[i]  # [Pn, T, 4]
            neighbors_future = neighbor_pred.unsqueeze(0)  # [1, Pn, T, 4]
            T_n = neighbors_future.shape[2]
            if T_n > future_len:
                neighbors_future = neighbors_future[:, :, :future_len, :]
            elif T_n < future_len:
                pad_n = torch.zeros(1, Pn, future_len - T_n, 4, device=device)
                neighbors_future = torch.cat([neighbors_future, pad_n], dim=2)
            neighbor_mask = torch.sum(torch.ne(neighbors_future[..., :2], 0), dim=-1) == 0

        all_neighbor_gt.append(neighbors_future)
        all_neighbor_mask.append(neighbor_mask)

    # Stack all targets: [N, T, 4], [N, Pn, T, 4], [N, Pn, T]
    ego_gt_all = torch.cat(all_ego_gt, dim=0)
    neighbor_gt_all = torch.cat(all_neighbor_gt, dim=0)
    neighbor_mask_all = torch.cat(all_neighbor_mask, dim=0)
    ego_gt_real_all = torch.cat(all_ego_gt_real, dim=0) if use_ego_il else None
    del all_ego_gt, all_neighbor_gt, all_neighbor_mask, all_ego_gt_real

    # 7. Batched training with SFT diffusion loss
    # sft_batch_size: scenes per forward pass (1 = sequential, same as original)
    # accum_steps: how many forward passes before optimizer step
    # Effective batch per step = sft_batch_size * accum_steps = grad_accum_groups
    sft_bs = max(1, config.sft_batch_size)
    if config.grad_accum_groups % sft_bs != 0:
        raise ValueError(
            "grad_accum_groups must be divisible by sft_batch_size: "
            f"grad_accum_groups={config.grad_accum_groups}, sft_batch_size={sft_bs}."
        )
    accum_steps = config.grad_accum_groups // sft_bs
    scenes_per_step = sft_bs * accum_steps  # for proper loss/metric weighting
    # Shuffle scene order, filter by selective training mask
    indices = [i for i in range(N) if scene_train_mask[i]]
    _random.shuffle(indices)
    N_train = len(indices)

    print(f"  Training on {N_train}/{N} scenes (ranked SFT, mode={mode}, "
          f"sft_batch_size={sft_bs}, accum_steps={accum_steps})...")
    model.train()
    optimizer.zero_grad()

    all_metrics = {}
    n_scenes = 0
    accum_count = 0

    for batch_start in range(0, N_train, sft_bs):
        batch_idx = indices[batch_start:batch_start + sft_bs]
        bs = len(batch_idx)

        # Slice raw observation data for this mini-batch
        mini_data = {
            k: v[batch_idx] if isinstance(v, torch.Tensor) and v.shape[0] == N else v
            for k, v in batch_data.items()
        }
        mini_ego_gt = ego_gt_all[batch_idx]           # [bs, T, 4]
        mini_neighbor_gt = neighbor_gt_all[batch_idx]  # [bs, Pn, T, 4]
        mini_neighbor_mask = neighbor_mask_all[batch_idx]  # [bs, Pn, T]
        mini_ego_gt_real = ego_gt_real_all[batch_idx] if ego_gt_real_all is not None else None

        loss, metrics = _compute_sft_diffusion_loss(
            model=model,
            model_args=model_args,
            data=mini_data,
            ego_gt=mini_ego_gt,
            neighbor_gt=mini_neighbor_gt,
            neighbor_mask=mini_neighbor_mask,
            device=device,
            K=config.diffusion_k_steps,
            neighbor_reg_weight=config.neighbor_reg_weight,
            neighbor_reg_only=config.neighbor_reg_only,
            ego_il_weight=config.ego_il_weight,
            ego_il_mode=config.ego_il_mode,
            ego_gt_real=mini_ego_gt_real,
        )

        # Scale loss to preserve per-scene gradient magnitude:
        # loss is a batch-mean over bs scenes; we want the gradient contribution
        # proportional to bs/scenes_per_step so the optimizer step averages over
        # scenes_per_step scenes total.
        scaled_loss = loss * (bs / scenes_per_step)
        scaled_loss.backward()
        accum_count += 1

        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0.0) + v * bs
        n_scenes += bs

        if accum_count >= accum_steps:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=5.0,
            )
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

    # Flush remaining gradients
    if accum_count > 0:
        if accum_count < accum_steps:
            scale_fix = accum_steps / accum_count
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
    avg_metrics.update(explorer_metrics)
    return avg_metrics
