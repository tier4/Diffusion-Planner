"""DPO loss computation for trajectory preference optimization."""

import contextlib
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear


def compute_trajectory_loss(
    model: Diffusion_Planner,
    data: dict[str, torch.Tensor],
    trajectory: np.ndarray,
    model_args,
    noise: torch.Tensor,
    t: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute MSE loss of a trajectory under the diffusion model.

    Args:
        model: The diffusion planner model
        data: Input observation data
        trajectory: Ground truth trajectory [T, 4]
        model_args: Model configuration arguments
        noise: Pre-generated noise [B, P, T, 4]
        t: Diffusion time [B]
        device: Computation device

    Returns:
        MSE loss value
    """
    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    # Convert trajectory to tensor and normalize
    gt_trajectory = torch.tensor(trajectory, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, 4]

    # Normalize using ego stats only
    ego_mean = model_args.state_normalizer.mean[0].to(device)
    ego_std = model_args.state_normalizer.std[0].to(device)
    gt_trajectory_norm = (gt_trajectory - ego_mean) / ego_std  # [1, T, 4]

    # Create full ground truth with ego + neighbors (neighbors are zeros)
    gt_future = torch.zeros(B, P, future_len, 4, device=device)
    gt_future[:, 0, :, :] = gt_trajectory_norm  # Only ego has ground truth

    # Get current states and normalize using the same ego_mean/ego_std already
    # computed above. This matches the normal training path in decoder.py where
    # inputs are already normalized before compute_training_loss is called.
    # Without this, the anchor point at index 0 of all_gt sits in raw coordinate
    # space while all future timesteps are normalized, causing a 10m forward
    # offset (ego x-mean=10, std=20 → raw [0,0,...] ≠ normalized [-0.5,0,...]).
    # Ego and neighbor share the same normalization stats (confirmed in normalization.json).
    ego_current = data["ego_current_state"][:, :4]
    if P > 1:
        neighbors_current = data["neighbor_agents_past"][:, : P - 1, -1, :4]
        neighbors_current_norm = (neighbors_current - ego_mean) / ego_std  # [B, P-1, 4]
    else:
        neighbors_current_norm = torch.zeros(B, 0, 4, device=device)
    ego_current_norm = (ego_current - ego_mean) / ego_std  # [B, 4]
    current_states = torch.cat([ego_current_norm[:, None], neighbors_current_norm], dim=1)  # [B, P, 4]

    # Concatenate current state with future
    all_gt = torch.cat([current_states[:, :, None, :], gt_future], dim=2)  # [B, P, 1+T, 4]

    # Add noise to future part only
    # Use VPSDE for marginal probability (matching training code in decoder.py)
    mean, std = VPSDE_linear().marginal_prob(all_gt[..., 1:, :], t)
    std = std.view(-1, *([1] * (len(all_gt[..., 1:, :].shape) - 1)))

    if model_args.diffusion_model_type == "flow_matching":
        t_expanded = t.reshape(-1, *([1] * (len(all_gt.shape) - 1)))  # [B, 1, 1, 1]
        xT = (1 - t_expanded) * noise + t_expanded * all_gt[:, :, 1:, :]  # [B, P, T, 4]
    else:
        xT = mean + std * noise

    # Concatenate current state with noisy future
    xT_full = torch.cat([all_gt[:, :, :1, :], xT], dim=2)  # [B, P, 1+T, 4]

    # Clone and normalize observations
    data_for_norm = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    data_normalized = model_args.observation_normalizer(data_for_norm)

    # Prepare model inputs
    merged_inputs = {**data_normalized}
    merged_inputs["gt_trajectories"] = all_gt
    merged_inputs["sampled_trajectories"] = xT_full
    merged_inputs["diffusion_time"] = t

    # Run model
    _, outputs = model(merged_inputs)

    # Extract model output
    if "model_output" in outputs:
        model_output = outputs["model_output"][:, 0, 1:, :]  # [B, T, 4] - ego only
    else:
        model_output = outputs["prediction"][:, 0, :, :]  # [B, T, 4] - ego only

    # Compute loss based on model type
    if model_args.diffusion_model_type == "score":
        # Score matching loss
        mse_loss = F.mse_loss(
            (model_output * std[:, 0] + noise[:, 0]),
            torch.zeros_like(model_output),
            reduction="mean",
        )
    elif model_args.diffusion_model_type == "x_start":
        # Direct prediction loss
        mse_loss = F.mse_loss(model_output, gt_trajectory_norm, reduction="mean")
    elif model_args.diffusion_model_type == "flow_matching":
        # Flow matching loss
        target_v = all_gt[:, 0, 1:, :] - noise[:, 0]
        mse_loss = F.mse_loss(model_output, target_v, reduction="mean")
    else:
        raise ValueError(f"Unknown diffusion model type: {model_args.diffusion_model_type}")

    return mse_loss


def compute_dpo_loss(
    policy_model: Diffusion_Planner,
    reference_model: Optional[nn.Module],
    batch: list[dict],
    beta: float,
    model_args,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute DPO loss for a batch of preference pairs.

    DPO Loss (adapted for MSE loss where lower is better):
        L = -log(sigma(-beta * ((l_w - l_ref_w) - (l_l - l_ref_l))))

    where:
        - l_w: MSE loss for winning trajectory
        - l_l: MSE loss for losing trajectory
        - l_ref_w: reference MSE loss for winning trajectory
        - l_ref_l: reference MSE loss for losing trajectory
        - beta: regularization parameter
        - sigma: sigmoid function

    Args:
        policy_model: Policy model being trained
        reference_model: Frozen reference model. When None, reference logprobs are
            computed by disabling LoRA adapters on the policy model (shared-model DPO).
            When provided, a separate frozen model is used as the reference.
        batch: List of preference samples
        beta: DPO regularization parameter
        model_args: Model configuration arguments
        device: Computation device

    Returns:
        Tuple of (loss, metrics_dict)
    """
    total_loss = 0.0
    metrics = {
        "accuracy": 0.0,
        "avg_log_ratio": 0.0,
        "avg_reward_margin": 0.0,
    }

    B = 1  # Batch size per sample
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    # Number of (t, noise) samples to average per preference pair.
    # Averaging over K samples reduces variance in the loss_diff estimate by sqrt(K).
    # A single sample is dominated by noise when MSE values are small (well-trained model).
    K = 8
    eps = 1e-3

    ref_model = reference_model if reference_model is not None else policy_model
    inner_ref = ref_model.module if hasattr(ref_model, "module") else ref_model
    use_lora_disable = hasattr(inner_ref, "disable_adapter")

    for sample in batch:
        data_raw = sample["data"]
        traj_w = sample["trajectory_w"]
        traj_l = sample["trajectory_l"]

        l_w_sum = torch.tensor(0.0, device=device)
        l_l_sum = torch.tensor(0.0, device=device)
        l_ref_w_sum = torch.tensor(0.0, device=device, requires_grad=False)
        l_ref_l_sum = torch.tensor(0.0, device=device, requires_grad=False)

        for _ in range(K):
            # Use the same noise and t for winner and loser so that l_w - l_l
            # is a clean comparison under identical noisy conditions. Averaging
            # over K draws reduces variance in the loss_diff estimate.
            noise = torch.randn(B, P, future_len, 4, device=device)
            t = torch.rand(B, device=device) * (1 - eps) + eps

            data_w = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_raw.items()}
            data_l = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_raw.items()}
            l_w_sum = l_w_sum + compute_trajectory_loss(policy_model, data_w, traj_w, model_args, noise, t, device)
            l_l_sum = l_l_sum + compute_trajectory_loss(policy_model, data_l, traj_l, model_args, noise, t, device)

            disable_ctx = inner_ref.disable_adapter() if use_lora_disable else contextlib.nullcontext()
            with disable_ctx, torch.no_grad():
                data_ref_w = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_raw.items()}
                data_ref_l = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_raw.items()}
                l_ref_w_sum = l_ref_w_sum + compute_trajectory_loss(ref_model, data_ref_w, traj_w, model_args, noise.clone(), t, device)
                l_ref_l_sum = l_ref_l_sum + compute_trajectory_loss(ref_model, data_ref_l, traj_l, model_args, noise.clone(), t, device)

        l_w = l_w_sum / K
        l_l = l_l_sum / K
        l_ref_w = l_ref_w_sum / K
        l_ref_l = l_ref_l_sum / K

        # Compute DPO loss
        # Since MSE loss is lower-is-better, we want l_w < l_l relative to reference
        loss_diff = (l_w - l_ref_w) - (l_l - l_ref_l)
        loss = -F.logsigmoid(-beta * loss_diff)
        total_loss += loss

        # Compute metrics
        with torch.no_grad():
            reward_margin = -beta * loss_diff
            metrics["avg_reward_margin"] += reward_margin.item()
            metrics["avg_log_ratio"] += (-loss_diff).item()
            metrics["accuracy"] += (loss_diff < 0).float().item()

    # Average over batch
    batch_size = len(batch)
    total_loss = total_loss / batch_size
    metrics = {k: v / batch_size for k, v in metrics.items()}

    return total_loss, metrics
