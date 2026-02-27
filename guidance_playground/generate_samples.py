"""Focused inference wrapper for the Guidance Playground.

Generates N independent trajectory samples from the Diffusion Planner under
a specified noise scale and optional GuidanceComposer configuration.

This module intentionally does NOT reuse generate_trajectory_pair from
preference_optimization/utils.py because that function has DPO-specific
pairing, threshold, and pruning logic that is irrelevant here.
"""

import numpy as np
import torch

from diffusion_planner.model.guidance.composer import GuidanceComposer


@torch.no_grad()
def generate_samples(
    model,
    model_args,
    data: dict[str, torch.Tensor],
    noise_scale: float,
    n_samples: int,
    composer: GuidanceComposer | None,
    device: torch.device,
) -> np.ndarray:
    """
    Generate n_samples independent ego trajectories under a shared configuration.

    Args:
        model: Loaded Diffusion_Planner instance.
        model_args: Config object returned by model_utils.load_model.
        data: Observation dict already normalised by model_args.observation_normalizer.
              Must have batch dimension B=1.
        noise_scale: Standard deviation of Gaussian noise for the initial latent
                     (xT). 0.0 = fully deterministic (MAP output).
        n_samples: Number of independent samples to draw.
        composer: GuidanceComposer instance to inject for this call, or None for
                  unguided sampling.
        device: Torch device.

    Returns:
        (n_samples, OUTPUT_T, 4) float32 numpy array.
        Each row: [x, y, cos_yaw, sin_yaw] in ego-centric metres.
    """
    # Save and override decoder guidance for this call.
    _orig_fn = model.decoder._guidance_fn
    _orig_scale = model.decoder._guidance_scale

    model.decoder._guidance_fn = composer
    if composer is not None:
        model.decoder._guidance_scale = composer._set_config.global_scale
    else:
        model.decoder._guidance_scale = 0.5

    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    # Precompute current-state tensor used as the pinned first element of xT.
    ego_current = data["ego_current_state"][:, :4]                         # [B, 4]
    neighbors_current = data["neighbor_agents_past"][:, :P - 1, -1, :4]   # [B, P-1, 4]
    current_states = torch.cat(
        [ego_current[:, None], neighbors_current], dim=1
    )  # [B, P, 4]

    results = []
    try:
        for _ in range(n_samples):
            # Build noisy initial latent xT.
            xT = current_states[:, :, None, :].expand(-1, -1, future_len + 1, -1).clone()
            if noise_scale > 0.0:
                xT[:, :, 1:, :] = noise_scale * torch.randn(
                    B, P, future_len, 4, device=device
                )
            # data["sampled_trajectories"] shape expected by decoder: (B, P, (T+1)*4) or (B, P, T+1, 4)
            # Existing working code in generate_trajectory_pair passes (B, P, T+1, 4) directly.
            data["sampled_trajectories"] = xT

            _, decoder_output = model(data)
            ego_trajectory = decoder_output["prediction"][0, 0].cpu().numpy()  # (OUTPUT_T, 4)
            results.append(ego_trajectory)
    finally:
        # Always restore the original guidance configuration.
        model.decoder._guidance_fn = _orig_fn
        model.decoder._guidance_scale = _orig_scale

    return np.stack(results, axis=0)  # (n_samples, OUTPUT_T, 4)
