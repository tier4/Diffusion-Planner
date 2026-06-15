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
    use_dit_memo: bool = True,
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
        use_dit_memo: When a composer is active, reuse its x0-refinement DiT
                      forward for the solver's noise prediction at the same
                      (x, t) (numerically equivalent, ~halves the guided step).
                      False = escape hatch for A/B verification. No effect when
                      composer is None.

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

    # Function-scope import (not module-top): rlvr.closed_loop.batched_rollout
    # pulls in modules that import back here, so a top-level import would create
    # an import cycle. Imported once per call here (not per loop iteration).
    from rlvr.closed_loop.batched_rollout import make_initial_latent
    from rlvr.guidance_batched import dit_memo

    B = data["ego_current_state"].shape[0]
    P = 1 + model_args.predicted_neighbor_num
    future_len = model_args.future_len

    results = []
    try:
        for _ in range(n_samples):
            data["sampled_trajectories"] = make_initial_latent(
                B,
                P,
                future_len,
                device,
                noise_scale,
            )

            if composer is not None and use_dit_memo:
                # dit_memo: the solver reuses the composer's x0-refinement
                # forward at the same (x, t) — ~25% off guided frames.
                with dit_memo(model.decoder):
                    _, decoder_output = model(data)
            else:
                _, decoder_output = model(data)
            ego_trajectory = decoder_output["prediction"][0, 0].cpu().numpy()  # (OUTPUT_T, 4)
            results.append(ego_trajectory)
    finally:
        # Always restore the original guidance configuration.
        model.decoder._guidance_fn = _orig_fn
        model.decoder._guidance_scale = _orig_scale

    return np.stack(results, axis=0)  # (n_samples, OUTPUT_T, 4)
