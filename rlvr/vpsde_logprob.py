"""VPSDE denoising step with Gaussian log-probability computation.

Adapts the DDV2 DDIMScheduler_with_logprob approach to our VPSDE_linear noise schedule.
The key idea: at each reverse step, the model predicts x_0, and we compute the
Gaussian transition distribution p(x_{t_prev} | x_0) = N(mean, std) using
VPSDE marginal_prob. The log-probability of the actual sample under this
distribution provides the policy gradient signal for GRPO.

Reference: DiffusionDriveV2 diffusiondrivev2_model_sel.py:629-783
"""

import math
from typing import Optional, Tuple

import torch

from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear


def vpsde_denoising_step_with_logprob(
    x0_pred: torch.Tensor,
    t_prev: torch.Tensor,
    sde: VPSDE_linear,
    min_std: float = 0.1,
    prev_sample: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One VPSDE reverse step with Gaussian log-probability.

    Given a model's x_0 prediction and a target time t_prev, computes the
    marginal distribution p(x_{t_prev} | x_0) = N(mean_coeff * x_0, std)
    and either samples from it or evaluates the log-probability of a given sample.

    Args:
        x0_pred: Model's clean trajectory prediction. Shape: arbitrary, but
            typically (B, P, T, D) for the future timesteps (no current state).
        t_prev: Target diffusion time for the reverse step. Shape: broadcastable
            to x0_pred (e.g., (B, 1, 1, 1) or scalar tensor).
        sde: VPSDE_linear instance.
        min_std: Minimum standard deviation to prevent log-prob explosion near t=0.
            DDV2 uses 0.1 for log-prob evaluation (supplementary material).
        prev_sample: If provided, compute log-prob for this fixed sample instead
            of drawing a new one. Used in the optimization pass (Stage 2).
            Must have same shape as x0_pred.

    Returns:
        sample: The (possibly new) sample x_{t_prev}. Same shape as x0_pred.
        log_prob: Gaussian log-probability, summed over all dims except batch.
            Shape: (B,) where B = x0_pred.shape[0].
        mean: The predicted mean of the transition distribution. Same shape as x0_pred.
    """
    # VPSDE marginal: p(x_t | x_0) = N(mean_coeff * x_0, std)
    # mean_coeff = exp(-0.25 * t^2 * (beta_max - beta_min) - 0.5 * beta_min * t)
    # std = sqrt(1 - exp(2 * log(mean_coeff)))
    mean, std = sde.marginal_prob(x0_pred, t_prev)
    std = std.clamp(min=min_std)

    if prev_sample is None:
        # Collection pass: sample new x_{t_prev}
        noise = torch.randn_like(mean)
        sample = mean + std * noise
    else:
        # Optimization pass: use stored sample
        sample = prev_sample

    # Gaussian log-probability (gradient flows through mean only, not sample)
    # log N(x | mu, sigma) = -((x - mu)^2) / (2 * sigma^2) - log(sigma) - 0.5 * log(2*pi)
    log_prob = (
        -((sample.detach() - mean) ** 2) / (2 * std**2)
        - torch.log(std)
        - 0.5 * math.log(2 * math.pi)
    )

    # Sum over all dimensions except batch → (B,)
    # This sums over agents, timesteps, and features
    # Mean (not sum) over dims to normalize by trajectory length.
    # Without this, log-probs are ~-160M for our 320-dim trajectories,
    # causing vanishing RL gradients. DDV2 sums over only 16 dims.
    log_prob = log_prob.reshape(log_prob.shape[0], -1).mean(dim=-1)

    return sample, log_prob, mean


def create_timestep_schedule(
    t_start: float,
    num_steps: int,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Create a decreasing timestep schedule for the denoising rollout.

    Args:
        t_start: Starting noise level (e.g., 0.01 for truncated diffusion).
        num_steps: Number of denoising steps.
        eps: Minimum time value (avoid t=0 singularity).

    Returns:
        Tensor of shape (num_steps + 1,) with decreasing timesteps from t_start to eps.
        The first num_steps values are the step inputs, the last is the final target.
    """
    return torch.linspace(t_start, eps, num_steps + 1)


def compute_discount_weights(
    num_steps: int,
    discount: float = 0.8,
) -> torch.Tensor:
    """Compute per-step discount weights for advantage weighting.

    Earlier denoising steps (high noise) get less weight than later steps (near clean).
    Matches DDV2: discount[i] = gamma^(num_steps - i - 1)

    Args:
        num_steps: Number of denoising steps.
        discount: Discount factor (DDV2 uses 0.8).

    Returns:
        Tensor of shape (num_steps,) with increasing weights.
    """
    return torch.tensor(
        [discount ** (num_steps - i - 1) for i in range(num_steps)]
    )
