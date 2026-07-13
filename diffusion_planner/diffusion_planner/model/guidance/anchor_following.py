"""Anchor-following (MTR-style) guidance for the Diffusion Planner.

Guides the ego trajectory toward a pre-clustered prototype trajectory
(a "motion mode") extracted from the training set. The anchor defines
a desired shape in the ego-centric frame (straight, turn-left, etc.)
and the guidance energy attracts the predicted trajectory toward it.

Two modes (params["mode"]):

  "path" (default) — the anchor is treated as a PATH: each predicted
      waypoint is pulled toward the nearest point on the (extrapolated)
      anchor polyline with a Huber cost, so the guidance shapes WHERE the
      trajectory goes without fighting the model over speed. The energy is
      time-compensated (multiplied by alpha_t / sigma_t^2) so the effective
      x0 nudge per solver step is constant across the whole denoising
      schedule, and the guidance window covers the mode-forming high-noise
      steps. This is what makes the guidance able to actually switch modes.

  "waypoint" — the legacy behaviour: per-timestep squared distance to the
      anchor waypoints, active only in the base t-window (0.005, 0.1).
      Kept for reproducing older runs. Known failure modes: the quadratic
      pull is dominated by longitudinal (speed) mismatch against the
      anchor's own timing, the unbounded gradient overshoots by tens of
      metres through the sigma^2/alpha * std^2 amplification, and the late
      window cannot switch modes because the trajectory is already
      committed.
"""

import functools

import numpy as np
import torch

from .base import BaseGuidance
from .registry import register

# The VP-SDE linear schedule constants (NoiseScheduleVP defaults). Used to
# compute the per-step time compensation alpha_t / sigma_t^2 in path mode.
_BETA_0 = 0.1
_BETA_1 = 20.0

# xy std of the state normalizer (normalization.json convention used by all
# checkpoints in this repo). The DPM classifier correction shifts x0 by
# guidance_scale * sigma^2/alpha * std^2 * dE/dx_phys metres per coordinate;
# path mode divides by std^2 so `step_gain` is calibrated in metres.
_STATE_STD_XY = 20.0


@functools.lru_cache(maxsize=4)
def _load_prototypes(path: str) -> np.ndarray:
    """Load a prototypes .npy file, cached by path to avoid repeated disk I/O."""
    return np.load(path)


@register
class AnchorFollowingGuidance(BaseGuidance):
    """Soft guidance toward a prototype trajectory shape.

    The anchor is one row of a prototypes array of shape (K, 80, 2) loaded
    from a .npy file.

    Required params in GuidanceConfig.params:
        prototypes_path (str): Path to .npy file of shape (K, 80, 2).
        anchor_index (int):    Index of the prototype to follow (0 <= idx < K).

    Optional params (path mode):
        mode (str):        "path" (default) or "waypoint" (legacy).
        dist_cap (float):  Huber delta in metres; bounds the per-waypoint
                           gradient so the correction saturates instead of
                           exploding with distance. Default 2.0.
        extend_len (float): Metres to extrapolate the anchor polyline along
                           its final heading, so egos faster than the
                           prototype library are not dragged backward.
                           Default 100.0.
        step_gain (float): Target x0 nudge in metres per active solver step
                           per metre of (capped) offset, at config.scale=1
                           and global_scale=1. Default 1.0.
        t_min / t_max (float): Active guidance window in diffusion time.
                           Defaults 0.005 / 1.0 — unlike the legacy window
                           this includes the mode-forming high-noise steps.
    """

    name = "anchor_following"
    _energy_scale = 0.05  # legacy waypoint mode only

    def __init__(self, config: "GuidanceConfig"):  # noqa: F821
        super().__init__(config)
        protos = _load_prototypes(config.params["prototypes_path"])  # (K, 80, 2)
        idx = config.params["anchor_index"]
        self._anchor = torch.tensor(protos[idx], dtype=torch.float32)  # (80, 2)

        self._mode = config.params.get("mode", "path")
        if self._mode not in ("path", "waypoint"):
            raise ValueError(
                f"anchor_following mode must be 'path' or 'waypoint', got {self._mode!r}"
            )
        self._dist_cap = float(config.params.get("dist_cap", 2.0))
        self._extend_len = float(config.params.get("extend_len", 100.0))
        self._step_gain = float(config.params.get("step_gain", 1.0))
        self._path_t_min = float(config.params.get("t_min", 0.005))
        self._path_t_max = float(config.params.get("t_max", 1.0))
        self._anchor_path = self._build_anchor_path(self._anchor)  # (S, 2)

    def _build_anchor_path(self, anchor: torch.Tensor) -> torch.Tensor:
        """Anchor polyline extrapolated along its final heading.

        Drops near-duplicate trailing points (stop modes) before computing
        the final heading; a fully degenerate (stationary) anchor is
        extended along +x so it still defines a usable path.
        """
        seg = anchor[1:] - anchor[:-1]
        seg_len = seg.norm(dim=-1)
        valid = seg_len > 0.05
        if valid.any():
            last = int(torch.nonzero(valid)[-1])
            direction = seg[last] / seg_len[last]
        else:
            direction = torch.tensor([1.0, 0.0])
        tail = anchor[-1] + direction * self._extend_len
        return torch.cat([anchor, tail.unsqueeze(0)], dim=0)

    # ------------------------------------------------------------------
    # path mode
    # ------------------------------------------------------------------

    def _path_distance(self, ego_pred: torch.Tensor) -> torch.Tensor:
        """Min distance from each waypoint to the anchor polyline. [B,T] <- [B,T,2]."""
        path = self._anchor_path.to(ego_pred.device)  # (S, 2)
        a, b = path[:-1], path[1:]  # (S-1, 2)
        ab = b - a
        ab_sq = (ab * ab).sum(-1).clamp_min(1e-8)  # (S-1,)
        ap = ego_pred.unsqueeze(2) - a  # [B, T, S-1, 2]
        t_proj = ((ap * ab).sum(-1) / ab_sq).clamp(0.0, 1.0)  # [B, T, S-1]
        closest = a + t_proj.unsqueeze(-1) * ab  # [B, T, S-1, 2]
        d = (ego_pred.unsqueeze(2) - closest).norm(dim=-1)  # [B, T, S-1]
        return d.min(dim=-1).values  # [B, T]

    def _compute_path(self, x: torch.Tensor) -> torch.Tensor:
        """Huber path-distance energy. Returns [B]; higher = closer to path."""
        ego_pred = x[:, 0, 1:, :2]  # [B, T, 2]
        d = self._path_distance(ego_pred)
        cap = self._dist_cap
        huber = torch.where(d <= cap, 0.5 * d * d, cap * (d - 0.5 * cap))
        return -huber.sum(dim=-1)

    def energy(self, x: torch.Tensor, t: torch.Tensor, inputs: dict) -> torch.Tensor:
        if self._mode == "waypoint":
            return super().energy(x, t, inputs)

        t_scalar = t.reshape(t.shape[0], -1)[:, 0] if t.dim() > 1 else t
        mask = (t_scalar < self._path_t_max) * (t_scalar > self._path_t_min)
        mask_x = mask.view(x.shape[0], *([1] * (x.dim() - 1)))
        x_gated = torch.where(mask_x, x, x.detach())
        raw = self._compute_path(x_gated)

        # Per-step compensation: the DPM classifier correction shifts x0 by
        # guidance_scale * (sigma_t^2 / alpha_t) * std^2 * grad metres.
        # Multiplying the energy by alpha_t / sigma_t^2 makes the shift
        # magnitude t-independent: step_gain * scale metres per metre of
        # capped offset, at every active step.
        log_alpha = -0.25 * t_scalar**2 * (_BETA_1 - _BETA_0) - 0.5 * t_scalar * _BETA_0
        alpha = torch.exp(log_alpha)
        sigma_sq = (1.0 - torch.exp(2.0 * log_alpha)).clamp_min(1e-6)
        comp = alpha / sigma_sq

        gain = self._step_gain / (_STATE_STD_XY**2)
        return gain * self.config.scale * comp * raw

    # ------------------------------------------------------------------
    # shared _compute (used directly by reward(), and by energy() in
    # legacy waypoint mode)
    # ------------------------------------------------------------------

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        x: [B, P, T+1, 4] physical ego-centric metres.
        inputs: observation dict (unused by this function).

        Returns [B] unscaled reward (higher = closer to anchor).
        """
        if self._mode == "path":
            return self._compute_path(x)
        T = x.shape[2] - 1  # number of future timesteps
        ego_pred = x[:, 0, 1:, :2]  # [B, T, 2]
        anchor = self._anchor.to(x.device)[:T]  # [T, 2]
        sq_dist = ((ego_pred - anchor.unsqueeze(0)) ** 2).sum(dim=-1)  # [B, T]
        return -sq_dist.sum(dim=-1)  # [B]
