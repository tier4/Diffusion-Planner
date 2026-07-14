"""Anchor-following (MTR-style) guidance for the Diffusion Planner.

Guides the ego trajectory toward a pre-clustered prototype PATH
(a "motion mode") extracted from the training set. The anchor defines a
desired shape in the ego-centric frame (straight, turn-left, ...) and each
predicted waypoint is pulled toward the nearest point on the (extrapolated)
anchor polyline with a Huber cost, so the guidance shapes WHERE the
trajectory goes without fighting the model over speed. The energy is
time-compensated (multiplied by alpha_t / sigma_t^2) so the effective x0
nudge per solver step is constant across the whole denoising schedule, and
the guidance window covers the mode-forming high-noise steps — this is what
makes the guidance able to actually switch trajectory modes.

The legacy per-timestep squared-distance behaviour ("waypoint mode") was
REMOVED: a scale sweep showed it has no usable operating point — near-inert
at low scale, and at scales high enough to steer it pastes the prototype
shape over the scene regardless of the map. Configs passing params["mode"]
fail loudly.
"""

import functools

import numpy as np
import torch

from .base import BaseGuidance
from .registry import register

# The VP-SDE linear schedule constants (NoiseScheduleVP defaults). Used to
# compute the per-step time compensation alpha_t / sigma_t^2.
_BETA_0 = 0.1
_BETA_1 = 20.0

# Default xy std of the state normalizer (normalization.json convention used
# by all current checkpoints). The DPM classifier correction shifts x0 by
# guidance_scale * sigma^2/alpha * std^2 * dE/dx_phys metres per coordinate;
# the energy divides by std^2 so `step_gain` is calibrated in metres. If a
# checkpoint normalizes xy differently, pass params["state_std_xy"].
_STATE_STD_XY = 20.0


@functools.lru_cache(maxsize=4)
def _load_prototypes(path: str) -> np.ndarray:
    """Load a prototypes .npy file, cached by path to avoid repeated disk I/O."""
    return np.load(path)


@register
class AnchorFollowingGuidance(BaseGuidance):
    """Soft guidance toward a prototype path shape.

    The anchor is one row of a prototypes array of shape (K, M, 2) loaded
    from a .npy file — any per-row point count M works; the row is treated
    as a polyline, not as per-timestep waypoints. Build libraries with
    ``rlvr.autoresearch.tools.build_path_prototypes`` (arc-length clustered,
    speed-free).

    Required params in GuidanceConfig.params:
        prototypes_path (str): Path to .npy file of shape (K, M, 2).
        anchor_index (int):    Index of the prototype to follow (0 <= idx < K).

    Optional params:
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
                           Defaults 0.005 / 1.0. The window is inclusive at
                           t_max: the solver's FIRST model evaluation sits at
                           exactly t=1.0 and is the most mode-forming step.
        state_std_xy (float): xy std of the checkpoint's state normalizer.
                           Default 20.0 (this repo's normalization.json
                           convention); override for checkpoints normalized
                           differently or the metre calibration is off by
                           (std/20)^2.
    """

    name = "anchor_following"

    def __init__(self, config: "GuidanceConfig"):  # noqa: F821
        super().__init__(config)
        if "mode" in config.params:
            raise ValueError(
                "anchor_following params['mode'] was removed: the legacy 'waypoint' "
                "behaviour had no usable operating point and path following is now "
                "the only behaviour. Drop the param."
            )
        protos = _load_prototypes(config.params["prototypes_path"])  # (K, M, 2)
        idx = config.params["anchor_index"]
        self._anchor = torch.tensor(protos[idx], dtype=torch.float32)  # (M, 2)

        self._dist_cap = float(config.params.get("dist_cap", 2.0))
        self._extend_len = float(config.params.get("extend_len", 100.0))
        self._step_gain = float(config.params.get("step_gain", 1.0))
        self._path_t_min = float(config.params.get("t_min", 0.005))
        self._path_t_max = float(config.params.get("t_max", 1.0))
        self._state_std_xy = float(config.params.get("state_std_xy", _STATE_STD_XY))
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

    def _path_distance(self, ego_pred: torch.Tensor) -> torch.Tensor:
        """Min distance from each waypoint to the anchor polyline. [B,T] <- [B,T,2]."""
        if self._anchor_path.device != ego_pred.device:
            # One-time move per device; energy() runs once per solver step.
            self._anchor_path = self._anchor_path.to(ego_pred.device)
        path = self._anchor_path  # (S, 2)
        a, b = path[:-1], path[1:]  # (S-1, 2)
        ab = b - a
        ab_sq = (ab * ab).sum(-1).clamp_min(1e-8)  # (S-1,)
        ap = ego_pred.unsqueeze(2) - a  # [B, T, S-1, 2]
        t_proj = ((ap * ab).sum(-1) / ab_sq).clamp(0.0, 1.0)  # [B, T, S-1]
        closest = a + t_proj.unsqueeze(-1) * ab  # [B, T, S-1, 2]
        d = (ego_pred.unsqueeze(2) - closest).norm(dim=-1)  # [B, T, S-1]
        return d.min(dim=-1).values  # [B, T]

    def energy(self, x: torch.Tensor, t: torch.Tensor, inputs: dict) -> torch.Tensor:
        t_scalar = t.reshape(t.shape[0], -1)[:, 0] if t.dim() > 1 else t
        # Inclusive at t_max: the solver's first (and most mode-forming) model
        # evaluation is at exactly t = 1.0.
        mask = (t_scalar <= self._path_t_max) * (t_scalar > self._path_t_min)
        mask_x = mask.view(x.shape[0], *([1] * (x.dim() - 1)))
        x_gated = torch.where(mask_x, x, x.detach())
        raw = self._compute(x_gated, inputs)

        # Per-step compensation: the DPM classifier correction shifts x0 by
        # guidance_scale * (sigma_t^2 / alpha_t) * std^2 * grad metres.
        # Multiplying the energy by alpha_t / sigma_t^2 makes the shift
        # magnitude t-independent: step_gain * scale metres per metre of
        # capped offset, at every active step.
        log_alpha = -0.25 * t_scalar**2 * (_BETA_1 - _BETA_0) - 0.5 * t_scalar * _BETA_0
        alpha = torch.exp(log_alpha)
        sigma_sq = (1.0 - torch.exp(2.0 * log_alpha)).clamp_min(1e-6)
        comp = alpha / sigma_sq

        gain = self._step_gain / (self._state_std_xy**2)
        return gain * self.config.scale * comp * raw

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """Huber path-distance energy.

        x: [B, P, T+1, 4] physical ego-centric metres.
        inputs: observation dict (unused by this function).

        Returns [B]; higher = closer to the anchor path.
        """
        ego_pred = x[:, 0, 1:, :2]  # [B, T, 2]
        d = self._path_distance(ego_pred)
        cap = self._dist_cap
        huber = torch.where(d <= cap, 0.5 * d * d, cap * (d - 0.5 * cap))
        return -huber.sum(dim=-1)
