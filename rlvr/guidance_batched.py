"""Per-sample (batched) guidance functions for explorer-driven generation.

These live OUTSIDE ``diffusion_planner/`` (read-only) but register into the
same guidance registry, following the ``guidance_gui/custom_guidance.py``
pattern. They differ from their scalar counterparts in ONE way: the action
parameter may be a ``[B]`` tensor, giving each batch element its own guidance
strength — required when K explorer samples are generated in a single batch.

Provided:
    collision_swerve_batched -- directional, proximity-gated lateral swerve
        with a SIGNED per-sample ``eta_col``: +1 = full swerve LEFT,
        -1 = full swerve RIGHT (ego frame, +y = left — note the Scene Branch
        Editor slider shows the SCREEN direction, where + is right),
        0 = exactly inert (zero energy, zero gradient).
    speed_stretch_batched -- per-sample displacement stretch: ``stretch`` < 1
        slows/shortens the trajectory, > 1 speeds it up, 1 = inert. Same
        surrogate-gradient formulation as the scalar ``speed`` guidance's
        stretch mode, without its scalar-only ``abs(stretch - 1)`` branch.
"""

import torch

from diffusion_planner.model.guidance.base import BaseGuidance
from diffusion_planner.model.guidance.registry import register


def _as_batch_param(value, B: int, device) -> torch.Tensor:
    """Normalize a scalar-or-[B]-tensor param to a [B] tensor."""
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return value.reshape(1).expand(B).to(device)
        if value.shape[0] != B:
            raise ValueError(
                f"batched guidance param has shape {tuple(value.shape)}, "
                f"expected scalar or [{B}]"
            )
        return value.to(device)
    return torch.full((B,), float(value), device=device)


@register
class CollisionSwerveBatchedGuidance(BaseGuidance):
    """Directional, proximity-gated lateral swerve with per-sample signed eta.

    Params:
        eta_col (float | Tensor[B]): signed swerve command in [-1, 1].
            sign: +1 = swerve LEFT, -1 = swerve RIGHT (ego frame, +y = left).
            magnitude: push strength. 0 = inert (zero energy AND gradient).
        range (float): proximity radius in metres within which the swerve
            activates (centroid gap ego<->neighbour). Default 8.0.
        head_protect (int): zero the guidance weight on the FIRST N future
            steps. The closed loop executes the plan head; gradient guidance
            distorting it stalls the ego (sideways pull eats the first-step
            forward displacement). Default 0 = original behavior.

    Energy (maximised by the solver):
        E = eta_col * Σ_t  w_t * y_t
    where y_t is the ego lateral position at future step t and
    w_t = max(0, 1 - d_t / range) is the (detached) proximity to the nearest
    valid neighbour: a clean lateral push toward the chosen side, strongest
    next to the obstacle, self-gating to zero when no neighbour is in range.
    """

    name = "collision_swerve_batched"
    _energy_scale = 1.0

    def __init__(self, config: "GuidanceConfig", **kwargs):  # noqa: F821
        super().__init__(config)
        self._eta_col = config.params.get("eta_col", 0.0)
        self._range = float(config.params.get("range", 8.0))
        self._head_protect = int(config.params.get("head_protect", 0))

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        x: [B, P, T+1, 4] physical ego-centric metres.
        inputs: observation dict in physical units (needs neighbor_agents_past).

        Returns [B] energy (higher = more displacement toward the chosen side
        near obstacles).
        """
        B = x.shape[0]
        device = x.device

        nb = inputs.get("neighbor_agents_past")
        if nb is None:
            return torch.zeros(B, device=device)

        eta = _as_batch_param(self._eta_col, B, device)

        # Ego future positions (keep grad on xy).
        ego_xy = x[:, 0, 1:, :2]  # [B, T, 2]

        # Static neighbour current positions + validity mask.
        nb_cur = nb[:, :, -1, :4].detach()         # [B, Pn, 4]
        nb_xy = nb_cur[..., :2]                    # [B, Pn, 2]
        nb_valid = nb_cur.abs().sum(dim=-1) > 0    # [B, Pn]

        if nb_valid.sum().item() == 0:
            return torch.zeros(B, device=device)

        # Centroid gap ego_t <-> neighbour (detached -> proximity gate only).
        d = torch.cdist(ego_xy.detach(), nb_xy)    # [B, T, Pn]
        big = torch.full_like(d, 1e6)
        d = torch.where(nb_valid[:, None, :], d, big)
        w = torch.clamp(1.0 - d / self._range, min=0.0)   # [B, T, Pn]
        w = w.max(dim=-1).values                          # [B, T]
        if self._head_protect > 0:
            w = w.clone()
            w[:, : self._head_protect] = 0.0

        y = ego_xy[..., 1]                                # [B, T] lateral, +y = left
        reward = (eta[:, None] * w.detach() * y).sum(dim=-1)  # [B]
        return reward


@register
class SpeedStretchBatchedGuidance(BaseGuidance):
    """Per-sample trajectory stretch (tensor-safe ``speed`` stretch mode).

    Params:
        stretch (float | Tensor[B]): per-step displacement scale factor.
            < 1 = slow down / shorten, > 1 = speed up, 1 = inert (the
            correction is exactly zero, unlike the scalar ``speed`` guidance
            whose stretch mode falls back to band clamping at 1.0).

    Surrogate gradient: correction = disp * (stretch - 1), energy =
    dot(correction.detach(), pos) so grad_x(E) = correction.
    """

    name = "speed_stretch_batched"
    _energy_scale = 1.0

    def __init__(self, config: "GuidanceConfig", **kwargs):  # noqa: F821
        super().__init__(config)
        self._stretch = config.params.get("stretch", 1.0)

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        x: [B, P, T+1, 4] physical ego-centric metres.

        Returns [B] surrogate energy whose gradient pushes each point along
        its travel direction by (stretch - 1) * displacement.
        """
        B = x.shape[0]
        stretch = _as_batch_param(self._stretch, B, x.device)

        pos = x[:, 0, 1:, :2]                    # [B, T, 2]
        disp = pos[:, 1:, :] - pos[:, :-1, :]    # [B, T-1, 2]
        correction = disp * (stretch[:, None, None] - 1.0)
        reward = torch.sum(correction.detach() * pos[:, 1:, :2], dim=(1, 2))
        return reward


@register
class LateralBatchedGuidance(BaseGuidance):
    """PlannerRFT Eq.2 lateral guidance with head protection.

    Identical energy to the stock ``lateral`` (which already takes [B] eta)
    EXCEPT the first ``head_protect`` future steps carry zero weight — the
    closed loop executes the plan head, and bending it sideways stalls the
    ego. head_protect=0 reproduces the stock function exactly.

    Params: lambda_lat (float), eta_lat (float | Tensor[B]),
            head_protect (int, default 0).
    """

    name = "lateral_batched"
    _energy_scale = 10.0

    def __init__(self, config: "GuidanceConfig", **kwargs):  # noqa: F821
        super().__init__(config)
        self._lambda_lat = config.params.get("lambda_lat", 3.0)
        self._eta_lat = config.params.get("eta_lat", 0.0)
        self._head_protect = int(config.params.get("head_protect", 0))

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        B, P, T_plus1, _ = x.shape
        T = T_plus1 - 1
        device = x.device
        ref = inputs.get("reference_trajectory")
        if ref is None:
            return torch.zeros(B, device=device)
        ref = ref[:, :T, :]

        cos_h = ref[..., 2]
        sin_h = ref[..., 3]
        h_norm = (cos_h ** 2 + sin_h ** 2).sqrt().clamp_min(1e-6)
        n_perp_x = -sin_h / h_norm
        n_perp_y = cos_h / h_norm

        ego_pos = x[:, 0, 1:, :2]
        dx = ego_pos[..., 0] - ref[..., 0]
        dy = ego_pos[..., 1] - ref[..., 1]
        lateral_proj = n_perp_x * dx + n_perp_y * dy  # [B, T]

        target = self._lambda_lat * self._eta_lat
        if isinstance(target, torch.Tensor) and target.dim() >= 1:
            target = target.unsqueeze(-1)

        err2 = (lateral_proj - target) ** 2  # [B, T]
        if self._head_protect > 0:
            w = torch.ones_like(err2)
            w[:, : self._head_protect] = 0.0
            return -(err2 * w).sum(dim=-1) / w.sum(dim=-1).clamp_min(1.0)
        return -err2.mean(dim=-1)
