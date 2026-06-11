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


def build_head_composer(
    etas,
    *,
    lambda_lat: float = 5.0,
    lat_scale: float = 2.0,
    col_scale: float = 9.0,
    col_range: float = 8.0,
    lambda_spd: float = 0.2,
    stretch_scale: float = 1.0,
    guidance_scale: float = 0.5,
    head_protect: int = 0,
    lambda_lon: float = 0.25,
    envelope: str = "v1",
    lambda_col: float = 3.0,
    ramp_steps: int = 20,
):
    """Build a GuidanceComposer from a head->eta dict (scalar or [B] tensor).

    Single source of truth for the head-name -> guidance-function mapping
    (lateral / collision / stretch / legacy longitudinal), shared by the
    eval tools and the closed-loop rollout managers. Defaults are
    envelope-v2; a legacy lateral+longitudinal trainer config reproduces
    exactly with lat_scale=1.0, lambda_lat=2.5, lambda_lon=0.25.
    """
    from diffusion_planner.model.guidance.composer import GuidanceComposer
    from diffusion_planner.model.guidance.config import (
        GuidanceConfig,
        GuidanceSetConfig,
    )

    hp = int(head_protect)
    fns = []
    if envelope == "v2":
        # v2 set: ramped lateral target + ramp-and-hold bounded swerve.
        # head_protect is not implemented for v2 (the ramp already spares
        # the plan head) — fail loudly rather than silently ignore it.
        if hp > 0:
            raise ValueError("head_protect is a v1-envelope option")
        if "lateral" in etas:
            fns.append(GuidanceConfig(
                name="lateral_ramp_batched", enabled=True, scale=lat_scale,
                params={"lambda_lat": lambda_lat, "eta_lat": etas["lateral"],
                        "ramp_steps": ramp_steps},
            ))
        if "collision" in etas:
            fns.append(GuidanceConfig(
                name="collision_swerve_v2_batched", enabled=True,
                scale=col_scale,
                params={"eta_col": etas["collision"], "lambda_col": lambda_col,
                        "range": col_range},
            ))
    else:
        if "lateral" in etas:
            if hp > 0:
                fns.append(GuidanceConfig(
                    name="lateral_batched", enabled=True, scale=lat_scale,
                    params={"lambda_lat": lambda_lat, "eta_lat": etas["lateral"],
                            "head_protect": hp},
                ))
            else:
                fns.append(GuidanceConfig(
                    name="lateral", enabled=True, scale=lat_scale,
                    params={"lambda_lat": lambda_lat, "eta_lat": etas["lateral"]},
                ))
        if "collision" in etas:
            fns.append(GuidanceConfig(
                name="collision_swerve_batched", enabled=True, scale=col_scale,
                params={"eta_col": etas["collision"], "range": col_range,
                        "head_protect": hp},
            ))
    if "stretch" in etas:
        fns.append(GuidanceConfig(
            name="speed_stretch_batched", enabled=True, scale=stretch_scale,
            params={"stretch": 1.0 + lambda_spd * etas["stretch"]},
        ))
    if "longitudinal" in etas:
        # Legacy head (known-broken for speed control; kept so old
        # lateral+longitudinal configs reproduce bit-for-bit).
        fns.append(GuidanceConfig(
            name="longitudinal", enabled=True, scale=1.0,
            params={"lambda_lon": lambda_lon, "eta_lon": etas["longitudinal"]},
        ))
    set_cfg = GuidanceSetConfig(functions=fns, global_scale=guidance_scale)
    return GuidanceComposer(set_cfg)


@register
class CollisionSwerveV2BatchedGuidance(BaseGuidance):
    """Bounded-target, support-normalized swerve (v2 of collision_swerve).

    Fixes two defects of ``collision_swerve_batched`` measured on the
    response-curve probe (gain varied ~40x between scenes):
      1. The v1 energy ``eta * sum_t(w_t * y_t)`` is LINEAR in y with no
         target — gradient magnitude scales with how many steps are in
         range, and it keeps pushing forever (one scene reached +31 m).
      2. No support normalization — slow approaches (many in-range steps)
         get a huge cumulative gradient, brief encounters almost none.

    v2 energy (maximised by the solver):
        E = - sum_t( w_t * (y_t - sign(eta) * lambda_col * |eta|)^2 ) / sum_t(w_t)
    A given eta now means the SAME bounded target offset in every scene,
    applied only where the proximity gate w_t is active. eta = 0 keeps the
    v1 contract: exactly zero energy and gradient.

    Params:
        eta_col (float | Tensor[B]): signed command in [-1, 1].
        lambda_col (float): max target lateral offset in metres. Default 3.0.
        range (float): proximity radius in metres. Default 8.0.
        head_protect (int): zero weights on the first N future steps. Default 0.
    """

    name = "collision_swerve_v2_batched"
    _energy_scale = 10.0

    def __init__(self, config: "GuidanceConfig", **kwargs):  # noqa: F821
        super().__init__(config)
        self._eta_col = config.params.get("eta_col", 0.0)
        self._lambda_col = float(config.params.get("lambda_col", 3.0))
        self._range = float(config.params.get("range", 8.0))
        self._head_protect = int(config.params.get("head_protect", 0))

    def _compute(self, x: torch.Tensor, inputs: dict) -> torch.Tensor:
        B = x.shape[0]
        device = x.device
        nb = inputs.get("neighbor_agents_past")
        ref = inputs.get("reference_trajectory")
        if nb is None or ref is None:
            return torch.zeros(B, device=device)

        eta = _as_batch_param(self._eta_col, B, device)

        ego_xy = x[:, 0, 1:, :2]                   # [B, T, 2] (grad kept)
        T = ego_xy.shape[1]
        nb_cur = nb[:, :, -1, :4].detach()
        nb_xy = nb_cur[..., :2]
        nb_valid = nb_cur.abs().sum(dim=-1) > 0
        if nb_valid.sum().item() == 0:
            return torch.zeros(B, device=device)

        # Proximity bump from the (detached) REFERENCE trajectory — using the
        # current noisy x here would gate on garbage early in denoising.
        ref = ref[:, :T, :].detach()
        d = torch.cdist(ref[..., :2], nb_xy)       # [B, T, Pn]
        big = torch.full_like(d, 1e6)
        d = torch.where(nb_valid[:, None, :], d, big)
        w = torch.clamp(1.0 - d / self._range, min=0.0).max(dim=-1).values  # [B, T]
        if self._head_protect > 0:
            w = w.clone()
            w[:, : self._head_protect] = 0.0
        support = w.sum(dim=-1)                    # [B]

        # Lateral deviation from the reference (handles curved routes),
        # identical machinery to the stock lateral energy.
        cos_h = ref[..., 2]
        sin_h = ref[..., 3]
        h_norm = (cos_h ** 2 + sin_h ** 2).sqrt().clamp_min(1e-6)
        n_perp_x, n_perp_y = -sin_h / h_norm, cos_h / h_norm
        dx = ego_xy[..., 0] - ref[..., 0]
        dy = ego_xy[..., 1] - ref[..., 1]
        lateral_proj = n_perp_x * dx + n_perp_y * dy            # [B, T]

        # Ramp-up-and-HOLD target profile: cummax of the proximity gate
        # rises on approach and holds its peak after passing — the offset is
        # demanded over the remaining horizon like the (stable) stock
        # lateral target, instead of a bump that forces return-to-zero
        # mid-pass (bump variants needed unstable scales to act at all:
        # weak at scale 8, sign-inverting/divergent at 16).
        s = torch.cummax(w, dim=-1).values                      # [B, T]
        target = (self._lambda_col * eta)[:, None] * s          # [B, T]
        psi = ((lateral_proj - target) ** 2).mean(dim=-1)
        # eta == 0 -> exactly inert; negligible support -> exactly inert
        gate = (eta.abs() > 1e-6).float() * (support > 0.05).float()
        return -psi * gate


@register
class LateralRampBatchedGuidance(BaseGuidance):
    """Lateral guidance (Eq. 2) with a kinematic feasibility RAMP (v2).

    The stock ``lateral`` energy demands the full lambda*eta offset at EVERY
    future step including t = 0.1 s — kinematically impossible, producing
    huge near-field gradients (measured: plan-head distortion, backwards-
    bent leading points at low speed, scene-dependent gain 2.3-6.8 m at the
    same eta). v2 ramps the target linearly from 0 to lambda*eta over
    ``ramp_steps`` future steps, then holds.

    Params:
        lambda_lat (float): max lateral offset in metres. Default 3.0.
        eta_lat (float | Tensor[B]): command in [-1, 1].
        ramp_steps (int): steps to reach the full target. Default 20 (2 s).
    """

    name = "lateral_ramp_batched"
    _energy_scale = 10.0

    def __init__(self, config: "GuidanceConfig", **kwargs):  # noqa: F821
        super().__init__(config)
        self._lambda_lat = float(config.params.get("lambda_lat", 3.0))
        self._eta_lat = config.params.get("eta_lat", 0.0)
        self._ramp_steps = int(config.params.get("ramp_steps", 20))

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
        cos_h, sin_h = cos_h / h_norm, sin_h / h_norm
        n_perp_x, n_perp_y = -sin_h, cos_h

        ego_pos = x[:, 0, 1:, :2]
        dx = ego_pos[..., 0] - ref[..., 0]
        dy = ego_pos[..., 1] - ref[..., 1]
        lateral_proj = n_perp_x * dx + n_perp_y * dy        # [B, T]

        eta = _as_batch_param(self._eta_lat, B, device)
        ramp = torch.arange(1, T + 1, device=device, dtype=torch.float32)
        ramp = (ramp / max(self._ramp_steps, 1)).clamp(max=1.0)  # [T]
        target = (self._lambda_lat * eta)[:, None] * ramp[None, :]  # [B, T]

        psi = ((lateral_proj - target) ** 2).mean(dim=-1)
        return -psi
