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
                f"batched guidance param has shape {tuple(value.shape)}, expected scalar or [{B}]"
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
        nb_cur = nb[:, :, -1, :4].detach()  # [B, Pn, 4]
        nb_xy = nb_cur[..., :2]  # [B, Pn, 2]
        nb_valid = nb_cur.abs().sum(dim=-1) > 0  # [B, Pn]

        if nb_valid.sum().item() == 0:
            return torch.zeros(B, device=device)

        # Centroid gap ego_t <-> neighbour (detached -> proximity gate only).
        d = torch.cdist(ego_xy.detach(), nb_xy)  # [B, T, Pn]
        big = torch.full_like(d, 1e6)
        d = torch.where(nb_valid[:, None, :], d, big)
        w = torch.clamp(1.0 - d / self._range, min=0.0)  # [B, T, Pn]
        w = w.max(dim=-1).values  # [B, T]
        if self._head_protect > 0:
            w = w.clone()
            w[:, : self._head_protect] = 0.0

        y = ego_xy[..., 1]  # [B, T] lateral, +y = left
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

        pos = x[:, 0, 1:, :2]  # [B, T, 2]
        disp = pos[:, 1:, :] - pos[:, :-1, :]  # [B, T-1, 2]
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
        h_norm = (cos_h**2 + sin_h**2).sqrt().clamp_min(1e-6)
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
    fast: bool = True,
    guidance_strength=None,
):
    """Build a GuidanceComposer from a head->eta dict (scalar or [B] tensor).

    Single source of truth for the head-name -> guidance-function mapping
    (lateral / collision / stretch / legacy longitudinal), shared by the
    eval tools and the closed-loop rollout managers. The scale defaults are
    the v1-envelope calibration (lambda_lat 5.0 / lat_scale 2.0 /
    col_scale 9.0), matching envelope="v1"; pass envelope="v2" (+ lambda_col,
    ramp_steps) for the ramped v2 functions. A legacy lateral+longitudinal
    trainer config reproduces exactly with lat_scale=1.0, lambda_lat=2.5,
    lambda_lon=0.25.
    """
    from diffusion_planner.model.guidance.composer import GuidanceComposer
    from diffusion_planner.model.guidance.config import (
        GuidanceConfig,
        GuidanceSetConfig,
    )

    unknown = set(etas) - {"lateral", "collision", "stretch", "longitudinal"}
    if unknown:
        raise ValueError(
            f"unknown guidance head(s) {sorted(unknown)} — a head with no "
            "function mapping would train/act as a dead head silently"
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
            fns.append(
                GuidanceConfig(
                    name="lateral_ramp_batched",
                    enabled=True,
                    scale=lat_scale,
                    params={
                        "lambda_lat": lambda_lat,
                        "eta_lat": etas["lateral"],
                        "ramp_steps": ramp_steps,
                    },
                )
            )
        if "collision" in etas:
            fns.append(
                GuidanceConfig(
                    name="collision_swerve_v2_batched",
                    enabled=True,
                    scale=col_scale,
                    params={
                        "eta_col": etas["collision"],
                        "lambda_col": lambda_col,
                        "range": col_range,
                    },
                )
            )
    else:
        if "lateral" in etas:
            if hp > 0:
                fns.append(
                    GuidanceConfig(
                        name="lateral_batched",
                        enabled=True,
                        scale=lat_scale,
                        params={
                            "lambda_lat": lambda_lat,
                            "eta_lat": etas["lateral"],
                            "head_protect": hp,
                        },
                    )
                )
            else:
                fns.append(
                    GuidanceConfig(
                        name="lateral",
                        enabled=True,
                        scale=lat_scale,
                        params={"lambda_lat": lambda_lat, "eta_lat": etas["lateral"]},
                    )
                )
        if "collision" in etas:
            fns.append(
                GuidanceConfig(
                    name="collision_swerve_batched",
                    enabled=True,
                    scale=col_scale,
                    params={"eta_col": etas["collision"], "range": col_range, "head_protect": hp},
                )
            )
    if "stretch" in etas:
        fns.append(
            GuidanceConfig(
                name="speed_stretch_batched",
                enabled=True,
                scale=stretch_scale,
                params={"stretch": 1.0 + lambda_spd * etas["stretch"]},
            )
        )
    if "longitudinal" in etas:
        # Legacy head (known-broken for speed control; kept so old
        # lateral+longitudinal configs reproduce bit-for-bit).
        fns.append(
            GuidanceConfig(
                name="longitudinal",
                enabled=True,
                scale=1.0,
                params={"lambda_lon": lambda_lon, "eta_lon": etas["longitudinal"]},
            )
        )
    set_cfg = GuidanceSetConfig(functions=fns, global_scale=guidance_scale)
    if fast:
        # Equivalence-certified against GuidanceComposer (bit-identical
        # active trajectories; inert frames short-circuit to ~unguided cost).
        return FastGuidanceComposer(set_cfg, guidance_strength=guidance_strength)
    if guidance_strength is not None:
        # The strength gate scales the summed energy inside FastGuidanceComposer;
        # the read-only diffusion_planner GuidanceComposer has no such hook.
        raise ValueError(
            "guidance_strength requires fast=True (FastGuidanceComposer); the "
            "slow GuidanceComposer does not support the strength gate"
        )
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

        ego_xy = x[:, 0, 1:, :2]  # [B, T, 2] (grad kept)
        T = ego_xy.shape[1]
        nb_cur = nb[:, :, -1, :4].detach()
        nb_xy = nb_cur[..., :2]
        nb_valid = nb_cur.abs().sum(dim=-1) > 0
        if nb_valid.sum().item() == 0:
            return torch.zeros(B, device=device)

        # Proximity bump from the (detached) REFERENCE trajectory — using the
        # current noisy x here would gate on garbage early in denoising.
        ref = ref[:, :T, :].detach()
        d = torch.cdist(ref[..., :2], nb_xy)  # [B, T, Pn]
        big = torch.full_like(d, 1e6)
        d = torch.where(nb_valid[:, None, :], d, big)
        w = torch.clamp(1.0 - d / self._range, min=0.0).max(dim=-1).values  # [B, T]
        if self._head_protect > 0:
            w = w.clone()
            w[:, : self._head_protect] = 0.0
        support = w.sum(dim=-1)  # [B]

        # Lateral deviation from the reference (handles curved routes),
        # identical machinery to the stock lateral energy.
        cos_h = ref[..., 2]
        sin_h = ref[..., 3]
        h_norm = (cos_h**2 + sin_h**2).sqrt().clamp_min(1e-6)
        n_perp_x, n_perp_y = -sin_h / h_norm, cos_h / h_norm
        dx = ego_xy[..., 0] - ref[..., 0]
        dy = ego_xy[..., 1] - ref[..., 1]
        lateral_proj = n_perp_x * dx + n_perp_y * dy  # [B, T]

        # Ramp-up-and-HOLD target profile: cummax of the proximity gate
        # rises on approach and holds its peak after passing — the offset is
        # demanded over the remaining horizon like the (stable) stock
        # lateral target, instead of a bump that forces return-to-zero
        # mid-pass (bump variants needed unstable scales to act at all:
        # weak at scale 8, sign-inverting/divergent at 16).
        s = torch.cummax(w, dim=-1).values  # [B, T]
        target = (self._lambda_col * eta)[:, None] * s  # [B, T]
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
        h_norm = (cos_h**2 + sin_h**2).sqrt().clamp_min(1e-6)
        cos_h, sin_h = cos_h / h_norm, sin_h / h_norm
        n_perp_x, n_perp_y = -sin_h, cos_h

        ego_pos = x[:, 0, 1:, :2]
        dx = ego_pos[..., 0] - ref[..., 0]
        dy = ego_pos[..., 1] - ref[..., 1]
        lateral_proj = n_perp_x * dx + n_perp_y * dy  # [B, T]

        eta = _as_batch_param(self._eta_lat, B, device)
        ramp = torch.arange(1, T + 1, device=device, dtype=torch.float32)
        ramp = (ramp / max(self._ramp_steps, 1)).clamp(max=1.0)  # [T]
        target = (self._lambda_lat * eta)[:, None] * ramp[None, :]  # [B, T]

        psi = ((lateral_proj - target) ** 2).mean(dim=-1)
        return -psi


class FastGuidanceComposer:
    """Drop-in faster GuidanceComposer (same classifier_fn interface).

    Three measured wins over diffusion_planner's GuidanceComposer, with
    identical math:
      1. inputs inverse-normalisation cached ONCE per generation (the base
         composer inverse-normalises the full observation dict — lanes,
         neighbours — at EVERY denoise step although it never changes).
      2. eta~0 short-circuit: if every active function reports an inert
         command (|eta| < eta_eps), return a zero surrogate immediately —
         no x0-correction model forward, no energy autograd. With an inert
         policy (the common case on a route) the guided generation cost
         collapses to the unguided baseline.
         CAVEAT: the v1 ``lateral`` head AND the v2 ``lateral_ramp``
         variant are not mathematically inert at eta=0 — their quadratic
         energies still pull toward the reference trajectory. The
         short-circuit is exact when the reference IS the model's own det
         trajectory (the standard explorer setup, pull ~ 0); with any
         other reference the slow composer would apply a residual
         centering pull this fast path skips.
      3. optional skip_x0_correction: evaluate energies on the solver's
         current x directly instead of running an EXTRA full model forward
         to refine x0 first. Off by default (changes results slightly —
         enable only after the equivalence battery passes).

    Construction mirrors GuidanceComposer (GuidanceSetConfig), so
    build_head_composer can emit either.
    """

    def __init__(
        self,
        set_config,
        eta_eps: float = 1e-3,
        skip_x0_correction: bool = False,
        guidance_strength=None,
        **build_kwargs,
    ):
        from diffusion_planner.model.guidance.registry import build as _build

        self._set_config = set_config
        self._functions = [
            _build(fn_cfg, **build_kwargs) for fn_cfg in set_config.active_functions()
        ]
        self._eta_eps = float(eta_eps)
        self._skip_x0 = bool(skip_x0_correction)
        self._inputs_phys = None  # per-generation cache
        # Learned guidance-strength gate g in (0,1): scales the summed guidance
        # energy (hence the guidance gradient) per scene. None -> no gating
        # (full envelope). [B] tensor (or scalar) broadcast over the batch.
        self._strength = guidance_strength

    def _strength_inert(self) -> bool:
        """True if the strength gate forces zero guidance for the whole batch."""
        s = self._strength
        if s is None:
            return False
        if isinstance(s, torch.Tensor):
            return bool(s.abs().max().item() <= self._eta_eps)
        return abs(float(s)) <= self._eta_eps

    def reset_cache(self):
        self._inputs_phys = None

    def _all_inert(self) -> bool:
        for fn in self._functions:
            recognized = False
            for attr in ("_eta_lat", "_eta_col", "_eta_lon"):
                v = getattr(fn, attr, None)
                if v is None:
                    continue
                recognized = True
                if isinstance(v, torch.Tensor):
                    if v.abs().max().item() > self._eta_eps:
                        return False
                elif abs(float(v)) > self._eta_eps:
                    return False
            stretch = getattr(fn, "_stretch", None)
            if stretch is not None:
                # Stock SpeedGuidance carries _v_low/_v_high and at
                # stretch==1.0 runs in BAND mode (speed clamping — active
                # regardless of stretch). Only the dedicated stretch-mode
                # variants (no band attrs) are inert at 1.0.
                if (
                    getattr(fn, "_v_low", None) is not None
                    or getattr(fn, "_v_high", None) is not None
                ):
                    return False
                recognized = True
                if isinstance(stretch, torch.Tensor):
                    if (stretch - 1.0).abs().max().item() > self._eta_eps:
                        return False
                elif abs(float(stretch) - 1.0) > self._eta_eps:
                    return False
            if not recognized:
                # A function exposing no known eta attribute must be treated
                # as active — short-circuiting would silently drop it.
                return False
        return True

    def __call__(self, x_in, t_input, cond, *args, **kwargs):
        B = x_in.shape[0]
        if self._all_inert() or self._strength_inert():
            # zero surrogate with a real graph so the solver's autograd
            # produces an (all-zero) gradient of the right shape
            return (x_in * 0.0).sum()

        state_normalizer = kwargs["state_normalizer"]
        P = x_in.shape[1]
        x_4d = x_in.reshape(B, P, -1, 4)
        if self._skip_x0:
            x_corrected = x_4d
        else:
            model = kwargs["model"]
            model_condition = kwargs["model_condition"]
            x_fix = model(x_4d, t_input, **model_condition).detach() - x_4d.detach()
            x_fix[:, :, 0] = 0.0
            x_corrected = x_4d + x_fix

        x_phys = state_normalizer.inverse(x_corrected.detach())
        if self._inputs_phys is None:
            self._inputs_phys = kwargs["observation_normalizer"].inverse(kwargs["inputs"])
        inputs = self._inputs_phys

        t_scalar = t_input.reshape(B, -1)[:, 0] if t_input.dim() > 1 else t_input
        x_phys_grad = x_phys.detach().requires_grad_(True)
        raw_energy = torch.zeros(B, device=x_in.device)
        for fn in self._functions:
            e = fn.energy(x_phys_grad, t_scalar, inputs)
            if torch.isnan(e).any():
                continue
            raw_energy = raw_energy + e
        if self._strength is not None:
            # Per-scene learned gate: scale the summed energy (g is constant
            # w.r.t. x, so the guidance gradient is scaled by g per sample).
            s = self._strength
            if isinstance(s, torch.Tensor):
                # detached: only x is differentiated; s is a constant scale.
                s = s.detach().to(raw_energy.device).reshape(-1)
                if s.numel() == 1:
                    s = s.expand(B)
                elif s.numel() != B:
                    raise ValueError(
                        f"guidance_strength has {s.numel()} elements; expected 1 or B={B}"
                    )
            raw_energy = raw_energy * s
        if raw_energy.requires_grad:
            grad_phys = torch.autograd.grad(raw_energy.sum(), x_phys_grad)[0]
        else:
            grad_phys = torch.zeros_like(x_phys_grad)
        std = state_normalizer.std.to(grad_phys.device)
        grad_flat = (grad_phys * std).detach().reshape(x_in.shape)
        return (grad_flat * x_in).sum()


class DiTForwardMemo(torch.nn.Module):
    """Single-slot forward memo for the decoder's DiT during guided sampling.

    In ``dpm.model_wrapper``'s classifier branch every solver evaluation runs
    the DiT twice on the same input: first inside the guidance composer's
    x̂0-refinement (``cond_grad_fn`` → composer → ``model(x_4d, t, ...)``),
    then again in ``noise_pred_fn`` for the same ``(x, t)``. Both call sites
    reshape x to ``[B, P, -1, 4]`` and pass the solver's time tensor straight
    through, so the computations are value-identical — the second call can
    return the first call's output. That removes one of the two DiT forwards
    per guided solver evaluation: the DiT work per guided step roughly halves,
    which measures as ~25% off the whole active guided frame (the frame also
    pays the encoder, policy and energy autograd).

    Matching is exact: positional tensor args must be value-equal
    (``torch.equal`` against a detached CLONE — a plain detached view would
    alias the caller's storage and could report stale equality after an
    in-place update), keyword tensors (the per-generation conditioning) must
    be the same objects, and any non-tensor arg must compare ``==``. Any
    mismatch falls through to a real forward, so a miss costs only the
    comparison.

    The wrapped forward runs under ``no_grad``: every classifier_fn in this
    repo detaches the DiT output (straight-through x̂0 correction), and the
    sampling call sites are already no_grad — the graph the composer's call
    would otherwise build inside ``cond_grad_fn``'s enable_grad block is pure
    waste. Consequently this wrapper is for SAMPLING only; never install it
    around a training forward.
    """

    def __init__(self, dit: torch.nn.Module):
        super().__init__()
        # Plain-list indirection keeps the wrapped DiT out of _modules, so
        # while the memo is installed the decoder's state_dict/apply never
        # see a "dit.<wrapped>" level.
        self._wrapped = [dit]
        self._args = None
        self._kwargs = None
        self._out = None
        self.hits = 0
        self.misses = 0

    def _match(self, args, kwargs) -> bool:
        if self._out is None or len(args) != len(self._args):
            return False
        if set(kwargs.keys()) != set(self._kwargs.keys()):
            return False
        for a, c in zip(args, self._args):
            if torch.is_tensor(a) != torch.is_tensor(c):
                return False
            if torch.is_tensor(a):
                if a.shape != c.shape or not torch.equal(a, c):
                    return False
            elif a != c:
                return False
        for k, v in kwargs.items():
            c = self._kwargs[k]
            if torch.is_tensor(v):
                if v is not c:
                    return False
            elif v != c:
                return False
        return True

    def forward(self, *args, **kwargs):
        if self._match(args, kwargs):
            self.hits += 1
            return self._out
        with torch.no_grad():
            out = self._wrapped[0](*args, **kwargs)
        self._args = tuple(a.detach().clone() if torch.is_tensor(a) else a for a in args)
        self._kwargs = dict(kwargs)
        self._out = out
        self.misses += 1
        return out


class dit_memo:
    """Context manager: swap ``decoder.dit`` for a :class:`DiTForwardMemo`.

    Usage::

        with dit_memo(model.decoder) as memo:
            _, out = model(batch_data)   # guided x_start inference
        # memo.hits / memo.misses available after the block

    Scoped install/restore mirrors the ``_guidance_fn`` swap pattern the
    rollout helpers already use, so nothing outside the guided generation
    ever sees the wrapper.
    """

    def __init__(self, decoder: torch.nn.Module):
        self._decoder = decoder
        self._orig = None
        self.memo = None

    def __enter__(self) -> DiTForwardMemo:
        self._orig = self._decoder.dit
        self.memo = DiTForwardMemo(self._orig)
        self._decoder.dit = self.memo
        return self.memo

    def __exit__(self, exc_type, exc, tb):
        self._decoder.dit = self._orig
        return False
