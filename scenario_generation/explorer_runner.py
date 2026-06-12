"""Exploration-policy guidance for the closed-loop replay sim.

Wraps a trained ExplorationPolicy (frozen planner, learned per-scene guidance
etas) so `run_route_replay` can generate the EGO trajectory through the
guidance composer each step, while all other agents keep the plain forward.

Per step:
  1. x_ref = the ego's plain det prediction from the shared batched forward
     (already computed by the replay loop — no extra pass).
  2. policy(scene_encoding, x_ref) -> deterministic etas per head.
  3. One extra ego-only forward with decoder._guidance_fn = composer
     (same private-attr pattern as rlvr.closed_loop.batched_rollout).

The guidance envelope (lambda_lat / col_scale / ...) MUST match the sweep
envelope the policy was trained against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.model.guidance.composer import GuidanceComposer
from diffusion_planner.model.guidance.config import GuidanceConfig, GuidanceSetConfig

import rlvr.guidance_batched  # noqa: F401 -- registers batched guidance
from exploration_policy.model import ExplorationPolicy, ExplorationPolicyConfig
from exploration_policy.utils import run_frozen_encoder


def plan_static_clearance(
    plan_ego_frame: np.ndarray,
    static_boxes_ego_frame: list[tuple[float, float, float, float, float]],
    ego_shape_wlw: tuple[float, float, float],
    device,
) -> float:
    """Min OBB clearance of an ego-frame plan to static boxes (canonical fn).

    Args:
        plan_ego_frame: (T, 4) [x, y, cos, sin] ego-centric plan.
        static_boxes_ego_frame: list of (x, y, heading, length, width) in the
            SAME ego frame.
        ego_shape_wlw: (wheelbase, length, width).

    Returns min clearance in metres (99.0 when no static boxes). Geometry is
    entirely compute_static_collision_penalty (no hand-rolled OBB math).
    """
    from rlvr.reward import RewardConfig, compute_static_collision_penalty

    if not static_boxes_ego_frame:
        return 99.0
    T = plan_ego_frame.shape[0]
    ego_trajs = torch.from_numpy(np.ascontiguousarray(plan_ego_frame)).float()
    ego_trajs = ego_trajs.unsqueeze(0).to(device)
    S = len(static_boxes_ego_frame)
    nb = torch.zeros(S, T, 4, device=device)
    shapes = torch.zeros(S, 2, device=device)
    for i, (x, y, h, length, w) in enumerate(static_boxes_ego_frame):
        nb[i, :, 0] = x
        nb[i, :, 1] = y
        nb[i, :, 2] = float(np.cos(h))
        nb[i, :, 3] = float(np.sin(h))
        shapes[i, 0] = w
        shapes[i, 1] = length
    valid = torch.ones(S, T, dtype=torch.bool, device=device)
    ego_shape = torch.tensor(ego_shape_wlw, device=device, dtype=torch.float32)
    res = compute_static_collision_penalty(
        ego_trajs, ego_shape, nb, shapes, valid, RewardConfig(),
    )
    return float(res["per_timestep_min"][0, 1:].min().item())


@dataclass
class ExplorerEnvelope:
    """Guidance strengths; defaults = the campaign sweep envelope."""

    lambda_lat: float = 5.0
    lat_scale: float = 2.0
    col_scale: float = 9.0
    col_range: float = 8.0
    lambda_spd: float = 0.2
    stretch_scale: float = 1.0
    guidance_scale: float = 0.5


class ExplorerGuidanceRunner:
    """Per-step explorer inference + guided ego generation for the sim."""

    def __init__(
        self,
        policy_dir: str | Path,
        model_args,
        device,
        envelope: ExplorerEnvelope | None = None,
        eta_smooth: float = 0.5,
    ):
        pdir = Path(policy_dir)
        cfg_path = pdir / "exploration_policy_config.json"
        ckpt_path = pdir / "exploration_policy.pth"
        if not cfg_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(
                f"explorer dir {pdir} must contain exploration_policy.pth and "
                f"exploration_policy_config.json"
            )
        cfg = ExplorationPolicyConfig.from_json(cfg_path)
        self.heads = cfg.heads
        self.policy = ExplorationPolicy(cfg, ref_seq_len=model_args.future_len).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.policy.load_state_dict(state, strict=True)
        self.policy.eval()
        self.device = device
        self.envelope = envelope or ExplorerEnvelope()
        # Exponential smoothing on etas across steps: damps left/right
        # flip-flop when the policy sits near a decision boundary mid-pass.
        # 0.0 = no smoothing (use raw per-step eta), 1.0 = frozen first eta.
        self.eta_smooth = float(eta_smooth)
        self._eta_prev: dict[str, float] | None = None

    def reset(self) -> None:
        self._eta_prev = None

    def _composer(self, etas: dict[str, torch.Tensor]) -> GuidanceComposer:
        env = self.envelope
        unmapped = set(etas) - {"lateral", "collision", "stretch"}
        if unmapped:
            raise ValueError(
                f"policy heads {sorted(unmapped)} have no guidance mapping "
                "in explorer_runner — running without them would silently "
                "change the deployed behaviour")
        fns = []
        if "lateral" in etas:
            fns.append(GuidanceConfig(
                name="lateral", enabled=True, scale=env.lat_scale,
                params={"lambda_lat": env.lambda_lat, "eta_lat": etas["lateral"]},
            ))
        if "collision" in etas:
            fns.append(GuidanceConfig(
                name="collision_swerve_batched", enabled=True, scale=env.col_scale,
                params={"eta_col": etas["collision"], "range": env.col_range},
            ))
        if "stretch" in etas:
            fns.append(GuidanceConfig(
                name="speed_stretch_batched", enabled=True, scale=env.stretch_scale,
                params={"stretch": 1.0 + env.lambda_spd * etas["stretch"]},
            ))
        return GuidanceComposer(GuidanceSetConfig(
            functions=fns, global_scale=env.guidance_scale,
        ))

    @torch.no_grad()
    def guided_ego_prediction(
        self,
        model,
        ego_tensor_dict: dict,
        x_ref: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Run policy + one guided ego forward.

        Args:
            model: the (frozen) planner, eval() mode.
            ego_tensor_dict: model-ready (normalized) tensor dict for the ego,
                as built by tensor_converter.to_model_tensors — the same dict
                _predict_batch would feed to model(data).
            x_ref: (T, 4) the ego's plain det prediction this step (physical
                ego-centric), used as the policy's reference trajectory and
                the lateral-guidance reference.

        Returns:
            (guided_traj (T, 4) physical ego-centric, etas dict)
        """
        x_ref_t = torch.from_numpy(np.ascontiguousarray(x_ref)).float()
        x_ref_t = x_ref_t.unsqueeze(0).to(self.device)
        data = dict(ego_tensor_dict)
        data["reference_trajectory"] = x_ref_t

        enc = run_frozen_encoder(model, data)
        out = self.policy(enc, x_ref_t, deterministic=True)
        etas_raw = {h: float(2.0 * out.dists[h].mean.item() - 1.0) for h in self.heads}

        if self.eta_smooth > 0.0 and self._eta_prev is not None:
            etas_f = {
                h: self.eta_smooth * self._eta_prev[h] + (1 - self.eta_smooth) * v
                for h, v in etas_raw.items()
            }
        else:
            etas_f = etas_raw
        self._eta_prev = etas_f

        etas_t = {h: torch.tensor([v], device=self.device) for h, v in etas_f.items()}
        composer = self._composer(etas_t)

        inner = model.module if hasattr(model, "module") else model
        decoder = inner.decoder
        saved_fn = decoder._guidance_fn
        saved_scale = decoder._guidance_scale
        decoder._guidance_fn = composer
        decoder._guidance_scale = composer._set_config.global_scale
        try:
            _, outputs = model(data)
            guided = outputs["prediction"][0, 0].cpu().numpy()
        finally:
            decoder._guidance_fn = saved_fn
            decoder._guidance_scale = saved_scale

        return guided, etas_f
