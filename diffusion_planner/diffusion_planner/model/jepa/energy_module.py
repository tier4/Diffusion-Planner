"""JEPAEnergy: frozen ego-only latent-consistency energy for the training-time aux loss.

Encapsulates the planner-facing contract: given the predicted ego world trajectory, build
the K-step state prefix, apply the JEPA pose normalisation, derive the action (Δ normalised
pose — the velocity rep the predictor was trained on), and return E(τ) ∈ [B]. Encoder +
predictor are frozen; gradient flows only through the trajectory (into the planner). Energy
state is ego-only (the design that won the AUROC comparison; context degraded it).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusion_planner.model.jepa.encoder import TrajStateEncoder
from diffusion_planner.model.jepa.energy import compute_traj_energy
from diffusion_planner.model.jepa.predictor import ACLatentPredictor

__all__ = ["JEPAEnergy"]


class JEPAEnergy(nn.Module):
    def __init__(self, encoder: nn.Module, predictor: nn.Module,
                 pose_mean: torch.Tensor, pose_std: torch.Tensor, prefix_K: int = 10):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.register_buffer("pose_mean", torch.as_tensor(pose_mean, dtype=torch.float32))
        self.register_buffer("pose_std", torch.as_tensor(pose_std, dtype=torch.float32))
        self.prefix_K = int(prefix_K)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def energy(self, ego_current: torch.Tensor, ego_pred_world: torch.Tensor,
               K: int | None = None) -> torch.Tensor:
        """ego_current: [B, 4] physical ego pose; ego_pred_world: [B, T, 4] predicted ego
        world poses. Returns E: [B] (lower = more dynamically self-consistent)."""
        K = self.prefix_K if K is None else int(K)
        s = torch.cat([ego_current[:, None, :], ego_pred_world[:, :K, :]], dim=1)  # [B, K+1, 4]
        s_norm = (s - self.pose_mean) / self.pose_std
        a = s_norm[:, 1:, :] - s_norm[:, :-1, :]  # [B, K, 4] action = Δ normalised pose
        return compute_traj_energy(self.encoder, self.predictor, s_norm, a, ctx=None, K=K)

    @classmethod
    def from_ckpts(cls, encoder_ckpt: str, predictor_ckpt: str, device, prefix_K: int = 10) -> "JEPAEnergy":
        ec = torch.load(encoder_ckpt, map_location=device, weights_only=False)
        pc = torch.load(predictor_ckpt, map_location=device, weights_only=False)
        enc = TrajStateEncoder(ec["state_dim"], ec["embed_dim"], ec["enc_hidden"], ec["enc_layers"])
        enc.load_state_dict(ec["encoder"])
        pred = ACLatentPredictor(z_dim=pc["z_dim"], a_dim=pc["a_dim"], hidden=pc["hidden"],
                                 layers=pc["layers"], nhead=pc["nhead"])
        pred.load_state_dict(pc["predictor"])
        return cls(enc, pred, ec["pose_mean"], ec["pose_std"], prefix_K).to(device)
