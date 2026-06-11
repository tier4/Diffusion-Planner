#!/usr/bin/env python3
"""Canonical valid_predictor ego/neighbor L2 with the explorer in the loop.

Wraps the frozen planner in a shim whose forward runs: plain det generation ->
exploration policy (deterministic etas) -> guidance composer -> guided
generation, and feeds that shim to the UNMODIFIED
diffusion_planner.valid_predictor.validate_model loop — so the reported
ego/neighbor avg_loss is the exact canonical metric (squared error of the
generated trajectory vs GT, neighbor-validity masked), directly comparable to
the memorized baselines (J6 12k: ego=1.920, neighbor=3.435).

Usage:
    python -m rlvr.autoresearch.tools.valid_predictor_guided \
        --model_path <base.pth> --policy_dir <dir> \
        --args_json_path <args.json> --valid_set_list <val.json> \
        [--batch_size 32] [--lambda_lat 5.0] [--lat_scale 2.0] [--col_scale 9.0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

import rlvr.guidance_batched  # noqa: F401
from diffusion_planner.utils.config import Config
from diffusion_planner.utils.dataset import DiffusionPlannerData
from exploration_policy.utils import run_frozen_encoder
from rlvr.autoresearch.tools.eval_det_avoidance import load_model
from rlvr.autoresearch.tools.eval_policy_avoidance import load_policy, make_composer


def _load_validate_model():
    """valid_predictor.py is a SCRIPT under <repo>/diffusion_planner/, not a
    package module — load it by file path (same code the canonical torchrun
    invocation executes)."""
    import importlib.util

    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "diffusion_planner" / "valid_predictor.py"
    spec = importlib.util.spec_from_file_location("dp_valid_predictor", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.validate_model


class GuidedPlannerShim(nn.Module):
    """model(inputs) -> guided (enc, outputs); inputs arrive pre-normalized."""

    def __init__(self, model, policy, heads, env_args):
        super().__init__()
        self.model = model
        self.policy = policy
        self.heads = heads
        self.env_args = env_args

    @torch.no_grad()
    def forward(self, inputs: dict):
        decoder = self.model.module.decoder if hasattr(self.model, "module") else self.model.decoder
        saved_fn, saved_scale = decoder._guidance_fn, decoder._guidance_scale

        # 1. plain det generation (guidance off) = the policy's reference
        decoder._guidance_fn = None
        _, det_out = self.model(inputs)
        det = det_out["prediction"][:, 0].detach()  # [B, T, 4] physical

        # 2. policy -> per-scene etas -> composer
        data = dict(inputs)
        data["reference_trajectory"] = det
        enc = run_frozen_encoder(self.model, data)
        pout = self.policy(enc, det, deterministic=True)
        etas = {h: (2.0 * pout.dists[h].mean - 1.0) for h in self.heads}
        composer = make_composer(etas, self.env_args)

        # 3. guided generation, restoring decoder state afterwards
        decoder._guidance_fn = composer
        decoder._guidance_scale = composer._set_config.global_scale
        try:
            result = self.model(data)
        finally:
            decoder._guidance_fn = saved_fn
            decoder._guidance_scale = saved_scale
        return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--policy_dir", required=True)
    parser.add_argument("--args_json_path", required=True)
    parser.add_argument("--valid_set_list", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lambda_lat", type=float, default=5.0)
    parser.add_argument("--lat_scale", type=float, default=2.0)
    parser.add_argument("--col_scale", type=float, default=9.0)
    parser.add_argument("--col_range", type=float, default=8.0)
    parser.add_argument("--lambda_spd", type=float, default=0.2)
    parser.add_argument("--stretch_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = load_model(args.model_path, device)
    policy, heads = load_policy(args.policy_dir, model_args, device)

    cfg = Config(args.args_json_path)
    cfg.device = device

    shim = GuidedPlannerShim(model, policy, heads, args).to(device)

    valid_set = DiffusionPlannerData(args.valid_set_list)
    loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"[valid_guided] {len(valid_set)} scenes, batch {args.batch_size}, "
          f"heads={heads}")

    validate_model = _load_validate_model()
    ego_loss, nbr_loss = validate_model(shim, loader, cfg)

    report = {
        "ego_avg_loss": float(ego_loss),
        "neighbor_avg_loss": float(nbr_loss),
        "n_scenes": len(valid_set),
        "guidance_args": {k: getattr(args, k) for k in (
            "lambda_lat", "lat_scale", "col_scale", "col_range",
            "lambda_spd", "stretch_scale", "guidance_scale")},
        "policy_dir": args.policy_dir,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "valid_guided_l2.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
