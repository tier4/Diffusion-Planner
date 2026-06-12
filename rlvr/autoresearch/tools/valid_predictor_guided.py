#!/usr/bin/env python3
"""Canonical valid_predictor ego/neighbor L2: det vs explorer-guided, one pass.

Replicates the EXACT loss computation of diffusion_planner/valid_predictor.py
(squared error of the generated trajectory vs GT, cos/sin headings, neighbor
validity masking) for BOTH the plain det trajectory and the explorer-guided
one, computed in the same batch — the det numbers double as the baseline
column AND as a validation that this loop reproduces the canonical script
(the det column must match a plain valid_predictor run on the same list).

Usage:
    python -m rlvr.autoresearch.tools.valid_predictor_guided \
        --model_path <base.pth> --policy_dir <dir> \
        --args_json_path <args.json> --valid_set_list <val.json> \
        --output_dir <dir> [--limit 500] [--batch_size 64] \
        [--lambda_lat 5.0] [--lat_scale 2.0] [--col_scale 9.0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusion_planner.dimensions import MAX_NUM_AGENTS, OUTPUT_T, POSE_DIM
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.config import Config
from diffusion_planner.utils.dataset import DiffusionPlannerData
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import rlvr.guidance_batched  # noqa: F401
from exploration_policy.utils import run_frozen_encoder
from rlvr.autoresearch.tools.eval_det_avoidance import load_model
from rlvr.autoresearch.tools.eval_policy_avoidance import load_policy, make_composer


class _LossAccum:
    """Canonical avg_loss accumulation (valid_predictor lines 120-127)."""

    def __init__(self):
        self.ego = self.ego_n = self.nbr = self.nbr_n = 0.0

    def add(self, prediction, all_gt, neighbors_future_valid, B):
        loss_tensor = (prediction - all_gt) ** 2
        self.ego += loss_tensor[:, 0, :].mean().item() * B
        self.ego_n += B
        loss_nei = loss_tensor[:, 1:, :][neighbors_future_valid]
        if loss_nei.shape[0] > 0:
            self.nbr += loss_nei.mean().item() * loss_nei.shape[0]
            self.nbr_n += loss_nei.shape[0]

    def result(self):
        return {
            "ego_avg_loss": self.ego / max(self.ego_n, 1),
            "neighbor_avg_loss": self.nbr / max(self.nbr_n, 1),
        }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--policy_dir", required=True)
    parser.add_argument("--args_json_path", required=True)
    parser.add_argument("--valid_set_list", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 = all scenes")
    parser.add_argument("--lambda_lat", type=float, default=5.0)
    parser.add_argument("--lat_scale", type=float, default=2.0)
    parser.add_argument("--col_scale", type=float, default=9.0)
    parser.add_argument("--col_range", type=float, default=8.0)
    parser.add_argument("--lambda_spd", type=float, default=0.2)
    parser.add_argument("--stretch_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument("--envelope", choices=["v1", "v2"], default="v1")
    parser.add_argument("--lambda_col", type=float, default=3.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = load_model(args.model_path, device)
    policy, heads = load_policy(args.policy_dir, model_args, device)
    cfg = Config(args.args_json_path)

    valid_set = DiffusionPlannerData(args.valid_set_list)
    if args.limit:
        valid_set = Subset(valid_set, range(min(args.limit, len(valid_set))))
    loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"[valid_guided] {len(valid_set)} scenes, batch {args.batch_size}, heads={heads}")

    decoder = model.module.decoder if hasattr(model, "module") else model.decoder
    det_acc, gui_acc = _LossAccum(), _LossAccum()

    for inputs in tqdm(loader, desc="valid_guided"):
        # ---- canonical preprocessing (valid_predictor lines 53-80) ----
        inputs = {k: v.to(device) for k, v in inputs.items()}
        B = inputs["ego_current_state"].shape[0]
        inputs["sampled_trajectories"] = torch.zeros(
            B, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM,
            dtype=torch.float32, device=device,
        )
        inputs["delay"] = torch.full((B,), 0, dtype=torch.float32, device=device)
        inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
        inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])

        ego_future = heading_to_cos_sin(inputs["ego_agent_future"])
        neighbors_future = inputs["neighbor_agents_future"]
        neighbor_future_mask = (
            torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
        )
        neighbors_future = heading_to_cos_sin(neighbors_future)
        neighbors_future[neighbor_future_mask] = 0.0
        B, Pn, T, _ = neighbors_future.shape
        ego_current = inputs["ego_current_state"][:, :4]
        neighbors_current = inputs["neighbor_agents_past"][:, :Pn, -1, :4]
        inputs = cfg.observation_normalizer(inputs)

        neighbor_current_mask = (
            torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        )
        neighbor_mask = torch.cat(
            (neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1
        )
        gt_future = torch.cat([ego_future[:, None], neighbors_future], dim=1)
        current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)
        all_gt = torch.cat([current_states[:, :, None, :], gt_future], dim=2)
        all_gt[:, 1:][neighbor_mask] = 0.0
        all_gt = all_gt[:, :, 1:, :]
        neighbors_future_valid = ~neighbor_future_mask

        # ---- pass 1: plain det (baseline column + policy reference) ----
        saved_fn, saved_scale = decoder._guidance_fn, decoder._guidance_scale
        decoder._guidance_fn = None
        _, det_out = model(inputs)
        det_pred = det_out["prediction"]
        det_acc.add(det_pred, all_gt, neighbors_future_valid, B)

        # ---- pass 2: policy -> composer -> guided ----
        det_ego = det_pred[:, 0].detach()
        data = dict(inputs)
        data["reference_trajectory"] = det_ego
        enc = run_frozen_encoder(model, data)
        pout = policy(enc, det_ego, deterministic=True)
        etas = {h: (2.0 * pout.dists[h].mean - 1.0) for h in heads}
        composer = make_composer(etas, args)
        decoder._guidance_fn = composer
        decoder._guidance_scale = composer._set_config.global_scale
        try:
            _, gui_out = model(data)
        finally:
            decoder._guidance_fn = saved_fn
            decoder._guidance_scale = saved_scale
        gui_acc.add(gui_out["prediction"], all_gt, neighbors_future_valid, B)

    det_r, gui_r = det_acc.result(), gui_acc.result()
    report = {
        "n_scenes": len(valid_set),
        "det": det_r,
        "guided": gui_r,
        "delta_pct": {
            k: (gui_r[k] - det_r[k]) / det_r[k] * 100 for k in det_r
        },
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
