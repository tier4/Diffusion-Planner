"""Fine-tune the Diffusion Planner for temporal stability (reduce frame-to-frame flicker)
with a cross-frame temporal-consistency loss.

Idea (validated): the planner re-plans each frame independently, so its output flickers —
a good trajectory one frame, a bad one the next. We add a loss that makes the prediction at
frame t, propagated forward by g steps, AGREE with the prediction at frame t+g on the overlap
(``planner_metrics.temporal_consistency_loss``). Training (not inference filtering) is the right
lever: it makes the model *intrinsically* temporally stable rather than gaming the metric by
averaging consecutive plans.

Two details that make it work (else it fights accuracy / is meaningless noise):
  * the consistency term reads a near-CLEAN prediction via a fixed low diffusion noise level
    (``--fixed_t``), not the random-t x_start estimate;
  * the metres-scale consistency is normalised into planning-loss units (``--cons_scale``).

Measured (erga/879_hiratsuka held-out, 240 pairs, 600 steps, coeff 0.5): replan p90 (flicker
tail) -17% at flat ego-L2; the consistency term decreases over training. The coefficient trades
tail-reduction vs mean-accuracy; ~0.3-0.5 is the accuracy-safe sweet spot.

Run (OnePlanner venv; needs diffusion_planner + planner_metrics on PYTHONPATH):
    python -m diffusion_planner.finetune_temporal_consistency \
        --ckpt_dir <model_dir> --data_root <basic_dataset> --coeff 0.5 --steps 600 --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.model.module.decoder import compute_training_loss
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.config import Config
from planner_metrics.replan_consistency import (
    consecutive_frame_pairs,
    ego_future_to_4col,
    inter_frame_transform,
    temporal_consistency_loss,
)


def build_pairs(data_root: str, locations: list[str], step_g: int, exclude: str, limit: int):
    pairs = []
    for loc in locations:
        base = os.path.join(data_root, loc)
        for root, _, files in os.walk(base):
            if exclude and exclude in root:
                continue
            if not any(f.endswith(".npz") for f in files):
                continue
            ps = sorted(glob.glob(os.path.join(root, "*.npz")))
            for ia, pa, ib, pb, g in consecutive_frame_pairs(ps):
                if g == step_g:
                    pairs.append((pa, pb))
            if len(pairs) > limit:
                break
        if len(pairs) > limit:
            break
    random.Random(0).shuffle(pairs)
    return pairs[:limit]


class PairDS(Dataset):
    def __init__(self, pairs, step_g):
        self.pairs = pairs
        self.g = step_g

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pa, pb = self.pairs[i]
        da = dict(np.load(pa, allow_pickle=True)); da.pop("version", None)
        db = dict(np.load(pb, allow_pickle=True)); db.pop("version", None)
        rp, rh = inter_frame_transform(ego_future_to_4col(da["ego_agent_future"]), self.g)
        return da, db, rp, rh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="model_dir with args.json + best_model.pth")
    ap.add_argument("--data_root", required=True, help="basic_dataset root")
    ap.add_argument("--locations", nargs="*", default=["erga/879_hiratsuka"])
    ap.add_argument("--exclude", default="", help="substring of sessions to hold out")
    ap.add_argument("--out", required=True, help="output dir for the fine-tuned model")
    ap.add_argument("--coeff", type=float, default=0.5, help="consistency weight (0 = control)")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--step_g", type=int, default=3, help="frame cadence == trajectory-step offset")
    ap.add_argument("--fixed_t", type=float, default=0.5, help="low diffusion-t for the clean forward")
    ap.add_argument("--cons_scale", type=float, default=10.0)
    ap.add_argument("--w_heading", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init", default="ema_state_dict", choices=["ema_state_dict", "model"])
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    cfg = Config(os.path.join(args.ckpt_dir, "args.json"))
    cfg.coeff_jepa_consistency_loss = 0.0
    model = Diffusion_Planner(cfg).to(dev).train()
    sd = torch.load(os.path.join(args.ckpt_dir, "best_model.pth"), map_location=dev, weights_only=False)
    model.load_state_dict({k.replace("module.", "", 1): v for k, v in sd[args.init].items()}, strict=True)

    pairs = build_pairs(args.data_root, args.locations, args.step_g, args.exclude,
                        args.steps * args.batch_size + args.batch_size)
    print(f"[data] {len(pairs)} consecutive pairs (g={args.step_g})", flush=True)

    def collate(batch):
        das, dbs, rps, rhs = zip(*batch)
        stack = lambda ds: {k: torch.stack([torch.as_tensor(d[k]) for d in ds]) for k in ds[0]}
        return stack(das), stack(dbs), torch.stack(rps), torch.stack(rhs)

    dl = DataLoader(PairDS(pairs, args.step_g), batch_size=args.batch_size, shuffle=True,
                    num_workers=4, drop_last=True, collate_fn=collate,
                    generator=torch.Generator().manual_seed(0))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def prep(inp):
        inp = {k: v.to(dev) for k, v in inp.items()}
        inp["ego_agent_past"] = heading_to_cos_sin(inp["ego_agent_past"])
        inp["goal_pose"] = heading_to_cos_sin(inp["goal_pose"])
        ego_future = heading_to_cos_sin(inp["ego_agent_future"])
        nf = inp["neighbor_agents_future"]
        mask = torch.sum(torch.ne(nf[..., :3], 0), dim=-1) == 0
        nf = heading_to_cos_sin(nf); nf[mask] = 0.0
        return cfg.observation_normalizer(inp), (ego_future, nf, mask)

    def plan_loss(loss):
        return (getattr(cfg, "alpha_neighbor_loss", 0.1) * loss["neighbor_prediction_loss"]
                + getattr(cfg, "alpha_planning_loss", 1.0) * loss["ego_planning_loss"]
                + loss["turn_indicator_loss"]
                + getattr(cfg, "coeff_road_border_loss", 0.0) * loss["road_border_loss"]
                + getattr(cfg, "coeff_neighbor_collision_loss", 0.0) * loss["neighbor_collision_loss"])

    cfg._return_ego_pred_world = True
    step = 0; t0 = time.time(); crun = []
    while step < args.steps:
        for da, db, rp, rh in dl:
            if step >= args.steps:
                break
            rp, rh = rp.to(dev), rh.to(dev)
            ia, fa = prep(da); ib, fb = prep(db)
            cfg._fixed_diffusion_t = None
            la = compute_training_loss(model, ia, fa, cfg)
            lb = compute_training_loss(model, ib, fb, cfg)
            tot = plan_loss(la) + plan_loss(lb)
            cval = 0.0
            if args.coeff > 0:
                cfg._fixed_diffusion_t = args.fixed_t
                ca = compute_training_loss(model, ia, fa, cfg)
                cb = compute_training_loss(model, ib, fb, cfg)
                cfg._fixed_diffusion_t = None
                cons = temporal_consistency_loss(ca["ego_pred_world"], cb["ego_pred_world"],
                                                 args.step_g, rp, rh, w_heading=args.w_heading,
                                                 stop_grad_a=True) / args.cons_scale
                tot = tot + args.coeff * cons
                cval = float(cons)
            opt.zero_grad(set_to_none=True); tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            crun.append(cval); step += 1
            if step % 100 == 0:
                print(f"  step {step}/{args.steps} ego={la['ego_planning_loss'].item():.4f} "
                      f"cons={np.mean(crun[-100:]):.4f} ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    torch.save({"model": model.state_dict(), "coeff": args.coeff, "steps": args.steps,
                "fixed_t": args.fixed_t}, os.path.join(args.out, "best_model.pth"))
    import shutil
    shutil.copy(os.path.join(args.ckpt_dir, "args.json"), os.path.join(args.out, "args.json"))
    print(f"[saved] {args.out}/best_model.pth (+ args.json)", flush=True)


if __name__ == "__main__":
    main()
