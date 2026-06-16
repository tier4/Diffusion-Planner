"""Build curated CL-GUIDED centered targets for a bias-fix graft.

For each scene: generate K trajectories with strong route-centerline guidance (the
generation_variant's cl_spd slots), score every slot with reward.compute_centerline_score_batch
(baselink), pick the MOST-CENTERED slot, and write it into ego_agent_future as the curated
SFT target. Unlike plain det (build_baseline_det_target), which stays near an off-center start,
the CL-guided trajectory steers toward the route center — a real centering signal for curated SFT.

Reuses ONLY existing fns: eval_det_avoidance.{load_model,load_npz_data},
grpo_trainer_batched.{_stack_scene_data,_normalize_batch,generate_all_scenes_batched},
reward.compute_centerline_score_batch.
"""

import argparse
import json
import os

import numpy as np
import torch

from rlvr.autoresearch.tools.eval_det_avoidance import load_model, load_npz_data
from rlvr.grpo_trainer_batched import (
    _normalize_batch,
    _stack_scene_data,
    generate_all_scenes_batched,
)
from rlvr.reward import compute_centerline_score_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="centered source (baseline)")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--ego_shape", required=True, help="WB,L,W — validated against each NPZ")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_list", required=True)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--variant", default="rl_cl_soft_sweep_stretch")
    ap.add_argument("--gt_max_speed", type=float, default=9.0)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, margs = load_model(args.model, dev)
    cli_es = np.array([float(x) for x in args.ego_shape.split(",")])
    paths = json.load(open(args.scenes))
    os.makedirs(args.out_dir, exist_ok=True)
    written, n_cen = [], 0
    for p in paths:
        d = load_npz_data(p, dev)
        es = d["ego_shape"]
        es_one = es[0] if es.dim() > 1 else es
        npz_es = es_one.detach().cpu().numpy().reshape(-1)[:3]
        if not np.allclose(npz_es, cli_es, atol=1e-2):
            raise ValueError(
                f"{p}: --ego_shape {cli_es.tolist()} != NPZ ego_shape "
                f"{npz_es.tolist()} (platform mismatch)"
            )
        nb = _normalize_batch(_stack_scene_data([d], dev), margs)
        trajs = generate_all_scenes_batched(
            model,
            margs,
            nb,
            K=args.K,
            noise_range=(0.5, 2.0),
            device=dev,
            gen_chunk_size=args.K,
            gt_max_speed=args.gt_max_speed,
            generation_variant=args.variant,
            use_route_cl_guidance=True,
        )[0]  # (K,T,4)
        sc = compute_centerline_score_batch(trajs, es_one, d, usage_mode="baselink")
        best = int(torch.argmax(sc).item())
        if float(sc[best]) > -0.05:
            n_cen += 1
        raw = dict(np.load(p, allow_pickle=True))
        raw["ego_agent_future"] = trajs[best].detach().cpu().numpy().astype(np.float32)
        out_p = os.path.join(args.out_dir, os.path.basename(p))
        np.savez(out_p, **raw)
        written.append(out_p)
    json.dump(written, open(args.out_list, "w"), indent=1)
    print(
        f"wrote {len(written)} CL-guided curated targets -> {args.out_dir} (centerline>-0.05: {n_cen}/{len(written)})"
    )


if __name__ == "__main__":
    main()
