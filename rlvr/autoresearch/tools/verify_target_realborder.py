"""Verify a curated scene's TARGET trajectory does not cross the ROAD BORDER,
using reward.py's canonical `compute_road_border_penalty` with CORRECT road-border
line_strings rebuilt from the map (the parsed NPZ line_strings conflate lane lines
with borders, so they're unreliable for RB).

Pipeline rule (GRAFT/HEAL RB fix): before training toward a curated target
(baseline-det / PRiSM rank-1 / RB-guided), confirm the target itself stays off the
ROAD BORDER; DISCARD scenes whose target crosses (training toward a border-crossing
target can't fix RB). Writes the target-clean subset.

Per scene: recover world ego pose (goal_pose + route), transform target
`ego_agent_future` (ego frame) to WORLD, rebuild road-border line_strings from the
map at the ego world pos (`LaneletSceneBuilder.build_line_strings_tensor`, channel 3
= road_border), then `reward.compute_road_border_penalty` (same frame, geometry is
frame-agnostic). Crossing = its t>=1 gate fires at `RewardConfig.rb_cross_thresh`.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np, torch
from scenario_generation.tools._heatmap_common import load_route, recover_ego_world_pose_from_goal
from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
from rlvr.reward import compute_road_border_penalty, RewardConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--ego_shape", required=True, help="WB,L,W")
    ap.add_argument("--out_clean", default=None)
    args = ap.parse_args()
    ego_shape = torch.tensor([float(x) for x in args.ego_shape.split(",")], dtype=torch.float32)
    cfg = RewardConfig()
    route = load_route(Path(args.route))
    b = LaneletSceneBuilder(str(route.map_path))

    scenes = json.load(open(args.scenes))
    clean, crossed, mind = [], 0, []
    for sp in scenes:
        d = np.load(sp, allow_pickle=True)
        ex, ey, eyaw = recover_ego_world_pose_from_goal(np.asarray(d["goal_pose"]), route)
        fut = np.asarray(d["ego_agent_future"]).astype(np.float32)  # ego-frame
        if fut.shape[-1] == 3:  # (x,y,heading) -> (x,y,cos,sin); explicit, not a silent default
            h = fut[:, 2]
            fut = np.stack([fut[:, 0], fut[:, 1], np.cos(h), np.sin(h)], -1).astype(np.float32)
        elif fut.shape[-1] != 4:
            raise ValueError(f"{sp}: ego_agent_future last dim must be 3 or 4, got {fut.shape}")
        c, s = math.cos(eyaw), math.sin(eyaw)
        x, y = fut[:, 0], fut[:, 1]
        wx = ex + c * x - s * y
        wy = ey + s * x + c * y
        wcos = c * fut[:, 2] - s * fut[:, 3]
        wsin = s * fut[:, 2] + c * fut[:, 3]
        world_traj = torch.tensor(np.stack([wx, wy, wcos, wsin], -1), dtype=torch.float32).unsqueeze(0)  # (1,T,4)
        ls = b.build_line_strings_tensor(np.array([ex, ey], dtype=float))  # (60,20,4) world, ch3=road_border
        data = {"line_strings": torch.tensor(np.asarray(ls), dtype=torch.float32).unsqueeze(0)}  # (1,60,20,4)
        out = compute_road_border_penalty(world_traj, ego_shape, data, cfg)
        crossing_gate = out[0]            # (N,) 1.0 safe, 0.0 crossing
        per_ts_min = out[5]               # (N,T) min border dist (return order: gate,near,wide,first_steps,cont,per_ts_min)
        is_cross = bool(crossing_gate[0].item() < 0.5)
        mind.append(float(per_ts_min[0, 1:].min().item()))  # exclude t=0 like the gate
        if is_cross:
            crossed += 1
        else:
            clean.append(sp)
    mind = np.array(mind)
    print(f"ROAD-BORDER target check (reward.compute_road_border_penalty, real map borders):")
    print(f"  {len(scenes)} scenes | target min-border(t>=1): min={mind.min():.2f} p5={np.percentile(mind,5):.2f} "
          f"p25={np.percentile(mind,25):.2f} p50={np.percentile(mind,50):.2f}")
    print(f"  target CROSSES road border: {crossed}/{len(scenes)}  |  CLEAN (usable target): {len(clean)}/{len(scenes)}")
    if args.out_clean:
        json.dump(clean, open(args.out_clean, "w"), indent=2)
        print(f"  wrote {len(clean)} target-clean scenes -> {args.out_clean}")


if __name__ == "__main__":
    main()
