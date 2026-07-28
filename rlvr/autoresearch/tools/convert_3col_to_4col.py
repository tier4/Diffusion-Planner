"""Convert NPZ scenes with 3-col (x,y,heading) futures to 4-col (x,y,cos,sin).

The current reward.py (compute_reward_batch) REQUIRES neighbor_agents_future in 4-col
(x, y, cos, sin) form — a requirement added after the original SFT/champion era. The
curated ranked-SFT batch stacks prob + normal via torch.cat, so all scenes must share a
column count. This tool brings canonical 3-col scenes up to 4-col so they can be mixed
with 4-col branch-editor scenes and pass the reward-based in-training eval.

cos = cos(heading), sin = sin(heading); padded rows (xy exactly zero) stay fully zero,
matching the trainer's validity mask. 4-col passes through.
"""

import argparse
import json
import os

import numpy as np

from planner_metrics.scene_format import future_to_4col as _to_4col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_list", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_list", required=True)
    args = ap.parse_args()

    with open(args.scene_list) as f:
        paths = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)

    out_paths = []
    conv = 0
    for p in paths:
        d = dict(np.load(p, allow_pickle=True))
        for k, zero_is_pad in (("ego_agent_future", False), ("neighbor_agents_future", True)):
            if k not in d:
                raise KeyError(f"{p} missing {k}")
            if d[k].shape[-1] == 3:
                conv += 1
            d[k] = _to_4col(d[k], zero_rows_are_padding=zero_is_pad)
        out_p = os.path.join(args.out_dir, os.path.basename(p))
        np.savez(out_p, **d)
        out_paths.append(out_p)

    with open(args.out_list, "w") as f:
        json.dump(out_paths, f, indent=2)
    print(f"Wrote {len(out_paths)} scenes ({conv} field-conversions) to {args.out_dir}")
    print(f"List: {args.out_list}")


if __name__ == "__main__":
    main()
