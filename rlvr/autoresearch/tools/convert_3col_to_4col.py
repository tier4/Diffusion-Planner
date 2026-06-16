"""Convert NPZ scenes with 3-col (x,y,heading) futures to 4-col (x,y,cos,sin).

The current reward.py (compute_reward_batch) REQUIRES neighbor_agents_future in 4-col
(x, y, cos, sin) form — a requirement added after the original SFT/champion era. The
curated ranked-SFT batch stacks prob + normal via torch.cat, so all scenes must share a
column count. This tool brings canonical 3-col scenes up to 4-col so they can be mixed
with 4-col branch-editor scenes and pass the reward-based in-training eval.

cos = cos(heading), sin = sin(heading), masked to (0,0) on padded steps (xy≈0) to match
the validity convention used by grpo_sft_trainer's 3-col curated path. 4-col passes through.
"""

import argparse
import json
import os

import numpy as np


def _to_4col(arr: np.ndarray) -> np.ndarray:
    """(.., T, 3) [x,y,heading] -> (.., T, 4) [x,y,cos,sin]. 4-col passes through."""
    if arr.shape[-1] == 4:
        return arr.astype(np.float32)
    if arr.shape[-1] != 3:
        raise ValueError(f"expected last dim 3 or 4, got {arr.shape}")
    x = arr[..., 0].astype(np.float32)
    y = arr[..., 1].astype(np.float32)
    h = arr[..., 2]
    valid = (np.abs(x) + np.abs(y)) > 0.1
    cos = np.where(valid, np.cos(h), 0.0).astype(np.float32)
    sin = np.where(valid, np.sin(h), 0.0).astype(np.float32)
    return np.stack([x, y, cos, sin], axis=-1)


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
        for k in ("ego_agent_future", "neighbor_agents_future"):
            if k not in d:
                raise KeyError(f"{p} missing {k}")
            if d[k].shape[-1] == 3:
                conv += 1
            d[k] = _to_4col(d[k])
        out_p = os.path.join(args.out_dir, os.path.basename(p))
        np.savez(out_p, **d)
        out_paths.append(out_p)

    with open(args.out_list, "w") as f:
        json.dump(out_paths, f, indent=2)
    print(f"Wrote {len(out_paths)} scenes ({conv} field-conversions) to {args.out_dir}")
    print(f"List: {args.out_list}")


if __name__ == "__main__":
    main()
