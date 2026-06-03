"""Convert NPZ scenes with 4-col (x,y,cos,sin) futures back to canonical 3-col (x,y,heading).

The branch-editor / disturb_and_replay pipeline emits ego_agent_future and
neighbor_agents_future as 4 columns (x, y, cos_heading, sin_heading). The canonical
SFT/curated data format (used by the SFT baseline, the L2 val sets, and the champion's
avoidance curated run) is 3 columns (x, y, heading_radians).

Curated ranked-SFT mixes prob + normal scenes in one batch via torch.cat, which requires
a uniform column count. Mixing 4-col prob with 3-col real-platform normal is impossible, so the
whole batch must share a format. The reference model trained entirely in 3-col; this tool brings
4-col scenes back to that canonical format so they can be mixed with native 3-col normal
scenes and fed through the well-tested 3-col curated path.

heading = atan2(sin, cos); padded steps (cos=sin=0 or cos=1,sin=0) map to ~0, matching
the canonical convention. No silent fallbacks: missing fields raise.
"""
import argparse
import json
import os

import numpy as np


def _to_3col(arr: np.ndarray) -> np.ndarray:
    """(.., T, 4) [x,y,cos,sin] -> (.., T, 3) [x,y,heading]. 3-col passes through."""
    if arr.shape[-1] == 3:
        return arr.astype(np.float32)
    if arr.shape[-1] != 4:
        raise ValueError(f"expected last dim 3 or 4, got {arr.shape}")
    x = arr[..., 0]
    y = arr[..., 1]
    cos = arr[..., 2]
    sin = arr[..., 3]
    heading = np.arctan2(sin, cos).astype(np.float32)
    return np.stack([x.astype(np.float32), y.astype(np.float32), heading], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_list", required=True, help="JSON list of source NPZ paths")
    ap.add_argument("--out_dir", required=True, help="dir to write converted NPZs")
    ap.add_argument("--out_list", required=True, help="path to write converted scene-list JSON")
    args = ap.parse_args()

    with open(args.scene_list) as f:
        paths = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)

    out_paths = []
    converted = 0
    for p in paths:
        d = dict(np.load(p, allow_pickle=True))
        for k in ("ego_agent_future", "neighbor_agents_future"):
            if k not in d:
                raise KeyError(f"{p} missing {k}")
            before = d[k].shape[-1]
            d[k] = _to_3col(d[k])
            if before == 4:
                converted += 1
        out_p = os.path.join(args.out_dir, os.path.basename(p))
        np.savez(out_p, **d)
        out_paths.append(out_p)

    with open(args.out_list, "w") as f:
        json.dump(out_paths, f, indent=2)
    print(f"Wrote {len(out_paths)} scenes ({converted} field-conversions) to {args.out_dir}")
    print(f"List: {args.out_list}")


if __name__ == "__main__":
    main()
