"""In-place pad NPZ neighbor_agents_past/future from 32-slot to 320-slot.

parse_rosbag.py emits NPZs with MAX_NUM_NEIGHBORS=32 (from
diffusion_planner.dimensions), but the 320-neighbor model checkpoint
(PhaseG ep4 + descendants) expects 320 slots. Real bag scenes have
≤ 32 tracked neighbors, so the extra slots are just zeros.

This script rewrites each NPZ with padded neighbor arrays (and a
matching `static_objects` count if needed — but static_objects_num=5
matches between configs so no change there).
"""

import argparse
import json
from pathlib import Path

import numpy as np


TARGET_NEIGHBORS = 320


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes_json", type=Path, required=True,
                   help="JSON list of NPZ paths to pad in place.")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def pad(arr: np.ndarray, axis_size: int, axis: int = 0) -> np.ndarray:
    if arr.shape[axis] >= axis_size:
        return arr if arr.shape[axis] == axis_size else arr.take(range(axis_size), axis=axis)
    pad_shape = list(arr.shape)
    pad_shape[axis] = axis_size - arr.shape[axis]
    pad_block = np.zeros(pad_shape, dtype=arr.dtype)
    return np.concatenate([arr, pad_block], axis=axis)


def main() -> None:
    args = parse_args()
    with args.scenes_json.open() as f:
        scenes = json.load(f)

    n_pad = 0
    n_skip = 0
    for npz_path in scenes:
        npz_path = Path(npz_path)
        if not npz_path.exists():
            n_skip += 1
            continue
        with np.load(npz_path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}

        n_past = data["neighbor_agents_past"].shape[0]
        n_future = data["neighbor_agents_future"].shape[0]
        if n_past == TARGET_NEIGHBORS and n_future == TARGET_NEIGHBORS:
            n_skip += 1
            continue

        data["neighbor_agents_past"] = pad(data["neighbor_agents_past"], TARGET_NEIGHBORS, axis=0)
        data["neighbor_agents_future"] = pad(data["neighbor_agents_future"], TARGET_NEIGHBORS, axis=0)

        if not args.dry_run:
            np.savez(str(npz_path), **data)
        n_pad += 1

    print(
        f"Padded {n_pad} NPZs to neighbor_agents_(past|future).shape[0]="
        f"{TARGET_NEIGHBORS}. Skipped {n_skip} (already-padded or missing)."
    )


if __name__ == "__main__":
    main()
