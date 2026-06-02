"""Neighbor-pattern database for random neighbor augmentation during GRPO training.

Idea: before training, scan every scene in the dataset once and collect the *closest*
1..K neighbors (their full past + future tracks, in the ego-centric frame). At training
time we randomly copy a handful of these patterns into the empty neighbor slots of each
scene. This synthesises additional, plausible-but-novel traffic for the collision-based
reward to react to, without needing new recordings.

Only the closest neighbors are stored because (a) they are the ones the ego actually has
to negotiate with, and (b) far-away copied agents would never enter the collision margin
and so would add no learning signal.

Layout of the on-disk DB (a single ``.npz``):
    past:   [M, INPUT_T + 1, 11] float32  - neighbor past/current states (raw npz format)
    future: [M, OUTPUT_T,    3] float32  - neighbor future (x, y, heading), raw npz format

The columns match ``neighbor_agents_past`` / ``neighbor_agents_future`` exactly, so injected
patterns flow through the normal preprocessing (heading_to_cos_sin, normalization, masking)
with no special handling.
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
from tqdm import tqdm

from diffusion_planner.utils.train_utils import openjson

# Column indices within a neighbor past row (see neighbor preprocessing / loss.py).
_PAST_X = 0
_PAST_Y = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a neighbor-pattern DB for random neighbor augmentation during GRPO."
    )
    parser.add_argument("--train_set_list", type=str, required=True,
                        help="JSON list (or {'files': [...]}) of scene .npz paths")
    parser.add_argument("--output_path", type=str, required=True,
                        help="destination .npz for the pattern database")
    parser.add_argument("--max_per_scene", type=int, default=5,
                        help="keep at most this many of the closest valid neighbors per scene")
    parser.add_argument("--max_patterns", type=int, default=200_000,
                        help="global cap on stored patterns (random subsample if exceeded)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                        help="parallel worker processes for scanning (default: all cores)")
    parser.add_argument("--chunk_size", type=int, default=256,
                        help="scenes handed to a worker per task (IPC granularity)")
    return parser.parse_args()


def _valid_slot_mask(neighbor_past: np.ndarray) -> np.ndarray:
    """[Pn] bool mask: a slot is valid if its past track is not all zeros (padding)."""
    return np.any(neighbor_past != 0.0, axis=(1, 2))


def _current_distance(neighbor_past: np.ndarray) -> np.ndarray:
    """[Pn] distance of each neighbor's current position from the ego origin."""
    current_xy = neighbor_past[:, -1, [_PAST_X, _PAST_Y]]  # [Pn, 2]
    return np.linalg.norm(current_xy, axis=-1)


def _scan_scene(path: str, max_per_scene: int):
    """Return the closest ``max_per_scene`` valid neighbor patterns of one scene.

    Returns ``(past, future)`` with shapes ``[k, 31, 11]`` / ``[k, 80, 3]`` (``k`` up to
    ``max_per_scene``), or ``None`` if the scene has no valid neighbor.
    """
    scene = np.load(path, allow_pickle=True)
    neighbor_past = np.asarray(scene["neighbor_agents_past"], dtype=np.float32)  # [Pn, 31, 11]
    neighbor_future = np.asarray(scene["neighbor_agents_future"], dtype=np.float32)  # [Pn, 80, 3]

    valid = _valid_slot_mask(neighbor_past)
    valid_indices = np.nonzero(valid)[0]
    if valid_indices.size == 0:
        return None

    distances = _current_distance(neighbor_past)[valid_indices]
    order = np.argsort(distances)
    chosen = valid_indices[order[:max_per_scene]]
    return neighbor_past[chosen].copy(), neighbor_future[chosen].copy()


def _scan_chunk(paths: list[str], max_per_scene: int):
    """Worker entry point: scan a chunk of scenes and return their concatenated patterns.

    Concatenating inside the worker keeps inter-process traffic to two arrays per task
    instead of one tiny array per neighbor.
    """
    pasts, futures = [], []
    for path in paths:
        result = _scan_scene(path, max_per_scene)
        if result is None:
            continue
        pasts.append(result[0])
        futures.append(result[1])
    if not pasts:
        return None
    return np.concatenate(pasts, axis=0), np.concatenate(futures, axis=0)


def build_neighbor_db(
    data_list_path: str,
    output_path: str,
    max_per_scene: int,
    max_patterns: int,
    seed: int,
    num_workers: int = None,
    chunk_size: int = 256,
) -> None:
    """Scan every scene (in parallel) and persist the closest-neighbor patterns.

    Args:
        data_list_path: JSON list (or ``{"files": [...]}``) of scene ``.npz`` paths.
        output_path: destination ``.npz`` for the pattern database.
        max_per_scene: keep at most this many of the closest valid neighbors per scene.
        max_patterns: global cap on stored patterns (uniform random subsample if exceeded).
        seed: RNG seed for the subsample.
        num_workers: parallel worker processes (defaults to all CPU cores).
        chunk_size: scenes handed to a worker per task (IPC granularity).
    """
    data = openjson(data_list_path)
    files = data["files"] if isinstance(data, dict) else data

    num_workers = num_workers or os.cpu_count()
    chunks = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]

    past_patterns: list[np.ndarray] = []
    future_patterns: list[np.ndarray] = []
    num_patterns = 0

    worker = partial(_scan_chunk, max_per_scene=max_per_scene)
    print(f"[build_neighbor_db] scanning {len(files)} scenes with {num_workers} workers")
    with Pool(processes=num_workers) as pool:
        progress = tqdm(
            pool.imap_unordered(worker, chunks),
            total=len(chunks),
            desc="build_neighbor_db",
            unit="chunk",
        )
        for result in progress:
            if result is not None:
                past_patterns.append(result[0])
                future_patterns.append(result[1])
                num_patterns += result[0].shape[0]
            progress.set_postfix(patterns=num_patterns)

    if num_patterns == 0:
        raise RuntimeError("No valid neighbor patterns found while building the DB.")

    past_arr = np.concatenate(past_patterns, axis=0)  # [M, 31, 11]
    future_arr = np.concatenate(future_patterns, axis=0)  # [M, 80, 3]

    if past_arr.shape[0] > max_patterns:
        rng = np.random.default_rng(seed)
        keep = rng.choice(past_arr.shape[0], size=max_patterns, replace=False)
        past_arr = past_arr[keep]
        future_arr = future_arr[keep]

    np.savez(output_path, past=past_arr, future=future_arr)
    print(f"[build_neighbor_db] saved {past_arr.shape[0]} patterns to {output_path}")


class NeighborPatternDB:
    """Loaded neighbor-pattern DB with random injection into empty neighbor slots."""

    def __init__(self, db_path: str):
        data = np.load(db_path, allow_pickle=True)
        self.past = torch.from_numpy(np.asarray(data["past"], dtype=np.float32))  # [M, 31, 11]
        self.future = torch.from_numpy(np.asarray(data["future"], dtype=np.float32))  # [M, 80, 3]
        self.num_patterns = self.past.shape[0]

    @torch.no_grad()
    def inject(
        self,
        inputs: dict[str, torch.Tensor],
        inject_max: int,
        inject_prob: float,
    ) -> dict[str, torch.Tensor]:
        """Randomly copy patterns into empty neighbor slots of a *raw* batch (in place).

        Must be called BEFORE ``heading_to_cos_sin`` / observation normalization, while the
        batch still holds raw npz-format neighbor tensors.

        Args:
            inputs: batch dict containing ``neighbor_agents_past`` [B, Pn, 31, 11] and
                ``neighbor_agents_future`` [B, Pn, 80, 3].
            inject_max: maximum number of patterns to inject per scene (actual count is
                uniform in ``[1, inject_max]``).
            inject_prob: per-scene probability of injecting any neighbors at all.

        Returns:
            The same ``inputs`` dict (mutated in place) for convenience.
        """
        neighbor_past = inputs["neighbor_agents_past"]
        neighbor_future = inputs["neighbor_agents_future"]
        device = neighbor_past.device
        B, Pn = neighbor_past.shape[0], neighbor_past.shape[1]

        # Empty slot == all-zero past track.
        empty = (neighbor_past != 0.0).any(dim=(2, 3)).logical_not()  # [B, Pn]

        for b in range(B):
            if torch.rand(1, device=device).item() > inject_prob:
                continue
            empty_slots = torch.nonzero(empty[b], as_tuple=False).squeeze(-1)
            if empty_slots.numel() == 0:
                continue
            k = int(torch.randint(1, inject_max + 1, (1,), device=device).item())
            k = min(k, empty_slots.numel())

            slot_perm = empty_slots[torch.randperm(empty_slots.numel(), device=device)[:k]]
            pattern_idx = torch.randint(0, self.num_patterns, (k,), device=device)

            neighbor_past[b, slot_perm] = self.past[pattern_idx.cpu()].to(device)
            neighbor_future[b, slot_perm] = self.future[pattern_idx.cpu()].to(device)

        return inputs


if __name__ == "__main__":
    args = parse_args()
    build_neighbor_db(
        data_list_path=args.train_set_list,
        output_path=args.output_path,
        max_per_scene=args.max_per_scene,
        max_patterns=args.max_patterns,
        seed=args.seed,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
    )
