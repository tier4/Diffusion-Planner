"""Real-neighbor database for collision-search augmentation during GRPO training.

Two phases:

  1. **Build** (offline): scan every scene once and collect real neighbor tracks (full past +
     future, in the ego-centric frame, raw npz format). Only neighbors with a usable future
     track are kept. The result is a single ``.npz`` "pattern database".

  2. **Inject** (train time): for each scene, *search* the database for real neighbor tracks
     that -- copied in verbatim (no transform) -- already collide with this scene's ego GT
     future at an avoidable point, and paste a few of them into neighbor slots.

The key idea (vs the synthetic constant-accel collider): the injected agents move with real
recorded kinematics (real speed / curvature / size / type). Because both the DB tracks and the
target scene are in the same ego-centric frame (origin = ego at t=0, heading +x), a real
neighbor that crossed *its* ego's path will, pasted verbatim, cross *this* ego's path whenever
the two GT paths happen to coincide in space-time -- which a large enough DB makes common.
"Situation is ignored": the pasted agent keeps its own motion regardless of the new scene's map.

On-disk DB layout (single ``.npz``):
    past:   [M, INPUT_T + 1, 11] float32  - neighbor past/current states (raw npz format)
    future: [M, OUTPUT_T,    3] float32  - neighbor future (x, y, heading), raw npz format

The columns match ``neighbor_agents_past`` / ``neighbor_agents_future`` exactly, so injected
patterns flow through the normal preprocessing (heading_to_cos_sin, normalization, masking).
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
from tqdm import tqdm

from diffusion_planner.dimensions import OUTPUT_T
from diffusion_planner.utils.train_utils import openjson

# Default real-neighbor pattern DB used as the collision-search augmentation source (training
# and visualization both default to DB-based / copy-paste colliders).
DEFAULT_NEIGHBOR_DB_PATH = "/mnt/storage_rdma/diffusion_planner/dataset/basic_dataset/"

# Column indices within a neighbor past row (see neighbor preprocessing / loss.py).
_PAST_X = 0
_PAST_Y = 1
# One-hot agent type occupies columns 8..10 = [vehicle, pedestrian, bicycle]
# (matches synthetic_neighbors._TYPE_BASE).
_TYPE_BASE = 8
_TYPE_PEDESTRIAN = 1
_TYPE_BICYCLE = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a real-neighbor DB for collision-search augmentation during GRPO."
    )
    parser.add_argument(
        "--train_set_list",
        type=str,
        required=True,
        help="JSON list (or {'files': [...]}) of scene .npz paths",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="destination .npz for the pattern database"
    )
    parser.add_argument(
        "--max_per_scene",
        type=int,
        default=10,
        help="keep at most this many of the closest valid neighbors per scene",
    )
    parser.add_argument(
        "--min_future_steps",
        type=int,
        default=40,
        help="require at least this many non-padding future waypoints to keep a "
        "neighbor (so the pasted agent actually moves through the scene)",
    )
    parser.add_argument(
        "--max_patterns",
        type=int,
        default=50_000,
        help="global cap on stored patterns (random subsample if exceeded)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="parallel worker processes for scanning (default: all cores)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="scenes handed to a worker per task (IPC granularity)",
    )
    return parser.parse_args()


def _valid_slot_mask(neighbor_past: np.ndarray) -> np.ndarray:
    """[Pn] bool mask: a slot is valid if its past track is not all zeros (padding)."""
    return np.any(neighbor_past != 0.0, axis=(1, 2))


def _full_future_mask(neighbor_future: np.ndarray, min_future_steps: int) -> np.ndarray:
    """[Pn] bool mask: a slot has enough non-padding future waypoints to be a usable mover."""
    valid_steps = (np.abs(neighbor_future[:, :, :2]).sum(axis=-1) > 1e-6).sum(axis=-1)
    return valid_steps >= min_future_steps


def _current_distance(neighbor_past: np.ndarray) -> np.ndarray:
    """[Pn] distance of each neighbor's current position from the ego origin."""
    current_xy = neighbor_past[:, -1, [_PAST_X, _PAST_Y]]  # [Pn, 2]
    return np.linalg.norm(current_xy, axis=-1)


def _scan_scene(path: str, max_per_scene: int, min_future_steps: int):
    """Return the closest valid+full-future neighbor patterns of one scene.

    Returns ``(past, future)`` with shapes ``[k, 31, 11]`` / ``[k, 80, 3]`` (``k`` up to
    ``max_per_scene``), or ``None`` if the scene has no usable neighbor.
    """
    scene = np.load(path, allow_pickle=True)
    neighbor_past = np.asarray(scene["neighbor_agents_past"], dtype=np.float32)  # [Pn, 31, 11]
    neighbor_future = np.asarray(scene["neighbor_agents_future"], dtype=np.float32)  # [Pn, 80, 3]

    usable = _valid_slot_mask(neighbor_past) & _full_future_mask(neighbor_future, min_future_steps)
    usable_indices = np.nonzero(usable)[0]
    if usable_indices.size == 0:
        return None

    distances = _current_distance(neighbor_past)[usable_indices]
    order = np.argsort(distances)
    chosen = usable_indices[order[:max_per_scene]]
    return neighbor_past[chosen].copy(), neighbor_future[chosen].copy()


def _scan_chunk(paths: list[str], max_per_scene: int, min_future_steps: int):
    """Worker entry point: scan a chunk of scenes and return their concatenated patterns."""
    pasts, futures = [], []
    for path in paths:
        result = _scan_scene(path, max_per_scene, min_future_steps)
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
    min_future_steps: int,
    max_patterns: int,
    seed: int,
    num_workers: int,
    chunk_size: int,
) -> None:
    """Scan every scene (in parallel) and persist real neighbor patterns."""
    data = openjson(data_list_path)
    files = data["files"] if isinstance(data, dict) else data

    chunks = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]
    past_patterns: list[np.ndarray] = []
    future_patterns: list[np.ndarray] = []
    num_patterns = 0

    worker = partial(_scan_chunk, max_per_scene=max_per_scene, min_future_steps=min_future_steps)
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
        raise RuntimeError("No usable neighbor patterns found while building the DB.")

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
    """Loaded real-neighbor DB that injects DB tracks which collide with the ego GT (verbatim)."""

    def __init__(
        self,
        db_path: str,
        collision_margin: float,
        keep_clear_radius: float,
        min_collision_time: float,
        search_subsample: int,
    ):
        data = np.load(db_path, allow_pickle=True)
        self.past = torch.from_numpy(np.asarray(data["past"], dtype=np.float32))  # [M, 31, 11]
        self.future = torch.from_numpy(np.asarray(data["future"], dtype=np.float32))  # [M, 80, 3]
        self.num_patterns = self.past.shape[0]

        # Precompute future xy + validity for fast collision search.
        self.future_xy = self.future[:, :, :2].contiguous()  # [M, 80, 2]
        self.future_valid = self.future_xy.abs().sum(dim=-1) > 1e-6  # [M, 80]

        # Precompute past xy + validity + agent type. Pedestrians/bicycles are only injected when
        # their *past* track overlaps the ego's future path (see inject); vehicles are unrestricted.
        self.past_xy = self.past[:, :, :2].contiguous()  # [M, 31, 2]
        self.past_valid = self.past_xy.abs().sum(dim=-1) > 1e-6  # [M, 31]
        type_idx = self.past[:, -1, _TYPE_BASE : _TYPE_BASE + 3].argmax(dim=-1)  # [M]
        self.is_vru = (type_idx == _TYPE_PEDESTRIAN) | (type_idx == _TYPE_BICYCLE)  # [M]

        # Closest approach of each track to the origin (= the ego's t=0 pose), over the current
        # pose and every valid future step. A track that ever comes within keep_clear_radius of
        # the origin would hit a *stationary* ego, so the forced collision would be unavoidable;
        # such patterns are excluded from the search below. Intrinsic to the pattern (ego frame).
        cur_dist = self.past[:, -1, :2].norm(dim=-1)  # [M] neighbor current pos -> origin
        fut_dist = torch.where(
            self.future_valid,
            self.future_xy.norm(dim=-1),
            torch.full_like(self.future_valid, float("inf"), dtype=torch.float32),
        )
        self.track_clear = torch.minimum(cur_dist, fut_dist.min(dim=1).values)  # [M]

        self.collision_margin = collision_margin
        self.keep_clear_radius = keep_clear_radius
        self.min_collision_time = min_collision_time
        # 0 (or >= M) searches the whole DB each scene; a positive value caps the per-scene
        # search to a random subsample (caps cost for very large DBs).
        self.search_subsample = search_subsample

        dt = 0.1
        self._tau_future = torch.arange(1, OUTPUT_T + 1, dtype=torch.float32) * dt  # [80] in [.1,8]
        self._device = None  # tensors moved to the batch device on first inject

    def _to(self, device):
        if self._device == device:
            return
        self.past = self.past.to(device)
        self.future = self.future.to(device)
        self.future_xy = self.future_xy.to(device)
        self.future_valid = self.future_valid.to(device)
        self.track_clear = self.track_clear.to(device)
        self.past_xy = self.past_xy.to(device)
        self.past_valid = self.past_valid.to(device)
        self.is_vru = self.is_vru.to(device)
        self._tau_future = self._tau_future.to(device)
        self._device = device

    @torch.no_grad()
    def inject(self, inputs: dict[str, torch.Tensor], inject_max: int, inject_prob: float) -> dict:
        """Paste DB tracks that collide with each scene's ego GT into neighbor slots (in place).

        Must be called on a *raw* batch (before heading_to_cos_sin / normalization). The mask of
        slots actually written is stored on ``self.last_injected_mask`` ([B, Pn]).
        """
        neighbor_past = inputs["neighbor_agents_past"]  # [B, Pn, 31, 11]
        neighbor_future = inputs["neighbor_agents_future"]  # [B, Pn, 80, 3]
        ego_future = inputs["ego_agent_future"]  # [B, 80, 3] (x, y, heading)
        device = neighbor_past.device
        self._to(device)
        B, Pn = neighbor_past.shape[:2]

        late = self._tau_future >= self.min_collision_time  # [80]
        empty = (neighbor_past != 0.0).any(dim=(2, 3)).logical_not()  # [B, Pn]
        injected = torch.zeros(B, Pn, dtype=torch.bool, device=device)

        for b in range(B):
            if torch.rand((), device=device).item() > inject_prob:
                continue

            ego_xy = ego_future[b, :, :2]  # [80, 2]
            # ego GT points that make an avoidable, meaningful collision target:
            # non-padding, late enough, and far enough from the ego t=0 pose (origin).
            ego_ok = (
                (ego_xy.abs().sum(dim=-1) > 1e-6)
                & late
                & (ego_xy.norm(dim=-1) >= self.keep_clear_radius)
            )  # [80]
            if not ego_ok.any():
                continue

            # search the DB (optionally a random subsample) for tracks that collide.
            if 0 < self.search_subsample < self.num_patterns:
                sidx = torch.randperm(self.num_patterns, device=device)[: self.search_subsample]
                fxy, fvalid, clear = (
                    self.future_xy[sidx],
                    self.future_valid[sidx],
                    self.track_clear[sidx],
                )
            else:
                sidx = None
                fxy, fvalid, clear = self.future_xy, self.future_valid, self.track_clear

            dist = torch.linalg.norm(fxy - ego_xy[None], dim=-1)  # [S, 80]
            hit = (dist < self.collision_margin) & fvalid & ego_ok[None]  # [S, 80]
            # require an avoidable track: it must never come within keep_clear of the ego t=0 pose.
            qualifies = hit.any(dim=1) & (clear >= self.keep_clear_radius)  # [S]
            cand = torch.nonzero(qualifies, as_tuple=False).squeeze(-1)
            if cand.numel() == 0:
                continue
            if sidx is not None:
                cand = sidx[cand]  # map back to global pattern indices

            # Pedestrians/bicycles: keep only patterns whose *past* track overlaps the ego's
            # future path (a VRU already on the ego's route), so we don't paste VRUs that merely
            # happen to cross the route in their future. Vehicles are unrestricted. Computed on
            # the (small) candidate set only, to avoid an [M, 31, 80] distance tensor.
            is_vru = self.is_vru[cand]  # [C]
            if is_vru.any():
                ego_valid = ego_xy.abs().sum(dim=-1) > 1e-6  # [80]
                cand_past_xy = self.past_xy[cand]  # [C, 31, 2]
                cand_past_valid = self.past_valid[cand]  # [C, 31]
                past_dist = torch.linalg.norm(
                    cand_past_xy[:, :, None, :] - ego_xy[None, None, :, :], dim=-1
                )  # [C, 31, 80]
                past_overlap = (
                    (past_dist < self.collision_margin)
                    & cand_past_valid[:, :, None]
                    & ego_valid[None, None, :]
                ).any(dim=(1, 2))  # [C]
                keep = (~is_vru) | past_overlap
                cand = cand[keep]
                if cand.numel() == 0:
                    continue

            # write into a random index in [0, first_empty] inclusive (overwrite a real neighbor
            # or take the first empty slot), so injected agents are interspersed, not back-packed.
            empty_idx = torch.nonzero(empty[b], as_tuple=False).squeeze(-1)
            first_empty = int(empty_idx.min().item()) if empty_idx.numel() > 0 else Pn - 1
            n_positions = first_empty + 1

            k = int(torch.randint(1, inject_max + 1, (), device=device).item())
            k = min(k, n_positions, cand.numel())
            slots = torch.randperm(n_positions, device=device)[:k]
            chosen = cand[torch.randperm(cand.numel(), device=device)[:k]]

            neighbor_past[b, slots] = self.past[chosen]
            neighbor_future[b, slots] = self.future[chosen]
            injected[b, slots] = True

        self.last_injected_mask = injected
        return inputs


if __name__ == "__main__":
    args = parse_args()
    build_neighbor_db(
        data_list_path=args.train_set_list,
        output_path=args.output_path,
        max_per_scene=args.max_per_scene,
        min_future_steps=args.min_future_steps,
        max_patterns=args.max_patterns,
        seed=args.seed,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
    )
