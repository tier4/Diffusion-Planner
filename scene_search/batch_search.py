"""Search logic for finding contiguous scene batches by position + heading.

Uses the search_scenes.py backend for spatial/heading filtering, then expands
matches into contiguous batches of n_before + 1 + n_after frames.
"""

import os
import re
from dataclasses import dataclass, field

import sys
from pathlib import Path

import numpy as np

# search_scenes.py lives in diffusion_planner/util_scripts/ which is not a pip package
_UTIL_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "diffusion_planner" / "util_scripts")
if _UTIL_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _UTIL_SCRIPTS_DIR)

from search_scenes import (
    build_index,
    filter_heading,
    filter_radius,
    group_sequences,
    load_index_parquet,
    read_sidecar,
    save_index_parquet,
)


@dataclass
class Batch:
    """A contiguous sequence of NPZ scenes centered around a search match."""
    bag_prefix: str                    # bag/sequence identifier (path prefix before frame number)
    scenes: list[str]                  # ordered NPZ paths (full batch)
    central_indices: list[int]         # indices within scenes that were actual search matches
    metadata: dict = field(default_factory=dict)

    @property
    def n_scenes(self) -> int:
        return len(self.scenes)

    @property
    def central_scenes(self) -> list[str]:
        return [self.scenes[i] for i in self.central_indices]

    def summary(self) -> str:
        bag_short = self.bag_prefix.split("/")[-1][:20]
        return f"{bag_short}... ({self.n_scenes} scenes, {len(self.central_indices)} matches)"


def _parse_frame_info(npz_path: str) -> tuple[str, int]:
    """Extract bag prefix and frame number from an NPZ path.

    Expected format: .../prefix_0000000000000431.npz
    Returns (prefix, frame_number).
    """
    match = re.search(r"^(.+)_(\d+)\.npz$", npz_path)
    if not match:
        raise ValueError(f"Cannot parse frame number from: {npz_path}")
    return match.group(1), int(match.group(2))


def _frame_path(prefix: str, frame: int, pad: int = 19) -> str:
    """Reconstruct NPZ path from prefix and frame number."""
    return f"{prefix}_{frame:0{pad}d}.npz"


def _expand_contiguous(prefix: str, central_frame: int, n_before: int, n_after: int, pad: int = 19) -> list[str]:
    """Expand outward from central_frame, stopping when frames don't exist.

    This ensures we only include scenes from a continuous recording segment —
    gaps in frame numbers (recording stopped/restarted) truncate the batch.
    """
    scenes_before = []
    for offset in range(1, n_before + 1):
        candidate = _frame_path(prefix, central_frame - offset, pad)
        if os.path.exists(candidate):
            scenes_before.append(candidate)
        else:
            break  # Recording boundary — stop expanding backward
    scenes_before.reverse()

    scenes_after = []
    for offset in range(1, n_after + 1):
        candidate = _frame_path(prefix, central_frame + offset, pad)
        if os.path.exists(candidate):
            scenes_after.append(candidate)
        else:
            break  # Recording boundary — stop expanding forward

    central = _frame_path(prefix, central_frame, pad)
    return scenes_before + [central] + scenes_after


def find_batches(
    index: list[dict],
    center_x: float,
    center_y: float,
    heading_deg: float,
    radius: float = 50.0,
    heading_tolerance: float = 30.0,
    n_before: int = 30,
    n_after: int = 80,
    constraint_filters: list = None,
) -> list[Batch]:
    """Find contiguous scene batches matching the arrow query.

    Args:
        index: Spatial index from build_index() — list of dicts with npz_path, x, y, heading_deg.
        center_x, center_y: MGRS world coordinates of arrow start.
        heading_deg: Arrow direction in degrees [-180, 180).
        radius: Search radius in meters.
        heading_tolerance: Heading match tolerance in degrees (±).
        n_before: Max frames to include before central match.
        n_after: Max frames to include after central match.
        constraint_filters: List of (constraint, params) tuples for additional filtering.

    Returns:
        List of Batch objects, each representing a contiguous scene sequence.
    """
    # 1. Spatial filter
    filtered = filter_radius(index, center_x, center_y, radius)
    if not filtered:
        return []

    # 2. Heading filter (handles wraparound)
    hmin = heading_deg - heading_tolerance
    hmax = heading_deg + heading_tolerance
    # Normalize to [-180, 180) range
    hmin = (hmin + 180) % 360 - 180
    hmax = (hmax + 180) % 360 - 180
    filtered = filter_heading(filtered, hmin, hmax)
    if not filtered:
        return []

    # 3. Apply constraint filters (load NPZ as needed)
    if constraint_filters:
        passing = []
        for entry in filtered:
            npz_data = np.load(entry["npz_path"])
            passes_all = True
            for constraint, params in constraint_filters:
                if not constraint.filter(entry["npz_path"], npz_data, params):
                    passes_all = False
                    break
            if passes_all:
                passing.append(entry)
        filtered = passing

    if not filtered:
        return []

    # 4. Group matching scenes by bag sequence and expand to batches
    # Parse frame info for all matches
    match_frames: dict[str, list[int]] = {}  # prefix → [frame_numbers]
    pad_length = 19  # Default padding length
    for entry in filtered:
        try:
            prefix, frame = _parse_frame_info(entry["npz_path"])
            # Detect padding length from the actual filename
            match = re.search(r"_(\d+)\.npz$", entry["npz_path"])
            if match:
                pad_length = len(match.group(1))
            match_frames.setdefault(prefix, []).append(frame)
        except ValueError:
            continue

    # 5. For each prefix, merge overlapping expansions
    batches = []
    for prefix, frames in match_frames.items():
        frames.sort()

        # Expand each central frame and merge overlapping ranges
        expanded_ranges: list[tuple[int, int, list[int]]] = []  # (start, end, central_frames)
        for f in frames:
            expanded = _expand_contiguous(prefix, f, n_before, n_after, pad_length)
            if not expanded:
                continue
            first_prefix, first_frame = _parse_frame_info(expanded[0])
            last_prefix, last_frame = _parse_frame_info(expanded[-1])

            if expanded_ranges and first_frame <= expanded_ranges[-1][1] + 1:
                # Overlaps with previous — merge
                prev_start, prev_end, prev_centrals = expanded_ranges[-1]
                merged_end = max(prev_end, last_frame)
                prev_centrals.append(f)
                expanded_ranges[-1] = (prev_start, merged_end, prev_centrals)
            else:
                expanded_ranges.append((first_frame, last_frame, [f]))

        # Build Batch objects from merged ranges
        for start_frame, end_frame, central_frames in expanded_ranges:
            scenes = []
            for frame_num in range(start_frame, end_frame + 1):
                path = _frame_path(prefix, frame_num, pad_length)
                if os.path.exists(path):
                    scenes.append(path)

            if not scenes:
                continue

            # Find central indices within the scene list
            scene_set = {s: i for i, s in enumerate(scenes)}
            central_indices = []
            for cf in central_frames:
                cp = _frame_path(prefix, cf, pad_length)
                if cp in scene_set:
                    central_indices.append(scene_set[cp])

            # Compute metadata from sidecar of first central scene
            meta = {"n_matches": len(central_frames)}
            first_central = _frame_path(prefix, central_frames[0], pad_length)
            sidecar = read_sidecar(first_central)
            if sidecar:
                meta["x"] = sidecar["x"]
                meta["y"] = sidecar["y"]
                meta["heading_deg"] = sidecar["heading_deg"]

            batches.append(Batch(
                bag_prefix=prefix,
                scenes=scenes,
                central_indices=central_indices,
                metadata=meta,
            ))

    return batches
