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

    # 3. Apply constraint filters. Every entry still triggers an NPZ load
    #    because legacy constraints (neighbor_count, speed_range,
    #    travel_distance) read npz_data; the entry dict is passed through
    #    for metric-based constraints (reward_threshold) that read
    #    precomputed fields instead of deriving them from the NPZ.
    #    Skipping the NPZ load when only entry-only constraints are active
    #    is a follow-up (would need a capability flag on the constraint).
    if constraint_filters:
        passing = []
        for entry in filtered:
            with np.load(entry["npz_path"]) as npz_data:
                passes_all = True
                for constraint, params in constraint_filters:
                    if not constraint.filter(entry["npz_path"], npz_data,
                                             params, entry=entry):
                        passes_all = False
                        break
            if passes_all:
                passing.append(entry)
        filtered = passing

    if not filtered:
        return []

    # 4. Score each match by distance + heading similarity to the arrow,
    #    pick the single best match per bag prefix, expand exactly n_before + 1 + n_after.
    import math

    def _score(entry):
        """Lower = better match. Combines position distance and heading difference."""
        dx = entry["x"] - center_x
        dy = entry["y"] - center_y
        pos_dist = math.sqrt(dx * dx + dy * dy)
        # Heading difference (handles wraparound)
        h_diff = abs(entry["heading_deg"] - heading_deg)
        if h_diff > 180:
            h_diff = 360 - h_diff
        # Weight heading difference as 1 deg ≈ 1 meter
        return pos_dist + h_diff

    # Group by bag prefix, pick best per bag
    best_per_bag: dict[str, tuple[float, dict]] = {}  # prefix → (score, entry)
    pad_length = 19
    for entry in filtered:
        try:
            prefix, frame = _parse_frame_info(entry["npz_path"])
            m = re.search(r"_(\d+)\.npz$", entry["npz_path"])
            if m:
                pad_length = len(m.group(1))
        except ValueError:
            continue
        s = _score(entry)
        if prefix not in best_per_bag or s < best_per_bag[prefix][0]:
            best_per_bag[prefix] = (s, entry)

    # 5. Expand each best match into a batch
    batches = []
    for prefix, (score, entry) in best_per_bag.items():
        _, central_frame = _parse_frame_info(entry["npz_path"])
        m = re.search(r"_(\d+)\.npz$", entry["npz_path"])
        pl = len(m.group(1)) if m else pad_length

        scenes = _expand_contiguous(prefix, central_frame, n_before, n_after, pl)
        if not scenes:
            continue

        # Find central index within the scene list
        central_path = _frame_path(prefix, central_frame, pl)
        central_idx = next((i for i, s in enumerate(scenes) if s == central_path), 0)

        meta = {
            "n_matches_in_radius": sum(1 for e in filtered if e["npz_path"].startswith(prefix)),
            "best_score": round(score, 2),
            "x": entry["x"],
            "y": entry["y"],
            "heading_deg": entry["heading_deg"],
        }

        batches.append(Batch(
            bag_prefix=prefix,
            scenes=scenes,
            central_indices=[central_idx],
            metadata=meta,
        ))

    return batches
