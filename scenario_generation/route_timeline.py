"""RouteTimeline: load a route's recorded frames (NPZ + pose sidecar) for the
closed-loop Perception Reproducer.

A *route* is a sequence of consecutive 10 Hz frames from one bag (grouped by the
shared filename prefix, e.g. ``<bag-time>_<frameidx>.npz``). Each frame carries:

* the **model-input arrays** baked by the converter (neighbors, lanes, route,
  polygons, line_strings, traffic-in-lanes, goal, ego_shape, turn_indicators) —
  ego-centric at the *recorded* ego pose, exactly the training NPZ format; and
* the **absolute map ego pose** from the per-frame JSON sidecar
  (``x, y, z, qx, qy, qz, qw`` — written by parse_rosbag.py / the cpp
  frame_writer).

The reproducer's cursor compares the *live* simulated ego world position to the
recorded ego world positions (cKDTree here), picks a recorded frame, and the
rollout re-centers that frame's baked tensors onto the live ego — so no lanelet
map is needed in the hot path.

Sidecar resolution: by default the ``.json`` sits next to the ``.npz`` (fresh
converter output). When a padded corpus dropped the sidecars, pass ``sidecar_dir``
pointing at the original conversion tree (nested ``<date>/<bag-time>/``) — frames
are matched by filename stem.

The cursor snaps to whole recorded frames (autoware-faithful; both the log and the
sim are 10 Hz). The render path additionally interpolates neighbor poses between
their real detections using the sidecar ``neighbor_ids`` (track UUIDs) to smooth
the perception's freeze-then-jump stutter — see ``reproducer_rollout``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from scenario_generation.perf_timer import Timers
from scenario_generation.transforms import yaw_from_quat

# Trailing integer in "<prefix>_<frameidx>" filenames (zero-padded frame index).
_FRAME_IDX_RE = re.compile(r"_(\d+)$")

# Keys the reproducer actually consumes. .npz is a zip whose members decompress
# lazily, so loading only these skips the unused GT-future arrays
# (neighbor_agents_future ~0.29 MB + ego_agent_future) — ~25-30% less IO/decompress
# per frame, since neighbors are replayed from the cursor, not the stored future.
_NEEDED_KEYS = (
    "ego_agent_past",
    "ego_current_state",
    "neighbor_agents_past",
    "lanes",
    "lanes_speed_limit",
    "lanes_has_speed_limit",
    "route_lanes",
    "route_lanes_speed_limit",
    "route_lanes_has_speed_limit",
    "polygons",
    "line_strings",
    "static_objects",
    "ego_shape",
    "turn_indicators",
    "goal_pose",
)


def _frame_index(npz_path: Path) -> int:
    """Parse the trailing zero-padded frame index from the filename stem."""
    m = _FRAME_IDX_RE.search(npz_path.stem)
    if m is None:
        raise ValueError(
            f"Cannot parse frame index from {npz_path.name!r} (expected '<prefix>_<frameidx>.npz')"
        )
    return int(m.group(1))


def route_prefix(npz_path: Path) -> str:
    """Route key = filename stem with the trailing '_<frameidx>' removed.

    e.g. '13-42-45_0000000000000031' -> '13-42-45'; 'AAAA_0' would also collapse
    to 'AAAA' (multi-bag routes that share a prefix group together).
    """
    return _FRAME_IDX_RE.sub("", npz_path.stem)


def group_routes(npz_paths: list[Path]) -> dict[str, list[Path]]:
    """Group NPZ files into routes by shared prefix, each sorted by frame index."""
    routes: dict[str, list[Path]] = {}
    for p in npz_paths:
        routes.setdefault(route_prefix(p), []).append(p)
    for key in routes:
        routes[key].sort(key=_frame_index)
    return routes


# stem -> path index per sidecar_dir, built once and reused across every frame and
# every route (avoids an O(N_files) rglob per frame on large nested corpora).
_SIDECAR_INDEX_CACHE: dict[Path, dict[str, Path]] = {}


def _sidecar_index(sidecar_dir: Path) -> dict[str, Path]:
    key = sidecar_dir.resolve()
    idx = _SIDECAR_INDEX_CACHE.get(key)
    if idx is None:
        idx = {p.stem: p for p in sidecar_dir.rglob("*.json")}
        _SIDECAR_INDEX_CACHE[key] = idx
    return idx


def _resolve_sidecar(npz_path: Path, sidecar_dir: Path | None) -> Path:
    """Locate the pose JSON for an NPZ: sibling first, then sidecar_dir by stem.

    The under-``sidecar_dir`` lookup uses a one-time stem->path index (built by a
    single rglob, cached per directory) so nested corpora don't pay a recursive glob
    per frame."""
    sib = npz_path.with_suffix(".json")
    if sib.is_file():
        return sib
    if sidecar_dir is not None:
        # Match by stem anywhere under sidecar_dir (handles <date>/<bag>/ nesting).
        cand = sidecar_dir / f"{npz_path.stem}.json"
        if cand.is_file():
            return cand
        hit = _sidecar_index(sidecar_dir).get(npz_path.stem)
        if hit is not None:
            return hit
    raise FileNotFoundError(
        f"No pose sidecar for {npz_path.name} "
        f"(looked next to it{' and under ' + str(sidecar_dir) if sidecar_dir else ''})"
    )


class RouteTimeline:
    """Ordered recorded frames of one route, with a KDTree over ego world xy.

    Poses are loaded eagerly (cheap: one small JSON per frame). The bulky NPZ
    model-input arrays are loaded lazily and cached, so a 60 s segment touches at
    most ~600 NPZs and memory stays bounded.
    """

    def __init__(
        self,
        npz_paths: list[Path],
        sidecar_dir: Path | None = None,
        timers: Timers | None = None,
    ) -> None:
        if not npz_paths:
            raise ValueError("RouteTimeline needs at least one frame")
        self.timers = timers or Timers()
        self.npz_paths = sorted(npz_paths, key=_frame_index)
        self.frame_indices = np.array([_frame_index(p) for p in self.npz_paths], dtype=np.int64)
        self._sidecar_paths = [_resolve_sidecar(p, sidecar_dir) for p in self.npz_paths]

        with self.timers("timeline_load_poses"):
            poses = np.empty((len(self.npz_paths), 3), dtype=np.float64)  # x, y, yaw
            for i, sc in enumerate(self._sidecar_paths):
                d = json.loads(sc.read_text())
                poses[i, 0] = d["x"]
                poses[i, 1] = d["y"]
                poses[i, 2] = yaw_from_quat(d["qx"], d["qy"], d["qz"], d["qw"])
            self.poses = poses
            self.kdtree = cKDTree(poses[:, :2])

        # Recorded ego speed per frame, derived from pose deltas. Gap-aware: the
        # real dt between two stored frames is (Δframe_index * 0.1 s), so skipped
        # frames don't inflate the speed. Used by the cursor's speed-gap guard.
        self.speeds = self._compute_speeds()

        self._npz_cache: dict[int, dict[str, np.ndarray]] = {}

    def _compute_speeds(self, base_dt: float = 0.1) -> np.ndarray:
        n = len(self.npz_paths)
        speeds = np.zeros(n, dtype=np.float64)
        if n < 2:
            return speeds
        dxy = np.linalg.norm(np.diff(self.poses[:, :2], axis=0), axis=1)
        didx = np.maximum(np.diff(self.frame_indices), 1).astype(np.float64)
        seg_speed = dxy / (didx * base_dt)
        speeds[:-1] = seg_speed
        speeds[-1] = seg_speed[-1]
        return speeds

    def __len__(self) -> int:
        return len(self.npz_paths)

    @classmethod
    def from_npz_dir(
        cls,
        npz_dir: str | Path,
        sidecar_dir: str | Path | None = None,
        timers: Timers | None = None,
    ) -> "RouteTimeline":
        """Build from a single bag/route directory of ``*.npz`` frames."""
        npz_dir = Path(npz_dir)
        paths = sorted(npz_dir.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No .npz files under {npz_dir}")
        return cls(paths, Path(sidecar_dir) if sidecar_dir else None, timers)

    def npz(self, idx: int) -> dict[str, np.ndarray]:
        """Lazy-load + cache the model-input arrays for recorded frame ``idx``."""
        cached = self._npz_cache.get(idx)
        if cached is not None:
            return cached
        with self.timers("timeline_load_npz"):
            with np.load(self.npz_paths[idx], allow_pickle=True) as z:
                # Only decompress the keys we need (skip GT futures) — lazy npz access.
                data = {k: z[k] for k in _NEEDED_KEYS if k in z.files}
        self._npz_cache[idx] = data
        return data

    def prefetch(self, indices) -> None:
        """Decompress + cache the given recorded frames if not already cached.

        Called from a background thread while the GPU runs the model forward, so the
        next rollout tick's input build hits the cache instead of paying the np.load
        decompress on the critical path. ``npz`` already caches; loading an
        already-cached or out-of-range frame is a benign no-op (double-decompress at
        worst, same value), so this is safe to call concurrently with the build
        threads."""
        n = len(self.npz_paths)
        for i in indices:
            if 0 <= i < n and i not in self._npz_cache:
                self.npz(i)

    def pose(self, idx: int) -> np.ndarray:
        """World ego pose [x, y, yaw] at recorded frame ``idx``."""
        return self.poses[idx]

    def neighbor_ids(self, idx: int) -> list[str]:
        """Per-neighbor track UUIDs (hex) for frame ``idx``, aligned to the
        neighbor_past slot order. Read from the sidecar's ``neighbor_ids`` field
        (present on corpora regenerated by the updated cpp converter); empty list
        if the sidecar predates that field."""
        if not hasattr(self, "_nid_cache"):
            self._nid_cache: dict[int, list[str]] = {}
        cached = self._nid_cache.get(idx)
        if cached is not None:
            return cached
        try:
            d = json.loads(self._sidecar_paths[idx].read_text())
            ids = list(d.get("neighbor_ids", []))
        except (OSError, json.JSONDecodeError):
            ids = []
        self._nid_cache[idx] = ids
        return ids

    def query_radius(self, xy: np.ndarray, radius: float) -> np.ndarray:
        """Recorded-frame indices whose ego world xy is within ``radius`` of ``xy``."""
        return np.asarray(self.kdtree.query_ball_point(xy, radius), dtype=np.int64)

    def nearest(self, xy: np.ndarray) -> int:
        """Index of the recorded frame whose ego world xy is nearest ``xy``."""
        _, i = self.kdtree.query(xy)
        return int(i)

    def iter_segments(self, seg_len: int, stride: int | None = None):
        """Yield (start, end) frame-index windows of length ``seg_len`` (10 Hz).

        ``stride`` defaults to ``seg_len`` (non-overlapping). The final partial
        window is included if it has at least 2 frames.
        """
        stride = stride or seg_len
        n = len(self)
        start = 0
        while start < n:
            end = min(start + seg_len, n)
            if end - start >= 2:
                yield (start, end)
            if end >= n:
                break
            start += stride
