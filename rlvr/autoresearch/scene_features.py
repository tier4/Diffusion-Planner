"""Standalone per-scene feature & label library for training-format NPZ scenes.

This is the scene-identification core used by
``rlvr.autoresearch.tools.scene_feature_index`` (the sharded-parquet extractor),
factored out so it can be used **on its own** — score a single scene, filter a
list, or label scenes ad hoc — without the dataset-building machinery.

Quick start::

    from rlvr.autoresearch.scene_features import extract_scene_features
    row = extract_scene_features("scene.npz")
    print(row["label"])          # e.g. "low|turn-L|has-ped"
    print(row["nearest_ped_path_m"], row["gt_yaw_deg"], row["peak_lat_accel"])

Or compose the pieces::

    from rlvr.autoresearch.scene_features import (
        FeatureConfig, features_from_npz, derive_labels,
    )
    feats = features_from_npz(np.load(p, allow_pickle=True))
    labels = derive_labels(feats, FeatureConfig(straight_yaw_deg=20.0))

**Interaction flags are ego-path-relevant, not scene-presence.** An agent counts
as an interaction only when it comes within a corridor of the ego's GT future
trajectory (min distance point→polyline, reusing
``planner_metrics.geometry._point_to_segments_min_dist``). So ``has-ped`` means "a
pedestrian the ego will actually pass close to", not "a pedestrian exists
somewhere in the 320 slots".

All feature bodies REUSE the canonical implementations (``lat_accel_curvature``,
``read_sidecar``, ``future_to_4col``, ``_point_to_segments_min_dist``) — never
reimplemented. The features that live
inline inside ``scene_search`` constraint ``.filter()`` methods (ego speed, GT
travel distance, ego-origin neighbor count) use the exact documented formula with
a ``file:line`` citation. Heavy imports (torch) are deferred to first use so
``import scene_features`` stays cheap.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import re
from dataclasses import dataclass

import numpy as np

# Repo root = <root>/rlvr/autoresearch/scene_features.py -> up 2 from this dir.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_FRAME_RE = re.compile(r"_(\d+)\.npz$")

# Neighbor past last-dim layout (npz_loader.py:208): [x, y, cos, sin, vx, vy,
# width, length, is_veh, is_ped, is_bike]. Type one-hot is columns 8,9,10.
_NB_VEH_COL = 8
_NB_PED_COL = 9

# Bump when the per-scene NPZ feature/format contract changes in a way that makes
# previously-extracted shards or previously-written canonical NPZs incompatible.
# Included in the scene_feature_index resume fingerprint (so an index built with an
# older extractor semantics can't be silently resumed/appended into) and stamped as
# the canonical NPZ ``version`` field.
FEATURE_SCHEMA_VERSION = 3

# Ego/goal pose widths accepted in a canonical NPZ. The repo supports BOTH the raw
# 3-col form [x, y, heading] AND the converted 4-col form [x, y, cos, sin]
# (``convert_3col_to_4col.py``): the model-input loader / ``heading_to_cos_sin`` are
# idempotent for 4-col input, so both are valid on disk. An axis whose expected value
# is a tuple means "one of these" (see ``validate_canonical_scene``).
_EGO_POSE_WIDTHS = (3, 4)  # 4 == dimensions.POSE_DIM (cos/sin); 3 == raw yaw


def _canonical_expected_shapes() -> dict[str, tuple]:
    """Per-field expected canonical-NPZ shape (per-scene, NO leading batch axis),
    derived from ``diffusion_planner.dimensions`` — the single source of truth for the
    model's tensor contract. Each axis is one of: an int (asserted exactly), a tuple
    of ints (one-of, e.g. the 3-or-4-col ego pose width), or ``None`` (only where the
    contract genuinely does not fix it). These are the shapes the converter emits and
    the encoder later slices/indexes (e.g. it slices turn_indicators ``[:, :-1]`` into
    ``FloatsEncoder(num_float=INPUT_T)``, so the stored length is ``INPUT_T + 1``, and
    map points are (x, y) + a type one-hot), so a malformed tensor — scalar, wrong
    count, wrong feature width — is caught here instead of crashing mid-training."""
    from diffusion_planner import dimensions as D

    return {
        "ego_agent_past": (D.INPUT_T + 1, _EGO_POSE_WIDTHS),
        "ego_current_state": (D.EGO_CURRENT_STATE_SHAPE[-1],),
        "ego_agent_future": (D.OUTPUT_T, _EGO_POSE_WIDTHS),
        "neighbor_agents_past": (D.MAX_NUM_NEIGHBORS, D.INPUT_T + 1, D.NEIGHBOR_SHAPE[-1]),
        "neighbor_agents_future": (D.MAX_NUM_NEIGHBORS, D.OUTPUT_T, D.POSE_DIM),
        "static_objects": (D.NUM_STATIC_OBJECTS, D.STATIC_OBJECTS_SHAPE[-1]),
        "lanes": (D.NUM_SEGMENTS_IN_LANE, D.POINTS_PER_LANELET, D.SEGMENT_POINT_DIM),
        "lanes_speed_limit": (D.NUM_SEGMENTS_IN_LANE, 1),
        "lanes_has_speed_limit": (D.NUM_SEGMENTS_IN_LANE, 1),
        "route_lanes": (D.NUM_SEGMENTS_IN_ROUTE, D.POINTS_PER_LANELET, D.SEGMENT_POINT_DIM),
        "route_lanes_speed_limit": (D.NUM_SEGMENTS_IN_ROUTE, 1),
        "route_lanes_has_speed_limit": (D.NUM_SEGMENTS_IN_ROUTE, 1),
        # map points = (x, y) + a per-type one-hot: polygon width 2 + POLYGON_TYPE_NUM,
        # line-string width 2 + LINE_STRING_TYPE_NUM.
        "polygons": (D.NUM_POLYGONS, D.POINTS_PER_POLYGON, 2 + D.POLYGON_TYPE_NUM),
        "line_strings": (D.NUM_LINE_STRINGS, D.POINTS_PER_LINE_STRING, 2 + D.LINE_STRING_TYPE_NUM),
        "goal_pose": (_EGO_POSE_WIDTHS,),
        "turn_indicators": (D.INPUT_T + 1,),
        "ego_shape": (D.EGO_SHAPE_SHAPE[-1],),
    }


def canonical_required_fields() -> tuple[str, ...]:
    """The model-input field names every canonical NPZ must carry (the keys of the
    canonical shape contract) — the single source both the builder and the training
    verifier use, so the required-field list can't drift between them."""
    return tuple(_canonical_expected_shapes().keys())


def validate_canonical_scene(data, ctx: str = "") -> None:
    """Fail loud unless a loaded canonical scene carries EVERY required model-input
    field with a shape compatible with the encoder. ``data`` is an npz mapping
    (``np.load`` result or a dict) indexable by field name. Raises ``ValueError`` on
    the first missing field or shape mismatch. This is the shared schema validator
    the builder runs on each output and the R2LPL verifier runs on sampled scenes, so
    a scene the real pipeline cannot consume never passes as loadable."""
    keys = data.files if hasattr(data, "files") else data
    expected = _canonical_expected_shapes()
    missing = [k for k in expected if k not in keys]
    if missing:
        raise ValueError(f"{ctx}: missing required model-input field(s) {missing}")
    for name, exp in expected.items():
        shp = getattr(data[name], "shape", None)
        if shp is None or len(shp) != len(exp):
            raise ValueError(f"{ctx}: {name} has shape {shp}, expected {len(exp)} dim(s) {exp}")
        for axis, (got, want) in enumerate(zip(shp, exp)):
            # want: None = any; tuple = one-of; int = exact.
            if want is None:
                continue
            allowed = want if isinstance(want, tuple) else (want,)
            if got not in allowed:
                raise ValueError(
                    f"{ctx}: {name} axis {axis} = {got}, expected {want} (got {shp}, want {exp})"
                )


@dataclass(frozen=True)
class FeatureConfig:
    """Thresholds for deriving the categorical scene labels.

    Declared defaults (NOT silent fallbacks) — override explicitly to retune the
    label taxonomy. ``is_stopped`` uses the <0.5 m/s test from
    eval_driving_metrics.py:163 (independent of ``speed_bins``)."""

    speed_bins: tuple[float, float, float] = (0.5, 3.0, 8.0)  # stopped|low|mid|high
    straight_yaw_deg: float = 15.0  # |signed total GT yaw| < this => straight
    dense_count: int = 4  # agents interacting with the ego path >= this => dense
    neighbor_radius_m: float = 30.0  # ego-origin proximity (informational count)
    lead_range_m: float = 40.0  # lead vehicle: ahead within this range (ego +x)
    lead_half_width_m: float = 2.0  # ... and within this distance of the ego PATH
    interaction_corridor_m: float = 8.0  # agent within this of ego GT path => interacting
    ped_corridor_m: float = 6.0  # pedestrian within this of ego GT path => has-ped


# Stable column order for the parquet feature table (shards concat cleanly).
FEATURE_COLUMNS = [
    "npz_path",
    "bag",
    "frame",
    "ego_speed_ms",
    "is_stopped",
    "travel_dist_m",
    "gt_yaw_deg",  # signed total GT yaw
    "gt_path_m",
    "peak_lat_accel",  # max |speed*omega| over ego future
    "peak_abs_yaw_deg",  # unsigned total yaw magnitude (width-aware unsigned_total_yaw_deg)
    "neighbor_count",  # active neighbors within radius of ego ORIGIN (informational)
    "nearest_neighbor_m",  # nearest active neighbor to ego ORIGIN
    "nearest_agent_path_m",  # nearest active neighbor to the ego GT PATH
    "n_interacting",  # active neighbors within interaction_corridor of the ego PATH
    "has_lead",  # vehicle on the ego PATH corridor, ahead within range
    "lead_dist_m",
    "nearest_ped_path_m",  # nearest pedestrian to the ego GT PATH (None if no peds)
    "has_pedestrian",  # a pedestrian within ped_corridor of the ego PATH
    "static_count",
    "goal_dist_m",
    "goal_heading_delta_deg",
    "world_x",  # nullable (sidecar)
    "world_y",
    "world_heading_deg",
    "speed_bin",  # stopped|low|mid|high
    "maneuver",  # straight|turn-L|turn-R
    "interaction",  # has-ped|has-lead|dense|none (all ego-path-relevant)
    "label",  # "<speed_bin>|<maneuver>|<interaction>" (bucket triple)
]


def _load_util_script(module_name: str, rel_path: str):
    """Load a self-contained ``diffusion_planner/util_scripts/*.py`` helper by file
    path. Those files are NOT importable as a package: the editable install maps
    ``diffusion_planner`` to the nested ``diffusion_planner/diffusion_planner/``
    package, so ``diffusion_planner.util_scripts`` (a sibling) has no module path."""
    path = os.path.join(_REPO_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {rel_path} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_sidecar(npz_path: str):
    """World-pose sidecar for ``npz_path`` (``{x,y,heading_deg,timestamp}``) or None.
    Reuses ``search_scenes.read_sidecar``."""
    fn = _load_util_script(
        "search_scenes", "diffusion_planner/util_scripts/search_scenes.py"
    ).read_sidecar
    return fn(npz_path)


def min_dist_to_polyline(
    points_xy: np.ndarray, polyline_xy: np.ndarray, *, device: str = "cpu"
) -> np.ndarray:
    """Min distance from each of ``points_xy`` (Q,2) to a polyline ``polyline_xy``
    (P,2), as a (Q,) numpy array. Reuses the canonical
    ``planner_metrics.geometry._point_to_segments_min_dist`` (torch; deferred
    import) on ``device``. A degenerate polyline (<2 vertices, e.g. a stopped ego
    whose GT future barely moves) is treated as point-to-point distance to its
    first vertex."""
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    poly = np.asarray(polyline_xy, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if poly.shape[0] < 2:
        poly = (
            np.repeat(poly[:1], 2, axis=0) if poly.shape[0] == 1 else np.zeros((2, 2), np.float32)
        )

    import torch

    from planner_metrics.geometry import _point_to_segments_min_dist

    p = torch.from_numpy(pts).to(device)
    poly_t = torch.from_numpy(poly).to(device)
    d = _point_to_segments_min_dist(p, poly_t[:-1], poly_t[1:])
    return d.cpu().numpy()


def batch_min_dist_to_paths(points, seg_p1, seg_p2, point_mask=None):
    """Batched point→polyline min distance for GPU throughput over many scenes.

    Vectorized analog of ``planner_metrics.geometry._point_to_segments_min_dist`` over
    a scene batch — same projection-onto-segment math, one kernel for the whole
    batch instead of a Python loop of tiny per-scene calls (which is what makes the
    CPU path slow). Validated bit-for-bit against the canonical per-scene helper in
    the unit tests.

    Args:
        points: (B, Q, 2) padded neighbor positions per scene.
        seg_p1, seg_p2: (B, S, 2) segment endpoints per scene (ego path). The ego
            future is a fixed length so S is constant and no segment mask is needed.
        point_mask: optional (B, Q) bool; padded points get +inf so they never win
            a per-scene min.

    Returns:
        (B, Q) min distance from each point to its scene's polyline (on the input
        tensors' device).
    """
    import torch

    seg = seg_p2 - seg_p1  # (B,S,2)
    seg_len2 = (seg**2).sum(-1).clamp(min=1e-10)  # (B,S)
    diff = points[:, :, None, :] - seg_p1[:, None, :, :]  # (B,Q,S,2)
    t = ((diff * seg[:, None, :, :]).sum(-1) / seg_len2[:, None, :]).clamp(0, 1)  # (B,Q,S)
    closest = seg_p1[:, None, :, :] + t[..., None] * seg[:, None, :, :]  # (B,Q,S,2)
    d = (points[:, :, None, :] - closest).norm(dim=-1)  # (B,Q,S)
    out = d.min(dim=-1).values  # (B,Q)
    if point_mask is not None:
        out = out.masked_fill(~point_mask, float("inf"))
    return out


# --------------------------------------------------------------------------- #
# filename parsing
# --------------------------------------------------------------------------- #
def session_key(npz_path: str) -> str:
    """Injective, stable identifier for a source **session directory** (the dir the
    NPZ lives in). ``<readable-tail>_<hash8>`` where the hash is over the FULL
    directory path, so distinct source dirs never collide — unlike joining the last
    two path components with ``_`` (``/A/a_b/c`` and ``/B/a/b_c`` both -> ``a_b_c``),
    or the basename alone (same date/session under two roots). Used identically for
    contiguous-window grouping (``bag_and_frame``) and output lineage
    (``build_small_dataset.contig_relpath``) so a selected window maps to exactly one
    output dir."""
    d = os.path.dirname(npz_path)
    tail = re.sub(r"[^A-Za-z0-9_.-]+", "-", os.path.basename(d)) or "bag"
    h = hashlib.md5(d.encode()).hexdigest()[:8]
    return f"{tail}_{h}"


def bag_and_frame(npz_path: str) -> tuple[str, int]:
    """(bag id, frame index) from the path. Frame index is the trailing ``_(\\d+)``
    of the filename (search_scenes.py:317-ff); -1 if absent.

    The bag id is ``<session_key>/<filename prefix>``. Using the filename prefix
    alone conflates different bags that share a basename, which lets contiguous-window
    selection cross source bags and then breaks reproducer lineage;
    :func:`session_key` is an injective per-directory key (see its docstring)."""
    base = os.path.basename(npz_path)
    m = _FRAME_RE.search(base)
    frame = int(m.group(1)) if m else -1
    prefix = _FRAME_RE.sub("", base) if m else base[:-4]
    return f"{session_key(npz_path)}/{prefix}", frame


# --------------------------------------------------------------------------- #
# individual feature computations (pure numpy)
# --------------------------------------------------------------------------- #
def ego_speed_ms(ego_current_state: np.ndarray) -> float:
    """Ego speed (m/s) from ``ego_current_state[4:6] = (vx, vy)``.
    speed_range.py:37-42 (state = [x,y,cos,sin,vx,vy,ax,ay,steer,yaw_rate])."""
    s = np.asarray(ego_current_state, dtype=np.float32).reshape(-1)
    return math.hypot(float(s[4]), float(s[5]))


def is_stopped(speed_ms: float) -> bool:
    """<0.5 m/s (eval_driving_metrics.py:163)."""
    return bool(speed_ms < 0.5)


def gt_travel_distance_m(ego_future: np.ndarray) -> float:
    """Summed step-to-step GT displacement over the ego future. travel_distance.py:37-40."""
    fut = np.asarray(ego_future, dtype=np.float32)
    dx = np.diff(fut[:, 0])
    dy = np.diff(fut[:, 1])
    return float(np.sqrt(dx * dx + dy * dy).sum())


def signed_total_yaw_deg(ego_future: np.ndarray) -> float:
    """Signed cumulative GT heading change (deg), wrapped per-step. Same per-step
    wrap as curate_curve_scenes.compute_gt_stats (lines 40-42) but keeps the sign so
    turn direction can be classified.

    Width-aware: a 3-col ego future is ``[x, y, yaw]`` (col 2 is the yaw angle); a
    4-col ego future is ``[x, y, cos, sin]`` (recover yaw via ``atan2(sin, cos)``).
    Reading col 2 as a yaw on 4-col input integrates changes in *cosine*, which
    silently mislabels the maneuver and can flip the turn direction. Any other
    width fails loudly."""
    dh = _ego_yaw_steps(ego_future)
    return float(np.degrees(dh.sum()))


def _ego_yaw_steps(ego_future: np.ndarray) -> np.ndarray:
    """Per-step wrapped heading deltas (radians), width-aware. 3-col ego is
    ``[x,y,yaw]`` (col 2); 4-col is ``[x,y,cos,sin]`` (``atan2(sin,cos)``); any other
    width fails loudly. Shared by the signed and unsigned total-yaw computations so
    both decode width consistently."""
    fut = np.asarray(ego_future, dtype=np.float32)
    w = fut.shape[1]
    if w == 3:
        yaw = fut[:, 2]
    elif w == 4:
        yaw = np.arctan2(fut[:, 3], fut[:, 2])
    else:
        raise ValueError(
            f"ego_future must be 3-col [x,y,yaw] or 4-col [x,y,cos,sin]; got width {w}"
        )
    dh = np.diff(yaw)
    return np.arctan2(np.sin(dh), np.cos(dh))


def unsigned_total_yaw_deg(ego_future: np.ndarray) -> float:
    """Unsigned cumulative heading change (deg), width-aware. Matches
    curate_curve_scenes.compute_gt_stats' magnitude but decodes 4-col ego futures
    correctly (that legacy helper reads col 2 as radians, wrong for [x,y,cos,sin])."""
    return float(np.degrees(np.abs(_ego_yaw_steps(ego_future)).sum()))


def peak_lat_accel(ego_future: np.ndarray) -> float:
    """max |speed*omega| over the ego future (proxy peak lateral accel / curvature).
    REUSES eval_driving_metrics.lat_accel_curvature on a 4-col trajectory
    (future_to_4col with zero_rows_are_padding=False: the t=0 origin row is real,
    not padding). Heavy import deferred to first call."""
    from planner_metrics.scene_format import future_to_4col
    from rlvr.autoresearch.tools.eval_driving_metrics import lat_accel_curvature

    fut = np.asarray(ego_future, dtype=np.float32)
    fut4 = future_to_4col(fut, zero_rows_are_padding=False)
    la = lat_accel_curvature(fut4)
    return float(np.max(np.abs(la))) if la.size else 0.0


def active_neighbor_info(neighbor_agents_past: np.ndarray):
    """``(pos (Q,2), ped_mask (Q,), veh_mask (Q,), dists_origin (Q,))`` for the ACTIVE
    neighbors at the last timestep of ``neighbor_agents_past`` (N,T,11). "Active" =
    any of the 11 cols non-zero at t=0 (the interaction is judged at the present).
    ``ped_mask``/``veh_mask`` are the pedestrian/vehicle type one-hots (cols 9/8). The
    ordering is deterministic so a path distance computed separately (e.g. batched on
    GPU) stays aligned with ``pos``. Q=0 arrays when there are no active neighbors.

    Note: this "present at t=0, all-cols" activeness differs from ``npz_loader``'s
    "any timestep, cols 0:5" — they haven't diverged on real data, but a neighbor
    seen only in history would count there and not here. "Interaction = present now"
    is the intended semantics for labeling."""
    nap = np.asarray(neighbor_agents_past, dtype=np.float32)
    cur = nap[:, -1, :]  # (N, 11)
    active = np.any(cur != 0, axis=-1)
    pos = cur[active, :2]
    ped_mask = cur[active, _NB_PED_COL] == 1
    veh_mask = cur[active, _NB_VEH_COL] == 1
    dists_origin = np.linalg.norm(pos, axis=-1) if pos.shape[0] else np.empty((0,), np.float32)
    return pos, ped_mask, veh_mask, dists_origin


def _empty_neighbor_stats() -> dict:
    return {
        "neighbor_count": 0,
        "nearest_neighbor_m": None,
        "nearest_agent_path_m": None,
        "n_interacting": 0,
        "has_lead": False,
        "lead_dist_m": None,
        "nearest_ped_path_m": None,
        "has_pedestrian": False,
    }


def finalize_neighbor_stats(
    pos: np.ndarray,
    ped_mask: np.ndarray,
    veh_mask: np.ndarray,
    dists_origin: np.ndarray,
    dpath: np.ndarray,
    *,
    radius_m: float,
    lead_range_m: float,
    lead_half_width_m: float,
    interaction_corridor_m: float,
    ped_corridor_m: float,
) -> dict:
    """Assemble the neighbor feature dict from precomputed per-active-neighbor
    quantities: ego-origin distances ``dists_origin`` and ego-PATH distances
    ``dpath`` (both (Q,), aligned with ``pos``). This is the single finalize used by
    BOTH the CPU per-scene path and the batched GPU path, so their outputs are
    identical by construction.

    All interaction signals are ego-future-PATH relevant: ``n_interacting`` (any
    agent within ``interaction_corridor_m`` of the path), ``has_pedestrian`` (a
    pedestrian within ``ped_corridor_m`` of the path), and ``has_lead`` (a VEHICLE
    within ``lead_half_width_m`` of the path AND ahead of the ego within
    ``lead_range_m``). The lateral gate is distance-to-the-actual-path, so a lead
    around a curve is caught (a straight t=0 front cone would miss it)."""
    out = _empty_neighbor_stats()
    if pos.shape[0] == 0:
        return out
    out["neighbor_count"] = int(np.sum(dists_origin <= radius_m))  # ego-origin (informational)
    out["nearest_neighbor_m"] = float(dists_origin.min())
    out["nearest_agent_path_m"] = float(dpath.min())
    out["n_interacting"] = int(np.sum(dpath <= interaction_corridor_m))
    # lead vehicle: a VEHICLE on the ego path corridor, ahead within range. "Ahead"
    # uses ego-frame +x (path leaves the origin forward); the lateral gate is the
    # distance to the actual (possibly curved) future path, not a straight cone.
    fx = pos[:, 0]
    lead_mask = veh_mask & (dpath <= lead_half_width_m) & (fx > 0) & (fx <= lead_range_m)
    if np.any(lead_mask):
        out["has_lead"] = True
        out["lead_dist_m"] = float(fx[lead_mask].min())
    # pedestrian relevance against the ego path
    if np.any(ped_mask):
        ped_path = float(dpath[ped_mask].min())
        out["nearest_ped_path_m"] = ped_path
        out["has_pedestrian"] = bool(ped_path <= ped_corridor_m)
    return out


def neighbor_stats(
    neighbor_agents_past: np.ndarray,
    ego_future: np.ndarray,
    *,
    radius_m: float,
    lead_range_m: float,
    lead_half_width_m: float,
    interaction_corridor_m: float,
    ped_corridor_m: float,
    dpath: np.ndarray | None = None,
    device: str = "cpu",
) -> dict:
    """Neighbor interaction features scored for RELEVANCE to the ego's GT future
    path. ``neighbor_count`` / ``nearest_neighbor_m`` are ego-ORIGIN proximity
    (informational); ``nearest_agent_path_m`` / ``n_interacting`` /
    ``nearest_ped_path_m`` / ``has_pedestrian`` / ``has_lead`` are all measured
    against the ego PATH (has_lead = a vehicle on the path corridor, ahead).

    ``dpath`` (Q,) may be supplied precomputed (e.g. batched on GPU, aligned with
    :func:`active_neighbor_info`'s ordering); otherwise it is computed per-scene via
    :func:`min_dist_to_polyline` on ``device``."""
    pos, ped_mask, veh_mask, dists_origin = active_neighbor_info(neighbor_agents_past)
    if pos.shape[0] == 0:
        return _empty_neighbor_stats()
    if dpath is None:
        dpath = min_dist_to_polyline(pos, np.asarray(ego_future)[:, :2], device=device)
    return finalize_neighbor_stats(
        pos,
        ped_mask,
        veh_mask,
        dists_origin,
        np.asarray(dpath, dtype=np.float32),
        radius_m=radius_m,
        lead_range_m=lead_range_m,
        lead_half_width_m=lead_half_width_m,
        interaction_corridor_m=interaction_corridor_m,
        ped_corridor_m=ped_corridor_m,
    )


def static_count(static_objects) -> int:
    """Number of non-zero-row static objects."""
    if static_objects is None:
        return 0
    s = np.asarray(static_objects, dtype=np.float32)
    return int(np.sum(np.any(s != 0, axis=-1))) if s.size else 0


def goal_features(goal_pose: np.ndarray) -> dict:
    """{'goal_dist_m', 'goal_heading_delta_deg'} from goal_pose [x,y,heading] (ego frame)."""
    g = np.asarray(goal_pose, dtype=np.float32).reshape(-1)
    return {
        "goal_dist_m": float(math.hypot(float(g[0]), float(g[1]))),
        "goal_heading_delta_deg": float(
            abs(math.degrees(math.atan2(math.sin(float(g[2])), math.cos(float(g[2])))))
        ),
    }


# --------------------------------------------------------------------------- #
# label derivation
# --------------------------------------------------------------------------- #
def speed_bin(speed_ms: float, bins: tuple[float, float, float]) -> str:
    lo, mid, hi = bins
    if speed_ms < lo:
        return "stopped"
    if speed_ms < mid:
        return "low"
    if speed_ms < hi:
        return "mid"
    return "high"


def maneuver(signed_yaw_deg: float, straight_yaw_deg: float) -> str:
    if abs(signed_yaw_deg) < straight_yaw_deg:
        return "straight"
    return "turn-L" if signed_yaw_deg > 0 else "turn-R"


def interaction(
    *,
    n_interacting: int,
    has_lead: bool,
    has_pedestrian: bool,
    dense_count: int,
) -> str:
    """Single interaction label, priority by interestingness for scene mining:
    has-ped > has-lead > dense > none. Every branch is ego-future-PATH relevant
    (pedestrian/vehicle/agents near the path), so a distant agent never drives the
    label. Pedestrian-near-path is the rarest/most safety-critical, so it wins.

    ``has-static`` is intentionally NOT a category: ``static_objects`` is unpopulated
    (all-zero) in the current data and its column layout is not documented anywhere
    in the repo, so a path-relevant static check can't be verified. Add a
    ``has-static`` branch here (static within a corridor of the path — same shape as
    the others) once static_objects is populated and its xy columns are confirmed;
    ``static_count`` is still emitted as an informational feature."""
    if has_pedestrian:
        return "has-ped"
    if has_lead:
        return "has-lead"
    if n_interacting >= dense_count:
        return "dense"
    return "none"


def derive_labels(features: dict, config: FeatureConfig = FeatureConfig()) -> dict:
    """Compute {'speed_bin','maneuver','interaction','label'} from a feature dict
    (as returned by :func:`features_from_npz`)."""
    sbin = speed_bin(features["ego_speed_ms"], config.speed_bins)
    man = maneuver(features["gt_yaw_deg"], config.straight_yaw_deg)
    inter = interaction(
        n_interacting=features["n_interacting"] or 0,
        has_lead=bool(features["has_lead"]),
        has_pedestrian=bool(features["has_pedestrian"]),
        dense_count=config.dense_count,
    )
    return {
        "speed_bin": sbin,
        "maneuver": man,
        "interaction": inter,
        "label": f"{sbin}|{man}|{inter}",
    }


# --------------------------------------------------------------------------- #
# top-level entry points
# --------------------------------------------------------------------------- #
def features_from_npz(
    d,
    config: FeatureConfig = FeatureConfig(),
    *,
    device: str = "cpu",
    precomputed_dpath: np.ndarray | None = None,
) -> dict:
    """Compute the numeric feature columns from an already-loaded NPZ mapping ``d``
    (``np.load(..., allow_pickle=True)`` or a dict). Does NOT read the sidecar or
    parse the filename — see :func:`extract_scene_features` for the full row.
    ``peak_abs_yaw_deg``/``gt_path_m`` are computed here from the loaded ego future
    (width-aware, no reload).

    ``device`` selects where the per-scene path geometry runs; ``precomputed_dpath``
    (aligned with :func:`active_neighbor_info`) lets a batched GPU caller supply the
    ego-path distances so no per-scene geometry runs here."""
    files = set(d.files) if hasattr(d, "files") else set(d.keys())
    ego_future = np.asarray(d["ego_agent_future"], dtype=np.float32)
    speed = ego_speed_ms(d["ego_current_state"])
    row: dict = {
        "ego_speed_ms": speed,
        "is_stopped": is_stopped(speed),
        "travel_dist_m": gt_travel_distance_m(ego_future),
        "gt_yaw_deg": signed_total_yaw_deg(ego_future),
        # width-aware (from the loaded ego future); no reload, no legacy col-2 decode
        "gt_path_m": gt_travel_distance_m(ego_future),
        "peak_abs_yaw_deg": unsigned_total_yaw_deg(ego_future),
        "peak_lat_accel": peak_lat_accel(ego_future),
    }
    row.update(
        neighbor_stats(
            d["neighbor_agents_past"],
            ego_future,
            radius_m=config.neighbor_radius_m,
            lead_range_m=config.lead_range_m,
            lead_half_width_m=config.lead_half_width_m,
            interaction_corridor_m=config.interaction_corridor_m,
            ped_corridor_m=config.ped_corridor_m,
            dpath=precomputed_dpath,
            device=device,
        )
    )
    row["static_count"] = static_count(d["static_objects"] if "static_objects" in files else None)
    row.update(goal_features(d["goal_pose"]))
    row.update(derive_labels(row, config))
    return row


def extract_scene_features(
    npz_path: str,
    config: FeatureConfig = FeatureConfig(),
    *,
    with_sidecar: bool = True,
    device: str = "cpu",
    precomputed_dpath: np.ndarray | None = None,
    loaded=None,
) -> dict:
    """Full per-scene feature row for one NPZ path — every :data:`FEATURE_COLUMNS`
    key. Loads the NPZ (or reuses ``loaded`` if given), computes numeric features
    (incl. width-aware ``gt_path_m`` / ``peak_abs_yaw_deg``), the bag/frame,
    optional world pose from the sidecar (nullable), and the derived labels.
    ``device`` / ``precomputed_dpath`` are forwarded to :func:`features_from_npz`."""
    d = loaded if loaded is not None else np.load(npz_path, allow_pickle=True)
    # Canonicalize to physical identity: resolve symlinks + relativeness so the same
    # source scene reached via different path spellings (relative vs absolute) maps to
    # ONE identity everywhere downstream — the stored npz_path, the collision-free
    # output name (md5 of this string), and the frame/val split exclusion — instead of
    # aliasing into both train and val.
    npz_path = os.path.realpath(npz_path)
    row: dict = dict.fromkeys(FEATURE_COLUMNS)
    row["npz_path"] = npz_path
    row["bag"], row["frame"] = bag_and_frame(npz_path)
    row.update(features_from_npz(d, config, device=device, precomputed_dpath=precomputed_dpath))
    # gt_path_m / peak_abs_yaw_deg are now computed width-aware inside
    # features_from_npz (no compute_gt_stats reload / legacy col-2 decode).

    if with_sidecar:
        side = read_sidecar(npz_path)
        if side is not None:
            row["world_x"] = float(side["x"])
            row["world_y"] = float(side["y"])
            row["world_heading_deg"] = float(side["heading_deg"])
    return row
