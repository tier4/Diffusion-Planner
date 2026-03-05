"""
Pure metrics module for multi-trajectory evaluation in TeraSim.

No GUI, no model, no TeraSim imports — only numpy/math computations.
Each trajectory episode produces a TrajectoryMetrics object via finalize_metrics(),
then rank_trajectories() and metrics_to_dataframe() format results for display.

Off-road detection uses SUMO snap distance rather than geometric lane proximity.
When the AV is teleported with moveToXY keepRoute=0, SUMO snaps it to the nearest
road edge. The distance between the commanded position and SUMO's reported AV
position is a direct proxy for how far off-road the commanded trajectory was.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NEAR_MISS_THRESH = 3.0   # metres — NPC centroid distance
OFF_ROAD_THRESH  = 3.0   # metres — snap distance threshold for off-road classification
SIM_DT           = 0.1   # seconds per step


@dataclass
class StepState:
    """Raw data captured at one simulation step."""
    step: int
    ego_xy_map: tuple[float, float]    # commanded position in MGRS map frame
    ego_xy_sumo: tuple[float, float]   # AV position reported by SUMO (snapped to nearest road)
    snap_distance: float               # ||commanded - reported|| — proxy for off-road distance
    ego_speed: float                   # m/s (from ego trajectory, not TeraSim)
    # NPC states: keyed by agent_id, each entry is the dict from TeraSim /state.
    # Fields per entry: x, y, orientation (rad), speed (m/s), acceleration (m/s²),
    #                   angular_velocity (rad/s), length, width
    vehicle_states: dict = field(default_factory=dict)
    vru_states:     dict = field(default_factory=dict)
    av_in_sim: bool = True


@dataclass
class TrajectoryMetrics:
    label: str
    collision: bool = False
    collision_step: int | None = None
    finish_reason: str = ""             # from /simulation_result endpoint
    completion_rate: float = 0.0        # steps_done / 80
    min_clearance_m: float = float("inf")
    near_miss_count: int = 0
    min_ttc_s: float = float("inf")
    off_road_fraction: float = 0.0      # fraction of steps where snap_distance > OFF_ROAD_THRESH
    max_snap_distance: float = 0.0      # peak snap distance (metres) — worst off-road deviation
    mean_jerk: float = 0.0              # from commanded trajectory (ego)
    fde_from_gt_m: float = 0.0
    score: float = 0.0


def npc_velocity(npc: dict) -> np.ndarray:
    """
    Decompose NPC speed + orientation into (vx, vy).

    Uses 'orientation' field (ROS rad, CCW from +X) from TeraSim state when
    available.  Falls back to sumo_angle conversion if orientation is absent.
    """
    if "orientation" in npc:
        angle = npc["orientation"]
    else:
        # sumo_angle: degrees CW from north → rad CCW from +X
        angle = math.radians(90.0 - npc["sumo_angle"])
    speed = npc["speed"]
    return np.array([speed * math.cos(angle), speed * math.sin(angle)])


def compute_ttc(
    ego_xy: np.ndarray,   # (2,)
    ego_vel: np.ndarray,  # (2,)
    npc_xy: np.ndarray,   # (2,)
    npc_vel: np.ndarray,  # (2,)
) -> float:
    """
    Constant-velocity time-to-collision.

    Returns inf if vehicles are diverging or stationary relative to each other.
    """
    rel_pos = npc_xy - ego_xy
    dist = np.linalg.norm(rel_pos)
    if dist < 1e-6:
        return 0.0
    rel_vel = ego_vel - npc_vel
    closing_speed = -np.dot(rel_pos / dist, rel_vel)
    if closing_speed > 0.0:
        return dist / closing_speed
    return float("inf")


def compute_ego_jerk(trajectory_map: np.ndarray, dt: float = SIM_DT) -> float:
    """
    Mean absolute jerk (m/s³) from commanded ego position sequence.

    trajectory_map: (T, 2+) array, first two columns are [x, y].
    Uses finite differences of position (TeraSim does not report AV acceleration).
    """
    if len(trajectory_map) < 4:
        return 0.0
    pos = trajectory_map[:, :2]
    vel  = np.diff(pos, axis=0) / dt
    acc  = np.diff(vel,  axis=0) / dt
    jerk = np.diff(acc,  axis=0) / dt
    return float(np.mean(np.linalg.norm(jerk, axis=1)))


def finalize_metrics(
    step_states: list[StepState],
    trajectory_map: np.ndarray,            # (T, 3): [x_map, y_map, yaw_rad]
    gt_trajectory_map: np.ndarray,         # (80, 3)
    total_steps: int = 80,
    near_miss_thresh: float = NEAR_MISS_THRESH,
    off_road_thresh: float = OFF_ROAD_THRESH,
) -> TrajectoryMetrics:
    """
    Aggregate per-step recorded states into a single TrajectoryMetrics object.

    Off-road detection uses SUMO snap distance (||commanded_xy - sumo_reported_xy||).
    When moveToXY keepRoute=0 is used, SUMO snaps the AV to the nearest road edge.
    A large snap distance means the commanded trajectory was off-road.

    Ego velocity is estimated from consecutive commanded positions (TeraSim does not
    report AV acceleration back).
    """
    m = TrajectoryMetrics(label="")

    min_clearance = float("inf")
    min_ttc       = float("inf")
    near_misses   = 0
    steps_done    = len(step_states)
    snap_dists    = []

    for i, s in enumerate(step_states):
        if not s.av_in_sim:
            m.collision = True
            m.collision_step = s.step
            break

        ego_xy = np.array(s.ego_xy_map)
        snap_dists.append(s.snap_distance)

        # Ego velocity from position diff (use step states for consistency)
        if i > 0:
            ego_vel = (np.array(step_states[i].ego_xy_map) -
                       np.array(step_states[i-1].ego_xy_map)) / SIM_DT
        else:
            ego_vel = np.zeros(2)

        all_npcs = {**s.vehicle_states, **s.vru_states}
        for nid, npc in all_npcs.items():
            npc_xy = np.array([npc["x"], npc["y"]])
            dist   = float(np.linalg.norm(ego_xy - npc_xy))

            min_clearance = min(min_clearance, dist)
            if dist < near_miss_thresh:
                near_misses += 1

            npc_vel = npc_velocity(npc)
            ttc = compute_ttc(ego_xy, ego_vel, npc_xy, npc_vel)
            min_ttc = min(min_ttc, ttc)

    m.completion_rate = (steps_done if not m.collision else m.collision_step) / total_steps
    m.min_clearance_m = min_clearance
    m.near_miss_count = near_misses
    m.min_ttc_s       = min_ttc
    m.mean_jerk       = compute_ego_jerk(trajectory_map)

    # Off-road via SUMO snap distance (no lane geometry needed)
    if snap_dists:
        snap_arr = np.array(snap_dists)
        m.off_road_fraction = float(np.mean(snap_arr > off_road_thresh))
        m.max_snap_distance = float(np.max(snap_arr))

    # FDE vs GT at last completed step
    last = min(steps_done, len(gt_trajectory_map)) - 1
    if last >= 0:
        m.fde_from_gt_m = float(np.linalg.norm(
            trajectory_map[last, :2] - gt_trajectory_map[last, :2]
        ))

    return m


def compute_score(m: TrajectoryMetrics, total_steps: int = 80) -> float:
    """
    Composite score in [0, 1]. Higher = better.
    Collision → 0 (hard fail).

    Weights: clearance 0.30, TTC 0.20, near-miss-free 0.10,
             completion 0.20, on-road 0.10, jerk 0.10.
    """
    if m.collision:
        return 0.0

    clearance_score  = min(m.min_clearance_m / 5.0, 1.0)
    ttc_score        = min(m.min_ttc_s / 5.0, 1.0) if m.min_ttc_s < float("inf") else 1.0
    safe_steps_score = 1.0 - m.near_miss_count / total_steps
    on_road_score    = 1.0 - m.off_road_fraction
    jerk_score       = max(0.0, 1.0 - m.mean_jerk / 5.0)

    return (
        0.30 * clearance_score  +
        0.20 * ttc_score        +
        0.10 * safe_steps_score +
        0.20 * m.completion_rate +
        0.10 * on_road_score    +
        0.10 * jerk_score
    )


def rank_trajectories(metrics: list[TrajectoryMetrics]) -> list[TrajectoryMetrics]:
    """Return copy of metrics sorted by score descending (best first)."""
    return sorted(metrics, key=lambda m: m.score, reverse=True)


def metrics_to_dataframe(ranked: list[TrajectoryMetrics]) -> list[list]:
    """
    Convert ranked metrics to list-of-lists for gr.Dataframe.

    Columns: Rank | Label | Score | Collision | Completion% |
             MinClear(m) | MinTTC(s) | NearMiss | OffRoad% | MaxSnap(m) | Jerk(m/s³) | FDE(m)
    """
    rows = []
    for rank, m in enumerate(ranked, 1):
        rows.append([
            rank,
            m.label,
            f"{m.score:.3f}",
            "YES" if m.collision else "no",
            f"{m.completion_rate * 100:.0f}%",
            f"{m.min_clearance_m:.2f}" if m.min_clearance_m < 1e9 else "—",
            f"{m.min_ttc_s:.2f}"       if m.min_ttc_s < float('inf') else "inf",
            str(m.near_miss_count),
            f"{m.off_road_fraction * 100:.0f}%",
            f"{m.max_snap_distance:.2f}",
            f"{m.mean_jerk:.3f}",
            f"{m.fde_from_gt_m:.2f}",
        ])
    return rows
