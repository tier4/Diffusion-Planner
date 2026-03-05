"""
Pure metrics module for multi-trajectory evaluation in TeraSim.

No GUI, no model, no TeraSim imports — only numpy/math computations.
Each trajectory episode produces a TrajectoryMetrics object via finalize_metrics(),
then rank_trajectories() and metrics_to_dataframe() format results for display.

Off-road detection uses SUMO's TraCI lane assignment exposed through TeraSim's
agent state (StepState.av_lane_id / av_lateral_lane_pos / av_lane_width).
A step is flagged off-road when:
  1. av_lane_id is "" (SUMO did not assign the AV to any road lane), OR
  2. the vehicle body extends beyond the lane boundary:
       |av_lateral_lane_pos| + av_width/2 > av_lane_width/2
     (requires SUMO sublane model; skipped when lateral_lane_pos is always 0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NEAR_MISS_THRESH = 3.0   # metres — NPC centroid distance for near-miss counting
SIM_DT           = 0.1   # seconds per step


@dataclass
class StepState:
    """Raw data captured at one simulation step."""
    step: int
    ego_xy_map: tuple[float, float]    # commanded position in MGRS map frame
    ego_speed: float                   # m/s (from ego trajectory, not TeraSim)
    # AV lane occupancy — sourced from SUMO TraCI via TeraSim agent state.
    # lane_id == "" means SUMO has not assigned the AV to any road lane (fully off-road).
    # lateral_lane_pos is the offset from lane centre (metres, positive = left);
    # requires SUMO sublane model — 0.0 when model is disabled.
    # lane_width is the current lane width (metres); 0.0 when lane_id is empty/junction.
    av_lane_id: str = ""
    av_lateral_lane_pos: float = 0.0
    av_lane_width: float = 0.0
    av_width: float = 2.0              # ego vehicle width (metres)
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
    finish_reason: str = ""
    completion_rate: float = 0.0        # steps_done / total_steps
    progress_frac: float = 0.0          # distance_traveled / gt_distance (capped at 1)
    distance_traveled_m: float = 0.0   # total ego path length (metres)
    min_clearance_m: float = float("inf")
    near_miss_count: int = 0
    min_ttc_s: float = float("inf")
    off_road_fraction: float = 0.0      # fraction of steps flagged off-road by SUMO
    mean_jerk: float = 0.0
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


def _step_is_off_road(s: StepState) -> bool:
    """
    Return True if the AV is off-road at this step according to SUMO's lane data.

    Fully off-road: lane_id is "" (no lane assignment) or a junction internal lane (":").
    Partially off-road: vehicle body extends outside lane boundary, i.e.
      |lateral_lane_pos| + av_width/2 > lane_width/2
    The lateral check only fires when lane_width > 0 (sublane model enabled).
    """
    on_road_lane = s.av_lane_id != "" and not s.av_lane_id.startswith(":")
    if not on_road_lane:
        return True
    if s.av_lane_width > 0:
        if abs(s.av_lateral_lane_pos) + s.av_width / 2.0 > s.av_lane_width / 2.0:
            return True
    return False


def finalize_metrics(
    step_states: list[StepState],
    trajectory_map: np.ndarray,   # (T, 3): [x_map, y_map, yaw_rad]
    gt_trajectory_map: np.ndarray, # (80, 3)
    total_steps: int = 80,
    near_miss_thresh: float = NEAR_MISS_THRESH,
) -> TrajectoryMetrics:
    """
    Aggregate per-step recorded states into a single TrajectoryMetrics object.

    Off-road detection uses SUMO's TraCI lane assignment (see _step_is_off_road).
    Progress is the fraction of GT trajectory distance covered by the ego.
    """
    m = TrajectoryMetrics(label="")

    min_clearance = float("inf")
    min_ttc       = float("inf")
    near_misses   = 0
    steps_done    = len(step_states)
    off_road_count = 0

    for i, s in enumerate(step_states):
        if not s.av_in_sim:
            m.collision = True
            m.collision_step = s.step
            break

        ego_xy = np.array(s.ego_xy_map)

        if _step_is_off_road(s):
            off_road_count += 1

        # Ego velocity from position diff
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

    m.completion_rate   = (steps_done if not m.collision else m.collision_step) / total_steps
    m.min_clearance_m   = min_clearance
    m.near_miss_count   = near_misses
    m.min_ttc_s         = min_ttc
    m.mean_jerk         = compute_ego_jerk(trajectory_map)
    m.off_road_fraction = off_road_count / max(steps_done, 1)

    # Progress: fraction of GT distance covered by the ego trajectory.
    ego_pos = trajectory_map[:steps_done, :2]
    gt_pos  = gt_trajectory_map[:, :2]
    if len(ego_pos) > 1:
        m.distance_traveled_m = float(np.sum(np.linalg.norm(np.diff(ego_pos, axis=0), axis=1)))
    gt_dist = float(np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)))
    if gt_dist > 0:
        m.progress_frac = min(m.distance_traveled_m / gt_dist, 1.0)

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

    Weights: progress 0.25, clearance 0.25, on-road 0.15, TTC 0.15,
             near-miss-free 0.10, jerk 0.10.
    """
    if m.collision:
        return 0.0

    clearance_score  = min(m.min_clearance_m / 5.0, 1.0)
    ttc_score        = min(m.min_ttc_s / 5.0, 1.0) if m.min_ttc_s < float("inf") else 1.0
    safe_steps_score = max(0.0, 1.0 - m.near_miss_count / total_steps)
    on_road_score    = 1.0 - m.off_road_fraction
    jerk_score       = max(0.0, 1.0 - m.mean_jerk / 5.0)
    progress_score   = m.progress_frac

    return (
        0.25 * progress_score   +
        0.25 * clearance_score  +
        0.15 * on_road_score    +
        0.15 * ttc_score        +
        0.10 * safe_steps_score +
        0.10 * jerk_score
    )


def rank_trajectories(metrics: list[TrajectoryMetrics]) -> list[TrajectoryMetrics]:
    """Return copy of metrics sorted by score descending (best first)."""
    return sorted(metrics, key=lambda m: m.score, reverse=True)


def metrics_to_dataframe(ranked: list[TrajectoryMetrics]) -> list[list]:
    """
    Convert ranked metrics to list-of-lists for gr.Dataframe.

    Columns: Rank | Label | Score | Collision | Progress% | Dist(m) |
             MinClear(m) | MinTTC(s) | NearMiss | OffRoad% | Jerk(m/s³) | FDE(m)
    """
    rows = []
    for rank, m in enumerate(ranked, 1):
        rows.append([
            rank,
            m.label,
            f"{m.score:.3f}",
            "YES" if m.collision else "no",
            f"{m.progress_frac * 100:.0f}%",
            f"{m.distance_traveled_m:.1f}",
            f"{m.min_clearance_m:.2f}" if m.min_clearance_m < 1e9 else "—",
            f"{m.min_ttc_s:.2f}"       if m.min_ttc_s < float('inf') else "inf",
            str(m.near_miss_count),
            f"{m.off_road_fraction * 100:.0f}%",
            f"{m.mean_jerk:.3f}",
            f"{m.fde_from_gt_m:.2f}",
        ])
    return rows
