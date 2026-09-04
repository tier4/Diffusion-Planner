"""Build a Diffusion-Planner ``SceneContext`` from OpenSCENARIO simulator truth.

The simulator reports poses in the map (MGRS) frame and ``LaneletSceneBuilder`` works in the
same frame, so no frame change is needed; ``to_model_tensors`` re-centers them onto the ego.
Within that frame the ego is stored at base_link and every other agent at its bbox centroid,
which is the pair ``SceneContext`` is defined in.

Entities are identified by the simulator's type field rather than by name, and each one
carries a rolling pose history because the model consumes a fixed-length past.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
from scenario_generation.scene_context import Agent, AgentType, SceneContext
from scenario_generation.tensor_converter import _INPUT_T

DT = 0.1  # sim + model timestep (10 Hz). Must match the interpreter's local_frame_rate.
_HISTORY_LEN = _INPUT_T + 1  # the past the model sees, plus the current pose.
_SIM_TYPE_EGO = 0  # get_entity_states()["type"]: 0=EGO 1=VEHICLE 2=PEDESTRIAN 3=MISC_OBJECT
# get_entity_states()["subtype"]. A two-wheeler is a VEHICLE by type; only the subtype tells
# it apart, and the model's agent-type one-hot has a class for it.
_SIM_SUBTYPE_MOTORCYCLE = 5
_SIM_SUBTYPE_BICYCLE = 6
# Not signalling, in the TurnIndicatorsReport space: it has no 0, so the NO_COMMAND the
# sim's setter would also accept is not a value this history may carry.
TURN_INDICATOR_DISABLE = 1


@dataclass
class SceneConfig:
    """Window and shape parameters for the SceneContext snapshot."""

    max_map_lanelets: int = 140
    map_mask_range_m: float = 100.0
    route_window_segments: int = 25


def resolve_ego_name(states: dict) -> str:
    """The ego's entity name as the simulator reports it.

    Identified by type, not by name: ``type == 0`` is authoritative whatever the scenario
    chose to call the entity. Matching on the name instead fails at spawn time with a message
    that points at the symptom rather than the cause.
    """
    egos = [name for name, st in states.items() if int(st["type"]) == _SIM_TYPE_EGO]
    if len(egos) == 1:
        return egos[0]
    if not egos:
        raise RuntimeError(
            f"no ego-typed (type={_SIM_TYPE_EGO}) entity spawned; entities="
            + str({n: st["type"] for n, st in states.items()})
        )
    raise RuntimeError(f"more than one ego-typed entity: {egos}")


def _agent_type(state: dict) -> AgentType | None:
    """Map a reported entity to ``AgentType``; ``None`` for non-agents."""
    sim_type = int(state["type"])
    if sim_type in (0, 1):  # EGO, VEHICLE
        subtype = int(state.get("subtype", 0))
        if subtype in (_SIM_SUBTYPE_MOTORCYCLE, _SIM_SUBTYPE_BICYCLE):
            return AgentType.BICYCLE
        return AgentType.VEHICLE
    if sim_type == 2:  # PEDESTRIAN
        return AgentType.PEDESTRIAN
    return None  # MISC_OBJECT is not a dynamic agent


class HistoryBuffers:
    """Per-entity rolling ``(x, y, heading)`` and ``(vx, vy)`` history keyed by the sim name.

    A newly seen entity has its buffers filled by repeating the current sample rather than
    zeros, so no teleport artifact enters the model input. ``age`` counts how many *previous*
    samples are real, which is the convention the tensor converter masks on, and lets scoring
    wait until the history is warm.
    """

    def __init__(self, length: int = _HISTORY_LEN):
        self.length = length
        self._buf: dict[str, deque] = {}
        self._vel: dict[str, deque] = {}
        self.age: dict[str, int] = {}

    def update(self, name: str, x: float, y: float, yaw: float, vx: float, vy: float) -> None:
        if name not in self._buf:
            self._buf[name] = deque([(x, y, yaw)] * self.length, maxlen=self.length)
            self._vel[name] = deque([(vx, vy)] * self.length, maxlen=self.length)
            self.age[name] = 0
        else:
            self._buf[name].append((x, y, yaw))
            self._vel[name].append((vx, vy))
            self.age[name] += 1

    def forget(self, live_names: set[str]) -> None:
        """Drop entities the simulator no longer reports.

        A scenario may delete an entity and spawn a new one under the same name, and the sim
        accepts that. Keeping the old buffer would append the new spawn pose onto it, putting
        a teleport into the model input with an age that says the history is warm.
        """
        for name in self._buf.keys() - live_names:
            del self._buf[name]
            del self._vel[name]
            del self.age[name]

    def trajectory(self, name: str) -> np.ndarray:
        return np.array(self._buf[name], dtype=np.float64)  # (length, 3)

    def velocities(self, name: str) -> np.ndarray:
        return np.array(self._vel[name], dtype=np.float64)  # (length, 2)


def baselink_xyh(state: dict) -> tuple[float, float, float]:
    """The reported pose as-is -- the entity's reference point, which for a vehicle catalogued
    the Autoware way is base_link. The one place the pose dict shape is read."""
    p = state["pose"]
    return float(p["x"]), float(p["y"]), float(p["yaw"])


def centroid_xyh(state: dict) -> tuple[float, float, float]:
    """The bbox-centre pose. The offset from base_link to the centre is reported separately, and
    is zero only for an entity whose two already coincide."""
    x, y, yaw = baselink_xyh(state)
    c = state["bounding_box"]["center"]
    cx, cy = float(c["x"]), float(c["y"])
    return (
        x + math.cos(yaw) * cx - math.sin(yaw) * cy,
        y + math.sin(yaw) * cx + math.cos(yaw) * cy,
        yaw,
    )


def velocity_xy(state: dict) -> tuple[float, float]:
    """Map-frame ``(vx, vy)`` from the reported twist.

    The twist is in the entity's own frame -- the headless ego only ever sets ``linear.x``
    from the vehicle model's longitudinal speed -- so it is rotated by the yaw rather than
    used as-is.
    """
    tw = state["twist"]
    vlong, vlat = float(tw["linear_x"]), float(tw["linear_y"])
    yaw = float(state["pose"]["yaw"])
    c, s_ = math.cos(yaw), math.sin(yaw)
    return vlong * c - vlat * s_, vlong * s_ + vlat * c


def update_history(buffers: HistoryBuffers, states: dict, ego_name: str) -> None:
    """Append this tick's truth pose to each dynamic entity's rolling buffer.

    The ego is stored at base_link and every other agent at its bbox centroid:
    ``to_model_tensors`` shifts a neighbour back by ``wheelbase / 2`` when it stands in as the
    ego, and the metric OBB builders read a neighbour's stored xy as the box centre.
    """
    live = {name for name, st in states.items() if _agent_type(st) is not None}
    buffers.forget(live)
    for name in live:
        st = states[name]
        pose = baselink_xyh(st) if name == ego_name else centroid_xyh(st)
        buffers.update(name, *pose, *velocity_xy(st))


def entity_shape(state: dict) -> tuple[float, float, float]:
    """``(length, width, axle_wheelbase)`` for ``Agent`` -- model input, not metrics.

    The axle spacing is the simulator's, off the entity's vehicle parameters; it is 0 for a
    non-vehicle, whose reference point already is its box centre. Metric geometry wants a
    different number for the same vehicle and takes it from :func:`ego_metric_box`.
    """
    dims = state["bounding_box"]["dimensions"]
    return float(dims["x"]), float(dims["y"]), float(state["wheel_base"])


def ego_metric_box(state: dict) -> tuple[float, float, float]:
    """``(length, width, box_wheelbase)`` for the metric OBB builders.

    ``box_wheelbase`` is not the axle spacing. The OBB builders treat the reported pose as the
    rear-axle midpoint and assume symmetric overhangs, placing the box centre ``wheelbase / 2``
    ahead of it -- so what they need is twice the true centre offset, which the simulator
    reports directly. Estimating it from ``length`` biases clearance in opposite directions
    fore and aft on a vehicle with asymmetric overhangs, which cannot be corrected afterwards.
    """
    bbox = state["bounding_box"]
    dims = bbox["dimensions"]
    return float(dims["x"]), float(dims["y"]), 2.0 * float(bbox["center"]["x"])


def build_scene(
    states: dict,
    buffers: HistoryBuffers,
    builder: LaneletSceneBuilder,
    ego_route_ids: list[int],
    goal_pose: np.ndarray,
    cfg: SceneConfig,
    ego_name: str,
    turn_indicators: np.ndarray,
) -> SceneContext:
    """Build a SceneContext snapshot in the map frame from this tick's sim truth."""
    ex, ey, _ = baselink_xyh(states[ego_name])
    ego_xy = np.array([ex, ey], dtype=np.float32)

    # Closest-N lanelets around the ego, with the ego route pinned first so route context can
    # never be dropped by the distance cut.
    closest = builder.closest_lanelets(
        ego_xy, cfg.max_map_lanelets, mask_range=cfg.map_mask_range_m
    )
    seen: set[int] = set()
    all_ids: list[int] = []
    for ll_id in list(ego_route_ids) + list(closest):
        if ll_id in seen or not builder.has_lanelet_id(ll_id):
            continue
        seen.add(ll_id)
        all_ids.append(ll_id)
        if len(all_ids) >= cfg.max_map_lanelets:
            break
    map_data = builder._build_map_data(all_ids, center_xy=ego_xy)

    # Ego route_lanes: a forward sliding window refreshed every tick.
    window = (
        builder.select_route_segment_indices(
            ego_route_ids, ego_xy, max_segments=cfg.route_window_segments
        )
        or ego_route_ids[: cfg.route_window_segments]
    )
    route_lanes, route_sl, route_hsl = builder._route_to_33dim(
        window, max_segments=cfg.route_window_segments
    )

    agents: list[Agent] = []
    for name, st in states.items():
        atype = _agent_type(st)
        if atype is None:
            continue
        length, width, wheelbase = entity_shape(st)
        traj = buffers.trajectory(name)
        is_ego = name == ego_name
        agents.append(
            Agent(
                id=name,
                agent_type=atype,
                length=length,
                width=width,
                wheelbase=wheelbase,
                past_trajectory=traj,
                past_velocities=buffers.velocities(name),
                goal_pose=goal_pose.astype(np.float32) if is_ego else None,
                route_lanes=route_lanes if is_ego else None,
                route_speed_limit=route_sl if is_ego else None,
                route_has_speed_limit=route_hsl if is_ego else None,
                turn_indicators=(turn_indicators if is_ego else None),
                route_lanelet_ids=list(ego_route_ids) if is_ego else None,
                age_steps=buffers.age[name],
            )
        )
    return SceneContext(agents=agents, map_data=map_data, ego_agent_id=ego_name, dt=DT)
