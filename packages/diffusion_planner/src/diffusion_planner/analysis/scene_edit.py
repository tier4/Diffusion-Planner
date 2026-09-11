"""Place, move and remove neighbor agents in a loaded planner frame.

A frame reserves ``MAX_NUM_NEIGHBORS`` neighbor slots and typically fills a
small fraction of them, so an agent that never existed in the recording can be
added by writing one free slot of three tensors: a pose history in
``neighbor_agents_past``, a size in ``agent_shape``, and a class one-hot in
``agent_label``. No map, route or scene-graph machinery is involved, because the
encoder reads exactly those three tensors and derives occupancy from the poses.

Edits are in **metric, ego-frame units**, matching what the frame loader
returns. Normalization happens downstream at inference time, so an edited frame
is fed to the model exactly like a recorded one.

This is the only way to obtain a genuinely unknown-labelled agent: recorded
shards contain none, and the training-time augmentation can only relabel an
agent that is already there, inheriting its motion and size.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..visualizer.schema import AgentShapeIndex, NeighborIndex
from .attention import AGENT_CLASS_NAMES, UNLABELED_NAME, neighbor_valid

__all__ = [
    "DEFAULT_TIMESTEP_S",
    "PlacedAgent",
    "agent_history",
    "edited_slots",
    "free_slots",
    "insert_agent",
    "occupied_slots",
    "read_agent",
    "remove_agent",
    "update_agent",
]

DEFAULT_TIMESTEP_S = 0.1
"""Sample spacing of the recorded pose history."""

Frame = dict[str, Any]


@dataclass(frozen=True)
class PlacedAgent:
    """An agent to write into a frame, in metric ego-frame units.

    Attributes:
        x_m: Longitudinal position at the current step.
        y_m: Lateral position at the current step.
        heading_rad: Heading at the current step.
        speed_mps: Constant speed used to synthesize the pose history.
        yaw_rate_rps: Constant yaw rate used to synthesize the pose history.
        width_m: Box width.
        length_m: Box length.
        agent_class: One of the label class names, or ``"unlabeled"`` for a
            valid agent carrying no class at all.
    """

    x_m: float
    y_m: float
    heading_rad: float = 0.0
    speed_mps: float = 0.0
    yaw_rate_rps: float = 0.0
    width_m: float = 2.0
    length_m: float = 4.5
    agent_class: str = "unknown"

    def __post_init__(self) -> None:
        allowed = (*AGENT_CLASS_NAMES, UNLABELED_NAME)
        if self.agent_class not in allowed:
            raise ValueError(f"agent_class must be one of {allowed}")
        if self.width_m <= 0.0 or self.length_m <= 0.0:
            raise ValueError("width_m and length_m must be positive")

    @property
    def label_index(self) -> int | None:
        """Column to set in ``agent_label``, or ``None`` when unlabeled."""
        if self.agent_class == UNLABELED_NAME:
            return None
        return AGENT_CLASS_NAMES.index(self.agent_class)


def agent_history(
    agent: PlacedAgent,
    length: int,
    timestep_s: float = DEFAULT_TIMESTEP_S,
) -> NDArray[np.float32]:
    """Synthesize a pose history ending at the agent's placed pose.

    The history is integrated backwards from the current step at constant speed
    and yaw rate, which is what the recorded histories look like over the short
    window the model sees, and is the fallback used when no lane to snap to is
    known.

    Returns:
        Poses with shape ``(length, 4)`` as ``x, y, cos(yaw), sin(yaw)``, the
        last row being the placed pose.
    """
    if length < 1:
        raise ValueError("length must be at least 1")
    poses = np.zeros((length, len(NeighborIndex)), dtype=np.float32)
    x, y, heading = agent.x_m, agent.y_m, agent.heading_rad
    poses[-1] = (x, y, math.cos(heading), math.sin(heading))
    for step in range(length - 2, -1, -1):
        heading -= agent.yaw_rate_rps * timestep_s
        x -= agent.speed_mps * math.cos(heading) * timestep_s
        y -= agent.speed_mps * math.sin(heading) * timestep_s
        poses[step] = (x, y, math.cos(heading), math.sin(heading))
    return poses


def _poses(frame: Mapping[str, Any]) -> NDArray[np.float32]:
    return np.asarray(frame["neighbor_agents_past"], dtype=np.float32)


def occupied_slots(frame: Mapping[str, Any]) -> NDArray[np.intp]:
    """Indices of neighbor slots holding an agent."""
    return np.flatnonzero(neighbor_valid(_poses(frame)))


def free_slots(frame: Mapping[str, Any]) -> NDArray[np.intp]:
    """Indices of neighbor slots available for a new agent."""
    return np.flatnonzero(~neighbor_valid(_poses(frame)))


def _copied(frame: Mapping[str, Any]) -> Frame:
    """Copy the three tensors an edit touches, sharing the rest."""
    output: Frame = dict(frame)
    for name in ("neighbor_agents_past", "agent_shape", "agent_label"):
        output[name] = np.array(frame[name], dtype=np.float32, copy=True)
    return output


def insert_agent(
    frame: Mapping[str, Any],
    agent: PlacedAgent,
    slot: int | None = None,
    timestep_s: float = DEFAULT_TIMESTEP_S,
) -> tuple[Frame, int]:
    """Write an agent into a free neighbor slot.

    Args:
        frame: A raw, unnormalized frame.
        agent: The agent to place.
        slot: A specific free slot, or ``None`` to take the lowest free one.
        timestep_s: Spacing used to synthesize the pose history.

    Returns:
        A new frame and the slot written. The input frame is not modified.

    Raises:
        ValueError: If the frame is full, or the requested slot is occupied.
    """
    available = free_slots(frame)
    if slot is None:
        if available.size == 0:
            raise ValueError(
                f"every one of {_poses(frame).shape[0]} neighbor slots is "
                "occupied; remove an agent before inserting one"
            )
        slot = int(available[0])
    elif slot not in available:
        raise ValueError(f"slot {slot} is already occupied")

    output = _copied(frame)
    history_length = output["neighbor_agents_past"].shape[1]
    output["neighbor_agents_past"][slot] = agent_history(
        agent, history_length, timestep_s
    )
    output["agent_shape"][slot, AgentShapeIndex.WIDTH] = agent.width_m
    output["agent_shape"][slot, AgentShapeIndex.LENGTH] = agent.length_m
    output["agent_label"][slot] = 0.0
    if agent.label_index is not None:
        output["agent_label"][slot, agent.label_index] = 1.0
    return output, slot


def remove_agent(frame: Mapping[str, Any], slot: int) -> Frame:
    """Clear a neighbor slot.

    Zeroing all three tensors is exactly what an empty slot means to the
    encoder, so a removed agent is indistinguishable from one that never
    existed.
    """
    if slot not in occupied_slots(frame):
        raise ValueError(f"slot {slot} holds no agent")
    output = _copied(frame)
    output["neighbor_agents_past"][slot] = 0.0
    output["agent_shape"][slot] = 0.0
    output["agent_label"][slot] = 0.0
    return output


def update_agent(
    frame: Mapping[str, Any],
    slot: int,
    agent: PlacedAgent,
    timestep_s: float = DEFAULT_TIMESTEP_S,
) -> Frame:
    """Replace whatever occupies a slot with a new agent."""
    if slot not in occupied_slots(frame):
        raise ValueError(f"slot {slot} holds no agent")
    cleared = remove_agent(frame, slot)
    updated, _ = insert_agent(cleared, agent, slot=slot, timestep_s=timestep_s)
    return updated


def read_agent(frame: Mapping[str, Any], slot: int) -> PlacedAgent | None:
    """Recover a slot's agent, or ``None`` when the slot is empty.

    Speed and yaw rate are estimated from the last two history steps, so a
    recorded agent round-trips approximately rather than exactly.
    """
    poses = _poses(frame)
    if not bool(neighbor_valid(poses)[slot]):
        return None
    history = poses[slot]
    current = history[-1]
    heading = math.atan2(
        float(current[NeighborIndex.SIN_YAW]), float(current[NeighborIndex.COS_YAW])
    )
    speed = 0.0
    yaw_rate = 0.0
    if history.shape[0] > 1:
        previous = history[-2]
        step = np.hypot(
            float(current[NeighborIndex.X] - previous[NeighborIndex.X]),
            float(current[NeighborIndex.Y] - previous[NeighborIndex.Y]),
        )
        speed = float(step / DEFAULT_TIMESTEP_S)
        previous_heading = math.atan2(
            float(previous[NeighborIndex.SIN_YAW]),
            float(previous[NeighborIndex.COS_YAW]),
        )
        delta = (heading - previous_heading + math.pi) % (2.0 * math.pi) - math.pi
        yaw_rate = float(delta / DEFAULT_TIMESTEP_S)

    labels = np.asarray(frame["agent_label"], dtype=np.float32)[slot]
    name = (
        UNLABELED_NAME
        if float(labels.sum()) <= 0.0
        else AGENT_CLASS_NAMES[int(np.argmax(labels))]
    )
    shape = np.asarray(frame["agent_shape"], dtype=np.float32)[slot]
    return PlacedAgent(
        x_m=float(current[NeighborIndex.X]),
        y_m=float(current[NeighborIndex.Y]),
        heading_rad=heading,
        speed_mps=speed,
        yaw_rate_rps=yaw_rate,
        width_m=max(float(shape[AgentShapeIndex.WIDTH]), 1e-3),
        length_m=max(float(shape[AgentShapeIndex.LENGTH]), 1e-3),
        agent_class=name,
    )


def edited_slots(
    original: Mapping[str, Any], edited: Mapping[str, Any]
) -> NDArray[np.intp]:
    """Slots whose pose, size or label differ between two frames."""
    changed = np.zeros(_poses(original).shape[0], dtype=bool)
    for name in ("neighbor_agents_past", "agent_shape", "agent_label"):
        before = np.asarray(original[name], dtype=np.float32)
        after = np.asarray(edited[name], dtype=np.float32)
        if before.shape != after.shape:
            raise ValueError(f"{name} changed shape between the two frames")
        axes = tuple(range(1, before.ndim))
        changed |= np.any(before != after, axis=axes)
    return np.flatnonzero(changed)
