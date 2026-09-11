"""Roll a scene forward under the planner's own predictions.

Each step samples a trajectory, advances the ego and every neighbor one
timestep along it, and re-expresses the scene around the ego's new pose. The
recentring reuses ``recenter_frame_to_pose``, the same rigid transform the
training augmentation applies, so the geometry stays consistent with what the
model was trained on.

Re-planning cadence matters more than it looks. Sampling afresh every single
step makes the ego stall; following one prediction for several steps tracks the
recorded speed closely. ``rollout``'s ``replan_every`` documents the measurements.

**The map does not regenerate.** Every map tensor is ego-frame, and this module
only moves the geometry the starting frame already contained. Nothing new comes
into view, so the rollout is trustworthy only while the ego stays inside the
window the original frame covered. :class:`RolloutStep` reports the remaining
coverage and flags the step where the ego leaves it, rather than quietly
producing confident nonsense. For longer horizons the map tensors would have to
be rebuilt from a map source, which lives in the C++ preprocessing.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from ..data.dimensions import TRAJECTORY_DIM, TRAJECTORY_LENGTH
from ..data.transforms import PlannerDataNormalizer
from ..data.transforms.rigid_augmentation import recenter_frame_to_pose
from ..models.diffusion_planner import DiffusionPlanner
from ..visualizer.schema import EgoIndex, NeighborIndex, PoseIndex
from .attention import neighbor_valid
from .scene_edit import DEFAULT_TIMESTEP_S

__all__ = ["RolloutStep", "map_coverage_m", "rollout"]


@dataclass(frozen=True)
class RolloutStep:
    """One closed-loop step and how far it can still be trusted."""

    step: int
    frame: dict[str, Any]
    """The scene re-expressed around the ego's new pose."""
    travelled_m: float
    """Cumulative distance the ego has moved since the starting frame."""
    heading_change_rad: float
    """Cumulative heading change since the starting frame."""
    map_coverage_m: float
    """Furthest lane geometry still ahead of the ego, in metres."""
    world_x_m: float
    """Ego x in the starting frame, accumulated across steps."""
    world_y_m: float
    """Ego y in the starting frame, accumulated across steps."""
    world_heading_rad: float
    """Ego heading in the starting frame, accumulated across steps."""
    beyond_map: bool
    """Whether the ego has left the region the starting frame's map covered."""


def map_coverage_m(frame: Mapping[str, Any]) -> float:
    """Distance to the furthest lane point still ahead of the ego.

    Used as the rollout's horizon: once the ego passes it, the starting frame's
    map no longer describes where the ego is.
    """
    lanes = np.asarray(frame.get("lanes"), dtype=np.float32)
    if lanes.size == 0:
        return 0.0
    points = lanes[..., : PoseIndex.COS_YAW].reshape(-1, 2)
    occupied = np.abs(lanes).sum(axis=-1).reshape(-1) > 0.0
    ahead = points[occupied & (points[:, 0] > 0.0)]
    if ahead.size == 0:
        return 0.0
    return float(ahead[:, 0].max())


def _sample(
    model: DiffusionPlanner,
    frame: Mapping[str, Any],
    normalizer: PlannerDataNormalizer,
    device: torch.device,
    *,
    num_steps: int,
    seed: int,
) -> NDArray[np.float32]:
    """Sample one denormalized trajectory set of shape ``(1 + N, T, 4)``."""
    normalized = normalizer({key: np.asarray(v) for key, v in frame.items()})
    input_data = {
        key: torch.as_tensor(value, device=device).unsqueeze(0)
        for key, value in normalized.items()
    }
    neighbor_count = int(normalized["neighbor_agents_past"].shape[0])
    generator = torch.Generator(device=device).manual_seed(seed)
    initial_noise = torch.randn(
        (1, neighbor_count + 1, TRAJECTORY_LENGTH, TRAJECTORY_DIM),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    with torch.inference_mode():
        prediction, _ = model.sample(
            input_data, initial_noise=initial_noise, num_steps=num_steps
        )
    trajectories = prediction[0].detach().float().cpu().numpy()
    return normalizer.denormalize_trajectory(trajectories).astype(np.float32)


def _relative_pose(
    origin: NDArray[np.float32], target: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Express ``target`` in the frame of ``origin``, both being poses."""
    cos_yaw = float(origin[PoseIndex.COS_YAW])
    sin_yaw = float(origin[PoseIndex.SIN_YAW])
    norm = math.hypot(cos_yaw, sin_yaw) or 1.0
    cos_yaw, sin_yaw = cos_yaw / norm, sin_yaw / norm
    delta_x = float(target[PoseIndex.X] - origin[PoseIndex.X])
    delta_y = float(target[PoseIndex.Y] - origin[PoseIndex.Y])
    out = np.zeros_like(target)
    out[PoseIndex.X] = cos_yaw * delta_x + sin_yaw * delta_y
    out[PoseIndex.Y] = -sin_yaw * delta_x + cos_yaw * delta_y
    target_cos = float(target[PoseIndex.COS_YAW])
    target_sin = float(target[PoseIndex.SIN_YAW])
    out[PoseIndex.COS_YAW] = cos_yaw * target_cos + sin_yaw * target_sin
    out[PoseIndex.SIN_YAW] = cos_yaw * target_sin - sin_yaw * target_cos
    return out


def _advance_histories(
    frame: dict[str, Any],
    prediction: NDArray[np.float32],
    index: int,
    offset: int,
    step_distance: float,
    timestep_s: float,
) -> dict[str, Any]:
    """Roll ego and neighbor histories forward by one predicted step.

    ``frame`` is already recentred on the ego's new pose, so poses taken from
    ``prediction`` must be expressed relative to the pose that recentring
    consumed. Between re-plans that is the previous index of the same
    prediction, not its origin, or every agent is placed with coordinates from a
    frame the scene has already left.
    """
    output = dict(frame)

    ego_past = np.array(frame["ego_agent_past"], dtype=np.float32, copy=True)
    current = np.zeros(ego_past.shape[-1], dtype=np.float32)
    current[EgoIndex.COS_YAW] = 1.0
    if ego_past.shape[-1] > EgoIndex.VELOCITY:
        current[EgoIndex.VELOCITY] = step_distance / timestep_s
    output["ego_agent_past"] = np.concatenate((ego_past[1:], current[None, :]), axis=0)

    neighbors = np.array(frame["neighbor_agents_past"], dtype=np.float32, copy=True)
    valid = neighbor_valid(neighbors)
    if offset == 0:
        predicted = prediction[1:, index, : len(NeighborIndex)]
    else:
        origin = prediction[0, index - 1]
        predicted = np.stack(
            [
                _relative_pose(origin, pose)[: len(NeighborIndex)]
                for pose in prediction[1:, index]
            ]
        )
    rolled = np.concatenate((neighbors[:, 1:], neighbors[:, -1:]), axis=1)
    count = min(rolled.shape[0], predicted.shape[0])
    rolled[:count, -1] = np.where(
        valid[:count, None], predicted[:count], rolled[:count, -1]
    )
    rolled[~valid] = 0.0
    output["neighbor_agents_past"] = rolled
    return output


def rollout(
    model: DiffusionPlanner,
    frame: Mapping[str, Any],
    *,
    steps: int,
    device: str | torch.device = "cpu",
    num_sampling_steps: int = 10,
    timestep_s: float = DEFAULT_TIMESTEP_S,
    horizon_index: int = 0,
    replan_every: int = 10,
    seed: int = 0,
    stop_beyond_map: bool = True,
) -> Iterator[RolloutStep]:
    """Advance a scene under the model's own predictions, one step at a time.

    Args:
        model: A planner in eval mode on ``device``.
        frame: The raw, unnormalized starting frame.
        steps: How many closed-loop steps to take.
        device: Torch device.
        num_sampling_steps: Denoising steps per closed-loop step.
        timestep_s: Seconds represented by one closed-loop step.
        horizon_index: Which predicted timestep to advance to, ``0`` being the
            next one.
        replan_every: Steps to follow one sampled trajectory before sampling
            again. Measured on a 13.1 m/s scene over 30 steps, against an ideal
            39.3 m: ``5`` gives 37.7 m, ``10`` gives 38.6 m, ``20`` gives
            38.1 m, ``40`` gives 38.1 m — every cadence within about 3%. But
            ``1`` gives **12.9 m**, the ego decelerating to a near-stop within
            two seconds, because re-planning from a freshly reconstructed
            history every step compounds the mismatch between that history and
            what the model was trained on. Hence the default of 10 rather than
            1; the old architecture kept an MPC tracker for the same reason.
        seed: Base seed; each step derives its own so the rollout is repeatable.
        stop_beyond_map: Stop once the ego leaves the starting frame's map
            coverage. Set ``False`` to keep going and read the flag yourself.

    Yields:
        One :class:`RolloutStep` per step taken.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if not 0 <= horizon_index < TRAJECTORY_LENGTH:
        raise ValueError(f"horizon_index must be 0..{TRAJECTORY_LENGTH - 1}")
    if replan_every < 1:
        raise ValueError("replan_every must be at least 1")
    if horizon_index + replan_every > TRAJECTORY_LENGTH:
        raise ValueError(
            f"horizon_index + replan_every must not exceed {TRAJECTORY_LENGTH}"
        )

    torch_device = torch.device(device)
    normalizer = PlannerDataNormalizer()
    current = {key: np.asarray(value) for key, value in frame.items()}
    coverage = map_coverage_m(current)
    travelled = 0.0
    heading_change = 0.0
    # Pose in the starting frame, composed step by step, so a rollout can be
    # drawn as a path rather than only summarised as a distance.
    world_x, world_y, world_heading = 0.0, 0.0, 0.0

    prediction = np.zeros((0, 0, 0), dtype=np.float32)
    for step in range(steps):
        offset = step % replan_every
        if offset == 0:
            prediction = _sample(
                model,
                current,
                normalizer,
                torch_device,
                num_steps=num_sampling_steps,
                seed=seed + step,
            )
        # Walk further along the same prediction between re-plans, and express
        # it relative to the pose already consumed from it.
        index = horizon_index + offset
        target = prediction[0, index]
        if offset > 0:
            target = _relative_pose(prediction[0, index - 1], target)
        position = target[: PoseIndex.COS_YAW].astype(np.float32)
        heading = target[PoseIndex.COS_YAW : PoseIndex.SIN_YAW + 1].astype(np.float32)

        step_x, step_y = float(position[0]), float(position[1])
        step_heading = float(math.atan2(float(heading[1]), float(heading[0])))
        travelled += float(np.hypot(step_x, step_y))
        heading_change += step_heading
        cos_w, sin_w = math.cos(world_heading), math.sin(world_heading)
        world_x += cos_w * step_x - sin_w * step_y
        world_y += sin_w * step_x + cos_w * step_y
        world_heading += step_heading

        recentred = recenter_frame_to_pose(current, position, heading)
        current = _advance_histories(
            dict(recentred),
            prediction,
            index,
            offset,
            float(np.hypot(position[0], position[1])),
            timestep_s,
        )

        remaining = coverage - travelled
        beyond = remaining <= 0.0
        yield RolloutStep(
            step=step,
            frame=current,
            travelled_m=travelled,
            heading_change_rad=heading_change,
            map_coverage_m=max(remaining, 0.0),
            beyond_map=beyond,
            world_x_m=world_x,
            world_y_m=world_y,
            world_heading_rad=world_heading,
        )
        if beyond and stop_beyond_map:
            return
