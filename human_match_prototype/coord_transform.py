"""SE(2) coordinate transforms between ego-centric frames using world poses."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WorldPose:
    x: float
    y: float
    heading_rad: float


def pose_from_sidecar(sidecar: dict) -> WorldPose:
    return WorldPose(
        x=sidecar["x"],
        y=sidecar["y"],
        heading_rad=math.radians(sidecar["heading_deg"]),
    )


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def transform_trajectory(traj: np.ndarray, from_pose: WorldPose, to_pose: WorldPose) -> np.ndarray:
    """Transform (T, 3) trajectory [x, y, yaw] from from_pose's ego frame to to_pose's ego frame.

    Combined: p_to = R(h_from - h_to) @ p_from + R(-h_to) @ ([x_from, y_from] - [x_to, y_to])
    """
    dh = from_pose.heading_rad - to_pose.heading_rad
    R_dh = _rotation_matrix(dh)
    R_neg_to = _rotation_matrix(-to_pose.heading_rad)
    translation = R_neg_to @ np.array([from_pose.x - to_pose.x, from_pose.y - to_pose.y])

    result = np.empty_like(traj)
    result[:, :2] = (R_dh @ traj[:, :2].T).T + translation
    result[:, 2] = traj[:, 2] + dh
    return result


def transform_trajectories(
    trajs: list[np.ndarray],
    from_poses: list[WorldPose],
    to_pose: WorldPose,
) -> list[np.ndarray]:
    return [transform_trajectory(t, fp, to_pose) for t, fp in zip(trajs, from_poses)]
