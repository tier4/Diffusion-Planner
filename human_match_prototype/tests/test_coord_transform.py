# human_match_prototype/tests/test_coord_transform.py
import math

import numpy as np

from human_match_prototype.coord_transform import (
    WorldPose,
    pose_from_sidecar,
    transform_trajectories,
    transform_trajectory,
)


def _straight(v=5.0, yaw=0.0, T=80, dt=0.1):
    t = np.arange(1, T + 1) * dt
    xy = np.stack([v * t * np.cos(yaw), v * t * np.sin(yaw)], -1)
    return np.concatenate([xy, np.full((T, 1), yaw)], -1)


def test_identity_transform():
    """Same pose -> trajectory unchanged."""
    pose = WorldPose(x=100.0, y=200.0, heading_rad=0.5)
    traj = _straight(5.0, yaw=0.3)
    result = transform_trajectory(traj, from_pose=pose, to_pose=pose)
    np.testing.assert_allclose(result, traj, atol=1e-10)


def test_pure_translation():
    """Poses with same heading but different position -> translation only."""
    from_pose = WorldPose(x=10.0, y=20.0, heading_rad=0.0)
    to_pose = WorldPose(x=15.0, y=20.0, heading_rad=0.0)
    traj = _straight(5.0, yaw=0.0, T=10)
    result = transform_trajectory(traj, from_pose, to_pose)
    np.testing.assert_allclose(result[:, 0], traj[:, 0] - 5.0, atol=1e-10)
    np.testing.assert_allclose(result[:, 1], traj[:, 1], atol=1e-10)
    np.testing.assert_allclose(result[:, 2], traj[:, 2], atol=1e-10)


def test_pure_rotation():
    """Same position, 90-deg heading difference."""
    from_pose = WorldPose(x=0.0, y=0.0, heading_rad=0.0)
    to_pose = WorldPose(x=0.0, y=0.0, heading_rad=math.pi / 2)
    traj = np.array([[1.0, 0.0, 0.0]])  # 1m ahead in from-ego
    result = transform_trajectory(traj, from_pose, to_pose)
    np.testing.assert_allclose(result[0, :2], [0.0, -1.0], atol=1e-10)
    np.testing.assert_allclose(result[0, 2], -math.pi / 2, atol=1e-10)


def test_round_trip():
    """ego->world->ego must be identity for any pose pair."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        p1 = WorldPose(
            rng.uniform(-100, 100), rng.uniform(-100, 100), rng.uniform(-math.pi, math.pi)
        )
        p2 = WorldPose(
            rng.uniform(-100, 100), rng.uniform(-100, 100), rng.uniform(-math.pi, math.pi)
        )
        traj = _straight(rng.uniform(1, 10), yaw=rng.uniform(-1, 1), T=20)
        fwd = transform_trajectory(traj, from_pose=p1, to_pose=p2)
        back = transform_trajectory(fwd, from_pose=p2, to_pose=p1)
        np.testing.assert_allclose(back, traj, atol=1e-9)


def test_pose_from_sidecar():
    sidecar = {
        "npz_path": "/a/b.npz",
        "x": 89135.0,
        "y": 42440.0,
        "heading_deg": 90.0,
        "timestamp": 123,
    }
    pose = pose_from_sidecar(sidecar)
    assert pose.x == 89135.0
    assert pose.y == 42440.0
    assert abs(pose.heading_rad - math.pi / 2) < 1e-10


def test_transform_trajectories_batch():
    """Batch version matches sequential."""
    p1 = WorldPose(10.0, 20.0, 0.3)
    p2 = WorldPose(15.0, 25.0, 0.8)
    p3 = WorldPose(0.0, 0.0, 0.0)
    trajs = [_straight(5.0, T=10), _straight(3.0, yaw=0.1, T=10)]
    from_poses = [p1, p2]
    results = transform_trajectories(trajs, from_poses, to_pose=p3)
    for i in range(2):
        expected = transform_trajectory(trajs[i], from_poses[i], p3)
        np.testing.assert_allclose(results[i], expected, atol=1e-10)
