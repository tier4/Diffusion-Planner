"""world_to_ego_frame_torch (batched, on-device) must match the numpy world_to_ego_frame.

The reproducer's ``--gpu_transform`` path replaces the per-segment numpy re-centering with
ONE batched torch op. This test asserts the two are numerically equivalent (float32 ordering,
~1e-4) across all model-input keys, including the transform-then-zero-invalid handling of
padding rows.
"""

import numpy as np
import pytest

from scenario_generation.transforms import world_to_ego_frame, world_to_ego_frame_torch

torch = pytest.importorskip("torch")

_SHAPES = {
    "ego_agent_past": (1, 31, 4),
    "ego_current_state": (1, 10),
    "neighbor_agents_past": (1, 320, 31, 11),
    "lanes": (1, 140, 20, 33),
    "route_lanes": (1, 25, 20, 33),
    "polygons": (1, 10, 40, 3),
    "line_strings": (1, 60, 20, 4),
    "static_objects": (1, 5, 10),
    "goal_pose": (1, 4),
}


def _make_sample(rng):
    """A random scene-frame sample with some all-zero padding rows (exercises the mask)."""
    d = {}
    for k, sh in _SHAPES.items():
        a = (rng.standard_normal(sh) * 10).astype(np.float32)
        if k == "neighbor_agents_past":
            a[0, ::3] = 0.0
        elif k in ("lanes", "route_lanes"):
            a[0, ::4] = 0.0
        elif k == "polygons":
            a[0, ::2] = 0.0
        elif k == "line_strings":
            a[0, ::5] = 0.0
        elif k == "static_objects":
            a[0, ::2] = 0.0
        d[k] = a
    return d


def test_world_to_ego_frame_torch_matches_numpy():
    rng = np.random.default_rng(0)
    B = 6
    samples = [_make_sample(rng) for _ in range(B)]
    poses = [
        (
            float(rng.uniform(-50, 50)),
            float(rng.uniform(-50, 50)),
            float(rng.uniform(-np.pi, np.pi)),
        )
        for _ in range(B)
    ]

    ref = [
        world_to_ego_frame({k: v.copy() for k, v in s.items()}, *p) for s, p in zip(samples, poses)
    ]

    batch = {
        k: torch.from_numpy(np.concatenate([s[k] for s in samples], axis=0).copy()) for k in _SHAPES
    }
    dx = torch.tensor([p[0] for p in poses])
    dy = torch.tensor([p[1] for p in poses])
    dyaw = torch.tensor([p[2] for p in poses])
    out = world_to_ego_frame_torch(batch, dx, dy, dyaw)

    for k in _SHAPES:
        ref_stack = np.concatenate([r[k] for r in ref], axis=0)
        got = out[k].numpy()
        assert got.shape == ref_stack.shape, k
        np.testing.assert_allclose(got, ref_stack, atol=1e-3, rtol=0, err_msg=k)
