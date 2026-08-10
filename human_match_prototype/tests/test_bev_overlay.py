"""Smoke test: bev_overlay produces a non-empty PNG without crashing."""

from unittest.mock import MagicMock

import numpy as np


def _fake_npz():
    """Minimal NPZ dict with the keys both render_frame and sampler need."""
    n_lanes = 5
    n_pts = 20
    n_nb = 10
    return {
        "ego_agent_past": np.random.randn(31, 3).astype(np.float32),
        "ego_agent_future": np.random.randn(80, 3).astype(np.float32),
        "neighbor_agents_past": np.zeros((n_nb, 31, 11), dtype=np.float32),
        "neighbor_agents_future": np.zeros((n_nb, 80, 3), dtype=np.float32),
        "lanes": np.zeros((n_lanes, n_pts, 33), dtype=np.float32),
        "route_lanes": np.zeros((n_lanes, n_pts, 33), dtype=np.float32),
        "line_strings": np.zeros((n_lanes, n_pts, 4), dtype=np.float32),
        "polygons": np.zeros((n_lanes, n_pts, 2), dtype=np.float32),
        "ego_shape": np.array([2.75, 4.34, 1.70], dtype=np.float32),
        "goal_pose": np.array([50.0, 0.0, 0.0], dtype=np.float32),
    }


def _fake_sample_result():
    from human_match_prototype.sampler import SampleResult

    return SampleResult(
        ego_samples=np.random.randn(4, 80, 3).astype(np.float32),
        embedding=np.random.randn(256).astype(np.float32),
        human_future=np.random.randn(80, 3).astype(np.float32),
    )


def test_bev_overlay_produces_png(tmp_path):
    from human_match_prototype.render_report import render_scene_overlay

    npz_path = tmp_path / "test.npz"
    np.savez(npz_path, **_fake_npz())

    out_png = tmp_path / "overlay.png"
    sampler = MagicMock()
    sampler.sample.return_value = _fake_sample_result()

    render_scene_overlay(sampler, str(npz_path), out_png, num_samples=4)

    assert out_png.exists()
    assert out_png.stat().st_size > 1000  # non-trivial PNG
    sampler.sample.assert_called_once_with(str(npz_path), num_samples=4, seed=0, temperature=1.0)


def test_render_scene_overlay_signature():
    """render_scene_overlay should accept seed and temperature kwargs."""
    import inspect

    from human_match_prototype.render_report import render_scene_overlay

    sig = inspect.signature(render_scene_overlay)
    assert "seed" in sig.parameters
    assert "temperature" in sig.parameters
