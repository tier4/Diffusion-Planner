"""Scene-generation futures are ALWAYS 4-col [x,y,cos,sin], never 3-col [x,y,heading].

Covers the extractor's widening/recenter helpers and that load_npz_data tolerates the
string `origin` metadata the extractor writes.
"""

import json

import numpy as np

from scenario_generation.reproducer_rollout import _future_to_4col, _recenter_neighbor_future


def test_future_to_4col_widens_and_passes_through():
    # 3-col [x,y,heading] -> 4-col [x,y,cos,sin]
    a = np.array([[[1.0, 2.0, 0.0], [3.0, 4.0, np.pi / 2]]], dtype=np.float32)  # (1,2,3)
    out = _future_to_4col(a)
    assert out.shape == (1, 2, 4)
    assert np.allclose(out[0, 0], [1, 2, 1, 0], atol=1e-6)  # heading 0 -> cos1,sin0
    assert np.allclose(out[0, 1], [3, 4, 0, 1], atol=1e-6)  # heading pi/2 -> cos0,sin1
    # 4-col passes through unchanged
    b = np.array([[[1.0, 2.0, 0.7, 0.7]]], dtype=np.float32)
    assert np.array_equal(_future_to_4col(b), b)


def test_future_to_4col_zero_rows_stay_zero():
    a = np.zeros((1, 3, 3), dtype=np.float32)
    a[0, 0] = [5.0, 0.0, 1.2]  # one valid row; others zero
    out = _future_to_4col(a)
    assert out.shape[-1] == 4
    assert np.all(out[0, 1] == 0) and np.all(out[0, 2] == 0)  # invalid rows stay zero
    assert not np.all(out[0, 0] == 0)


def test_recenter_neighbor_future_is_4col():
    # one neighbor, one timestep at (5,0) heading 0 in recorded frame
    naf3 = np.zeros((320, 1, 3), dtype=np.float32)
    naf3[0, 0] = [5.0, 0.0, 0.0]
    out = _recenter_neighbor_future(naf3, dx=0.0, dy=0.0, dyaw=0.0)
    assert out.shape == (320, 1, 4)  # NEVER 3-col
    assert np.allclose(out[0, 0], [5, 0, 1, 0], atol=1e-5)
    # +90deg live-ego yaw: neighbor (5,0) -> (0,-5), heading (cos,sin)=(0,-1)
    out2 = _recenter_neighbor_future(naf3, dx=0.0, dy=0.0, dyaw=np.pi / 2)
    assert np.allclose(out2[0, 0, :2], [0, -5], atol=1e-5)
    assert np.allclose(out2[0, 0, 2:4], [0, -1], atol=1e-5)
    # 4-col input also yields 4-col output
    naf4 = np.zeros((320, 1, 4), dtype=np.float32)
    naf4[0, 0] = [5.0, 0.0, 1.0, 0.0]
    assert _recenter_neighbor_future(naf4, 0.0, 0.0, 0.0).shape == (320, 1, 4)


def test_load_npz_data_skips_origin_string(tmp_path):
    import torch

    from preference_optimization.utils import load_npz_data

    p = tmp_path / "scene.npz"
    np.savez(
        p,
        ego_agent_past=np.zeros((31, 3), np.float32),
        ego_shape=np.array([4.76, 7.24, 2.29], np.float32),
        neighbor_agents_future=np.zeros((320, 80, 4), np.float32),
        origin=np.array("live"),  # string metadata the extractor writes
    )
    data = load_npz_data(p, torch.device("cpu"))  # must not crash on the string key
    assert "origin" not in data
    assert "neighbor_agents_future" in data and data["neighbor_agents_future"].shape[-1] == 4
