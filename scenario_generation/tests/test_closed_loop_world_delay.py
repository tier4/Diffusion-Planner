from types import SimpleNamespace

import numpy as np

from scenario_generation import reproducer_rollout as rollout


def _track(value: float, *, vx: float = 0.0, vy: float = 0.0) -> np.ndarray:
    track = np.zeros((3, 11), dtype=np.float32)
    for history_idx in range(track.shape[0]):
        track[history_idx] = [
            value + history_idx,
            -value - history_idx,
            1.0,
            0.0,
            vx,
            vy,
            value + 0.1,
            value + 0.2,
            1.0,
            0.0,
            0.0,
        ]
    return track


def test_clock_world_indices_lag_and_clamp_to_session_bounds():
    assert rollout._clock_world_indices(10, 0, 20, "lag_raw", 3) == (10, 10)
    assert rollout._clock_world_indices(10, 2, 20, "lag_raw", 3) == (12, 10)
    assert rollout._clock_world_indices(10, 4, 20, "lag_raw", 3) == (14, 11)
    assert rollout._clock_world_indices(10, 99, 20, "lag_raw", 3) == (19, 16)
    assert rollout._clock_world_indices(10, 4, 20, "none", 0) == (14, 14)


def test_neighbor_lag_realigns_the_complete_track_by_uuid():
    source = np.zeros((1, 4, 3, 11), dtype=np.float32)
    source[0, 0] = _track(10.0, vx=1.0)
    source[0, 1] = _track(20.0, vy=2.0)
    source[0, 2] = _track(30.0, vx=3.0)

    aligned = rollout._realign_neighbor_lag(
        source,
        lag_ids=["alpha", "bravo", "alpha", ""],
        truth_ids=["bravo", "alpha", "missing", "bravo"],
    )

    np.testing.assert_array_equal(aligned[0, 0], source[0, 1])
    np.testing.assert_array_equal(aligned[0, 1], source[0, 0])
    np.testing.assert_array_equal(aligned[0, 2], 0.0)
    np.testing.assert_array_equal(aligned[0, 3], 0.0)


def test_neighbor_lag_extrapolates_every_valid_history_row_only():
    source = np.zeros((1, 1, 3, 11), dtype=np.float32)
    source[0, 0, 0] = [1.0, 2.0, 1.0, 0.0, 3.0, -2.0, 2.1, 4.2, 1.0, 0.0, 0.0]
    source[0, 0, 1] = [4.0, 6.0, 0.0, 1.0, -1.0, 5.0, 2.2, 4.3, 0.0, 1.0, 0.0]

    extrapolated = rollout._extrapolate_neighbor_lag(source, 0.3)

    np.testing.assert_allclose(extrapolated[0, 0, 0, :2], [1.9, 1.4])
    np.testing.assert_allclose(extrapolated[0, 0, 1, :2], [3.7, 7.5])
    np.testing.assert_array_equal(extrapolated[0, 0, :, 2:], source[0, 0, :, 2:])
    np.testing.assert_array_equal(extrapolated[0, 0, 2], 0.0)
    np.testing.assert_array_equal(source[0, 0, 0, :2], [1.0, 2.0])


class _FakeTimeline:
    def __init__(self):
        self.frames = {
            2: {"frame": 2},
            5: {"frame": 5},
        }
        self.ids = {
            2: ["bravo", "alpha"],
            5: ["alpha", "bravo"],
        }

    def npz(self, idx):
        return self.frames[idx]

    def neighbor_ids(self, idx):
        return self.ids[idx]


def _fake_model_base(npz):
    frame = npz["frame"]
    neighbors = np.zeros((1, 2, 3, 11), dtype=np.float32)
    if frame == 2:
        neighbors[0, 0] = _track(20.0, vx=2.0)
        neighbors[0, 1] = _track(10.0, vx=1.0)
    else:
        neighbors[0, 0] = _track(100.0)
        neighbors[0, 1] = _track(200.0)
    lanes = np.zeros((1, 1, 1, 13), dtype=np.float32)
    lanes[..., 8:13] = frame
    return {"neighbor_agents_past": neighbors, "lanes": lanes}


def test_world_input_uses_raw_lag_and_keeps_traffic_signal_lagged(monkeypatch):
    monkeypatch.setattr(rollout, "_npz_to_model_base", _fake_model_base)
    tl = _FakeTimeline()

    raw = rollout._world_input_base(tl, 2, 5, "lag_raw", 3)
    extrapolated = rollout._world_input_base(tl, 2, 5, "lag_extrapolated", 3)

    # Current truth slot order is alpha, bravo; each complete lag track follows its UUID.
    np.testing.assert_array_equal(raw["neighbor_agents_past"][0, 0], _track(10.0, vx=1.0))
    np.testing.assert_array_equal(raw["neighbor_agents_past"][0, 1], _track(20.0, vx=2.0))
    np.testing.assert_allclose(
        extrapolated["neighbor_agents_past"][0, 0, :, 0],
        _track(10.0, vx=1.0)[:, 0] + 0.3,
    )
    np.testing.assert_allclose(
        extrapolated["neighbor_agents_past"][0, 1, :, 0],
        _track(20.0, vx=2.0)[:, 0] + 0.6,
    )
    # Traffic-light state lives in lane attributes and is intentionally not extrapolated.
    np.testing.assert_array_equal(raw["lanes"], extrapolated["lanes"])
    np.testing.assert_array_equal(extrapolated["lanes"][..., 8:13], 2.0)


def test_pre_step_keeps_metrics_on_truth_clock(monkeypatch):
    calls = []

    def fake_world_base(_tl, world_idx, truth_idx, mode, k_lag):
        calls.append(("world", world_idx, truth_idx, mode, k_lag))
        return {"source": "lag"}

    def fake_build(_tl, idx, _pose, _history, _dyn, *, base=None):
        calls.append(("build", idx, None if base is None else base["source"]))
        scene = {
            "neighbor_agents_past": np.zeros((1, 1, 1, 11), dtype=np.float32),
        }
        return scene, scene["neighbor_agents_past"][0, :, -1, :].copy()

    monkeypatch.setattr(rollout, "_world_input_base", fake_world_base)
    monkeypatch.setattr(rollout, "build_input_np", fake_build)
    cursor = SimpleNamespace(
        max_idx_reached=10,
        _update_base_state=lambda *, repeat: None,
    )
    state = SimpleNamespace(
        done=False,
        k=2,
        max_steps=20,
        live_pose=np.array([0.0, 0.0, 0.0]),
        goal_xy=np.array([100.0, 0.0]),
        goal_reach_m=1.0,
        replay_mode="clock",
        start=10,
        end=30,
        world_delay_mode="lag_raw",
        k_lag=3,
        cursor=cursor,
        prev_max_idx=10,
        stuck=0,
        max_stuck_steps=0,
        tl=object(),
        sim_time=0.2,
        dyn=SimpleNamespace(speed=1.0),
        ego_hist=np.zeros((31, 3), dtype=np.float32),
        nbr_tracker=None,
        turn_hist=np.zeros(31, dtype=np.int64),
    )

    _model_scene, _neighbors, truth_idx, _ids, _world = rollout._pre_step(state)

    assert truth_idx == 12
    assert state.clock_idx == 12
    assert state.world_idx == 10
    assert calls == [
        ("world", 10, 12, "lag_raw", 3),
        ("build", 12, None),
        ("build", 10, "lag"),
    ]
