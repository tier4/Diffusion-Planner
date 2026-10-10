"""Recorded speeds and frame times for NPZ routes named by frame index or by timestamp."""

import json

import numpy as np
import pytest

from scenario_generation.reproducer_rollout import DT, _gt_accels
from scenario_generation.route_timeline import (
    _NS_TIMESTAMP_MIN,
    RouteTimeline,
    _frame_times_s,
    _uses_ns_timestamps,
)

T0_NS = 1_776_387_369_838_338_599  # a real v0.0.2 frame name: 09-42-22_1776387369838338599


def _write_route(route_dir, names, xs):
    """Write empty NPZs plus pose sidecars; RouteTimeline only reads the sidecars up front."""
    route_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, x in zip(names, xs):
        npz = route_dir / f"{name}.npz"
        npz.write_bytes(b"")
        npz.with_suffix(".json").write_text(
            json.dumps({"x": float(x), "y": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0})
        )
        paths.append(npz)
    return paths


def test_timestamp_names_give_recorded_speeds(tmp_path):
    # 1 m per 100 ms, then a dropped frame (200 ms) with 2 m.
    stamps = [T0_NS, T0_NS + 100_000_000, T0_NS + 200_000_000, T0_NS + 400_000_000]
    tl = RouteTimeline(_write_route(tmp_path, [f"09-42-22_{s}" for s in stamps], [0, 1, 2, 4]))
    np.testing.assert_allclose(tl.frame_times_s, [0.0, 0.1, 0.2, 0.4], atol=1e-9)
    np.testing.assert_allclose(tl.speeds, [10.0, 10.0, 10.0, 10.0], rtol=1e-6)


def test_timestamp_names_keep_jitter(tmp_path):
    # 99 ms and 101 ms steps: 0.99 m / 0.099 s and 1.01 m / 0.101 s are both 10 m/s.
    stamps = [T0_NS, T0_NS + 99_000_000, T0_NS + 200_000_000]
    tl = RouteTimeline(_write_route(tmp_path, [f"09-42-22_{s}" for s in stamps], [0, 0.99, 2]))
    np.testing.assert_allclose(tl.frame_times_s, [0.0, 0.099, 0.2], atol=1e-9)
    np.testing.assert_allclose(tl.speeds, [10.0, 10.0, 10.0], rtol=1e-6)


def test_index_names_are_bit_identical_to_before(tmp_path):
    # Zero-padded frame indices with a one-frame gap and a repeated index.
    indices = [31, 32, 34, 34, 35]
    xs = [0.0, 1.0, 3.0, 3.0, 4.5]
    names = [f"13-42-45_{i:016d}" for i in indices]
    paths = _write_route(tmp_path / "a", names[:3], xs[:3]) + _write_route(
        tmp_path / "b", names[3:], xs[3:]
    )
    tl = RouteTimeline(paths)
    # The formula used before timestamp names were supported.
    didx = np.maximum(np.diff(tl.frame_indices), 1).astype(np.float64)
    old = np.linalg.norm(np.diff(tl.poses[:, :2], axis=0), axis=1) / (didx * 0.1)
    np.testing.assert_array_equal(tl.speeds[:-1], old)
    np.testing.assert_array_equal(tl.frame_times_s, tl.frame_indices.astype(np.float64) * DT)


def test_parse_rosbag_sequences_in_one_bag_are_unchanged(tmp_path):
    # parse_rosbag.py names frames "<bag>_<seq_id:08d><frame:08d>", so a second sequence of the
    # same bag continues the route ~1e8 "indices" later. That single jump must not be rejected.
    names = [f"13-42-45_{seq:08d}{frame:08d}" for seq, frame in [(0, 31), (0, 32), (0, 33)]]
    names += [f"13-42-45_{seq:08d}{frame:08d}" for seq, frame in [(1, 31), (1, 32)]]
    tl = RouteTimeline(_write_route(tmp_path, names, [0.0, 1.0, 2.0, 50.0, 51.0]))
    didx = np.maximum(np.diff(tl.frame_indices), 1).astype(np.float64)
    old = np.linalg.norm(np.diff(tl.poses[:, :2], axis=0), axis=1) / (didx * 0.1)
    np.testing.assert_array_equal(tl.speeds[:-1], old)
    assert tl.speeds[2] < 1e-5  # the sequence boundary, as before


def test_repeated_timestamp_names_count_as_one_frame(tmp_path):
    stamps = [T0_NS, T0_NS + 100_000_000]
    paths = _write_route(tmp_path / "a", [f"r_{s}" for s in stamps], [0, 1]) + _write_route(
        tmp_path / "b", [f"r_{stamps[1]}"], [1.5]
    )
    tl = RouteTimeline(paths)
    np.testing.assert_allclose(tl.speeds[:2], [10.0, 5.0], rtol=1e-6)


def test_single_frame_route(tmp_path):
    tl = RouteTimeline(_write_route(tmp_path, [f"r_{T0_NS}"], [0]))
    np.testing.assert_array_equal(tl.frame_times_s, [0.0])
    np.testing.assert_array_equal(tl.speeds, [0.0])


def test_threshold_between_indices_and_timestamps():
    assert not _uses_ns_timestamps(np.array([_NS_TIMESTAMP_MIN - 1]))
    assert _uses_ns_timestamps(np.array([_NS_TIMESTAMP_MIN]))
    np.testing.assert_array_equal(_frame_times_s(np.array([_NS_TIMESTAMP_MIN])), [0.0])


def test_mixed_names_are_rejected_with_the_route(tmp_path):
    paths = _write_route(tmp_path / "route", ["r_0000000000000031", f"r_{T0_NS}"], [0, 1])
    with pytest.raises(ValueError, match=r"route: frame names mix frame indices"):
        RouteTimeline(paths)


def test_unknown_clock_names_are_not_read_as_indices():
    # Nanoseconds of a clock that starts near zero would look like huge index steps.
    with pytest.raises(ValueError, match="look like timestamps of an unknown clock"):
        _frame_times_s(np.array([5_000_000_000, 5_100_000_000]))


def test_gt_accels_do_not_depend_on_the_naming(tmp_path):
    xs = np.cumsum(np.linspace(0.5, 1.5, 12))  # accelerating
    by_index = RouteTimeline(_write_route(tmp_path / "i", [f"r_{i:016d}" for i in range(12)], xs))
    by_stamp = RouteTimeline(
        _write_route(tmp_path / "s", [f"r_{T0_NS + i * 100_000_000}" for i in range(12)], xs)
    )
    np.testing.assert_allclose(by_stamp.speeds, by_index.speeds, rtol=1e-6)
    np.testing.assert_allclose(_gt_accels(by_stamp), _gt_accels(by_index), rtol=1e-5, atol=1e-6)
    assert np.abs(_gt_accels(by_index)[2:-2]).max() > 1.0  # really accelerating (~9 m/s^2)
