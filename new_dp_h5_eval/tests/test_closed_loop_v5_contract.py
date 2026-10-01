"""Small contract checks for v5 route windows and their NPZ timeline equivalent."""

import json

import h5py
import numpy as np
import pytest

from new_dp_h5_eval.closed_loop import NativeH5RouteTimeline
from new_dp_h5_eval.run_all_groups_closed_loop import _load_groups
from new_dp_h5_eval.schema import H5_FORMAT, MODEL_INPUT_NAMES
from scenario_generation.route_timeline import RouteTimeline


def _write_route(path, *, times=None, interval=0.1, frame="map", qw=1.0):
    times = times if times is not None else [1_000_000_000 + i * 100_000_000 for i in range(4)]
    count = len(times)
    with h5py.File(path, "w") as file:
        file.attrs["format"] = H5_FORMAT
        file.attrs["format_version"] = 5
        file.attrs["num_frames"] = count
        file.attrs["frame_interval_s"] = interval
        file.attrs["pose_frame_id"] = frame
        frames = file.create_group("frames")
        for name in MODEL_INPUT_NAMES:
            shape = {
                "ego_agent_past": (count, 31, 6),
                "neighbor_agents_past": (count, 320, 31, 4),
            }.get(name, (count, 1))
            frames.create_dataset(name, data=np.zeros(shape, dtype=np.float32))
        metadata = file.create_group("metadata")
        metadata.create_dataset("frame_time_ns", data=times)
        for name, values in {
            "x": np.arange(count, dtype=np.float64),
            "y": np.zeros(count),
            "z": np.zeros(count),
            "qx": np.zeros(count),
            "qy": np.zeros(count),
            "qz": np.zeros(count),
            "qw": np.full(count, qw),
        }.items():
            metadata.create_dataset(name, data=values)


def test_manifest_distinguishes_windows_and_passes_selection_metadata(tmp_path):
    shard = tmp_path / "frames.h5"
    _write_route(shard)
    manifest = tmp_path / "routes.json"
    entries = [
        {
            "h5_path": shard.name,
            "frame_start": start,
            "frame_stop": stop,
            "segment_start_ns": 1_000_000_000 + start * 100_000_000,
            "segment_end_ns": 1_000_000_000 + (stop - 1) * 100_000_000,
            "anchors": [{"eval_label": "departure", "timestamp": 1_100_000_000}],
        }
        for start, stop in ((0, 2), (2, 4))
    ]
    manifest.write_text(json.dumps({"departure": entries}))
    routes = _load_groups(manifest)["departure"]
    assert len({route["route_id"] for route in routes}) == 2
    assert routes[0]["anchors"] == entries[0]["anchors"]
    assert routes[1]["segment_start_ns"] == entries[1]["segment_start_ns"]
    manifest.write_text(
        json.dumps({"departure": [entries[0], {**entries[0], "h5_path": str(shard)}]})
    )
    with pytest.raises(ValueError, match="duplicate H5 route identities"):
        _load_groups(manifest)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"times": [1_000_000_000, 1_100_000_000, 3_100_000_000]}, "non-contiguous"),
        ({"interval": 0.5}, "0.1 s frame interval"),
        ({"frame": "odom"}, "map frame"),
        ({"qw": 1e-8}, "invalid closed-loop v5 pose"),
    ],
)
def test_rejects_invalid_closed_loop_contract(tmp_path, changes, message):
    shard = tmp_path / "frames.h5"
    _write_route(shard, **changes)
    with pytest.raises(ValueError, match=message):
        NativeH5RouteTimeline(shard)


def test_h5_and_npz_timeline_match_on_two_windows(tmp_path):
    shard = tmp_path / "frames.h5"
    _write_route(shard)
    paths = []
    for index in range(4):
        path = tmp_path / f"route_{index:08d}.npz"
        np.savez(
            path,
            ego_agent_past=np.zeros((31, 6), dtype=np.float32),
            neighbor_agents_past=np.zeros((320, 31, 4), dtype=np.float32),
        )
        path.with_suffix(".json").write_text(
            json.dumps({"x": float(index), "y": 0.0, "qx": 0, "qy": 0, "qz": 0, "qw": 1})
        )
        paths.append(path)
    for start, stop in ((0, 2), (2, 4)):
        with_h5 = NativeH5RouteTimeline(shard, start, stop)
        with_npz = RouteTimeline(paths[start:stop])
        try:
            np.testing.assert_allclose(with_h5.poses, with_npz.poses)
            np.testing.assert_allclose(with_h5.speeds, with_npz.speeds)
            for index in range(stop - start):
                for name in ("ego_agent_past", "neighbor_agents_past"):
                    np.testing.assert_array_equal(
                        with_h5.npz(index)[name], with_npz.npz(index)[name]
                    )
        finally:
            with_h5.close()
