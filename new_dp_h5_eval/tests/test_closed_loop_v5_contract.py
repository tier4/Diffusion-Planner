"""Contract checks for native H5 route windows and poses."""

import json

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_dp_h5_eval.closed_loop import NativeH5RouteTimeline
from new_dp_h5_eval.dataset import H5FrameIndex
from new_dp_h5_eval.run_all_groups_closed_loop import _load_groups
from new_dp_h5_eval.schema import H5_FORMAT, MODEL_INPUT_NAMES


def _write_route(path, *, times=None, interval=0.1, frame="map", qw=1.0, version=5, yaw=None):
    times = times if times is not None else [1_000_000_000 + i * 100_000_000 for i in range(4)]
    count = len(times)
    yaw = np.zeros(count) if yaw is None else np.asarray(yaw)
    with h5py.File(path, "w") as file:
        file.attrs["format"] = H5_FORMAT
        file.attrs["format_version"] = version
        file.attrs["num_frames"] = count
        file.attrs["frame_interval_s"] = interval
        file.attrs["pose_frame_id"] = frame
        frames = file.create_group("frames")
        for name in MODEL_INPUT_NAMES:
            shape = {
                "ego_agent_past": (count, 31, 6),
                "neighbor_agents_past": (count, 320, 31, 4),
                "agent_shape": (count, 320, 2),
                "agent_label": (count, 320, 3),
            }.get(name, (count, 1))
            frames.create_dataset(name, data=np.zeros(shape, dtype=np.float32))
        metadata = file.create_group("metadata")
        metadata.create_dataset("frame_time_ns", data=times)
        if version == 4:
            metadata.create_dataset("ego_x", data=np.arange(count, dtype=np.float64))
            metadata.create_dataset("ego_y", data=np.zeros(count))
            metadata.create_dataset("ego_yaw", data=yaw)
            return
        for name, values in {
            "x": np.arange(count, dtype=np.float64),
            "y": np.zeros(count),
            "z": np.zeros(count),
            "qx": np.zeros(count),
            "qy": np.zeros(count),
            "qz": np.sin(yaw / 2),
            "qw": np.cos(yaw / 2) * qw,
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


@pytest.mark.parametrize("version", [4, 5])
@pytest.mark.parametrize(
    "deltas_ns",
    [
        [54_000_000, 146_000_000, 100_000_000],
        [100_000_000, 2_000_000_000, 100_000_000],
        [],
    ],
)
def test_accepts_jitter_and_gaps_without_adding_frames(tmp_path, version, deltas_ns):
    times = 1_000_000_000 + np.cumsum([0, *deltas_ns], dtype=np.int64)
    shard = tmp_path / "frames.h5"
    _write_route(shard, version=version, times=times)
    timeline = NativeH5RouteTimeline(shard)
    try:
        np.testing.assert_array_equal(timeline.frame_times_ns, times)
        assert len(timeline.frame_times_ns) == len(times)
        assert np.isfinite(timeline.speeds).all()
    finally:
        timeline.close()


@pytest.mark.parametrize("version", [4, 5])
@pytest.mark.parametrize(
    ("delta_ns", "message"),
    [
        (0, "non-increasing"),
        (-1, "non-increasing"),
    ],
)
def test_rejects_non_increasing_timestamps(tmp_path, version, delta_ns, message):
    shard = tmp_path / "frames.h5"
    _write_route(
        shard, version=version, times=[1_000_000_000, 1_100_000_000, 1_100_000_000 + delta_ns]
    )
    with pytest.raises(ValueError, match=message):
        NativeH5RouteTimeline(shard)
    # Validation applies to the selected window, not invalid rows elsewhere in the shard.
    timeline = NativeH5RouteTimeline(shard, 0, 2)
    timeline.close()


@pytest.mark.parametrize("version", [4, 5])
def test_pose_and_indexed_reads(tmp_path, version):
    shard = tmp_path / "frames.h5"
    _write_route(shard, version=version, times=[1_000_000_000, 1_100_000_000], yaw=[0, np.pi / 2])
    timeline = NativeH5RouteTimeline(shard)
    try:
        np.testing.assert_allclose(timeline.poses, [[0, 0, 0], [1, 0, np.pi / 2]])
    finally:
        timeline.close()
    index = tmp_path / "index.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"h5_path": shard.name, "frame_index": 0, "frame_time_ns": 1_000_000_000}]
        ),
        index,
    )
    with H5FrameIndex(index) as frames:
        assert frames.frame(0)["ego_agent_past"].shape == (31, 6)
