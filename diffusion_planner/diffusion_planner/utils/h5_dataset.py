"""Train from the new-architecture H5 frame dataset without changing the model.

The new architecture (``new-architecture/main``) stores one ``frames.h5`` shard per rosbag
plus a split-level Parquet frame index, documented in ``docs/h5_dataset_schema.md`` of that
branch. Its scene dimensions are identical to the ones this branch's model was built for --
31 past steps, 80 future steps, 320 neighbours, 140 lanes / 25 route lanes of 20 points, 10
intersection polygons of 40 points, 60 line strings of 20 points -- but the tensors are
factored differently: boundary types and traffic lights live in their own arrays, neighbour
shape and class are separate from the neighbour tracks, and the ego pose carries cos/sin
instead of a raw yaw.

So the H5 layout is *re-assembled* here into the canonical NPZ layout the rest of this branch
already consumes, rather than teaching the encoder a second input contract. That keeps the
model, the ONNX export, the ROS 2 node and existing checkpoints untouched: the only thing that
changes is where a training batch comes from.

Fields H5 does not carry are emitted as all-zero tensors (``static_objects``) or as zeros in
the unused slots (neighbour vx/vy, ego lateral velocity / acceleration / steering). All-zero
is the padding convention on both sides, and ``ObservationNormalizer`` restores exact zeros
after normalizing, so a padded row stays padded end to end. Every attribute written onto a
geometry row is therefore masked by that row's own validity -- writing a type one-hot onto a
padded lane point would make the point look valid to the encoder.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401  (registers the zstd filter used by the shards)
import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray
from torch.utils.data import Dataset

from diffusion_planner import dimensions as dim

H5_FORMAT = "diffusion_planner_frame_dataset"
H5_FORMAT_VERSION = 4
REQUIRED_INDEX_COLUMNS = ("h5_path", "frame_index")

# Frame tensors this converter reads. A shard missing any of them is rejected up front
# instead of failing inside a DataLoader worker.
REQUIRED_H5_KEYS = (
    "ego_agent_past",
    "ego_agent_future",
    "neighbor_agents_past",
    "neighbor_agents_future",
    "agent_shape",
    "agent_label",
    "lanes",
    "lane_types",
    "lanes_speed_limit",
    "lane_traffic_light_past",
    "route_lanes",
    "route_lane_types",
    "route_lanes_speed_limit",
    "route_traffic_light_past",
    "intersection_area",
    "stop_lines",
    "road_borders",
    "goal_pose",
    "ego_shape",
    "turn_indicators",
)

# H5 traffic-light channels: [green, amber, red, unknown_or_unavailable, white_or_no_light,
# arrow_flag]. This branch's 5-wide encoding is [green, yellow, red, white, no_traffic_light].
# "white" has no separate H5 channel (it is folded into white_or_no_light), and the arrow flag
# has no slot here at all, so both H5 "unknown" and "white or no light" land on
# no_traffic_light and the arrow flag is dropped.
_H5_TRAFFIC_LIGHT_GREEN = 0
_H5_TRAFFIC_LIGHT_AMBER = 1
_H5_TRAFFIC_LIGHT_RED = 2
_H5_TRAFFIC_LIGHT_UNKNOWN = 3
_H5_TRAFFIC_LIGHT_WHITE_OR_NONE = 4

_NEIGHBOR_WIDTH_INDEX = 6
_NEIGHBOR_LENGTH_INDEX = 7
_NEIGHBOR_TYPE_START = 8


def _pose_yaw(pose: NDArray[Any]) -> NDArray[np.float32]:
    """Convert a trailing [x, y, cos, sin, ...] pose to the canonical [x, y, yaw] layout.

    ``StatePerturbation.interpolation_future_trajectory`` reads column 2 of the ego future as a
    raw heading angle, so the ego future must be handed over in the 3-wide layout even though
    H5 stores cos/sin. An all-zero padded row maps to an all-zero row because atan2(0, 0) is 0.
    """
    yaw = np.arctan2(pose[..., 3], pose[..., 2])
    return np.concatenate([pose[..., :2], yaw[..., None]], axis=-1).astype(np.float32)


def _traffic_light_to_canonical(state: NDArray[Any]) -> NDArray[np.float32]:
    """Map the trailing H5 6-wide traffic-light encoding onto this branch's 5-wide one."""
    out = np.zeros((*state.shape[:-1], dim.TRAFFIC_LIGHT_ONE_HOT_DIM), dtype=np.float32)
    out[..., 0] = state[..., _H5_TRAFFIC_LIGHT_GREEN]
    out[..., 1] = state[..., _H5_TRAFFIC_LIGHT_AMBER]
    out[..., 2] = state[..., _H5_TRAFFIC_LIGHT_RED]
    # index 3 (white) stays zero: H5 cannot distinguish it from "no light".
    out[..., 4] = np.maximum(
        state[..., _H5_TRAFFIC_LIGHT_UNKNOWN], state[..., _H5_TRAFFIC_LIGHT_WHITE_OR_NONE]
    )
    return out


def _ego_current_state(ego_agent_past: NDArray[Any]) -> NDArray[np.float32]:
    """Rebuild the 10-wide ego current state from the last H5 past step.

    H5 keeps [x, y, cos, sin, velocity, yaw_rate] per step, so vy / ax / ay / steering have no
    source and stay zero. The augmenter re-derives the steering angle from the yaw rate and the
    wheelbase anyway, and ``compute_training_loss`` only reads the pose and the longitudinal
    velocity.
    """
    current = ego_agent_past[-1]
    state = np.zeros(dim.EGO_CURRENT_STATE_SHAPE[-1], dtype=np.float32)
    state[dim.EGOSTATE.X] = current[0]
    state[dim.EGOSTATE.Y] = current[1]
    state[dim.EGOSTATE.COS] = current[2]
    state[dim.EGOSTATE.SIN] = current[3]
    state[dim.EGOSTATE.VX] = current[4]
    state[dim.EGOSTATE.YAW_RATE] = current[5]
    return state


def _neighbor_agents_past(
    tracks: NDArray[Any], shape: NDArray[Any], label: NDArray[Any]
) -> NDArray[np.float32]:
    """Fold the separate H5 neighbour track / shape / class arrays into one 11-wide tensor.

    Layout is [x, y, cos, sin, vx, vy, width, length, vehicle, pedestrian, bicycle]. vx and vy
    stay zero because H5 has no neighbour velocity -- and ``NeighborEncoder`` zeroes those two
    channels before encoding regardless of what is fed in, so nothing is lost. Shape and class
    are per agent in H5 but per timestep here, so they are broadcast over time and then masked
    by each step's own validity.
    """
    num_agents, num_steps, _ = tracks.shape
    out = np.zeros((num_agents, num_steps, dim.NEIGHBOR_SHAPE[-1]), dtype=np.float32)
    out[..., : dim.POSE_DIM] = tracks
    valid = np.any(tracks != 0.0, axis=-1)[..., None]
    out[..., _NEIGHBOR_WIDTH_INDEX : _NEIGHBOR_LENGTH_INDEX + 1] = np.where(
        valid, shape[:, None, :], 0.0
    )
    out[..., _NEIGHBOR_TYPE_START:] = np.where(valid, label[:, None, :], 0.0)
    return out


def _lane_tensor(
    points: NDArray[Any], types: NDArray[Any], traffic_light_past: NDArray[Any]
) -> NDArray[np.float32]:
    """Rebuild a 33-wide lane/route tensor from H5 geometry, boundary types and lights.

    H5 stores [centre_x, centre_y, left_dx, left_dy, right_dx, right_dy] per point, boundary
    types once per segment, and the traffic light as a history; this branch wants
    [x, y, dx, dy, left_dx, left_dy, right_dx, right_dy, light(5), left_type(10),
    right_type(10)] per point. The boundary offsets already match, the forward difference is
    derived from the centreline, and the light is taken at the current step (the last history
    entry) because there is only one slot for it here.
    """
    num_segments, num_points, _ = points.shape
    valid = np.any(points != 0.0, axis=-1)
    out = np.zeros((num_segments, num_points, dim.SEGMENT_POINT_DIM), dtype=np.float32)

    centre = points[..., 0:2]
    out[..., dim.X : dim.Y + 1] = centre

    # dx, dy point at the next centreline sample; the last valid point of a segment has no
    # successor and keeps zeros, matching how the line encoders pad their own differences.
    delta = np.zeros_like(centre)
    delta[:, :-1] = centre[:, 1:] - centre[:, :-1]
    pair_valid = np.zeros_like(valid)
    pair_valid[:, :-1] = valid[:, :-1] & valid[:, 1:]
    out[..., dim.dX : dim.dY + 1] = np.where(pair_valid[..., None], delta, 0.0)

    out[..., dim.LB_X : dim.LB_Y + 1] = points[..., 2:4]
    out[..., dim.RB_X : dim.RB_Y + 1] = points[..., 4:6]

    light = _traffic_light_to_canonical(traffic_light_past[:, -1, :])
    attribute = np.concatenate([light, types.astype(np.float32)], axis=-1)
    out[..., dim.TRAFFIC_LIGHT :] = np.where(valid[..., None], attribute[:, None, :], 0.0)

    # Guard the padding invariant: a padded point must stay all-zero across the full row, or
    # ObservationNormalizer will shift it off zero and the encoder will read it as valid.
    return np.where(valid[..., None], out, 0.0)


def _speed_limit(speed_limit: NDArray[Any]) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Split the H5 speed limit into a value and an availability flag (zero means unknown)."""
    limit = speed_limit.astype(np.float32)
    return limit, limit > 0.0


def _polygons(intersection_area: NDArray[Any]) -> NDArray[np.float32]:
    """Turn H5 intersection areas into polygon points carrying their one-wide type flag."""
    points = intersection_area.astype(np.float32)
    out = np.zeros((*points.shape[:2], 2 + dim.POLYGON_TYPE_NUM), dtype=np.float32)
    out[..., :2] = points
    out[..., 2] = np.any(points != 0.0, axis=-1)
    return out


def _line_strings(stop_lines: NDArray[Any], road_borders: NDArray[Any]) -> NDArray[np.float32]:
    """Merge H5 stop lines and road borders into one flagged line-string tensor.

    This branch keeps both in a single array whose columns 2 and 3 flag which one a row is;
    H5 splits them, with stop lines carrying only their two end points. The two H5 arrays add
    up to ``NUM_LINE_STRINGS`` rows, and stop lines are zero-padded out to the common length.
    """
    stop = stop_lines.astype(np.float32)
    borders = road_borders.astype(np.float32)
    num_stop = stop.shape[0]
    num_rows = num_stop + borders.shape[0]
    if num_rows != dim.NUM_LINE_STRINGS:
        raise ValueError(
            f"H5 stop_lines + road_borders is {num_rows} rows, "
            f"expected NUM_LINE_STRINGS={dim.NUM_LINE_STRINGS}"
        )
    out = np.zeros(
        (num_rows, dim.POINTS_PER_LINE_STRING, 2 + dim.LINE_STRING_TYPE_NUM), dtype=np.float32
    )
    for offset, points, flag in (
        (0, stop, dim.LINESTRING.STOP_LINE_FLAG),
        (num_stop, borders, dim.LINESTRING.ROAD_BORDER_FLAG),
    ):
        length = min(points.shape[1], dim.POINTS_PER_LINE_STRING)
        rows = slice(offset, offset + points.shape[0])
        out[rows, :length, :2] = points[:, :length]
        out[rows, :length, flag] = np.any(points[:, :length] != 0.0, axis=-1)
    return out


def _ego_shape(ego_shape: NDArray[Any], wheel_base: float | None) -> NDArray[np.float32]:
    """Convert the H5 ego shape to this branch's [wheelbase, length, width] layout.

    H5 stores [base_link_to_front, length, width]. Slot 0 is a genuine wheelbase here: the
    augmenter turns a yaw rate into a steering angle with it, and the collision loss shifts the
    bounding box forward by half of it. When ``wheel_base`` is unknown it is approximated as
    twice the base_link-to-box-centre distance, which makes the bounding box exact and leaves
    only the augmenter's bicycle model on an estimate. Pass the real per-project value in to
    remove that estimate.
    """
    base_link_to_front, length, width = (float(value) for value in ego_shape[:3])
    if wheel_base is None:
        wheel_base = 2.0 * (base_link_to_front - 0.5 * length)
    return np.array([wheel_base, length, width], dtype=np.float32)


def convert_h5_frame(
    frame: dict[str, NDArray[Any]], wheel_base: float | None = None
) -> dict[str, NDArray[Any]]:
    """Convert one H5 frame into the canonical model-input dictionary of this branch."""
    ego_agent_past = frame["ego_agent_past"].astype(np.float32)
    lanes_speed_limit, lanes_has_speed_limit = _speed_limit(frame["lanes_speed_limit"])
    route_speed_limit, route_has_speed_limit = _speed_limit(frame["route_lanes_speed_limit"])
    return {
        # H5 already carries cos/sin here, and heading_to_cos_sin passes a 4-wide pose through
        # unchanged, so these need no yaw round trip.
        "ego_agent_past": ego_agent_past[:, : dim.POSE_DIM],
        "ego_current_state": _ego_current_state(ego_agent_past),
        "neighbor_agents_past": _neighbor_agents_past(
            frame["neighbor_agents_past"].astype(np.float32),
            frame["agent_shape"].astype(np.float32),
            frame["agent_label"].astype(np.float32),
        ),
        # H5 has no static-object channel; an all-zero block is fully masked by StaticEncoder.
        "static_objects": np.zeros(
            (dim.NUM_STATIC_OBJECTS, dim.STATIC_OBJECTS_SHAPE[-1]), dtype=np.float32
        ),
        "lanes": _lane_tensor(
            frame["lanes"].astype(np.float32),
            frame["lane_types"],
            frame["lane_traffic_light_past"].astype(np.float32),
        ),
        "lanes_speed_limit": lanes_speed_limit,
        "lanes_has_speed_limit": lanes_has_speed_limit,
        "route_lanes": _lane_tensor(
            frame["route_lanes"].astype(np.float32),
            frame["route_lane_types"],
            frame["route_traffic_light_past"].astype(np.float32),
        ),
        "route_lanes_speed_limit": route_speed_limit,
        "route_lanes_has_speed_limit": route_has_speed_limit,
        "polygons": _polygons(frame["intersection_area"]),
        "line_strings": _line_strings(frame["stop_lines"], frame["road_borders"]),
        "goal_pose": frame["goal_pose"].astype(np.float32)[: dim.POSE_DIM],
        "ego_shape": _ego_shape(frame["ego_shape"], wheel_base),
        "turn_indicators": frame["turn_indicators"].astype(np.float32),
        # The ego future goes out 3-wide: StatePerturbation's quintic refinement reads its
        # column 2 as a heading angle. The neighbour future keeps cos/sin, which is what the
        # canonical contract fixes it to and what centric_transform rotates directly.
        "ego_agent_future": _pose_yaw(frame["ego_agent_future"].astype(np.float32)),
        "neighbor_agents_future": frame["neighbor_agents_future"].astype(np.float32)[
            ..., : dim.POSE_DIM
        ],
    }


def load_wheel_base_by_project(path: str | Path) -> dict[str, float]:
    """Read per-project wheelbases from a data-converter parameter JSON.

    The file is the one the NPZ pipeline already uses: a mapping of project id to a dict with
    an ``ego_wheel_base`` entry. Projects without that entry are skipped.
    """
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return {
        str(project): float(params["ego_wheel_base"])
        for project, params in data.items()
        if isinstance(params, dict) and "ego_wheel_base" in params
    }


class H5FrameData(Dataset):
    """Serve canonical model inputs from H5 shards addressed by a Parquet frame index.

    H5 handles are opened lazily and cached per process so DataLoader workers each keep their
    own file descriptors; ``__getstate__`` drops them so the dataset can be pickled to workers.
    """

    def __init__(
        self,
        index_path: str | Path,
        file_capacity: int = 8,
        wheel_base_by_project: dict[str, float] | None = None,
    ) -> None:
        self._index_path = Path(index_path).expanduser().resolve()
        if not self._index_path.is_file():
            raise FileNotFoundError(f"Parquet frame index not found: {self._index_path}")
        if file_capacity < 1:
            raise ValueError(f"file_capacity must be at least 1: {file_capacity}")

        table = pq.read_table(self._index_path)
        missing = [name for name in REQUIRED_INDEX_COLUMNS if name not in table.column_names]
        if missing:
            raise ValueError(f"Parquet index is missing columns: {', '.join(missing)}")
        if table.num_rows == 0:
            raise ValueError(f"Parquet frame index is empty: {self._index_path}")

        # h5_path is stored relative to the index directory so a dataset stays portable.
        relative = table["h5_path"].combine_chunks().to_numpy(zero_copy_only=False)
        self.data_list = [str(self._index_path.parent / str(value)) for value in relative]
        self._frame_indices = (
            table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
        )
        if np.any(self._frame_indices < 0):
            raise ValueError(f"Parquet index has a negative frame_index: {self._index_path}")

        self._file_capacity = file_capacity
        self._wheel_base_by_project = dict(wheel_base_by_project or {})
        self._files: OrderedDict[str, h5py.File] = OrderedDict()
        self._wheel_base_cache: dict[str, float | None] = {}

    def __len__(self) -> int:
        return len(self._frame_indices)

    def subsample(self, step: int) -> None:
        """Keep every ``step``-th frame, in place.

        The NPZ dataset is thinned by slicing ``data_list`` directly, which cannot work here:
        a frame is addressed by a (shard, frame index) pair, so both arrays have to be sliced
        together or the index desynchronizes from the paths.
        """
        if step < 1:
            raise ValueError(f"subsample step must be at least 1: {step}")
        if step == 1:
            return
        self.data_list = self.data_list[::step]
        self._frame_indices = self._frame_indices[::step]

    def __getitem__(self, idx: int) -> dict[str, NDArray[Any]]:
        path = self.data_list[idx]
        file = self._file_for(path)
        frame_index = int(self._frame_indices[idx])
        num_frames = int(file.attrs["num_frames"])
        if not 0 <= frame_index < num_frames:
            raise IndexError(f"frame_index {frame_index} outside {path} ({num_frames} frames)")
        frames = file["frames"]
        frame = {key: np.asarray(frames[key][frame_index]) for key in REQUIRED_H5_KEYS}
        return convert_h5_frame(frame, self._wheel_base_cache[path])

    def _file_for(self, path: str) -> h5py.File:
        file = self._files.pop(path, None)
        if file is not None:
            self._files[path] = file
            return file

        file = h5py.File(path, "r")
        try:
            self._validate(file, path)
        except BaseException:
            file.close()
            raise
        project = str(file.attrs.get("project_id", ""))
        self._wheel_base_cache[path] = self._wheel_base_by_project.get(project)
        self._files[path] = file
        while len(self._files) > self._file_capacity:
            _, evicted = self._files.popitem(last=False)
            evicted.close()
        return file

    @staticmethod
    def _validate(file: h5py.File, path: str) -> None:
        if file.attrs.get("format") != H5_FORMAT:
            raise ValueError(f"Not a diffusion-planner H5 shard: {path}")
        version = int(file.attrs.get("format_version", -1))
        if version != H5_FORMAT_VERSION:
            raise ValueError(
                f"H5 format version {version} is unsupported (expected {H5_FORMAT_VERSION}): {path}"
            )
        if "num_frames" not in file.attrs or "frames" not in file:
            raise ValueError(f"Incomplete H5 shard: {path}")
        absent = [key for key in REQUIRED_H5_KEYS if key not in file["frames"]]
        if absent:
            raise ValueError(f"H5 shard is missing tensors {', '.join(absent)}: {path}")

    def close(self) -> None:
        for file in self._files.values():
            file.close()
        self._files.clear()

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_files": OrderedDict()}

    def __del__(self) -> None:
        if getattr(self, "_files", None):
            self.close()
