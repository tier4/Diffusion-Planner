"""Convert C++ data_converter `.bin` output to 320-model NPZ schema.

The cpp tool writes a fixed-layout `TrainingDataBinary` struct per frame.
This script decodes it and saves an NPZ that matches what the 320-neighbor
diffusion planner pipeline (training, K=8 ranking, sim) expects — i.e. the
same schema as `parse_rosbag.py` v2 output and the `perturbed_padded_train/`
reference NPZs.

Key transforms:
- ego_agent_past:    cpp (31, 4) [x,y,cos,sin] → npz (31, 3) [x,y,yaw]
- ego_agent_future:  cpp (80, 4) [x,y,cos,sin] → npz (80, 4) [x,y,cos,sin]  (kept 4-col)
- goal_pose:         cpp (4,)   [x,y,cos,sin] → npz (3,)   [x,y,yaw]
- neighbor_agents_future: cpp (320, 80, 4) → npz (320, 80, 4) [x,y,cos,sin]  (kept 4-col)
- speed-limit fields: (N,) → (N, 1), int32 has-flag → bool

FUTURES are kept 4-col [x,y,cos,sin]: the trainable/reward schema requires it and the
reward path hard-fails on 3-col. ego_agent_past and goal_pose stay 3-col [x,y,yaw] (the
canonical schema — the loader/model widen them, and `heading_to_cos_sin` is idempotent so
a 4-col input would also pass through unchanged).
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

# Constants from
# cpp_tools/src/universe/autoware_universe/planning/autoware_diffusion_planner/include/...
INPUT_T_WITH_CURRENT = 31
OUTPUT_T = 80
POSE_DIM = 4
MAX_NUM_NEIGHBORS = 320
NEIGHBOR_PAST_DIM = 11
NEIGHBOR_FUTURE_DIM = 4
STATIC_N = 5
STATIC_DIM = 10
NUM_SEGMENTS_IN_LANE = 140
NUM_SEGMENTS_IN_ROUTE = 25
NUM_POLYGONS = 10
NUM_LINE_STRINGS = 60
POINTS_PER_SEGMENT = 20
POINTS_PER_POLYGON = 40
POINTS_PER_LINE_STRING = 20
SEGMENT_POINT_DIM = (
    33  # 13 + 10 + 10 = X,Y,dX,dY,LB_X,LB_Y,RB_X,RB_Y,TL(5) + LeftType(10) + RightType(10)
)
POLYGON_DIM = 3  # 2 + POLYGON_TYPE_NUM=1
LS_DIM = 4  # 2 + LINE_STRING_TYPE_NUM=2

EXPECTED_BIN_SIZE = (
    4  # version uint32
    + (
        INPUT_T_WITH_CURRENT * POSE_DIM  # ego_agent_past
        + 10  # ego_current_state
        + OUTPUT_T * POSE_DIM  # ego_agent_future
        + MAX_NUM_NEIGHBORS * INPUT_T_WITH_CURRENT * NEIGHBOR_PAST_DIM
        + MAX_NUM_NEIGHBORS * OUTPUT_T * NEIGHBOR_FUTURE_DIM
        + STATIC_N * STATIC_DIM
        + NUM_SEGMENTS_IN_LANE * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM
        + NUM_SEGMENTS_IN_LANE  # lanes_speed_limit
        + NUM_SEGMENTS_IN_ROUTE * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM
        + NUM_SEGMENTS_IN_ROUTE  # route_lanes_speed_limit
        + NUM_POLYGONS * POINTS_PER_POLYGON * POLYGON_DIM
        + NUM_LINE_STRINGS * POINTS_PER_LINE_STRING * LS_DIM
        + POSE_DIM  # goal_pose
        + 3  # ego_shape
    )
    * 4  # floats
    + (
        NUM_SEGMENTS_IN_LANE  # lanes_has_speed_limit int32
        + NUM_SEGMENTS_IN_ROUTE  # route_lanes_has_speed_limit int32
        + INPUT_T_WITH_CURRENT  # turn_indicators int32
    )
    * 4
)
assert EXPECTED_BIN_SIZE == 1309172, EXPECTED_BIN_SIZE


def _xyc_to_xyy(arr: np.ndarray) -> np.ndarray:
    """[..., 4] (x, y, cos, sin) -> [..., 3] (x, y, yaw)."""
    yaw = np.arctan2(arr[..., 3], arr[..., 2])
    out = np.empty(arr.shape[:-1] + (3,), dtype=arr.dtype)
    out[..., :2] = arr[..., :2]
    out[..., 2] = yaw
    return out


def decode_bin(raw: bytes) -> dict[str, np.ndarray]:
    """Decode one .bin file into a dict of arrays in 320-model NPZ schema."""
    if len(raw) != EXPECTED_BIN_SIZE:
        raise ValueError(
            f"bin size {len(raw)} != expected {EXPECTED_BIN_SIZE}; cpp struct mismatch?"
        )

    off = 0

    def take_f32(n: int) -> np.ndarray:
        nonlocal off
        a = np.frombuffer(raw, dtype=np.float32, count=n, offset=off)
        off += n * 4
        return a

    def take_i32(n: int) -> np.ndarray:
        nonlocal off
        a = np.frombuffer(raw, dtype=np.int32, count=n, offset=off)
        off += n * 4
        return a

    def take_u32(n: int) -> np.ndarray:
        nonlocal off
        a = np.frombuffer(raw, dtype=np.uint32, count=n, offset=off)
        off += n * 4
        return a

    version = take_u32(1)[0]  # noqa: F841  -- structural cross-check only

    ego_past_xyc = take_f32(INPUT_T_WITH_CURRENT * POSE_DIM).reshape(INPUT_T_WITH_CURRENT, POSE_DIM)
    ego_current = take_f32(10).copy()
    ego_future_xyc = take_f32(OUTPUT_T * POSE_DIM).reshape(OUTPUT_T, POSE_DIM)
    neighbor_past = (
        take_f32(MAX_NUM_NEIGHBORS * INPUT_T_WITH_CURRENT * NEIGHBOR_PAST_DIM)
        .reshape(MAX_NUM_NEIGHBORS, INPUT_T_WITH_CURRENT, NEIGHBOR_PAST_DIM)
        .copy()
    )
    neighbor_future_xyc = take_f32(MAX_NUM_NEIGHBORS * OUTPUT_T * NEIGHBOR_FUTURE_DIM).reshape(
        MAX_NUM_NEIGHBORS, OUTPUT_T, NEIGHBOR_FUTURE_DIM
    )
    static_objects = take_f32(STATIC_N * STATIC_DIM).reshape(STATIC_N, STATIC_DIM).copy()

    lanes = (
        take_f32(NUM_SEGMENTS_IN_LANE * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM)
        .reshape(NUM_SEGMENTS_IN_LANE, POINTS_PER_SEGMENT, SEGMENT_POINT_DIM)
        .copy()
    )
    lanes_speed_limit = take_f32(NUM_SEGMENTS_IN_LANE).reshape(NUM_SEGMENTS_IN_LANE, 1).copy()
    lanes_has_sl = take_i32(NUM_SEGMENTS_IN_LANE).astype(bool).reshape(NUM_SEGMENTS_IN_LANE, 1)

    route_lanes = (
        take_f32(NUM_SEGMENTS_IN_ROUTE * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM)
        .reshape(NUM_SEGMENTS_IN_ROUTE, POINTS_PER_SEGMENT, SEGMENT_POINT_DIM)
        .copy()
    )
    route_speed_limit = take_f32(NUM_SEGMENTS_IN_ROUTE).reshape(NUM_SEGMENTS_IN_ROUTE, 1).copy()
    route_has_sl = take_i32(NUM_SEGMENTS_IN_ROUTE).astype(bool).reshape(NUM_SEGMENTS_IN_ROUTE, 1)

    polygons = (
        take_f32(NUM_POLYGONS * POINTS_PER_POLYGON * POLYGON_DIM)
        .reshape(NUM_POLYGONS, POINTS_PER_POLYGON, POLYGON_DIM)
        .copy()
    )
    line_strings = (
        take_f32(NUM_LINE_STRINGS * POINTS_PER_LINE_STRING * LS_DIM)
        .reshape(NUM_LINE_STRINGS, POINTS_PER_LINE_STRING, LS_DIM)
        .copy()
    )
    goal_pose_xyc = take_f32(POSE_DIM)
    turn_indicators = take_i32(INPUT_T_WITH_CURRENT).copy()
    ego_shape = take_f32(3).copy()

    if off != EXPECTED_BIN_SIZE:
        raise RuntimeError(f"decoded {off} bytes != expected {EXPECTED_BIN_SIZE}")

    return {
        "ego_agent_past": _xyc_to_xyy(ego_past_xyc).astype(np.float32),
        "ego_current_state": ego_current.astype(np.float32),
        # Futures stay 4-col [x,y,cos,sin] (trainable/reward schema; never 3-col).
        "ego_agent_future": ego_future_xyc.astype(np.float32),
        "neighbor_agents_past": neighbor_past.astype(np.float32),
        "neighbor_agents_future": neighbor_future_xyc.astype(np.float32),
        "static_objects": static_objects.astype(np.float32),
        "lanes": lanes.astype(np.float32),
        "lanes_speed_limit": lanes_speed_limit.astype(np.float32),
        "lanes_has_speed_limit": lanes_has_sl,
        "route_lanes": route_lanes.astype(np.float32),
        "route_lanes_speed_limit": route_speed_limit.astype(np.float32),
        "route_lanes_has_speed_limit": route_has_sl,
        "polygons": polygons.astype(np.float32),
        "line_strings": line_strings.astype(np.float32),
        "goal_pose": _xyc_to_xyy(goal_pose_xyc).astype(np.float32),
        "turn_indicators": turn_indicators.astype(np.int32),
        "ego_shape": ego_shape.astype(np.float32),
        "version": np.int64(2),
    }


def convert_dir(src_dir: Path, dst_dir: Path, skip_existing: bool = True) -> dict[str, int]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    bins = sorted(glob.glob(str(src_dir / "*.bin")))
    stats = {"total": len(bins), "ok": 0, "skipped_existing": 0, "skipped_invalid": 0, "errors": 0}
    for fp in bins:
        stem = Path(fp).stem
        # Each .bin has a sibling .json with skipping_info — skip rejected frames
        jpath = Path(fp).with_suffix(".json")
        if jpath.exists():
            with open(jpath) as f:
                meta = json.load(f)
            if meta.get("is_skipped", False):
                stats["skipped_invalid"] += 1
                continue
        out = dst_dir / f"{stem}.npz"
        if skip_existing and out.exists():
            stats["skipped_existing"] += 1
            continue
        try:
            with open(fp, "rb") as f:
                raw = f.read()
            arrs = decode_bin(raw)
            np.savez_compressed(out, **arrs)
            stats["ok"] += 1
        except Exception as exc:
            print(f"ERROR on {fp}: {exc}")
            stats["errors"] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src", type=Path, required=True, help="dir of .bin/.json pairs from cpp data_converter"
    )
    ap.add_argument("--dst", type=Path, required=True, help="dir to write .npz files")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    stats = convert_dir(args.src, args.dst, skip_existing=not args.overwrite)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
