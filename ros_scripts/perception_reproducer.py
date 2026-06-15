#!/usr/bin/env python3
"""Perception Reproducer: closed-loop simulation of a Diffusion-Planner checkpoint on a rosbag.

The ego is driven by the planner's own predictions (perfect tracking of the predicted pose at
+0.1 s each step). Every other agent ("perception") is reproduced from the log: at each sim
step we pick the recorded frame whose ego pose is nearest to the current simulated ego, take
that frame's tracked objects (map frame) and express them in the simulated ego frame. Lanes,
route and traffic lights are queried from the map at the simulated ego pose.

Inputs are built with the SAME builders as ros_scripts/parse_rosbag.py, so they match the
format the model was trained on (time_len=31, 320 agents, 140 lanes, 60 line_strings, ...).

Example:
    python3 ros_scripts/perception_reproducer.py \
        /mnt/nvme/training_result/<run>/epoch0060 \
        /mnt/nvme/rosbags_from_label/x2_dev/<area>/train/<date>/<time> \
        --num_steps 100

Env: run under system python3.10 with /opt/ros/humble/setup.bash + cpp_tools/install/setup.bash
sourced and diffusion_planner + diffusion_planner_ros on PYTHONPATH (same as parse_rosbag.py).

`run_closed_loop(...)` is importable for later validation-time integration.
"""

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from diffusion_planner.dimensions import (
    INPUT_T,
    MAX_NUM_AGENTS,
    MAX_NUM_NEIGHBORS,
    NUM_LINE_STRINGS,
    NUM_POLYGONS,
    NUM_SEGMENTS_IN_LANE,
    NUM_SEGMENTS_IN_ROUTE,
    OUTPUT_T,
    POINTS_PER_LINE_STRING,
    POINTS_PER_POLYGON,
    POSE_DIM,
)
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.utils.visualize_input import visualize_inputs
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import (
    LINE_STRING_TYPE_MAP,
    LINE_STRING_TYPE_NUM,
    POLYGON_TYPE_MAP,
    POLYGON_TYPE_NUM,
    create_lane_tensor,
    create_line_tensor,
)
from diffusion_planner_ros.utils import (
    create_current_ego_state,
    filter_route_lanelets,
    rot3x3_to_heading_cos_sin,
)
from scipy.spatial.transform import Rotation

# venv-safe builder (no rosbag2_py); shares the parity-matched code with parse_rosbag.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproducer_inputs import PAST_TIME_STEPS, build_neighbor_past  # noqa: E402

DT = 0.1  # sim step (10 Hz)
MAKE_MP4 = Path.home() / "misc" / "ffmpeg_lib" / "make_mp4_from_unsequential_png.sh"

DEFAULT_WHEEL_BASE = 2.75
DEFAULT_EGO_LENGTH = 4.34
DEFAULT_EGO_WIDTH = 1.70


# --------------------------------------------------------------------------------------------
# Small geometry helpers
# --------------------------------------------------------------------------------------------
def rigid_inverse(mat: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    r = mat[:3, :3]
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ mat[:3, 3]
    return out


def pose_4x4_from_xy_cos_sin(x: float, y: float, cos: float, sin: float) -> np.ndarray:
    mat = np.eye(4)
    norm = float(np.hypot(cos, sin)) or 1.0
    c, s = cos / norm, sin / norm
    mat[0, 0], mat[0, 1] = c, -s
    mat[1, 0], mat[1, 1] = s, c
    mat[0, 3], mat[1, 3] = x, y
    return mat


def fake_kinematic_state(bl2map: np.ndarray, vx: float, vy: float, yaw_rate: float):
    """Build an Odometry-like object (only the fields the builders read)."""
    quat = Rotation.from_matrix(bl2map[:3, :3]).as_quat()  # x, y, z, w
    return SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=bl2map[0, 3], y=bl2map[1, 3], z=bl2map[2, 3]),
                orientation=SimpleNamespace(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
            )
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=vx, y=vy, z=0.0),
                angular=SimpleNamespace(x=0.0, y=0.0, z=yaw_rate),
            )
        ),
    )


def fake_acceleration(ax: float, ay: float):
    return SimpleNamespace(
        accel=SimpleNamespace(accel=SimpleNamespace(linear=SimpleNamespace(x=ax, y=ay, z=0.0)))
    )


def obb_corners(cx, cy, cos, sin, length, width):
    """Corners of an oriented box; heading along (cos, sin), length forward, width lateral."""
    fx, fy = cos, sin
    lx, ly = -sin, cos
    hl, hw = length / 2.0, width / 2.0
    return np.array(
        [
            [cx + fx * hl + lx * hw, cy + fy * hl + ly * hw],
            [cx + fx * hl - lx * hw, cy + fy * hl - ly * hw],
            [cx - fx * hl - lx * hw, cy - fy * hl - ly * hw],
            [cx - fx * hl + lx * hw, cy - fy * hl + ly * hw],
        ]
    )


def obb_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two convex quads."""
    for poly in (a, b):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            n = np.linalg.norm(axis)
            if n < 1e-9:
                continue
            axis /= n
            pa = a @ axis
            pb = b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


# --------------------------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------------------------
@dataclass
class StepRecord:
    sim_pos_map: np.ndarray  # (2,)
    nearest_recorded_idx: int
    deviation_from_recorded: float
    collision: bool
    offroute_lateral: float


@dataclass
class ReproducerResult:
    metrics: dict
    steps: list = field(default_factory=list)


# --------------------------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------------------------
def load_model(model_dir: Path, args_json: Path | None, device: torch.device):
    args_json = args_json or (model_dir / "args.json")
    ckpt_path = model_dir / "best_model.pth"
    if not args_json.is_file():
        raise FileNotFoundError(f"args.json not found: {args_json}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"best_model.pth not found: {ckpt_path}")
    config = Config(str(args_json))
    model = Diffusion_Planner(config)
    model.eval()
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt["model"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model, config


# --------------------------------------------------------------------------------------------
# Per-step input building
# --------------------------------------------------------------------------------------------
def build_ego_agent_past(sim_history: list, map2bl: np.ndarray, dev: torch.device) -> torch.Tensor:
    """(1, PAST_TIME_STEPS, 4) [x, y, cos, sin] in current ego frame from simulated history."""
    past = torch.zeros((1, PAST_TIME_STEPS, 4), dtype=torch.float32)
    n = len(sim_history)
    for t in range(PAST_TIME_STEPS):
        # oldest at index 0, current at index PAST_TIME_STEPS-1; fill-pad before the start.
        hist_idx = n - PAST_TIME_STEPS + t
        bl2map = sim_history[max(0, hist_idx)]
        pose_ego = map2bl @ bl2map
        cos, sin = rot3x3_to_heading_cos_sin(pose_ego[0:3, 0:3])
        past[0, t, 0] = pose_ego[0, 3]
        past[0, t, 1] = pose_ego[1, 3]
        past[0, t, 2] = cos
        past[0, t, 3] = sin
    return past.to(dev)


def build_input_dict(
    sim_history,
    sim_vel,
    sim_accel,
    recorded_frames,
    nearest_idx,
    vector_map,
    ego_shape_vec,
    wheel_base,
    dev,
):
    bl2map = sim_history[-1]
    map2bl = rigid_inverse(bl2map)
    center_x, center_y = bl2map[0, 3], bl2map[1, 3]

    # Ego (from the simulated state).
    kinematic = fake_kinematic_state(bl2map, sim_vel[0], sim_vel[1], sim_vel[2])
    accel = fake_acceleration(sim_accel[0], sim_accel[1])
    ego_current = create_current_ego_state(kinematic, accel, wheel_base).to(dev)
    ego_past = build_ego_agent_past(sim_history, map2bl, dev)

    # Neighbors reproduced from the nearest recorded frame, in the simulated ego frame.
    neighbor_np, _ids = build_neighbor_past(
        recorded_frames, nearest_idx, map2bl, MAX_NUM_NEIGHBORS, PAST_TIME_STEPS
    )
    neighbor = torch.from_numpy(neighbor_np).unsqueeze(0).to(dev)

    traffic_light_recognition = recorded_frames[nearest_idx].traffic_signals
    route = recorded_frames[nearest_idx].route

    # Past turn-indicator sequence at the nearest recorded frame (matches the npz construction).
    ti = np.zeros(PAST_TIME_STEPS, dtype=np.int64)
    for t in range(PAST_TIME_STEPS):
        idx = max(0, nearest_idx - INPUT_T + t)
        ti[t] = recorded_frames[idx].turn_indicator.report
    turn_indicators = torch.tensor(ti, dtype=torch.int64, device=dev).unsqueeze(0)

    lanes, lanes_speed, lanes_has_speed = create_lane_tensor(
        vector_map.lanelets.values(),
        map2bl_mat4x4=map2bl,
        center_x=center_x,
        center_y=center_y,
        traffic_light_recognition=traffic_light_recognition,
        num_segments=NUM_SEGMENTS_IN_LANE,
        dev=dev,
        do_sort=True,
    )

    route_lanelets = [
        vector_map.lanelets[seg.preferred_primitive.id]
        for seg in route.segments
        if seg.preferred_primitive.id in vector_map.lanelets
    ]
    route_lanelets = filter_route_lanelets(route_lanelets, kinematic)
    route_t, route_speed, route_has_speed = create_lane_tensor(
        route_lanelets,
        map2bl_mat4x4=map2bl,
        center_x=center_x,
        center_y=center_y,
        traffic_light_recognition=traffic_light_recognition,
        num_segments=NUM_SEGMENTS_IN_ROUTE,
        dev=dev,
        do_sort=False,
    )

    polygons = create_line_tensor(
        vector_map.polygons.values(),
        map2bl,
        center_x,
        center_y,
        NUM_POLYGONS,
        POINTS_PER_POLYGON,
        dev,
        POLYGON_TYPE_MAP,
        POLYGON_TYPE_NUM,
    )
    line_strings = create_line_tensor(
        vector_map.line_strings.values(),
        map2bl,
        center_x,
        center_y,
        NUM_LINE_STRINGS,
        POINTS_PER_LINE_STRING,
        dev,
        LINE_STRING_TYPE_MAP,
        LINE_STRING_TYPE_NUM,
    )

    # Goal pose from the route (deployment convention), in ego frame, [x, y, cos, sin].
    goal_ego = map2bl @ _pose_to_mat4x4(route.goal_pose)
    gcos, gsin = rot3x3_to_heading_cos_sin(goal_ego[0:3, 0:3])
    goal_pose = torch.tensor(
        [[goal_ego[0, 3], goal_ego[1, 3], gcos, gsin]], dtype=torch.float32, device=dev
    )

    input_dict = {
        "ego_agent_past": ego_past,
        "ego_current_state": ego_current,
        "neighbor_agents_past": neighbor,
        "lanes": lanes,
        "lanes_speed_limit": lanes_speed,
        "lanes_has_speed_limit": lanes_has_speed,
        "route_lanes": route_t,
        "route_lanes_speed_limit": route_speed,
        "route_lanes_has_speed_limit": route_has_speed,
        "polygons": polygons,
        "line_strings": line_strings,
        "static_objects": torch.zeros((1, 5, 10), dtype=torch.float32, device=dev),
        "goal_pose": goal_pose,
        "ego_shape": torch.tensor([ego_shape_vec], dtype=torch.float32, device=dev),
        "turn_indicators": turn_indicators,
        # Diffusion sampling seeds (zeros -> model samples internally); not normalized.
        "sampled_trajectories": torch.zeros(
            (1, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM), dtype=torch.float32, device=dev
        ),
        "delay": torch.zeros((1,), dtype=torch.float32, device=dev),
    }
    return input_dict, bl2map, map2bl


def _pose_to_mat4x4(pose) -> np.ndarray:
    mat = np.eye(4)
    q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    mat[:3, :3] = Rotation.from_quat(q).as_matrix()
    mat[0, 3] = pose.position.x
    mat[1, 3] = pose.position.y
    mat[2, 3] = pose.position.z
    return mat


# --------------------------------------------------------------------------------------------
# Closed loop
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def run_closed_loop(
    model,
    config,
    vector_map,
    sequence,
    device,
    num_steps: int | None = None,
    wheel_base: float = 2.75,
    ego_length: float = 4.34,
    ego_width: float = 1.70,
    result_dir: Path | None = None,
    make_video: bool = True,
    offroute_threshold: float = 5.0,
) -> ReproducerResult:
    """Run a closed-loop Perception Reproducer on one route sequence; return metrics + per-step log."""
    recorded_frames = sequence.data_list
    recorded_xy = np.array(
        [
            [f.kinematic_state.pose.pose.position.x, f.kinematic_state.pose.pose.position.y]
            for f in recorded_frames
        ]
    )
    n_rec = len(recorded_frames)
    if num_steps is None:
        num_steps = n_rec
    num_steps = min(num_steps, n_rec)

    ego_shape_vec = [wheel_base, ego_length, ego_width]
    frame_dir = None
    if make_video and result_dir is not None:
        frame_dir = result_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the simulated ego from the first recorded frame.
    init = recorded_frames[0].kinematic_state
    sim_history = [_pose_to_mat4x4(init.pose.pose)]
    sim_vel = [init.twist.twist.linear.x, init.twist.twist.linear.y, init.twist.twist.angular.z]
    prev_speed = float(np.hypot(sim_vel[0], sim_vel[1]))
    sim_accel = [0.0, 0.0]
    cursor = 0  # monotonic nearest-recorded-frame cursor

    steps = []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for k in range(num_steps):
        bl2map = sim_history[-1]
        sim_pos = np.array([bl2map[0, 3], bl2map[1, 3]])

        # Nearest recorded ego (search forward from the cursor; allow a small backward window).
        lo = max(0, cursor - 5)
        d = np.linalg.norm(recorded_xy[lo:] - sim_pos, axis=1)
        nearest_idx = lo + int(np.argmin(d))
        cursor = max(cursor, nearest_idx)

        input_dict, bl2map, map2bl = build_input_dict(
            sim_history,
            sim_vel,
            sim_accel,
            recorded_frames,
            nearest_idx,
            vector_map,
            ego_shape_vec,
            wheel_base,
            device,
        )
        raw_input = {key: val.detach().clone() for key, val in input_dict.items()}

        normed = config.observation_normalizer(
            {key: val.clone() for key, val in input_dict.items()}
        )
        out = model(normed)[1]
        pred = out["prediction"].detach().cpu().numpy()  # [1, P, T, 4] ego frame, metres
        ego_pred = pred[0, 0]  # (T, 4)

        # --- metrics for this step ---
        collision = _check_collision(raw_input, ego_length, ego_width)
        offroute = _offroute_lateral(raw_input)
        deviation = float(np.linalg.norm(sim_pos - recorded_xy[nearest_idx]))
        steps.append(StepRecord(sim_pos.copy(), nearest_idx, deviation, collision, offroute))

        # --- visualization ---
        if frame_dir is not None:
            fig, ax = plt.subplots(figsize=(10, 10))
            visualize_inputs(raw_input, ax=ax)
            ax.plot(
                ego_pred[:, 0],
                ego_pred[:, 1],
                "-",
                color="magenta",
                lw=2,
                label="planned",
                zorder=10,
            )
            ax.set_title(
                f"step {k:04d}  rec_idx={nearest_idx}  dev={deviation:.2f}m"
                f"  {'COLLISION' if collision else ''}"
            )
            fig.savefig(frame_dir / f"{k:08d}.png", bbox_inches="tight", dpi=80)
            plt.close(fig)

        # --- advance ego (perfect tracking of the +0.1 s predicted pose) ---
        pose_ego = pose_4x4_from_xy_cos_sin(*ego_pred[0])
        new_bl2map = bl2map @ pose_ego
        sim_history.append(new_bl2map)

        # Update velocity / acceleration from the followed step (body frame).
        disp_map = new_bl2map[:3, 3] - bl2map[:3, 3]
        disp_body = rigid_inverse(bl2map)[:3, :3] @ disp_map
        vx, vy = disp_body[0] / DT, disp_body[1] / DT
        yaw_rate = float(np.arctan2(ego_pred[0, 3], ego_pred[0, 2])) / DT
        sim_vel = [vx, vy, yaw_rate]
        speed = float(np.hypot(vx, vy))
        sim_accel = [(speed - prev_speed) / DT, 0.0]
        prev_speed = speed

    metrics = _summarize_metrics(steps, sim_history, recorded_frames, offroute_threshold)
    result = ReproducerResult(metrics=metrics, steps=steps)

    if frame_dir is not None and make_video:
        _make_mp4(frame_dir)
    return result


def _check_collision(raw_input, ego_length, ego_width) -> bool:
    ego_box = obb_corners(0.0, 0.0, 1.0, 0.0, ego_length, ego_width)
    neighbor = raw_input["neighbor_agents_past"][0].cpu().numpy()  # (P, T, 11)
    current = neighbor[:, -1, :]  # latest step
    for agent in current:
        if not np.any(agent[:4]):
            continue
        x, y, cos, sin, _, _, width, length = agent[:8]
        if length <= 0 or width <= 0:
            continue
        box = obb_corners(x, y, cos, sin, length, width)
        if obb_overlap(ego_box, box):
            return True
    return False


def _offroute_lateral(raw_input) -> float:
    route = raw_input["route_lanes"][0].cpu().numpy()  # (R, L, C)
    pts = route[:, :, :2].reshape(-1, 2)
    mask = np.any(route[:, :, :2] != 0, axis=-1).reshape(-1)
    pts = pts[mask]
    if len(pts) == 0:
        return float("nan")
    return float(np.linalg.norm(pts, axis=1).min())


def _summarize_metrics(steps, sim_history, recorded_frames, offroute_threshold) -> dict:
    sim_xy = np.array([[m[0, 3], m[1, 3]] for m in sim_history])
    progress = float(np.sum(np.linalg.norm(np.diff(sim_xy, axis=0), axis=1)))
    goal = recorded_frames[-1].kinematic_state.pose.pose.position
    dist_to_goal = float(np.linalg.norm(sim_xy[-1] - np.array([goal.x, goal.y])))
    deviations = np.array([s.deviation_from_recorded for s in steps])
    offroutes = np.array([s.offroute_lateral for s in steps])
    collisions = np.array([s.collision for s in steps])
    return {
        "num_steps": len(steps),
        "collision_rate": float(np.mean(collisions)) if len(collisions) else 0.0,
        "num_collision_steps": int(np.sum(collisions)),
        "offroute_lateral_mean": float(np.nanmean(offroutes)) if len(offroutes) else 0.0,
        "offroute_lateral_max": float(np.nanmax(offroutes)) if len(offroutes) else 0.0,
        "offroute_rate": float(np.mean(offroutes > offroute_threshold)) if len(offroutes) else 0.0,
        "progress_m": progress,
        "final_distance_to_goal_m": dist_to_goal,
        "deviation_from_recorded_mean": float(np.mean(deviations)) if len(deviations) else 0.0,
        "deviation_from_recorded_final": float(deviations[-1]) if len(deviations) else 0.0,
    }


def _make_mp4(frame_dir: Path):
    if not MAKE_MP4.is_file():
        print(f"Skip mp4: helper not found at {MAKE_MP4}")
        return
    if shutil.which("ffmpeg") is None:
        print("Skip mp4: ffmpeg not installed (PNGs are in frames/)")
        return
    try:
        subprocess.run([str(MAKE_MP4), str(frame_dir)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Skip mp4: ffmpeg failed ({e})")


# --------------------------------------------------------------------------------------------
# Scene reconstruction (from the plain pickle written by extract_scene.py)
# --------------------------------------------------------------------------------------------
def _ns_pose(d) -> SimpleNamespace:
    p, q = d["pos"], d["quat"]
    return SimpleNamespace(
        position=SimpleNamespace(x=p[0], y=p[1], z=p[2]),
        orientation=SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3]),
    )


def _reconstruct_object(o) -> SimpleNamespace:
    """Duck-typed TrackedObject so the (parity-matched) build_neighbor_past reads it unchanged."""
    return SimpleNamespace(
        object_id=SimpleNamespace(uuid=bytes(o["uuid"])),
        classification=[SimpleNamespace(label=lbl, probability=p) for lbl, p in o["cls"]],
        kinematics=SimpleNamespace(
            pose_with_covariance=SimpleNamespace(pose=_ns_pose(o)),
            twist_with_covariance=SimpleNamespace(
                twist=SimpleNamespace(linear=SimpleNamespace(x=o["vx"], y=o["vy"], z=0.0))
            ),
        ),
        shape=SimpleNamespace(dimensions=SimpleNamespace(x=o["dim_x"], y=o["dim_y"], z=0.0)),
    )


def reconstruct_sequence(scene) -> SimpleNamespace:
    """Rebuild a SequenceData-like object (.data_list of duck-typed FrameData) from the scene."""
    route = SimpleNamespace(
        segments=[
            SimpleNamespace(preferred_primitive=SimpleNamespace(id=lid))
            for lid in scene["route"]["lanelet_ids"]
        ],
        goal_pose=_ns_pose(scene["route"]["goal"]),
    )
    frames = []
    for fr in scene["frames"]:
        ego = fr["ego"]
        kinematic_state = SimpleNamespace(
            pose=SimpleNamespace(pose=_ns_pose(ego)),
            twist=SimpleNamespace(
                twist=SimpleNamespace(
                    linear=SimpleNamespace(x=ego["vx"], y=ego["vy"], z=0.0),
                    angular=SimpleNamespace(x=0.0, y=0.0, z=ego["yaw_rate"]),
                )
            ),
        )
        traffic = {
            gid: [SimpleNamespace(color=c, shape=s) for c, s in elems]
            for gid, elems in fr["traffic"].items()
        }
        frames.append(
            SimpleNamespace(
                kinematic_state=kinematic_state,
                acceleration=fake_acceleration(ego["ax"], ego["ay"]),
                tracked_objects=SimpleNamespace(
                    objects=[_reconstruct_object(o) for o in fr["objects"]]
                ),
                traffic_signals=traffic,
                turn_indicator=SimpleNamespace(report=fr["turn_indicator"]),
                route=route,
            )
        )
    return SimpleNamespace(data_list=frames, route=route)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="dir with args.json + best_model.pth")
    parser.add_argument("scene", type=Path, help="scene .pkl from extract_scene.py")
    parser.add_argument("--args_json", type=Path, default=None)
    parser.add_argument("--result_dir", type=Path, default=None)
    parser.add_argument("--num_steps", type=int, default=None, help="default: full sequence")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (auto if unset)")
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--wheel_base", type=float, default=DEFAULT_WHEEL_BASE)
    parser.add_argument("--ego_length", type=float, default=DEFAULT_EGO_LENGTH)
    parser.add_argument("--ego_width", type=float, default=DEFAULT_EGO_WIDTH)
    return parser.parse_args()


def run_reproducer(
    model_dir,
    args_json,
    scene_path,
    result_dir,
    num_steps,
    device,
    make_video,
    wheel_base,
    ego_length,
    ego_width,
) -> ReproducerResult:
    print(f"model : {model_dir}")
    print(f"scene : {scene_path}")
    print(f"device: {device}")

    with open(scene_path, "rb") as f:
        scene = pickle.load(f)
    vector_map = scene["map"]
    sequence = reconstruct_sequence(scene)
    print(f"sequence frames: {len(sequence.data_list)}")

    if "ego_shape" in scene["meta"]:
        ego_shape = scene["meta"]["ego_shape"]
        wheel_base = ego_shape["wheel_base"]
        ego_length = ego_shape["ego_length"]
        ego_width = ego_shape["ego_width"]
    print(f"ego_shape : wb={wheel_base} length={ego_length} width={ego_width}")

    if result_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path("/mnt/nvme/test") / f"{stamp}_reproducer_{scene['meta']['map_name']}"
    result_dir = result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"result dir : {result_dir}")

    model, config = load_model(model_dir, args_json, device)

    result = run_closed_loop(
        model,
        config,
        vector_map,
        sequence,
        device,
        num_steps=num_steps,
        wheel_base=wheel_base,
        ego_length=ego_length,
        ego_width=ego_width,
        result_dir=result_dir,
        make_video=make_video,
    )
    metrics_path = result_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result.metrics, f, indent=2)
    print("=== metrics ===")
    print(json.dumps(result.metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_reproducer(
        args.model_dir,
        args.args_json,
        args.scene,
        args.result_dir,
        args.num_steps,
        device,
        not args.no_video,
        args.wheel_base,
        args.ego_length,
        args.ego_width,
    )


if __name__ == "__main__":
    main()
