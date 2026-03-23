"""
Rosbagを解析して自車周囲のTrackedObjectの数をクラス別に計算・可視化するスクリプト。
デフォルトでは周囲200m（XY方向に正方形）のTrackedObjectをカウントする。
各オブジェクトはshapeとposeから回転した長方形として描画する。

出力:
  1. 統計情報 (コンソール)
  2. 時系列グラフ + ヒストグラム (tracked_objects_count.png)
  3. 各フレームのXY散布図 → mp4動画 (tracked_objects_video.mp4)
"""

import argparse
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
import yaml
from autoware_perception_msgs.msg import ObjectClassification, TrackedObjects
from matplotlib.patches import Polygon as MplPolygon
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# ObjectClassification label constants
# 0=UNKNOWN, 1=CAR, 2=TRUCK, 3=BUS, 4=TRAILER, 5=MOTORCYCLE, 6=BICYCLE, 7=PEDESTRIAN

LABEL_TO_CATEGORY: dict[int, str] = {
    ObjectClassification.CAR: "vehicle",
    ObjectClassification.TRUCK: "vehicle",
    ObjectClassification.BUS: "vehicle",
    ObjectClassification.TRAILER: "vehicle",
    ObjectClassification.MOTORCYCLE: "vehicle",
    ObjectClassification.BICYCLE: "bicycle",
    ObjectClassification.PEDESTRIAN: "pedestrian",
    ObjectClassification.UNKNOWN: "unknown",
}

CATEGORY_COLORS: dict[str, str] = {
    "vehicle": "#1E90FF",      # dodger blue
    "pedestrian": "#FF69B4",   # hot pink
    "bicycle": "#32CD32",      # lime green
    "unknown": "#AAAAAA",      # gray
}

CATEGORY_ORDER = ["vehicle", "pedestrian", "bicycle", "unknown"]


@dataclass
class ObjectBox:
    """1つのTrackedObjectの位置・姿勢・サイズ・カテゴリ。"""
    x: float
    y: float
    yaw: float  # [rad]
    length: float  # shape.dimensions.x
    width: float   # shape.dimensions.y
    category: str


@dataclass
class FrameResult:
    t_sec: float
    ego_x: float
    ego_y: float
    ego_yaw: float
    ego_length: float
    ego_width: float
    objects: list[ObjectBox] = field(default_factory=list)
    category_counts: dict[str, int] = field(default_factory=lambda: {c: 0 for c in CATEGORY_ORDER})
    total_in_range: int = 0
    total_all: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rosbagを解析して自車周囲のTrackedObjectの数をクラス別に計算・可視化する"
    )
    parser.add_argument("rosbag_path", type=Path)
    parser.add_argument("--range_m", type=float, default=200.0, help="周囲の範囲 [m] (正方形の一辺)")
    parser.add_argument("--save_dir", type=Path, default=None, help="保存先ディレクトリ（指定しない場合はrosbagと同階層）")
    parser.add_argument("--fps", type=int, default=10, help="動画のFPS")
    parser.add_argument("--frame_step", type=int, default=10, help="フレーム画像を何フレームおきに出力するか (デフォルト10=1秒おき)")
    parser.add_argument("--no_video", action="store_true", help="動画生成をスキップ")
    parser.add_argument("--ego_length", type=float, default=4.89, help="自車の長さ [m]")
    parser.add_argument("--ego_width", type=float, default=1.84, help="自車の幅 [m]")
    return parser.parse_args()


def parse_timestamp(stamp) -> int:
    return stamp.sec * 10**9 + stamp.nanosec


def get_nearest_kinematic_state(
    kinematic_states: list[Odometry],
    target_stamp,
    search_start: int,
) -> tuple[Odometry | None, int]:
    """target_stampに最も近いkinematic_stateを返す。"""
    target_ns = parse_timestamp(target_stamp)
    best_idx = search_start
    best_diff = abs(parse_timestamp(kinematic_states[search_start].header.stamp) - target_ns)
    for idx in range(search_start + 1, len(kinematic_states)):
        diff = abs(parse_timestamp(kinematic_states[idx].header.stamp) - target_ns)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx
        elif diff > best_diff:
            break
    if best_diff > int(0.5 * 1e9):
        return None, best_idx
    return kinematic_states[best_idx], best_idx


def get_object_category(obj) -> str:
    """TrackedObjectのclassificationからカテゴリを返す。"""
    if len(obj.classification) == 0:
        return "unknown"
    best_cls = obj.classification[0]
    for cls in obj.classification[1:]:
        if cls.probability > best_cls.probability:
            best_cls = cls
    return LABEL_TO_CATEGORY[best_cls.label]


def quaternion_to_yaw(orientation) -> float:
    """geometry_msgs/Quaternion -> yaw [rad]"""
    rot = Rotation.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
    return rot.as_euler("xyz")[2]


def make_box_corners(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    """中心(x,y), yaw, length, widthから4隅の座標を返す (4, 2)。"""
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    # ローカル座標での4隅 (前右, 前左, 後左, 後右)
    hl = length / 2.0
    hw = width / 2.0
    local_corners = np.array([
        [hl, -hw],
        [hl, hw],
        [-hl, hw],
        [-hl, -hw],
    ])
    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    return (rot @ local_corners.T).T + np.array([x, y])


def process_frame(
    tracking_msg: TrackedObjects,
    kin_msg: Odometry,
    half_range: float,
    t_sec: float,
    ego_length: float,
    ego_width: float,
) -> FrameResult:
    ego_x = kin_msg.pose.pose.position.x
    ego_y = kin_msg.pose.pose.position.y
    ego_yaw = quaternion_to_yaw(kin_msg.pose.pose.orientation)

    result = FrameResult(
        t_sec=t_sec, ego_x=ego_x, ego_y=ego_y,
        ego_yaw=ego_yaw, ego_length=ego_length, ego_width=ego_width,
    )
    result.total_all = len(tracking_msg.objects)

    for obj in tracking_msg.objects:
        pose = obj.kinematics.pose_with_covariance.pose
        ox = pose.position.x
        oy = pose.position.y
        if abs(ox - ego_x) > half_range or abs(oy - ego_y) > half_range:
            continue

        cat = get_object_category(obj)
        obj_yaw = quaternion_to_yaw(pose.orientation)
        obj_length = max(obj.shape.dimensions.x, 0.5)
        obj_width = max(obj.shape.dimensions.y, 0.5)

        result.objects.append(ObjectBox(
            x=ox, y=oy, yaw=obj_yaw,
            length=obj_length, width=obj_width, category=cat,
        ))
        result.category_counts[cat] += 1
        result.total_in_range += 1

    return result


def draw_frame(
    ax: plt.Axes,
    frame: FrameResult,
    half_range: float,
    range_m: float,
):
    """1フレーム分のXY散布図を描画する。オブジェクトは回転長方形。"""
    ax.set_aspect("equal")

    # 範囲の四角
    rect = plt.Rectangle(
        (frame.ego_x - half_range, frame.ego_y - half_range),
        range_m, range_m,
        linewidth=1, edgecolor="blue", facecolor="lightblue", alpha=0.15,
    )
    ax.add_patch(rect)

    # オブジェクトを長方形で描画
    drawn_categories: set[str] = set()
    for obj in frame.objects:
        corners = make_box_corners(obj.x, obj.y, obj.yaw, obj.length, obj.width)
        polygon = MplPolygon(
            corners, closed=True,
            facecolor=CATEGORY_COLORS[obj.category], edgecolor="black",
            linewidth=0.5, alpha=0.6, zorder=3,
        )
        ax.add_patch(polygon)
        drawn_categories.add(obj.category)

    # 自車を長方形で描画
    ego_corners = make_box_corners(
        frame.ego_x, frame.ego_y, frame.ego_yaw,
        frame.ego_length, frame.ego_width,
    )
    ego_polygon = MplPolygon(
        ego_corners, closed=True,
        facecolor="red", edgecolor="black",
        linewidth=1.0, alpha=0.8, zorder=4,
    )
    ax.add_patch(ego_polygon)

    # 凡例用ダミーパッチ
    legend_handles = [
        mpatches.Patch(facecolor="red", edgecolor="black", label="Ego"),
    ]
    for cat in CATEGORY_ORDER:
        if cat not in drawn_categories:
            continue
        cnt = frame.category_counts[cat]
        legend_handles.append(
            mpatches.Patch(facecolor=CATEGORY_COLORS[cat], edgecolor="black", label=f"{cat} ({cnt})")
        )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7)

    ax.set_xlim(frame.ego_x - half_range - 10, frame.ego_x + half_range + 10)
    ax.set_ylim(frame.ego_y - half_range - 10, frame.ego_y + half_range + 10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    count_text = ", ".join(
        f"{cat}: {frame.category_counts[cat]}" for cat in CATEGORY_ORDER if frame.category_counts[cat] > 0
    )
    ax.set_title(f"t={frame.t_sec:.1f}s | total={frame.total_in_range} ({count_text})")


def main():
    args = parse_args()
    rosbag_path: Path = args.rosbag_path
    range_m: float = args.range_m
    half_range = range_m / 2.0

    if args.save_dir is not None:
        save_dir: Path = args.save_dir
    else:
        save_dir = rosbag_path / "tracked_objects_analysis"

    save_dir.mkdir(parents=True, exist_ok=True)

    bag_name = rosbag_path.name  # e.g. "13-43-09"

    # --- rosbag読み込み ---
    serialization_format = "cdr"
    metadata_yaml_path = rosbag_path / "metadata.yaml"
    metadata_yaml = yaml.safe_load(metadata_yaml_path.read_text(encoding="utf-8"))
    storage_id = metadata_yaml["rosbag2_bagfile_information"]["storage_identifier"]
    storage_options = rosbag2_py.StorageOptions(uri=str(rosbag_path), storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format,
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    target_topic_list = [
        "/localization/kinematic_state",
        "/perception/object_recognition/tracking/objects",
    ]

    storage_filter = rosbag2_py.StorageFilter(topics=target_topic_list)
    reader.set_filter(storage_filter)

    topic_name_to_data: dict[str, list] = defaultdict(list)
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        if topic in target_topic_list:
            topic_name_to_data[topic].append(msg)

    tracking_msgs: list[TrackedObjects] = topic_name_to_data[
        "/perception/object_recognition/tracking/objects"
    ]
    kinematic_msgs: list[Odometry] = topic_name_to_data["/localization/kinematic_state"]

    print(f"TrackedObjects messages: {len(tracking_msgs)}")
    print(f"KinematicState messages: {len(kinematic_msgs)}")

    if len(tracking_msgs) == 0 or len(kinematic_msgs) == 0:
        print("No messages found. Exiting.")
        return

    # --- 各フレームで処理 ---
    frames: list[FrameResult] = []
    kin_search_start = 0
    first_timestamp_ns = parse_timestamp(tracking_msgs[0].header.stamp)

    for tracking_msg in tqdm(tracking_msgs, desc="Processing frames"):
        kin_msg, kin_search_start = get_nearest_kinematic_state(
            kinematic_msgs, tracking_msg.header.stamp, kin_search_start
        )
        if kin_msg is None:
            continue

        t_ns = parse_timestamp(tracking_msg.header.stamp)
        t_sec = (t_ns - first_timestamp_ns) / 1e9

        frame = process_frame(
            tracking_msg, kin_msg, half_range, t_sec,
            ego_length=args.ego_length, ego_width=args.ego_width,
        )
        frames.append(frame)

    if len(frames) == 0:
        print("No valid frames. Exiting.")
        return

    # --- 統計 ---
    total_counts = np.array([f.total_in_range for f in frames])
    cat_counts_dict: dict[str, np.ndarray] = {
        cat: np.array([f.category_counts[cat] for f in frames]) for cat in CATEGORY_ORDER
    }

    print(f"\n=== 範囲 {range_m}m 正方形内の TrackedObject 統計 ===")
    print(f"  フレーム数: {len(frames)}")
    print(f"  合計: min={total_counts.min()}, max={total_counts.max()}, "
          f"mean={total_counts.mean():.1f}, median={np.median(total_counts):.1f}")
    for cat in CATEGORY_ORDER:
        arr = cat_counts_dict[cat]
        if arr.sum() == 0:
            continue
        print(f"  {cat:12s}: min={arr.min()}, max={arr.max()}, "
              f"mean={arr.mean():.1f}, median={np.median(arr):.1f}")

    # --- 静的な可視化 (時系列 + ヒストグラム) ---
    timestamps = [f.t_sec for f in frames]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 時系列グラフ (カテゴリ別 stacked)
    ax = axes[0]
    bottom = np.zeros(len(frames))
    for cat in CATEGORY_ORDER:
        arr = cat_counts_dict[cat]
        if arr.sum() == 0:
            continue
        ax.fill_between(timestamps, bottom, bottom + arr, alpha=0.6,
                        color=CATEGORY_COLORS[cat], label=cat)
        bottom = bottom + arr
    ax.plot(timestamps, total_counts, color="black", linewidth=0.8, label="total", alpha=0.7)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Number of TrackedObjects")
    ax.set_title(f"TrackedObject count over time (range={range_m}m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ヒストグラム (カテゴリ別 stacked)
    ax = axes[1]
    hist_data = []
    hist_labels = []
    hist_colors = []
    for cat in CATEGORY_ORDER:
        arr = cat_counts_dict[cat]
        if arr.sum() == 0:
            continue
        hist_data.append(arr)
        hist_labels.append(cat)
        hist_colors.append(CATEGORY_COLORS[cat])
    bin_max = int(total_counts.max()) + 2
    ax.hist(hist_data, bins=range(0, bin_max), stacked=True,
            label=hist_labels, color=hist_colors, edgecolor="black", alpha=0.7)
    ax.set_xlabel(f"Number of TrackedObjects (in {range_m}m)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of TrackedObject count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    summary_path = save_dir / f"tracked_objects_count_{bag_name}.png"
    plt.savefig(summary_path, dpi=150)
    print(f"\nSaved summary: {summary_path}")
    plt.close(fig)

    # --- 動画生成 ---
    if args.no_video:
        print("Video generation skipped (--no_video).")
        return

    frames_dir = save_dir / f"tracked_objects_frames_{bag_name}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    step = args.frame_step
    selected_frames = frames[::step]
    print(f"\nGenerating {len(selected_frames)} frame images (step={step}, {len(frames)} total) to {frames_dir} ...")

    for i, frame in enumerate(tqdm(selected_frames, desc="Rendering frames")):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        draw_frame(ax, frame, half_range, range_m)
        frame_path = frames_dir / f"frame_{i:06d}.png"
        fig.savefig(frame_path, dpi=100)
        plt.close(fig)

    print(f"Saved {len(selected_frames)} frames to: {frames_dir}")

    # ffmpegで動画生成
    video_path = save_dir / f"tracked_objects_video_{bag_name}.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        str(video_path),
    ]
    print(f"Running: {' '.join(ffmpeg_cmd)}")
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg failed (exit code {result.returncode}):")
        print(result.stderr)
    else:
        print(f"Saved video: {video_path}")


if __name__ == "__main__":
    main()
