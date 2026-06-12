#!/usr/bin/env python3
"""
C++で保存されたTrainingDataBinaryバイナリファイルを読み込み、npz形式で保存するスクリプト
"""

import argparse
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from diffusion_planner.dimensions import *
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def cos_sin_to_heading(x: np.ndarray) -> np.ndarray:
    """
        Convert heading angle to cosine and sine.
    Args:
        x: [B, T, 4] where last dimension is (x, y, cos(heading), sin(heading))
    Output:
        x: [B, T, 3] where last dimension is (x, y, heading)
    """
    heading = np.arctan2(x[..., 3], x[..., 2])
    heading = np.expand_dims(heading, axis=-1)
    return np.concatenate([x[..., :2], heading], axis=-1).astype(np.float32)


class TrainingDataReader:
    """バイナリファイルから学習データを読み込むクラス"""

    def __init__(self):
        # データ構造のサイズ定義（C++の構造体と同じ）
        self.PAST_TIME_STEPS = INPUT_T + 1
        self.STATIC_NUM = 5

        # 各配列のサイズを計算
        self.sizes = {
            "version": 1,
            "ego_agent_past": self.PAST_TIME_STEPS * POSE_DIM,
            "ego_current_state": 10,
            "ego_agent_future": OUTPUT_T * POSE_DIM,
            "neighbor_agents_past": MAX_NUM_NEIGHBORS * self.PAST_TIME_STEPS * 11,
            "neighbor_agents_future": MAX_NUM_NEIGHBORS * OUTPUT_T * POSE_DIM,
            "static_objects": self.STATIC_NUM * 10,
            "lanes": NUM_SEGMENTS_IN_LANE * POINTS_PER_LANELET * SEGMENT_POINT_DIM,
            "lanes_speed_limit": NUM_SEGMENTS_IN_LANE,
            "lanes_has_speed_limit": NUM_SEGMENTS_IN_LANE,
            "route_lanes": NUM_SEGMENTS_IN_ROUTE * POINTS_PER_LANELET * SEGMENT_POINT_DIM,
            "route_lanes_speed_limit": NUM_SEGMENTS_IN_ROUTE,
            "route_lanes_has_speed_limit": NUM_SEGMENTS_IN_ROUTE,
            "polygons": NUM_POLYGONS * POINTS_PER_POLYGON * (2 + POLYGON_TYPE_NUM),
            "line_strings": NUM_LINE_STRINGS * POINTS_PER_LINE_STRING * (2 + LINE_STRING_TYPE_NUM),
            "goal_pose": POSE_DIM,
            "turn_indicators": self.PAST_TIME_STEPS,
            "ego_shape": 3,  # (wheel_base, length, width)
        }

    def read_binary_file(self, filepath: str) -> dict[str, object]:
        """
        バイナリファイルを読み込んでデータを辞書として返す

        Args:
            filepath: バイナリファイルのパス

        Returns:
            読み込んだデータの辞書
        """
        with open(filepath, "rb") as f:
            data = f.read()

        # バイナリデータを解析
        offset = 0
        result = {}

        # version (uint32_t)
        result["version"] = struct.unpack("<I", data[offset : offset + 4])[0]
        offset += 4

        # ego_agent_past (21, 4) -> (21, 3) to match expected format
        # 4次元: [x, y, cos(yaw), sin(yaw)] -> 3次元: [x, y, yaw]
        size = self.sizes["ego_agent_past"]
        ego_past_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        ego_past_array = np.array(ego_past_flat).reshape(self.PAST_TIME_STEPS, 4)
        result["ego_agent_past"] = cos_sin_to_heading(ego_past_array)
        offset += size * 4

        # ego_current_state (10,)
        size = self.sizes["ego_current_state"]
        ego_current_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        result["ego_current_state"] = np.array(ego_current_flat, dtype=np.float32)
        offset += size * 4

        # ego_agent_future (80, 4) -> (80, 3) to match expected format
        # 4次元: [x, y, cos(yaw), sin(yaw)] -> 3次元: [x, y, yaw]
        size = self.sizes["ego_agent_future"]
        ego_future_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        ego_future_array = np.array(ego_future_flat).reshape(OUTPUT_T, 4)
        result["ego_agent_future"] = cos_sin_to_heading(ego_future_array)
        offset += size * 4

        # neighbor_agents_past (32, 21, 11)
        size = self.sizes["neighbor_agents_past"]
        neighbor_past_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        result["neighbor_agents_past"] = np.array(neighbor_past_flat, dtype=np.float32).reshape(
            MAX_NUM_NEIGHBORS, self.PAST_TIME_STEPS, 11
        )
        offset += size * 4

        # neighbor_agents_future (32, 80, 4) -> (32, 80, 3)
        size = self.sizes["neighbor_agents_future"]
        neighbor_future_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        neighbor_future_array = np.array(neighbor_future_flat).reshape(
            MAX_NUM_NEIGHBORS, OUTPUT_T, 4
        )
        result["neighbor_agents_future"] = cos_sin_to_heading(neighbor_future_array)
        offset += size * 4

        # static_objects (5, 10)
        size = self.sizes["static_objects"]
        static_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        result["static_objects"] = np.array(static_flat, dtype=np.float32).reshape(
            self.STATIC_NUM, 10
        )
        offset += size * 4

        # lanes (70, 20, SEGMENT_POINT_DIM)
        size = self.sizes["lanes"]
        lanes_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        result["lanes"] = np.array(lanes_flat, dtype=np.float32).reshape(
            NUM_SEGMENTS_IN_LANE, POINTS_PER_LANELET, SEGMENT_POINT_DIM
        )
        offset += size * 4

        # lanes_speed_limit (70,) -> (70, 1) to match expected format
        size = self.sizes["lanes_speed_limit"]
        lanes_speed_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        result["lanes_speed_limit"] = np.array(lanes_speed_flat, dtype=np.float32).reshape(-1, 1)
        offset += size * 4

        # lanes_has_speed_limit (70,) -> (70, 1) to match expected format - int32_t
        size = self.sizes["lanes_has_speed_limit"]
        lanes_has_flat = struct.unpack(f"<{size}i", data[offset : offset + size * 4])
        result["lanes_has_speed_limit"] = np.array(lanes_has_flat, dtype=bool).reshape(-1, 1)
        offset += size * 4

        # route_lanes (25, 20, SEGMENT_POINT_DIM)
        size = self.sizes["route_lanes"]
        route_lanes_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        result["route_lanes"] = np.array(route_lanes_flat, dtype=np.float32).reshape(
            NUM_SEGMENTS_IN_ROUTE, POINTS_PER_LANELET, SEGMENT_POINT_DIM
        )
        offset += size * 4

        # route_lanes_speed_limit (25,) -> (25, 1) to match expected format
        size = self.sizes["route_lanes_speed_limit"]
        route_speed_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        result["route_lanes_speed_limit"] = np.array(route_speed_flat, dtype=np.float32).reshape(
            -1, 1
        )
        offset += size * 4

        # route_lanes_has_speed_limit (25,) -> (25, 1) to match expected format - int32_t
        size = self.sizes["route_lanes_has_speed_limit"]
        route_has_flat = struct.unpack(f"<{size}i", data[offset : offset + size * 4])
        result["route_lanes_has_speed_limit"] = np.array(route_has_flat, dtype=bool).reshape(-1, 1)
        offset += size * 4

        # polygons (10, 40, 2 + POLYGON_TYPE_NUM)
        size = self.sizes["polygons"]
        polygons_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        result["polygons"] = np.array(polygons_flat, dtype=np.float32).reshape(
            NUM_POLYGONS, POINTS_PER_POLYGON, 2 + POLYGON_TYPE_NUM
        )
        offset += size * 4

        # line_strings (10, 20, 2 + LINE_STRING_TYPE_NUM)
        size = self.sizes["line_strings"]
        line_strings_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        result["line_strings"] = np.array(line_strings_flat, dtype=np.float32).reshape(
            NUM_LINE_STRINGS, POINTS_PER_LINE_STRING, 2 + LINE_STRING_TYPE_NUM
        )
        offset += size * 4

        # goal_pose (4,) -> (3,)
        size = self.sizes["goal_pose"]
        goal_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        goal_flat = np.array(goal_flat, dtype=np.float32).reshape(1, 4)
        result["goal_pose"] = cos_sin_to_heading(goal_flat).reshape(3)
        offset += size * 4

        # turn_indicators (scalar) - int32_t
        size = self.sizes["turn_indicators"]
        turn_flat = struct.unpack(f"<{size}i", data[offset : offset + size * 4])
        result["turn_indicators"] = np.array(turn_flat, dtype=np.int32)
        offset += size * 4

        # ego_shape (3,)
        size = self.sizes["ego_shape"]
        shape_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        result["ego_shape"] = np.array(shape_flat, dtype=np.float32)
        offset += size * 4

        return result


def process_single_file_worker(args):
    """並列処理用のワーカー関数"""
    input_file, output_dir = args
    process_single_file(input_file, output_dir)
    return str(input_file)


def process_single_file(input_file: Path, output_dir: Path) -> None:
    """単一のバイナリファイルを処理する"""
    try:
        # 各プロセスで独自のreaderインスタンスを作成
        reader = TrainingDataReader()

        # バイナリファイルを読み込み
        data = reader.read_binary_file(str(input_file))

        # トークンを生成（ファイル名から拡張子を除いたもの）
        token = input_file.stem

        # 出力ファイルパスを生成
        output_file = output_dir / f"{token}.npz"

        # npz形式で保存
        np.savez_compressed(str(output_file), **data)

        # 元のバイナリファイルを削除
        input_file.unlink()

    except Exception as e:
        print(f"Error processing {input_file}: {e}")


if __name__ == "__main__":
    args = parse_args()

    input_path = args.input_path
    output_dir = args.output_dir if args.output_dir else input_path.parent
    num_workers = args.num_workers

    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        # 単一ファイルの処理
        if input_path.suffix == ".bin":
            process_single_file(input_path, output_dir)
        else:
            print(f"Error: {input_path} is not a .bin file")
    elif input_path.is_dir():
        # ディレクトリ内のすべての.binファイルを処理
        bin_files = sorted(input_path.glob("*.bin"))
        if not bin_files:
            print(f"No .bin files found in {input_path}")
            sys.exit(1)

        print(f"Found {len(bin_files)} .bin files in {input_path}")

        # 並列処理で全ファイルを処理
        print(
            f"Processing {len(bin_files)} files in parallel with {num_workers or 'CPU count'} workers..."
        )

        worker_args = [(bin_file, output_dir) for bin_file in bin_files]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_single_file_worker, worker_args),
                    total=len(bin_files),
                    desc="Processing files",
                )
            )

        print(f"Successfully processed {len(results)} files")
    else:
        print(f"Error: {input_path} does not exist")
