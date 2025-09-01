#!/usr/bin/env python3
"""
C++で保存されたTrainingDataBinaryバイナリファイルを読み込み、npz形式で保存するスクリプト
"""

import argparse
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


class TrainingDataReader:
    """バイナリファイルから学習データを読み込むクラス"""

    def __init__(self):
        # データ構造のサイズ定義（C++の構造体と同じ）
        self.PAST_TIME_STEPS = 21
        self.FUTURE_TIME_STEPS = 80
        self.NEIGHBOR_NUM = 32
        self.STATIC_NUM = 5
        self.LANE_NUM = 70
        self.LANE_LEN = 20
        self.ROUTE_NUM = 25
        self.ROUTE_LEN = 20
        self.SEGMENT_POINT_DIM = 13

        # 各配列のサイズを計算
        self.sizes = {
            "version": 1,
            "ego_agent_past": self.PAST_TIME_STEPS * 4,
            "ego_current_state": 10,
            "ego_agent_future": self.FUTURE_TIME_STEPS * 4,
            "neighbor_agents_past": self.NEIGHBOR_NUM * self.PAST_TIME_STEPS * 11,
            "neighbor_agents_future": self.NEIGHBOR_NUM * self.FUTURE_TIME_STEPS * 4,
            "static_objects": self.STATIC_NUM * 10,
            "lanes": self.LANE_NUM * self.LANE_LEN * self.SEGMENT_POINT_DIM,
            "lanes_speed_limit": self.LANE_NUM,
            "lanes_has_speed_limit": self.LANE_NUM,
            "route_lanes": self.ROUTE_NUM * self.ROUTE_LEN * self.SEGMENT_POINT_DIM,
            "route_lanes_speed_limit": self.ROUTE_NUM,
            "route_lanes_has_speed_limit": self.ROUTE_NUM,
            "goal_pose": 4,
            "turn_indicator": 1,
        }

    def read_binary_file(self, filepath: str) -> Dict[str, Any]:
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
        # cos(yaw), sin(yaw) から yaw を計算
        x = ego_past_array[:, 0]
        y = ego_past_array[:, 1]
        cos_yaw = ego_past_array[:, 2]
        sin_yaw = ego_past_array[:, 3]
        yaw = np.arctan2(sin_yaw, cos_yaw)
        result["ego_agent_past"] = np.column_stack([x, y, yaw]).astype(np.float32)
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
        ego_future_array = np.array(ego_future_flat).reshape(self.FUTURE_TIME_STEPS, 4)
        # cos(yaw), sin(yaw) から yaw を計算
        x = ego_future_array[:, 0]
        y = ego_future_array[:, 1]
        cos_yaw = ego_future_array[:, 2]
        sin_yaw = ego_future_array[:, 3]
        yaw = np.arctan2(sin_yaw, cos_yaw)
        result["ego_agent_future"] = np.column_stack([x, y, yaw]).astype(np.float32)
        offset += size * 4

        # neighbor_agents_past (32, 21, 11)
        size = self.sizes["neighbor_agents_past"]
        neighbor_past_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        result["neighbor_agents_past"] = np.array(neighbor_past_flat, dtype=np.float32).reshape(
            self.NEIGHBOR_NUM, self.PAST_TIME_STEPS, 11
        )
        offset += size * 4

        # neighbor_agents_future (32, 80, 4) -> (32, 80, 3)
        size = self.sizes["neighbor_agents_future"]
        neighbor_future_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        neighbor_future_array = np.array(neighbor_future_flat).reshape(
            self.NEIGHBOR_NUM, self.FUTURE_TIME_STEPS, 4
        )
        # cos(yaw), sin(yaw) から yaw を計算
        x = neighbor_future_array[:, :, 0]
        y = neighbor_future_array[:, :, 1]
        cos_yaw = neighbor_future_array[:, :, 2]
        sin_yaw = neighbor_future_array[:, :, 3]
        yaw = np.arctan2(sin_yaw, cos_yaw)
        result["neighbor_agents_future"] = np.stack([x, y, yaw], axis=-1).astype(np.float32)
        offset += size * 4

        # static_objects (5, 10)
        size = self.sizes["static_objects"]
        static_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        result["static_objects"] = np.array(static_flat, dtype=np.float32).reshape(
            self.STATIC_NUM, 10
        )
        offset += size * 4

        # lanes (70, 20, 13)
        size = self.sizes["lanes"]
        lanes_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        result["lanes"] = np.array(lanes_flat, dtype=np.float32).reshape(
            self.LANE_NUM, self.LANE_LEN, self.SEGMENT_POINT_DIM
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

        # route_lanes (25, 20, 13)
        size = self.sizes["route_lanes"]
        route_lanes_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        result["route_lanes"] = np.array(route_lanes_flat, dtype=np.float32).reshape(
            self.ROUTE_NUM, self.ROUTE_LEN, self.SEGMENT_POINT_DIM
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

        # goal_pose (4,) -> (3,)
        size = self.sizes["goal_pose"]
        goal_flat = struct.unpack(f"<{size}f", data[offset : offset + size * 4])
        goal_flat = np.array(goal_flat, dtype=np.float32)
        offset += size * 4
        # cos(yaw), sin(yaw) から yaw を計算
        x = goal_flat[0]
        y = goal_flat[1]
        cos_yaw = goal_flat[2]
        sin_yaw = goal_flat[3]
        yaw = np.arctan2(sin_yaw, cos_yaw)
        result["goal_pose"] = np.array([x, y, yaw], dtype=np.float32)

        # turn_indicator (scalar) - int32_t
        result["turn_indicator"] = struct.unpack("<i", data[offset : offset + 4])[0]
        offset += 4

        return result

    def save_as_npz(
        self,
        data: Dict[str, Any],
        output_path: str,
        token: str = "00000000",
    ) -> None:
        """
        データをnpz形式で保存する (parse_rosbag.pyと同じ形式)

        Args:
            data: 読み込んだデータの辞書
            output_path: 出力ファイルのパス
            token: トークン
        """
        # parse_rosbag.pyと同じ形式でデータを準備
        npz_data = {
            "token": token,
            "ego_agent_past": data["ego_agent_past"],  # (21, 3)
            "ego_current_state": data["ego_current_state"],  # (10,)
            "ego_agent_future": data["ego_agent_future"],  # (80, 3)
            "neighbor_agents_past": data["neighbor_agents_past"],  # (32, 21, 11)
            "neighbor_agents_future": data["neighbor_agents_future"],  # (32, 80, 3)
            "static_objects": data["static_objects"],  # (5, 10)
            "lanes": data["lanes"],  # (70, 20, 13)
            "lanes_speed_limit": data["lanes_speed_limit"],  # (70, 1)
            "lanes_has_speed_limit": data["lanes_has_speed_limit"],  # (70, 1)
            "route_lanes": data["route_lanes"],  # (25, 20, 13)
            "route_lanes_speed_limit": data["route_lanes_speed_limit"],  # (25, 1)
            "route_lanes_has_speed_limit": data["route_lanes_has_speed_limit"],  # (25, 1)
            "turn_indicator": data["turn_indicator"],  # scalar
            "goal_pose": data["goal_pose"],  # (3,)
        }

        # npzファイルとして保存
        np.savez(output_path, **npz_data)


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
        reader.save_as_npz(data, str(output_file), token)

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
        bin_files = list(input_path.glob("*.bin"))
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
