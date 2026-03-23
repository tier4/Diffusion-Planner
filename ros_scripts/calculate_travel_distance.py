#!/usr/bin/env python3
"""
指定したroot_dirから再帰的にディレクトリを探索し、
末端ディレクトリにあるJSONファイルを読み取って移動距離を計算するスクリプト
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


def find_leaf_directories(root_dir: Path) -> List[Path]:
    """
    再帰的に探索して末端ディレクトリ（サブディレクトリを持たないディレクトリ）を見つける

    Args:
        root_dir: 探索の起点となるディレクトリ

    Returns:
        末端ディレクトリのリスト
    """
    leaf_dirs = []

    for item in root_dir.rglob("*"):
        if item.is_dir():
            # サブディレクトリがあるかチェック
            has_subdirs = any(sub.is_dir() for sub in item.iterdir())
            if not has_subdirs:
                leaf_dirs.append(item)

    return sorted(leaf_dirs)


def load_json_data(directory: Path) -> List[Dict]:
    """
    ディレクトリ内のすべてのJSONファイルを読み込む

    Args:
        directory: JSONファイルがあるディレクトリ

    Returns:
        JSONデータのリスト（タイムスタンプでソート済み）
    """
    json_files = sorted(directory.glob("*.json"))
    data_list = []

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                data_list.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    # タイムスタンプでソート
    data_list.sort(key=lambda x: x.get("timestamp", 0))

    return data_list


def calculate_distance_2d(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """
    2点間の2D距離を計算

    Args:
        pos1: 1つ目の座標 (x, y)
        pos2: 2つ目の座標 (x, y)

    Returns:
        2点間の距離
    """
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return math.sqrt(dx * dx + dy * dy)


def calculate_distance_3d(
    pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]
) -> float:
    """
    2点間の3D距離を計算

    Args:
        pos1: 1つ目の座標 (x, y, z)
        pos2: 2つ目の座標 (x, y, z)

    Returns:
        2点間の距離
    """
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    dz = pos2[2] - pos1[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def calculate_travel_distance(data_list: List[Dict], use_3d: bool = False) -> float:
    """
    JSONデータから移動距離を計算

    Args:
        data_list: 位置情報を含むJSONデータのリスト
        use_3d: True の場合は3D距離、False の場合は2D距離を計算

    Returns:
        合計移動距離
    """
    if len(data_list) < 2:
        return 0.0

    total_distance = 0.0

    for i in range(len(data_list) - 1):
        curr_data = data_list[i]
        next_data = data_list[i + 1]

        if use_3d:
            pos1 = (curr_data["x"], curr_data["y"], curr_data["z"])
            pos2 = (next_data["x"], next_data["y"], next_data["z"])
            distance = calculate_distance_3d(pos1, pos2)
        else:
            pos1 = (curr_data["x"], curr_data["y"])
            pos2 = (next_data["x"], next_data["y"])
            distance = calculate_distance_2d(pos1, pos2)

        total_distance += distance

    return total_distance


def calculate_duration(data_list: List[Dict]) -> float:
    """
    JSONデータから時間の長さ（duration）を計算

    Args:
        data_list: タイムスタンプを含むJSONデータのリスト

    Returns:
        duration（秒）
    """
    if len(data_list) < 2:
        return 0.0

    # タイムスタンプはナノ秒単位と仮定
    start_time = data_list[0]["timestamp"]
    end_time = data_list[-1]["timestamp"]

    duration_ns = end_time - start_time
    duration_sec = duration_ns / 1e9  # ナノ秒から秒に変換

    return duration_sec


def save_results_to_csv(results: List[Tuple], output_path: Path, root_path: Path) -> None:
    """
    結果をCSVファイルに保存

    Args:
        results: (relative_path, distance, num_points, duration)のタプルのリスト
        output_path: 出力CSVファイルのパス
        root_path: ルートディレクトリパス（メタ情報用）
    """
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # ヘッダー行
        writer.writerow(["Directory", "Distance_m", "Duration_s", "Points", "Average_Speed_m_s"])

        # データ行
        total_distance = 0.0
        total_duration = 0.0
        total_points = 0

        for rel_path, distance, num_points, duration in results:
            avg_speed = distance / duration if duration > 0 else 0.0
            writer.writerow(
                [
                    str(rel_path),
                    f"{distance:.2f}",
                    f"{duration:.2f}",
                    num_points,
                    f"{avg_speed:.2f}",
                ]
            )
            total_distance += distance
            total_duration += duration
            total_points += num_points

        # 合計行
        avg_speed_total = total_distance / total_duration if total_duration > 0 else 0.0
        writer.writerow(
            [
                "Total",
                f"{total_distance:.2f}",
                f"{total_duration:.2f}",
                total_points,
                f"{avg_speed_total:.2f}",
            ]
        )

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="指定したディレクトリから再帰的にJSONファイルを探索し、移動距離を計算します"
    )
    parser.add_argument("root_dir", type=str, help="探索の起点となるディレクトリ")
    parser.add_argument(
        "--3d", dest="use_3d", action="store_true", help="3D距離を計算（デフォルトは2D距離）"
    )
    parser.add_argument(
        "--min-distance", type=float, default=0.0, help="表示する最小移動距離（メートル）"
    )
    parser.add_argument(
        "--csv", type=str, default=None, help="結果をCSVファイルに保存（ファイルパスを指定）"
    )

    args = parser.parse_args()

    root_path = Path(args.root_dir)

    if not root_path.exists():
        print(f"Error: Directory '{args.root_dir}' does not exist.")
        return

    if not root_path.is_dir():
        print(f"Error: '{args.root_dir}' is not a directory.")
        return

    print(f"Searching for leaf directories in: {root_path}")
    print(f"Distance calculation mode: {'3D' if args.use_3d else '2D'}")
    print("-" * 80)

    # 末端ディレクトリを見つける
    leaf_dirs = find_leaf_directories(root_path)

    if not leaf_dirs:
        print("No leaf directories found.")
        return

    print(f"Found {len(leaf_dirs)} leaf directories.\n")

    # 各ディレクトリの移動距離を計算
    results = []

    for leaf_dir in leaf_dirs:
        if leaf_dir.name == "for_sft":
            continue

        # JSONファイルを読み込む
        data_list = load_json_data(leaf_dir)

        if not data_list:
            continue

        # 移動距離を計算
        distance = calculate_travel_distance(data_list, args.use_3d)

        # 時間の長さを計算
        duration = calculate_duration(data_list)

        if distance >= args.min_distance:
            relative_path = leaf_dir.relative_to(root_path)
            results.append((relative_path, distance, len(data_list), duration))
            print(
                f"Directory: {relative_path}, Distance: {distance:.2f} m, Duration: {duration:.2f} s, Points: {len(data_list)}"
            )

    # 結果を表示
    if results:
        print(f"{'Directory':<60} {'Distance (m)':<15} {'Duration (s)':<15} {'Points':<10}")
        print("=" * 100)

        total_distance = 0.0
        total_duration = 0.0
        total_points = 0

        for rel_path, distance, num_points, duration in results:
            print(f"{str(rel_path):<60} {distance:>14.2f} {duration:>14.2f} {num_points:>9}")
            total_distance += distance
            total_duration += duration
            total_points += num_points

        print("=" * 100)
        print(f"{'Total:':<60} {total_distance:>14.2f} {total_duration:>14.2f} {total_points:>9}")
        print(f"\nTotal directories: {len(results)}")

        # CSV出力
        if args.csv:
            csv_path = Path(args.csv)
            save_results_to_csv(results, csv_path, root_path)
    else:
        print("No data found or all distances are below the minimum threshold.")


if __name__ == "__main__":
    main()
