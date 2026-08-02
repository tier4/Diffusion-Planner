#!/usr/bin/env python3
"""path_list*.json の npz を間引いて読み、normalization.json を計算し直す。

学習時 (train_epoch.py) の前処理を再現してから統計を取る。
  * ego_agent_past / ego_agent_future は heading -> (cos, sin) に変換
  * 全要素 0 の行 (padding) は ObservationNormalizer と同じ基準で除外
    (neighbor_agents_future のみ train_epoch.py と同じく先頭 3 列で判定)

出力キーは Diffusion-Planner/diffusion_planner/normalization.json と同じ構成。
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

# 出力キー -> (npz のキー, heading を cos/sin に変換するか)
# 並びは既存 normalization.json に合わせている
KEY_SPEC = {
    "ego": ("ego_agent_future", True),
    "neighbor": ("neighbor_agents_future", True),
    "ego_agent_past": ("ego_agent_past", True),
    "ego_current_state": ("ego_current_state", False),
    "neighbor_agents_past": ("neighbor_agents_past", False),
    "static_objects": ("static_objects", False),
    "lanes": ("lanes", False),
    "lanes_speed_limit": ("lanes_speed_limit", False),
    "route_lanes": ("route_lanes", False),
    "polygons": ("polygons", False),
    "line_strings": ("line_strings", False),
    "route_lanes_speed_limit": ("route_lanes_speed_limit", False),
    # goal_pose is not normalized (GoalPoseEncoder handles it based on distance),
    # so no statistics are collected for it
}

# padding 判定に先頭 3 列だけを使うキー (train_epoch.py の mask と同じ)
FIRST3_MASK_KEYS = {"neighbor"}


def load_sampled_paths(json_file: Path, stride: int) -> list[str]:
    """path_list json を読み、stride 個ごとに 1 つパスを取り出す。"""
    with json_file.open("r", encoding="utf-8") as f:
        paths = json.load(f)
    return paths[::stride]


def rows_of(array: np.ndarray, convert_heading: bool) -> np.ndarray:
    """npz の配列を [N, D] の行列に整形する (heading 変換込み)。"""
    if array.ndim == 1:
        array = array.reshape(1, -1)
    rows = array.reshape(-1, array.shape[-1]).astype(np.float64)
    if convert_heading and rows.shape[-1] == 3:
        rows = np.concatenate(
            [rows[:, :2], np.cos(rows[:, 2:3]), np.sin(rows[:, 2:3])],
            axis=-1,
        )
    return rows


def accumulate_chunk(paths: list[str]) -> tuple[dict[str, tuple[int, list, list]], int]:
    """npz 群から key ごとの (件数, 総和, 二乗和) を求める。並列実行の単位。"""
    counts: dict[str, int] = {}
    sums: dict[str, np.ndarray] = {}
    sq_sums: dict[str, np.ndarray] = {}
    num_failed = 0

    for path in paths:
        try:
            data = np.load(path, allow_pickle=True)
            arrays = {key: data[key] for key in data.files}
        except Exception:
            num_failed += 1
            continue

        for out_key, (npz_key, convert_heading) in KEY_SPEC.items():
            if npz_key not in arrays:
                continue
            rows = rows_of(arrays[npz_key], convert_heading)
            if out_key in FIRST3_MASK_KEYS:
                valid = np.any(rows[:, :3] != 0, axis=-1)
            else:
                valid = np.any(rows != 0, axis=-1)
            rows = rows[valid]
            if rows.shape[0] == 0:
                continue
            if out_key not in counts:
                counts[out_key] = 0
                sums[out_key] = np.zeros(rows.shape[-1], dtype=np.float64)
                sq_sums[out_key] = np.zeros(rows.shape[-1], dtype=np.float64)
            counts[out_key] += rows.shape[0]
            sums[out_key] += rows.sum(axis=0)
            sq_sums[out_key] += (rows**2).sum(axis=0)

    packed = {key: (counts[key], sums[key].tolist(), sq_sums[key].tolist()) for key in counts}
    return packed, num_failed


def chunk_list(items: list[str], chunk_size: int) -> list[list[str]]:
    """リストを chunk_size ごとに分割する。"""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_list_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--stride", type=int, required=True, help="何件ごとに 1 件使うか")
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument(
        "--min_std",
        type=float,
        required=True,
        help="この値未満の std は 1.0 に置き換える (定数次元の 0 除算対策)",
    )
    args = parser.parse_args()

    print(f"Loading path list: {args.path_list_json}")
    sampled_paths = load_sampled_paths(args.path_list_json, args.stride)
    print(f"Sampled {len(sampled_paths):,} npz files (stride={args.stride})")

    chunks = chunk_list(sampled_paths, 64)
    total_counts: dict[str, int] = {}
    total_sums: dict[str, np.ndarray] = {}
    total_sq_sums: dict[str, np.ndarray] = {}
    total_failed = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = executor.map(accumulate_chunk, chunks)
        for packed, num_failed in tqdm(results, total=len(chunks), unit="chunk"):
            total_failed += num_failed
            for key, (count, sum_list, sq_sum_list) in packed.items():
                if key not in total_counts:
                    total_counts[key] = 0
                    total_sums[key] = np.zeros(len(sum_list), dtype=np.float64)
                    total_sq_sums[key] = np.zeros(len(sq_sum_list), dtype=np.float64)
                total_counts[key] += count
                total_sums[key] += np.asarray(sum_list, dtype=np.float64)
                total_sq_sums[key] += np.asarray(sq_sum_list, dtype=np.float64)

    if total_failed > 0:
        print(f"読み込みに失敗した npz: {total_failed:,} 件 (スキップ)")

    result = {}
    for out_key in KEY_SPEC:
        if out_key not in total_counts:
            print(f"警告: {out_key} のデータが 1 件も無かったので出力から除外します")
            continue
        count = total_counts[out_key]
        mean = total_sums[out_key] / count
        variance = np.maximum(0.0, total_sq_sums[out_key] / count - mean**2)
        std = np.sqrt(variance)
        degenerate = std < args.min_std
        if np.any(degenerate):
            dims = np.flatnonzero(degenerate).tolist()
            print(f"警告: {out_key} の次元 {dims} は std < {args.min_std} なので 1.0 にします")
            std[degenerate] = 1.0
        result[out_key] = {
            "mean": np.round(mean, 4).tolist(),
            "std": np.round(std, 4).tolist(),
        }
        print(f"{out_key:26s} rows={count:12,}")
        print(f"  mean={np.round(mean, 3).tolist()}")
        print(f"  std ={np.round(std, 3).tolist()}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print()
    print(f"Saved to {args.output_json}")
