import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_path_json", type=Path)
    parser.add_argument("prediction_results_dir", type=Path)
    return parser.parse_args()


def calc_loss(inputs, prediction) -> tuple:
    ego_future = inputs["ego_agent_future"]  # (T, 4) - already (x, y, cos(yaw), sin(yaw))
    # 元の角度情報を復元 (x, y, yaw)
    ego_future_original = np.concatenate(
        [
            ego_future[..., :2],  # x, y
            np.arctan2(ego_future[..., 3:4], ego_future[..., 2:3]),  # yaw from cos, sin
        ],
        axis=-1,
    )  # (T, 3)

    neighbors_future = inputs[
        "neighbor_agents_future"
    ]  # (P32, T, 4) - already (x, y, cos(yaw), sin(yaw))
    neighbor_future_mask = np.sum((neighbors_future[..., :4] != 0), axis=-1) == 0  # (P32, T)
    # 元の角度情報を復元 (x, y, yaw)
    neighbors_future_original = np.concatenate(
        [
            neighbors_future[..., :2],  # x, y
            np.arctan2(neighbors_future[..., 3:4], neighbors_future[..., 2:3]),  # yaw from cos, sin
        ],
        axis=-1,
    )  # (P32, T, 3)
    neighbors_future[neighbor_future_mask] = 0.0

    P32, T, _ = neighbors_future.shape
    ego_current, neighbors_current = (
        inputs["ego_current_state"][:4],
        inputs["neighbor_agents_past"][:P32, -1, :4],
    )
    # 元の角度情報を復元 (x, y, yaw)
    ego_current_original = np.concatenate(
        [
            inputs["ego_current_state"][:2],
            [np.arctan2(inputs["ego_current_state"][3], inputs["ego_current_state"][2])],
        ]
    )  # (3,)
    neighbors_current_original = np.concatenate(
        [
            inputs["neighbor_agents_past"][:P32, -1, :2],
            np.arctan2(
                inputs["neighbor_agents_past"][:P32, -1, 3:4],
                inputs["neighbor_agents_past"][:P32, -1, 2:3],
            ),
        ],
        axis=-1,
    )  # (P32, 3)
    # inputs = args.observation_normalizer(inputs)

    neighbor_current_mask = np.sum((neighbors_current[..., :4] != 0), axis=-1) == 0  # (P32)
    neighbor_mask = np.concatenate(
        (neighbor_current_mask[:, None], neighbor_future_mask), axis=-1
    )  # (P32, T + 1)
    neighbor_mask = neighbor_mask[:, -1]  # (P32) same to time axis

    gt_future = np.concatenate(
        [ego_future[None, :, :], neighbors_future[..., :]], axis=0
    )  # (1 + P32, T, 4)
    current_states = np.concatenate([ego_current[None, :], neighbors_current], axis=0)
    # (1 + P32, 4)

    # 元の角度情報を含むGTを作成
    gt_future_original = np.concatenate(
        [ego_future_original[None, :, :], neighbors_future_original[..., :]], axis=0
    )  # (1 + P32, T, 3)
    current_states_original = np.concatenate(
        [ego_current_original[None, :], neighbors_current_original], axis=0
    )
    # (1 + P32, 3)

    all_gt = np.concatenate([current_states[:, None, :], gt_future], axis=1)  # (1 + P32, T + 1, 4)
    all_gt_original = np.concatenate(
        [current_states_original[:, None, :], gt_future_original], axis=1
    )  # (1 + P32, T + 1, 3)
    all_gt[1:][neighbor_mask] = 0.0  # (P32, T + 1, 4)
    all_gt_original[1:][neighbor_mask] = 0.0  # (P32, T + 1, 3)

    pred_size = prediction.shape[0]

    neighbors_future_valid = ~neighbor_future_mask  # (P32, T)
    neighbors_future_valid = neighbors_future_valid[: (pred_size - 1)]  # (P10, T)
    all_gt = all_gt[:, 1:, :]  # (1 + P32, T, 4)
    all_gt_original = all_gt_original[:, 1:, :]  # (1 + P32, T, 3)
    all_gt = all_gt[:pred_size, :, :]  # (1 + P10, T, 4)
    all_gt_original = all_gt_original[:pred_size, :, :]  # (1 + P10, T, 3)

    # 予測値から角度を復元
    prediction_original = prediction.copy()
    prediction_original[..., 2] = np.arctan2(
        prediction[..., 3], prediction[..., 2]
    )  # cos, sinから角度を復元

    loss_tensor = (prediction - all_gt) ** 2  # (1 + P10, T, 4)
    loss_ego = loss_tensor[0, :]  # (T, 4)
    loss_nei = loss_tensor[1:, :]  # (P10, T, 4)

    # 位置と角度の個別誤差を計算
    # poseから緯度経度方向の誤差を計算
    # prediction_original[..., :2] = [x, y], prediction_original[..., 2] = heading
    # all_gt_original[..., :2] = [x, y], all_gt_original[..., 2] = heading

    # 位置の差分ベクトルを計算
    position_diff = prediction_original[..., :2] - all_gt_original[..., :2]  # (1 + P10, T, 2)

    # GTの向き（heading）を使って、車両座標系での縦方向（lon）と横方向（lat）誤差を計算
    gt_heading = all_gt_original[..., 2]  # (1 + P10, T)
    cos_heading = np.cos(gt_heading)  # (1 + P10, T)
    sin_heading = np.sin(gt_heading)  # (1 + P10, T)

    # 車両座標系への変換：縦方向（進行方向）と横方向（進行方向に垂直）
    lon_error = (
        position_diff[..., 0] * cos_heading + position_diff[..., 1] * sin_heading
    )  # 縦方向誤差
    lat_error = (
        -position_diff[..., 0] * sin_heading + position_diff[..., 1] * cos_heading
    )  # 横方向誤差

    # 絶対値を取る
    lat_error = np.abs(lat_error)  # (1 + P10, T)
    lon_error = np.abs(lon_error)  # (1 + P10, T)

    lat_error_ego = lat_error[0, :]  # (T,) - ego lat error
    lon_error_ego = lon_error[0, :]  # (T,) - ego lon error
    lat_error_nei = lat_error[1:, :]  # (P10, T) - neighbor lat error
    lon_error_nei = lon_error[1:, :]  # (P10, T) - neighbor lon error

    # 角度誤差
    angle_error = np.abs(prediction_original[..., 2] - all_gt_original[..., 2])  # (1 + P10, T)
    # 角度誤差を[-π, π]の範囲に正規化
    angle_error = np.minimum(angle_error, 2 * np.pi - angle_error)
    angle_error_ego = angle_error[0, :]  # (T,) - ego angle error
    angle_error_nei = angle_error[1:, :]  # (P10, T) - neighbor angle error

    sum_neighbors_future_valid = np.sum(neighbors_future_valid, axis=1) > 0  # (P10)

    loss_nei = loss_nei[sum_neighbors_future_valid]  # (P_valid, T, 4)
    lat_error_nei = lat_error_nei[sum_neighbors_future_valid]  # (P_valid, T)
    lon_error_nei = lon_error_nei[sum_neighbors_future_valid]  # (P_valid, T)
    angle_error_nei = angle_error_nei[sum_neighbors_future_valid]  # (P_valid, T)
    neighbors_future_valid = neighbors_future_valid[sum_neighbors_future_valid]  # (P_valid, T)

    return (
        loss_ego,
        loss_nei,
        neighbors_future_valid,
        lat_error_ego,
        lon_error_ego,
        angle_error_ego,
        lat_error_nei,
        lon_error_nei,
        angle_error_nei,
    )


if __name__ == "__main__":
    args = parse_args()
    validation_path_json = args.validation_path_json
    prediction_results_dir = args.prediction_results_dir

    # Load the validation path JSON
    with open(validation_path_json, "r") as f:
        validation_paths = json.load(f)

    prediction_result_paths = sorted(prediction_results_dir.glob("*.npz"))

    assert len(validation_paths) == len(prediction_result_paths)

    ave_loss_ego = 0.0
    ave_loss_nei = 0.0
    num_loss_nei = 0

    # 個別誤差の累積値
    ave_lat_error_ego = 0.0
    ave_lon_error_ego = 0.0
    ave_angle_error_ego = 0.0
    ave_lat_error_nei = 0.0
    ave_lon_error_nei = 0.0
    ave_angle_error_nei = 0.0

    for validation_path, prediction_result_path in zip(validation_paths, prediction_result_paths):
        validation_data = np.load(validation_path)
        prediction_result = np.load(prediction_result_path)["prediction"]

        (
            loss_ego,
            loss_nei,
            neighbors_future_valid,
            lat_error_ego,
            lon_error_ego,
            angle_error_ego,
            lat_error_nei,
            lon_error_nei,
            angle_error_nei,
        ) = calc_loss(validation_data, prediction_result)

        print(
            f"Validation Path: {validation_path}, "
            f"Prediction Result Path: {prediction_result_path}, "
            f"Loss Ego: {loss_ego.mean():.4f}, Loss Nei: {loss_nei.mean():.4f} {loss_nei.shape=}, "
            f"Lat Error Ego: {lat_error_ego.mean():.4f}, Lon Error Ego: {lon_error_ego.mean():.4f}, "
            f"Angle Error Ego: {angle_error_ego.mean():.4f}"
        )

        # ego
        loss_ego_mean = loss_ego.mean()
        ave_loss_ego += loss_ego_mean

        # ego個別誤差
        ave_lat_error_ego += lat_error_ego.mean()
        ave_lon_error_ego += lon_error_ego.mean()
        ave_angle_error_ego += angle_error_ego.mean()

        # neighbor
        if loss_nei.shape[0] > 0:
            curr_num = loss_nei.shape[0]
            loss_nei_valid = loss_nei[neighbors_future_valid]
            loss_nei_mean = loss_nei_valid.mean() * curr_num
            ave_loss_nei += loss_nei_mean
            num_loss_nei += curr_num

            # neighbor個別誤差
            lat_error_nei_valid = lat_error_nei[neighbors_future_valid]
            lon_error_nei_valid = lon_error_nei[neighbors_future_valid]
            angle_error_nei_valid = angle_error_nei[neighbors_future_valid]

            ave_lat_error_nei += lat_error_nei_valid.mean() * curr_num
            ave_lon_error_nei += lon_error_nei_valid.mean() * curr_num
            ave_angle_error_nei += angle_error_nei_valid.mean() * curr_num

    ave_loss_ego /= len(validation_paths)
    ave_loss_nei /= num_loss_nei

    # 平均誤差を計算
    ave_lat_error_ego /= len(validation_paths)
    ave_lon_error_ego /= len(validation_paths)
    ave_angle_error_ego /= len(validation_paths)
    ave_lat_error_nei /= num_loss_nei
    ave_lon_error_nei /= num_loss_nei
    ave_angle_error_nei /= num_loss_nei

    print(f"Average Loss Ego: {ave_loss_ego:.4f}, Average Loss Nei: {ave_loss_nei:.4f}")
    print(
        f"Average Lat Error Ego: {ave_lat_error_ego:.4f}, Average Lon Error Ego: {ave_lon_error_ego:.4f}, Average Angle Error Ego: {ave_angle_error_ego:.4f}"
    )
    print(
        f"Average Lat Error Nei: {ave_lat_error_nei:.4f}, Average Lon Error Nei: {ave_lon_error_nei:.4f}, Average Angle Error Nei: {ave_angle_error_nei:.4f}"
    )
