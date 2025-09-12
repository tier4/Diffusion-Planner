import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

parser = argparse.ArgumentParser()
parser.add_argument("target_dir2", type=Path)
args = parser.parse_args()

target_dir1 = Path("/mnt/nvme0/sakoda/test/baseline_test_parse_rosbag_20250612_101935_py")
target_dir2 = args.target_dir2

npz_list1 = sorted(target_dir1.glob("**/*.npz"))
npz_list2 = sorted(target_dir2.glob("**/*.npz"))

filename_list1 = [f.name for f in npz_list1]
filename_list2 = [f.name for f in npz_list2]
print(filename_list1[0:5])
print(filename_list2[0:5])

and_list = sorted(set(filename_list1) & set(filename_list2))
print(len(and_list))

and_list = and_list[::20]

npz_list1 = [f for f in npz_list1 if f.name in and_list]
npz_list2 = [f for f in npz_list2 if f.name in and_list]

result_map = defaultdict(list)

for f1, f2 in zip(npz_list1, npz_list2):
    npz1 = np.load(f1)
    npz2 = np.load(f2)

    print(f"\n{f1}, {f2}")
    save_dir = f2.parent.parent / "compare"
    save_dir.mkdir(exist_ok=True, parents=True)
    for key in npz1.keys():
        if key == "map_name" or key == "token" or key != "lanes":
            continue
        data1 = npz1[key]
        data2 = npz2[key]
        diff = np.abs(data1.astype(np.float32) - data2.astype(np.float32))
        max_diff = np.max(diff)
        judge = "NG" if max_diff > 1e-5 else "OK"
        print(judge, key, max_diff, data1.shape, data2.shape)
        result_map[key].append(max_diff < 1e-5)

        num_segment, num_points, num_features = data1.shape

        # セグメント間の平均距離を計算（より効率的な方法）
        distance = np.zeros((num_segment, num_segment))

        for i1 in range(num_segment):
            for i2 in range(num_segment):
                # 各セグメント間のすべてのポイント間距離の最小値を計算
                segment1 = data1[i1]  # shape: (num_points, num_features)
                segment2 = data2[i2]  # shape: (num_points, num_features)

                # ブロードキャストを使用して効率的に距離行列を計算
                diff = (
                    segment1[:, np.newaxis, :2] - segment2[np.newaxis, :, :2]
                )  # (num_points, num_points, num_features)
                point_distances = np.linalg.norm(diff, axis=2)  # (num_points, num_points)

                # セグメント間の距離として最小値を使用
                distance[i1, i2] = np.mean(point_distances)

        # 最小二部マッチング
        row_ind, col_ind = linear_sum_assignment(distance)

        for i in range(num_segment):
            idx1 = row_ind[i]
            idx2 = col_ind[i]
            for j in range(num_points):
                for k in range(num_features):
                    curr_diff = data1[idx1, j, k] - data2[idx2, j, k]
                    if curr_diff < 1e-1:
                        continue
                    print(
                        f"  {i:02d}, {j:02d}, {k:02d}: {data1[idx1, j, k]:.3f} -> {data2[idx2, j, k]:.3f} (diff: {curr_diff:.3f})"
                    )
            plt.plot(data1[idx1, :, 0], data1[idx1, :, 1], "o-", label="py")
            plt.plot(data2[idx2, :, 0], data2[idx2, :, 1], "x--", label="cpp")
            plt.legend()
            plt.savefig(save_dir / f"{key}_segment_{i:02d}.png")
            print(save_dir / f"{key}_segment_{i:02d}.png")
            plt.clf()

    break

for key, val in result_map.items():
    total_num = len(val)
    ok_num = sum(val)
    ok_ratio = ok_num / total_num if total_num > 0 else 0
    print(f"{key}: {ok_num}/{total_num} = {ok_ratio:.4f}")
