import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

# カラーパレットを定義
colors = plt.cm.tab10.colors  # 10色のカラーパレット

parser = argparse.ArgumentParser()
parser.add_argument("target_dir2", type=Path)
args = parser.parse_args()

target_dir1 = Path("/mnt/nvme0/sakoda/test/baseline_test_parse_rosbag_py_2")
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

itr = 0
for f1, f2 in zip(npz_list1, npz_list2):
    npz1 = np.load(f1)
    npz2 = np.load(f2)

    print(f"\n{f1}, {f2}")
    save_dir = f2.parent.parent / "compare"
    save_dir.mkdir(exist_ok=True, parents=True)
    itr += 1
    plt.figure(figsize=(12, 9))  # デフォルト(8, 6)から(8, 12)に変更

    for key in npz1.keys():
        if key == "map_name" or key == "token":
            continue
        # if key != "neighbor_agents_past" and key != "neighbor_agents_future":
        #     continue
        data1 = npz1[key]
        data2 = npz2[key]
        diff = np.abs(data1.astype(np.float32) - data2.astype(np.float32))
        max_diff = np.max(diff)
        judge = "NG" if max_diff > 1e-5 else "OK"
        if "agent" not in key and judge == "NG":
            print(key)
            print(data1)
            print(data2)
            print(diff)
            exit(1)
        result_map[key].append(max_diff < 1e-5)
        continue

        num_agents, num_timesteps, num_features = data1.shape

        # エージェントTrajectory間の平均距離を計算
        distance = np.zeros((num_agents, num_agents))

        for i1 in range(num_agents):
            for i2 in range(num_agents):
                # 各エージェント間のすべてのポイント間距離の最小値を計算
                segment1 = data1[i1]  # shape: (num_timesteps, num_features)
                segment2 = data2[i2]  # shape: (num_timesteps, num_features)

                # ブロードキャストを使用して効率的に距離行列を計算
                diff = (
                    segment1[:, np.newaxis, :2] - segment2[np.newaxis, :, :2]
                )  # (num_timesteps, num_timesteps, num_features)
                point_distances = np.linalg.norm(diff, axis=2)  # (num_timesteps, num_timesteps)

                # エージェント間の距離として平均を使用
                distance[i1, i2] = np.mean(point_distances)

        # 最小二部マッチング
        # row_ind, col_ind = linear_sum_assignment(distance)

        for i in range(num_agents):
            # idx1 = row_ind[i]
            # idx2 = col_ind[i]
            curr_data1 = data1[i]
            curr_data2 = data2[i]
            for j in range(num_timesteps):
                for k in range(num_features):
                    curr_diff = curr_data1[j, k] - curr_data2[j, k]
                    if curr_diff < 1e-1:
                        continue
                    print(
                        f"  {i:02d}, {j:02d}, {k:02d}: {curr_data1[j, k]:.3f} -> {curr_data2[j, k]:.3f} (diff: {curr_diff:.3f})"
                    )
            nonzero1 = curr_data1[curr_data1[:, 0] != 0]
            nonzero2 = curr_data2[curr_data2[:, 0] != 0]
            if len(nonzero1) == 0 and len(nonzero2) == 0:
                break

            # カラーパレットから色を選択 (indexに基づく)
            color = colors[i % len(colors)]

            # マーカーを決定 (pastは'o', futureは'x')
            marker = "o" if "past" in key else "x"

            # 上下に分けてプロット
            plt.subplot(2, 1, 1)  # 上側: Python
            plt.plot(
                nonzero1[:, 0],
                nonzero1[:, 1],
                marker=marker,
                color=color,
                label=f"py_{key}_{i}",
            )
            RANGE = 50
            plt.xlim(-RANGE, RANGE)
            plt.ylim(-RANGE, RANGE)
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=2)
            plt.title("Python")

            plt.subplot(2, 1, 2)  # 下側: C++
            plt.plot(
                nonzero2[:, 0],
                nonzero2[:, 1],
                marker=marker,
                color=color,
                label=f"cpp_{key}_{i}",
            )
            plt.xlim(-RANGE, RANGE)
            plt.ylim(-RANGE, RANGE)
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=2)
            plt.title("C++")

    # 全体的なレイアウト調整
    plt.tight_layout()
    plt.savefig(save_dir / f"{itr:08d}.png", bbox_inches="tight")
    print(save_dir / f"{itr:08d}.png")
    plt.clf()

for key, val in result_map.items():
    total_num = len(val)
    ok_num = sum(val)
    ok_ratio = ok_num / total_num if total_num > 0 else 0
    assert ok_ratio == 1.0 or "agent" in key
    print(f"{key}: {ok_num}/{total_num} = {ok_ratio:.4f}")
