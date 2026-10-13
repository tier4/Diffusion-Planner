import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from scenario_generation.canvas import Figure
from scenario_generation.colormaps import qualitative

# カラーパレットを定義
colors = [qualitative("tab10", i) for i in range(10)]  # 10色のカラーパレット

parser = argparse.ArgumentParser()
parser.add_argument("target_dir2", type=Path)
args = parser.parse_args()

target_dir1 = Path("/mnt/nvme0/sakoda/test/baseline_ver38")
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
    fig = Figure((12, 9), dpi=100)  # デフォルト(8, 6)から(8, 12)に変更
    ax_py, ax_cpp = (row[0] for row in fig.grid(2, 1, margins=(0.06, 0.7, 0.05, 0.95)))

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
            print(max_diff)
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
            RANGE = 50
            for ax, nonzero, name, title in (
                (ax_py, nonzero1, "py", "Python"),  # 上側: Python
                (ax_cpp, nonzero2, "cpp", "C++"),  # 下側: C++
            ):
                ax.lines([nonzero[:, :2]], color, label=f"{name}_{key}_{i}")
                ax.markers(nonzero[:, :2], marker, size=6, color=color)
                ax.set_xlim(-RANGE, RANGE)
                ax.set_ylim(-RANGE, RANGE)
                ax.legend("upper right")
                ax.title(title)

    fig.save(save_dir / f"{itr:08d}.png")
    print(save_dir / f"{itr:08d}.png")

all_ok = True
for key, val in result_map.items():
    total_num = len(val)
    ok_num = sum(val)
    ok_ratio = ok_num / total_num if total_num > 0 else 0
    if not (ok_ratio == 1.0 or "agent" in key):
        print(f"{key} has NG results {ok_ratio:.4f}")
        all_ok = False

assert all_ok, "Some keys have NG results"
print("All keys are OK")
