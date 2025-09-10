import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("target_dir2", type=Path)
args = parser.parse_args()

target_dir1 = Path("/mnt/nvme0/sakoda/test/baseline_test_parse_rosbag_py")
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
        diff = np.abs(data1.astype(np.float32) - npz2[key].astype(np.float32))
        max_diff = np.max(diff)
        judge = "NG" if max_diff > 1e-5 else "OK"
        print(judge, key, max_diff, data1.shape, npz2[key].shape)
        result_map[key].append(max_diff < 1e-5)

        num_segment, num_points, num_features = data1.shape
        for i in range(num_segment):
            for j in range(num_points):
                for k in range(num_features):
                    if diff[i, j, k] < 1e-1:
                        continue
                    print(
                        f"  {i:02d}, {j:02d}, {k:02d}: {data1[i, j, k]:.3f} -> {npz2[key][i, j, k]:.3f} (diff: {diff[i, j, k]:.3f})"
                    )
            plt.plot(data1[i, :, 0], data1[i, :, 1], "o-", label="data1")
            plt.plot(npz2[key][i, :, 0], npz2[key][i, :, 1], "x--", label="data2")
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
