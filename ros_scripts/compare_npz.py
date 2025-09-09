import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("target_dir2", type=Path)
args = parser.parse_args()

target_dir1 = Path("/mnt/nvme0/sakoda/test/20250828_131552_test_parse_rosbag_py")
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
    for key in npz1.keys():
        if key == "map_name" or key == "token":
            continue
        diff = np.abs(npz1[key].astype(np.float32) - npz2[key].astype(np.float32))
        max_diff = np.max(diff)
        judge = "NG" if max_diff > 1e-5 else "OK"
        print(judge, key, max_diff, npz1[key].shape, npz2[key].shape)
        result_map[key].append(max_diff < 1e-5)

        # is_there_diff = False
        # for i in range(diff.shape[0]):
        #     if np.any(diff[i, 0, 8:] > 1e-5):
        #         is_there_diff = True
        # for i in range(diff.shape[0]):
        #     if is_there_diff:
        #         print("  ", i, diff[i, 0, 8:], npz1[key][i, 0, 8:], npz2[key][i, 0, 8:])
        # if is_there_diff:
        #     a = input()

for key, val in result_map.items():
    total_num = len(val)
    ok_num = sum(val)
    ok_ratio = ok_num / total_num if total_num > 0 else 0
    print(f"{key}: {ok_num}/{total_num} = {ok_ratio:.4f}")
