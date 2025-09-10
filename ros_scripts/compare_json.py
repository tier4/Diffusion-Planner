import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("target_dir2", type=Path)
args = parser.parse_args()

target_dir1 = Path("/mnt/nvme0/sakoda/test/baseline_test_parse_rosbag_py")
target_dir2 = args.target_dir2

json_list1 = sorted(target_dir1.glob("**/*.json"))
json_list2 = sorted(target_dir2.glob("**/*.json"))

filename_list1 = [f.name for f in json_list1]
filename_list2 = [f.name for f in json_list2]
print(filename_list1[0:5])
print(filename_list2[0:5])

and_list = sorted(set(filename_list1) & set(filename_list2))
print(len(and_list))

and_list = and_list[::20]

json_list1 = [f for f in json_list1 if f.name in and_list]
json_list2 = [f for f in json_list2 if f.name in and_list]

result_map = defaultdict(list)

for f1, f2 in zip(json_list1, json_list2):
    json1 = json.load(open(f1))
    json2 = json.load(open(f2))
    for key in json1.keys():
        diff = np.abs(np.array(json1[key]) - np.array(json2[key]))
        print(f"{key}: {diff}")
